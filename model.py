import math
import struct
import inspect
from dataclasses import dataclass
from typing import Any, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

@dataclass
class ModelArgs:
    # default hyperparameters for the Llama 7B model
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: Optional[int] = None
    vocab_size: int = 32000
    hidden_dim: Optional[int] = 200
    multiple_of: int = 256  # MLP hidden layer size will be multiple of
    norm_eps: float = 1e-5
    max_seq_len: int = 2048
    dropout: float = 0.0

    # 添加你的模型所需的超参数
    num_tasks: int = 2
    task_dim: int = 32
    num_experts: int = 32
    diversity_weight: float = 0.1
    attribute_dim: int = num_tasks * task_dim
    topk: int = 2
    intermediate_dim: int = 628


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    freqs_cos = torch.cos(freqs)  # real part
    freqs_sin = torch.sin(freqs)  # imaginary part
    return freqs_cos, freqs_sin

def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(shape)

def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cos: torch.Tensor,
    freqs_sin: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:

    # reshape xq and xk to match the complex representation
    xq_r, xq_i = xq.float().reshape(xq.shape[:-1] + (-1, 2)).unbind(-1)
    xk_r, xk_i = xk.float().reshape(xk.shape[:-1] + (-1, 2)).unbind(-1)

    # reshape freqs_cos and freqs_sin for broadcasting
    freqs_cos = reshape_for_broadcast(freqs_cos, xq_r)
    freqs_sin = reshape_for_broadcast(freqs_sin, xq_r)

    # apply rotation using real numbers
    xq_out_r = xq_r * freqs_cos - xq_i * freqs_sin
    xq_out_i = xq_r * freqs_sin + xq_i * freqs_cos
    xk_out_r = xk_r * freqs_cos - xk_i * freqs_sin
    xk_out_i = xk_r * freqs_sin + xk_i * freqs_cos

    # flatten last two dimensions
    xq_out = torch.stack([xq_out_r, xq_out_i], dim=-1).flatten(3)
    xk_out = torch.stack([xk_out_r, xk_out_i], dim=-1).flatten(3)

    return xq_out.type_as(xq), xk_out.type_as(xk)

def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )
    
class TaskEncoder(nn.Module):
    def __init__(self, hidden_dim, num_tasks, task_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_tasks = num_tasks
        self.task_dim = task_dim
        
        self.self_attn = nn.MultiheadAttention(hidden_dim, num_heads=4)
        self.task_division = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_tasks)
        )
        self.task_encoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, task_dim)
            ) for _ in range(num_tasks)
        ])
        
    def forward(self, input_seq):
                # input_seq: (batch_size, seq_len, hidden_dim)
        attn_output, _ = self.self_attn(input_seq, input_seq, input_seq)
                # attn_output: (batch_size, seq_len, hidden_dim)
        task_probs = F.softmax(self.task_division(attn_output), dim=-1)
                # task_probs: (batch_size, seq_len, num_tasks)
        task_embeddings = []
        for i in range(self.num_tasks):
            task_input = attn_output * task_probs[:, :, i].unsqueeze(-1)
                        # task_input: (batch_size, seq_len, hidden_dim)
            task_embeddings.append(self.task_encoders[i](task_input))
                        # task_embeddings[i]: (batch_size, seq_len, task_dim)
        task_embeddings = torch.stack(task_embeddings, dim=1)
                # task_embeddings: (batch_size, num_tasks, seq_len, task_dim)    
        task_embeddings = task_embeddings.transpose(1, 2)
                # task_embeddings: (batch_size, seq_len, num_tasks, task_dim)
        return task_embeddings

class TaskAwareRouter(nn.Module):
    def __init__(self, hidden_dim, num_experts, num_tasks, task_dim, topk):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.num_tasks = num_tasks
        self.task_dim = task_dim
        self.topk = topk
        self.attribute_proj = nn.Parameter(torch.randn(task_dim, num_experts))
        
        self.input_proj = nn.Linear(hidden_dim + task_dim * num_tasks, hidden_dim * 4)
        self.intermediate_proj = nn.Linear(hidden_dim * 4, hidden_dim)
        self.expert_proj = nn.Parameter(torch.randn(num_experts, hidden_dim))
        self.router = nn.Sequential(
            nn.Linear(hidden_dim, num_experts),
            nn.Softmax(dim=-1)
        )
        
    def forward(self, x, task_embeddings):
        input_vec = torch.cat([x, task_embeddings.reshape(x.size(0), x.size(1), -1)], dim=-1)
        input_vec = F.relu(self.input_proj(input_vec))
        input_vec = F.relu(self.intermediate_proj(input_vec))

        expert_probs = self.router(input_vec)

        attribute_scores = torch.matmul(task_embeddings, self.attribute_proj)
        attribute_probs = torch.softmax(attribute_scores, dim=-1)  
        attribute_probs = attribute_probs.mean(dim=2)              

        expert_probs = expert_probs * attribute_probs

        _, topk_indices = torch.topk(expert_probs, k=self.topk, dim=-1)
        mask = torch.zeros_like(expert_probs).scatter_(-1, topk_indices, 1.0)
        expert_probs = expert_probs * mask

        entropy_loss = -torch.mean(torch.sum(expert_probs * torch.log(expert_probs + 1e-8), dim=-1)) 

        return expert_probs, entropy_loss, mask

class MoAExpert(nn.Module):
    def __init__(self, hidden_dim, attribute_dim, intermediate_dim):
        super(MoAExpert, self).__init__()
        self.hidden_dim = hidden_dim
        self.attribute_dim = attribute_dim
        self.intermediate_dim = intermediate_dim
        
        self.attribute_embedding = nn.Parameter(torch.randn(1, attribute_dim))
        self.attribute_proj = nn.Linear(attribute_dim, hidden_dim)
        
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim, intermediate_dim),
            nn.ReLU(),
            nn.Linear(intermediate_dim, hidden_dim),
            nn.Sigmoid()
        )
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, intermediate_dim),
            nn.ReLU(),
            nn.Linear(intermediate_dim, hidden_dim)
        )
        
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, x):
        residual = x
        attribute_expanded = self.attribute_embedding.expand(x.size(0), x.size(1), -1)
        attribute_projected = self.attribute_proj(attribute_expanded)
        
        gate_input = torch.cat([x, attribute_projected], dim=-1)
        gate = self.gate(gate_input)
        
        x = x * gate + attribute_projected * (1 - gate)
        x = self.fc(x)
        x = self.layer_norm(x + residual)
        return x

class MoELayer(nn.Module):
    def __init__(self, hidden_dim, num_experts, task_dim, num_tasks, topk, intermediate_dim):
        super(MoELayer, self).__init__()
        self.num_experts = num_experts
        self.experts = nn.ModuleList([MoAExpert(hidden_dim, task_dim * num_tasks, intermediate_dim) for _ in range(num_experts)])
        self.router = TaskAwareRouter(hidden_dim, num_experts, num_tasks, task_dim, topk)
        
    def forward(self, x, task_embeddings):
        expert_probs, diversity_loss, mask = self.router(x, task_embeddings)
        expert_outputs = []
        for i, expert in enumerate(self.experts):
            if mask[:, :, i].any():
                expert_output = expert(x)
            else:
                expert_output = torch.zeros_like(x)
            expert_outputs.append(expert_output)
        expert_outputs = torch.stack(expert_outputs, dim=0)
        
        expert_probs = expert_probs.transpose(1, 2).unsqueeze(-1)
        expert_outputs = expert_outputs.transpose(0, 1)
        final_output = torch.sum(expert_probs * expert_outputs, dim=1)
        expert_probs = expert_probs.squeeze(-1).transpose(1, 2)
        expert_probs = torch.softmax(expert_probs, dim=-1)
        
        return final_output, diversity_loss

class LRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, batch_first=False):
        super(LRU, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.batch_first = batch_first
        self.weight_ih = nn.Parameter(torch.Tensor(hidden_dim, input_dim))
        self.weight_hh = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
        self.bias = nn.Parameter(torch.Tensor(hidden_dim))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_dim)
        for weight in self.parameters():
            nn.init.uniform_(weight, -stdv, stdv)

    def forward(self, input, hx=None):
        if self.batch_first:
            input = input.transpose(0, 1)
        seq_len, batch_size, _ = input.size()
        if hx is None:
            hx = torch.zeros(batch_size, self.hidden_dim, dtype=input.dtype, device=input.device)
        
        hy = []
        for i in range(seq_len):
            hx = F.linear(input[i], self.weight_ih, self.bias) + F.linear(hx, self.weight_hh)
            hy.append(hx)
        
        if self.batch_first:
            hy = torch.stack(hy, dim=1)
        else:
            hy = torch.stack(hy, dim=0)
        return hy
    
class LRUConnector(nn.Module):
    def __init__(self, input_dim, hidden_dim, mem_seq_len):
        super().__init__()
        self.mem_seq_len = mem_seq_len
        self.lru = LRU(input_dim, hidden_dim, batch_first=True)

    def forward(self, x):
        # x: (batch_size, seq_len, input_dim)
        batch_size, seq_len, _ = x.shape
        
        # 确保序列可以被 mem_seq_len 整除
        padded_len = ((seq_len - 1) // self.mem_seq_len + 1) * self.mem_seq_len
        padded_x = F.pad(x, (0, 0, 0, padded_len - seq_len))
        
        # 重塑为 (batch_size, num_chunks, mem_seq_len, input_dim)
        x_reshaped = padded_x.view(batch_size, -1, self.mem_seq_len, x.size(2))
        
        # 合并 batch_size 和 num_chunks 维度以适应 lru
        x_merged = x_reshaped.view(-1, self.mem_seq_len, x.size(2))
        
        # 通过 lru
        output = self.lru(x_merged)
        
        # 恢复原始的维度结构
        output = output.view(batch_size, -1, self.mem_seq_len, output.size(2))
        
        # 移除填充
        output = output[:, :, :seq_len, :]
        
        return output.reshape(batch_size, seq_len, -1)

class Transformer(nn.Module):
    def __init__(self, params: ModelArgs):
        super().__init__()
        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers

        self.tok_embeddings = nn.Embedding(params.vocab_size, params.dim)
        self.task_encoder = TaskEncoder(params.dim, params.num_tasks, params.task_dim)
        self.dropout = nn.Dropout(params.dropout)
        
        self.moe_layers = nn.ModuleList([
            MoELayer(params.dim, params.num_experts, params.task_dim, params.num_tasks, params.topk, params.intermediate_dim) for _ in range(params.n_layers)
        ])
        self.norm = RMSNorm(params.dim, eps=1e-5)
        self.output = nn.Linear(params.dim, params.vocab_size, bias=False)

        self.tok_embeddings.weight = self.output.weight
        self.apply(self._init_weights)
        self.last_loss = None
        self.lru_connector = LRUConnector(params.dim, params.dim, mem_seq_len=256)  # 假设 mem_seq_len=512

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, tokens: torch.Tensor, targets: Optional[torch.Tensor] = None) -> torch.Tensor:
                # tokens: (batch_size, seq_len)
        h = self.tok_embeddings(tokens)
        h = self.lru_connector(h)  # 应用LRU连接器处理
        task_embeddings = self.task_encoder(h)
                # h: (batch_size, seq_len, hidden_dim)
        h = self.dropout(h)
                # h: (batch_size, seq_len, hidden_dim)


        loss = 0
        for layer in self.moe_layers:
            h, diversity_loss = layer(h, task_embeddings)
                        # h: (batch_size, seq_len, hidden_dim)
            
            loss += diversity_loss
        h = self.norm(h)
                # h: (batch_size, seq_len, hidden_dim)
        
        if targets is not None:
            logits = self.output(h)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
            loss += self.params.diversity_weight * diversity_loss
            self.last_loss = loss
        else:
            logits = self.output(h[:, [-1], :])
            self.last_loss = None
        return logits

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """ estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS """
        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        N = sum(p.numel() for p in self.parameters())
        cfg = self.params
        L, H, Q, T = cfg.n_layers, cfg.n_heads, cfg.dim//cfg.n_heads, cfg.max_seq_len
        flops_per_token = 6*N + 12*L*H*Q*T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # express our flops throughput as ratio of A100 bfloat16 peak flops
        flops_achieved = flops_per_iter * (1.0/dt) # per second
        flops_promised = 312e12 # A100 GPU bfloat16 peak flops is 312 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu

    @torch.inference_mode()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        Also note this is a super inefficient version of sampling with no key/value cache.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.params.max_seq_len else idx[:, -self.params.max_seq_len:]
            # forward the model to get the logits for the index in the sequence
            logits = self(idx_cond)
            logits = logits[:, -1, :] # crop to just the final time step
            if temperature == 0.0:
                # "sample" the single most likely index
                _, idx_next = torch.topk(logits, k=1, dim=-1)
            else:
                # pluck the logits at the final step and scale by desired temperature
                logits = logits / temperature
                # optionally crop the logits to only the top k options
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float('Inf')
                # apply softmax to convert logits to (normalized) probabilities
                probs = F.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx
