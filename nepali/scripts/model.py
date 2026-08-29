import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
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
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis

def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)

def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor):
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)

class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, multiple_of: int = 256):
        super().__init__()
        # SwiGLU activation
        hidden_dim = int(2 * hidden_dim / 3)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

class Attention(nn.Module):
    def __init__(self, dim: int, n_heads: int):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        
        self.wq = nn.Linear(dim, dim, bias=False)
        self.wk = nn.Linear(dim, dim, bias=False)
        self.wv = nn.Linear(dim, dim, bias=False)
        self.wo = nn.Linear(dim, dim, bias=False)
        
        self.attn_dropout = nn.Dropout(0.1)
        self.resid_dropout = nn.Dropout(0.1)

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor, mask: torch.Tensor):
        bsz, seqlen, _ = x.shape
        
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.view(bsz, seqlen, self.n_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_heads, self.head_dim)

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        xq = xq.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)
        
        scores = torch.matmul(xq, xk.transpose(2, 3)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores + mask  # (bs, n_local_heads, seqlen, cache_len + seqlen)
            
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        attn_weights = scores.clone().detach() if hasattr(self, 'output_attentions') and self.output_attentions else None
        
        scores = self.attn_dropout(scores)
        
        output = torch.matmul(scores, xv)  # (bs, n_local_heads, seqlen, head_dim)
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        
        if hasattr(self, 'output_attentions') and self.output_attentions:
            return self.resid_dropout(self.wo(output)), attn_weights
        return self.resid_dropout(self.wo(output))

class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, dim: int, n_heads: int, hidden_dim: int):
        super().__init__()
        self.n_heads = n_heads
        self.dim = dim
        self.head_dim = dim // n_heads
        
        self.attention = Attention(dim, n_heads)
        self.feed_forward = FeedForward(dim, hidden_dim)
        
        self.attention_norm = RMSNorm(dim)
        self.ffn_norm = RMSNorm(dim)

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor, mask: torch.Tensor, output_attentions: bool = False):
        self.attention.output_attentions = output_attentions
        attn_out = self.attention(self.attention_norm(x), freqs_cis, mask)
        attn_weights = None
        if output_attentions:
            attn_out, attn_weights = attn_out
            
        h = x + attn_out
        out = h + self.feed_forward(self.ffn_norm(h))
        
        if output_attentions:
            return out, attn_weights
        return out

class LanguageModel(nn.Module):
    def __init__(self, vocab_size: int, dim: int = 512, n_layers: int = 6, n_heads: int = 8, hidden_dim: int = 2048, max_seq_len: int = 1024):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.n_layers = n_layers
        self.max_seq_len = max_seq_len

        self.tok_embeddings = nn.Embedding(vocab_size, dim)
        self.dropout = nn.Dropout(0.1)

        self.layers = nn.ModuleList()
        for layer_id in range(n_layers):
            self.layers.append(TransformerBlock(layer_id, dim, n_heads, hidden_dim))

        self.norm = RMSNorm(dim)
        self.output = nn.Linear(dim, vocab_size, bias=False)
        
        # Tie weights
        self.tok_embeddings.weight = self.output.weight

        self.freqs_cis = precompute_freqs_cis(self.dim // n_heads, self.max_seq_len * 2)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, tokens: torch.Tensor, targets: torch.Tensor = None, output_attentions: bool = False):
        _bsz, seqlen = tokens.shape
        h = self.tok_embeddings(tokens)
        h = self.dropout(h)
        
        freqs_cis = self.freqs_cis[:seqlen].to(h.device)

        mask = None
        if seqlen > 1:
            mask = torch.full((seqlen, seqlen), float("-inf"), device=tokens.device)
            mask = torch.triu(mask, diagonal=1)
            mask = mask.view(1, 1, seqlen, seqlen)

        all_attentions = []
        for layer in self.layers:
            if output_attentions:
                h, attn = layer(h, freqs_cis, mask, output_attentions=True)
                all_attentions.append(attn)
            else:
                h = layer(h, freqs_cis, mask, output_attentions=False)
            
        h = self.norm(h)
        logits = self.output(h)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            
        if output_attentions:
            return logits, loss, all_attentions
        return logits, loss

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters())
