import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (Pre-Norm)."""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

class FeedForward(nn.Module):
    """Position-wise Feed-Forward Network with SwiGLU non-linearity."""
    def __init__(self, dim: int, hidden_dim: int, multiple_of: int = 256):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

class AttentionNoPos(nn.Module):
    """Multi-Head Causal Self-Attention with NO positional embeddings (Ablated).
    
    In this ablated module, neither Rotary Position Embeddings (RoPE) nor learned
    absolute positional embeddings are applied. The queries and keys depend purely
    on token semantic content, rendering past tokens permutation-invariant except
    for the causal attention mask constraint.
    """
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
        self.output_attentions = False

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None):
        bsz, seqlen, _ = x.shape
        
        # Project inputs to Q, K, V without positional rotation (No RoPE)
        xq = self.wq(x)
        xk = self.wk(x)
        xv = self.wv(x)

        # Reshape into multiple heads: (bsz, seqlen, n_heads, head_dim)
        xq = xq.view(bsz, seqlen, self.n_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_heads, self.head_dim)

        # Transpose to (bsz, n_heads, seqlen, head_dim)
        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(xq, xk.transpose(2, 3)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores + mask
            
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        attn_weights = scores.clone().detach() if self.output_attentions else None
        
        scores = self.attn_dropout(scores)
        
        output = torch.matmul(scores, xv)  # (bsz, n_heads, seqlen, head_dim)
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        
        if self.output_attentions:
            return self.resid_dropout(self.wo(output)), attn_weights
        return self.resid_dropout(self.wo(output))

class TransformerBlockNoPos(nn.Module):
    """Transformer decoder block without positional embeddings."""
    def __init__(self, layer_id: int, dim: int, n_heads: int, hidden_dim: int):
        super().__init__()
        self.layer_id = layer_id
        self.attention = AttentionNoPos(dim, n_heads)
        self.feed_forward = FeedForward(dim, hidden_dim)
        self.attention_norm = RMSNorm(dim)
        self.ffn_norm = RMSNorm(dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None):
        if self.attention.output_attentions:
            attn_out, attn_weights = self.attention(self.attention_norm(x), mask=mask)
            h = x + attn_out
            out = h + self.feed_forward(self.ffn_norm(h))
            return out, attn_weights
        else:
            h = x + self.attention(self.attention_norm(x), mask=mask)
            out = h + self.feed_forward(self.ffn_norm(h))
            return out

class LanguageModelNoPos(nn.Module):
    """Decoder-only Transformer Language Model with No Positional Embeddings.
    
    Exact same architecture, dimensions, normalization, SwiGLU activations,
    and weight tying as the standard Phase 2 model, but completely lacking
    positional information.
    """
    def __init__(
        self,
        vocab_size: int = 32000,
        dim: int = 512,
        n_layers: int = 6,
        n_heads: int = 8,
        hidden_dim: int = 2048,
        max_seq_len: int = 512,
        dropout: float = 0.1
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.max_seq_len = max_seq_len

        # Token embeddings only — NO positional embeddings added
        self.tok_embeddings = nn.Embedding(vocab_size, dim)
        self.drop = nn.Dropout(dropout)
        
        self.layers = nn.ModuleList([
            TransformerBlockNoPos(i, dim, n_heads, hidden_dim) 
            for i in range(n_layers)
        ])
        
        self.norm = RMSNorm(dim)
        self.output = nn.Linear(dim, vocab_size, bias=False)
        
        # Explicit weight tying between input embedding and output linear head
        self.tok_embeddings.weight = self.output.weight

        # Precompute causal mask (upper triangular -inf)
        mask = torch.full((max_seq_len, max_seq_len), float("-inf"))
        mask = torch.triu(mask, diagonal=1)
        self.register_buffer("causal_mask", mask)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor, targets: torch.Tensor = None, output_attentions: bool = False):
        bsz, seqlen = x.shape
        assert seqlen <= self.max_seq_len, f"Sequence length {seqlen} exceeds max_seq_len {self.max_seq_len}"

        # Embedding lookup without positional addition
        h = self.drop(self.tok_embeddings(x))
        
        # Causal mask for the current sequence length
        mask = self.causal_mask[:seqlen, :seqlen].unsqueeze(0).unsqueeze(0)  # (1, 1, seqlen, seqlen)

        attentions = []
        for layer in self.layers:
            layer.attention.output_attentions = output_attentions
            if output_attentions:
                h, attn_w = layer(h, mask=mask)
                attentions.append(attn_w)
            else:
                h = layer(h, mask=mask)

        h = self.norm(h)
        logits = self.output(h)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, self.vocab_size), targets.view(-1))

        if output_attentions:
            return logits, loss, attentions
        return logits, loss

    def get_num_params(self, non_embedding=False):
        """Return total trainable parameters."""
        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        if non_embedding:
            n_params -= self.tok_embeddings.weight.numel()
        return n_params
