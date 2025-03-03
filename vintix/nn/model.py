import math
import warnings
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn

from vintix.nn.kv_cache import KVCache
from vintix.nn.nn import (GptNeoxMLP, LLaMAMLP,
                          get_alibi_slopes, get_mlp_class,
                          get_norm_class)

try:
    import flash_attn
except ImportError:
    warnings.warn("Missing FlashAttention Install", category=Warning)


class FlashAliBiCausalSelfAttention(nn.Module):
    """Casual attention layer with ALiBi and FlashAttention

    Args:
        hidden_dim: hidden dim of transformer
        num_heads: number of heads in attention
        dropout: dropout percentage
        normalize_qk: use qk normalization
        bias: use bias in Linear
        """

    def __init__(self,
                 hidden_dim: int,
                 num_heads: int,
                 dropout: float = 0.0,
                 normalize_qk: bool = False,
                 bias: bool = True):
        super().__init__()
        self.shape = (hidden_dim // num_heads) * num_heads
        self.in_proj = nn.Linear(hidden_dim,
                                 3 * self.shape,
                                 bias=bias)
        self.out_proj = nn.Linear(self.shape,
                                  hidden_dim,
                                  bias=bias)
        self.register_buffer(
            "alibi_slopes",
            torch.as_tensor(get_alibi_slopes(num_heads)),
            persistent=False,
        )
        if normalize_qk:
            self.q_norm = nn.LayerNorm(hidden_dim // num_heads)
            self.k_norm = nn.LayerNorm(hidden_dim // num_heads)

        self.dropout = dropout
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.normalize_qk = normalize_qk
        self.bias = bias

    def forward(
        self,
        x: torch.Tensor,
        k_cache: Optional[torch.Tensor] = None,
        v_cache: Optional[torch.Tensor] = None,
        cache_seqlens: Optional[int] = None
    ) -> torch.Tensor:
        """Forward pass

        Args:
            x: input (batch_size, seq_len, hidden_dim)
            k_cache: cache of keys for inference
            v_cache: cache of values for inference
            cache_seqlens: length of cache

        Returns:
            torch.Tensor: output (batch_size, seq_len, hidden_dim)
        """
        B, L, D = x.size()
        # (batch_size, seq_len, 3, num_heads, head_dim)
        qkv = self.in_proj(x).reshape(
            B, L, 3, self.num_heads, D // self.num_heads)

        q, k, v = qkv.unbind(2)

        # normalizing q,k, see: https://arxiv.org/abs/2302.05442
        if self.normalize_qk:
            q, k, v = qkv.unbind(2)
            q_norm, k_norm = self.q_norm(q), self.k_norm(k)
            qkv = torch.stack([q_norm, k_norm, v], dim=2).to(qkv.dtype)

        # (batch_size, seq_len, num_heads, head_dim)
        if k_cache is None or v_cache is None or cache_seqlens is None:
            out = flash_attn.flash_attn_qkvpacked_func(
                qkv=qkv,
                dropout_p=self.dropout if self.training else 0.0,
                causal=True,
                alibi_slopes=self.alibi_slopes,
            )
        else:
            assert not self.training
            q, k, v = qkv.unbind(2)
            out = flash_attn.flash_attn_with_kvcache(
                q=q,
                k=k,
                v=v,
                k_cache=k_cache,
                v_cache=v_cache,
                cache_seqlens=cache_seqlens,
                causal=True,
                alibi_slopes=self.alibi_slopes,
            )
        # (batch_size, seq_len, hidden_dim)
        out = self.out_proj(out.reshape(B, L, self.shape))
        return out


class TransformerBlock(nn.Module):
    """Transformer Block

    Args:
        hidden_dim: hidden dim of transformer
        num_heads: number of heads in attention
        attention_dropout: dropout in attention
        residual_dropout: dropout after attention
        normalize_qk: use qk normalization in attention
        bias: use bias in linear layers
        parallel_residual: use parallel residual (no attention
            in MLP)
        shared_attention_norm: share norm for attention and
            MLP (only if parallel_residual=True)
        norm_class: class for normalization layer (LayerNorm/RMSNorm)
        norm_eps: epsilon for normalization
        mlp_class: class for MLP layer (GptNeoxMLP/LLaMAMLP)
        intermediate_size: projection size in MLP layer
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        attention_dropout: float,
        residual_dropout: float,
        normalize_qk: bool = False,
        bias: bool = True,
        parallel_residual: bool = True,
        shared_attention_norm: bool = False,
        norm_class: str = "LayerNorm",
        norm_eps: float = 1e-5,
        mlp_class: str = "LLaMAMLP",
        intermediate_size: Optional[int] = None,
    ):
        super().__init__()
        self.shared_attention_norm = shared_attention_norm
        self.parallel_residual = parallel_residual
        if intermediate_size is None:
            intermediate_size = 4 * hidden_dim

        norm_class = get_norm_class(norm_class)
        self.norm1 = norm_class(hidden_dim,
                                eps=norm_eps)
        if not self.shared_attention_norm:
            self.norm2 = norm_class(hidden_dim,
                                    eps=norm_eps)

        self.drop = nn.Dropout(residual_dropout)

        self.attention = FlashAliBiCausalSelfAttention(
            hidden_dim,
            num_heads,
            attention_dropout,
            normalize_qk=normalize_qk,
            bias=bias
        )
        mlp_class = get_mlp_class(mlp_class)
        self.mlp = mlp_class(hidden_dim,
                             intermediate_size,
                             bias)

    def forward(
        self,
        x: torch.Tensor,
        k_cache: Optional[torch.Tensor] = None,
        v_cache: Optional[torch.Tensor] = None,
        cache_seqlens: Optional[int] = None
    ) -> torch.Tensor:
        """Forward pass

        Args:
            x: input (batch_size, seq_len, hidden_dim)
            k_cache: cache of keys for inference
            v_cache: cache of values for inference
            cache_seqlens: length of cache

        Returns:
            torch.Tensor: output (batch_size, seq_len, hidden_dim)
        """
        n_1 = self.norm1(x)
        h = self.attention(
            n_1,
            k_cache=k_cache,
            v_cache=v_cache,
            cache_seqlens=cache_seqlens,
        )
        h = self.drop(h)
        if self.parallel_residual:
            n_2 = n_1 if self.shared_attention_norm else self.norm2(x)
            x = x + h + self.mlp(n_2)
        else:
            if self.shared_attention_norm:
                raise NotImplementedError(
                    "Shared attention norm can't be used with parallel \
                    residual"
                )

            x = x + h
            x = x + self.mlp(self.norm2(x))
        return x


class Transformer(nn.Module):
    """Transformer class

    Inspired by TinyLlama (https://github.com/jzhang38/TinyLlama/tree/main)

    Args:
        hidden_dim: hidden dim of transformer
        num_layers: number of layers
        num_heads: number of heads in attention
        seq_len: expected length of context (used for inference with
            kv cache)
        attention_dropout: dropout in attention
        residual_dropout: dropout after attention
        normalize_qk: use qk normalization in attention
        bias: use bias in linear layers
        parallel_residual: use parallel residual (no attention
            in MLP)
        shared_attention_norm: share norm for attention and
            MLP (only if parallel_residual=True)
        norm_class: class for normalization layer (LayerNorm/RMSNorm)
        norm_eps: epsilon for normalization
        mlp_class: class for MLP layer (GptNeoxMLP/LLaMAMLP)
        intermediate_size: projection size in MLP layer
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        num_layers: int = 8,
        num_heads: int = 8,
        seq_len: int = 8192,
        attention_dropout: float = 0.0,
        residual_dropout: float = 0.0,
        normalize_qk: bool = False,
        bias: bool = True,
        parallel_residual: bool = False,
        shared_attention_norm: bool = False,
        norm_class: str = "LayerNorm",
        norm_eps: float = 1e-5,
        mlp_class: str = "LLaMAMLP",
        intermediate_size: Optional[int] = None,
    ):
        super().__init__()

        self.attention_dropout = attention_dropout
        self.residual_dropout = residual_dropout
        self.normalize_qk = normalize_qk

        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads

        self.norm_class = get_norm_class(norm_class)

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    hidden_dim=self.hidden_dim,
                    num_heads=self.num_heads,
                    attention_dropout=self.attention_dropout,
                    residual_dropout=self.residual_dropout,
                    normalize_qk=self.normalize_qk,
                    bias=bias,
                    parallel_residual=parallel_residual,
                    shared_attention_norm=shared_attention_norm,
                    norm_class=norm_class,
                    norm_eps=norm_eps,
                    mlp_class=mlp_class,
                    intermediate_size=intermediate_size
                )
                for _ in range(self.num_layers)
            ]
        )
        self.ln_f = self.norm_class(hidden_dim, eps=norm_eps)

        self.apply(self._init_weights)

    def _init_weights(self,
                      module: nn.Module) -> None:
        # GPT-NeoX  https://arxiv.org/pdf/2204.06745.pdf
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0,
                                  std=math.sqrt(2.0 / 5 / self.hidden_dim))
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

        for name, p in module.named_parameters():
            if (name == "proj.weight" and isinstance(module, LLaMAMLP)
                or name == "proj.weight" and isinstance(module, GptNeoxMLP)
                    or name == "out_proj.weight"
                        and isinstance(module, FlashAliBiCausalSelfAttention)):
                nn.init.normal_(p, mean=0.0, std=1 /
                                math.sqrt(self.hidden_dim) / self.num_layers)

    def init_cache(self,
                   batch_size: int,
                   dtype: torch.dtype,
                   device: torch.device) -> KVCache:
        """Initialize KV-cache for inference

        Args:
            batch_size: batch size of the cache
            dtype: torch dtype of the cache (float16/bfloat16)
            device: torch device

        Returns:
            KVCache: initialized kv-cache
        """
        cache = KVCache(
            batch_size=batch_size,
            max_seq_len=self.seq_len,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            head_dim=self.hidden_dim // self.num_heads,
            device=device,
            dtype=dtype,
        )
        return cache

    def forward(self,
                x: torch.Tensor,
                cache: Optional[KVCache] = None
                ) -> Union[torch.Tensor, Tuple[torch.Tensor, KVCache]]:
        """Forward pass

        Args:
            x: input (batch_size, seq_len, hidden_dim)
            cache: cache for inference

        Returns:
            torch.Tensor: output (batch_size, seq_len, hidden_dim)
            KVCache: updated KV-cache, returned only if it was
                passed in arguments
        """
        _cache = cache or [(None, None, None) for _ in range(self.num_layers)]

        for i, block in enumerate(self.blocks):
            x = block(x, *_cache[i])
        x = self.ln_f(x)

        if cache is not None:
            cache.update()
            return x, cache
        return x
