import torch


class KVCache:
    """KV-Cache class

    Args:
        batch_size: batch size of cache
        max_seq_len: maximum sequence length
        num_layers: number of layers in transformer
        num_heads: number of heads in transformer
        head_dim: hidden dim in transformer
        device: torch device
        dtype: dtype of cache
    """

    def __init__(
        self,
        batch_size: int,
        max_seq_len: int,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        device: torch.device,
        dtype: torch.dtype,
    ):
        self.cache_shape = (num_layers,
                            batch_size,
                            max_seq_len,
                            num_heads,
                            head_dim)
        self.k_cache = torch.full(
            self.cache_shape, fill_value=torch.nan, dtype=dtype, device=device
        ).detach()  # noqa
        self.v_cache = torch.full(
            self.cache_shape, fill_value=torch.nan, dtype=dtype, device=device
        ).detach()
        self.cache_seqlens = 0

    def __len__(self):
        """Get the number of layers"""
        return self.k_cache.shape[0]

    def __getitem__(self, layer_idx: int):
        """Get cache of the specific layer

        Args:
            layer_idx: index of the layer
        """
        return (
            self.k_cache[layer_idx],
            self.v_cache[layer_idx],
            self.cache_seqlens,
        )

    def reset(self) -> None:
        """Reset cache sequence length"""
        self.cache_seqlens = 0

    def update(self) -> None:
        """Update cache info"""
        self.cache_seqlens = self.cache_seqlens + 1
        if self.cache_seqlens == self.cache_shape[2]:
            self.k_cache = torch.roll(self.k_cache, -1, dims=2)
            self.v_cache = torch.roll(self.v_cache, -1, dims=2)
            self.cache_seqlens = self.cache_seqlens - 1
            assert self.cache_seqlens >= 0, "negative cache sequence length"
            self.k_cache[:, :, -1] = torch.nan
            self.v_cache[:, :, -1] = torch.nan
