"""
CausalConv2dCached that caches only along the width (last) dimension while keeping standard (non-causal, no cache) behavior along height. 
Like the 1D version, it guarantees that each forward over a new width chunk returns the outputs for that chunk only, matching offline causal results.

Tips
- Width stride must be 1 for streaming. Height stride/padding/dilation are free.
- Use get_state() / set_state() to checkpoint the width cache across segments or devices.
- Set detach_cache_grad=False only if you explicitly need gradients to flow across width chunks during training.
"""

import torch
import torch.nn as nn

class CausalConv2dCached(nn.Module):
    """
    Causal Conv2d with streaming cache *only along width*.
    Height dimension is handled by vanilla convolution (no caching, non-causal).

    Args:
      in_channels, out_channels: as usual
      kernel_size: int or (kh, kw)
      dilation: int or (dh, dw)
      stride_h: vertical stride (height). Horizontal (width) stride is fixed to 1 for streaming.
      padding_h: vertical padding (height). Width padding is handled manually for causality.
      groups, bias: as in nn.Conv2d

    Shapes:
      x: (B, C_in, H, W_chunk)
      out: (B, C_out, H_out, W_chunk)   # width is preserved; height depends on kh/stride_h/padding_h/dilation_h

    Notes:
      - Causality & cache are applied only along width.
      - Width stride is fixed to 1. Height stride is configurable.
      - Cache length = (kw - 1) * dw
    """
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        dilation=1,
        stride_h=1,
        padding_h=0,
        groups=1,
        bias=True,
    ):
        super().__init__()

        # Normalize tuples
        if isinstance(kernel_size, int):
            kh = kw = kernel_size
        else:
            kh, kw = kernel_size
        if isinstance(dilation, int):
            dh = dw = dilation
        else:
            dh, dw = dilation

        if kw < 1 or kh < 1:
            raise ValueError("kernel_size must be >= 1")
        if dw < 1 or dh < 1:
            raise ValueError("dilation must be >= 1")
        if stride_h < 1:
            raise ValueError("stride_h must be >= 1")

        self.kh, self.kw = int(kh), int(kw)
        self.dh, self.dw = int(dh), int(dw)
        self.stride_h = int(stride_h)
        self.padding_h = int(padding_h)
        self.groups = int(groups)

        # Cache only for width
        self.cache_len = (self.kw - 1) * self.dw

        # Conv2d with:
        #  - height padding handled by Conv2d (padding_h)
        #  - width padding handled manually (0 here)
        #  - width stride fixed to 1
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(self.kh, self.kw),
            stride=(self.stride_h, 1),
            padding=(self.padding_h, 0),
            dilation=(self.dh, self.dw),
            groups=self.groups,
            bias=bias,
        )

        # Internal width cache: (B, C_in, H, cache_len). Created lazily.
        self.register_buffer("_cache", None, persistent=False)

    # ---------------- Cache helpers ----------------
    def clear_cache(self):
        self._cache = None

    def has_cache(self):
        return (self._cache is not None) and (self.cache_len > 0)

    def _ensure_cache(self, x):
        if self.cache_len == 0:
            self._cache = None
            return
        B, C, H, _ = x.shape
        need_new = (
            self._cache is None
            or self._cache.size(0) != B
            or self._cache.size(1) != C
            or self._cache.size(2) != H
            or self._cache.device != x.device
            or self._cache.dtype != x.dtype
        )
        if need_new:
            self._cache = x.new_zeros(B, C, H, self.cache_len)

    def get_state(self):
        return None if self._cache is None else self._cache.detach().clone()

    def set_state(self, state):
        if state is None:
            self._cache = None
        else:
            if state.dim() != 4 or state.size(-1) != self.cache_len:
                raise ValueError(f"state must have shape (B, C_in, H, {self.cache_len})")
            self._cache = state.clone()

    @torch.no_grad()
    def _update_cache(self, cache, x):
        if self.cache_len == 0:
            return None
        W = x.size(-1)
        if cache is None:
            if W >= self.cache_len:
                return x[..., -self.cache_len:]
            pad = x.new_zeros(x.size(0), x.size(1), x.size(2), self.cache_len - W)
            return torch.cat([pad, x], dim=-1)

        if W >= self.cache_len:
            return x[..., -self.cache_len:]
        return torch.cat([cache[..., W:], x], dim=-1)

    # ---------------- Forward ----------------
    def forward(self, x, *, use_cache=True, reset_cache=False, detach_cache_grad=True):
        """
        x: (B, C_in, H, W_chunk)
        Returns outputs for this chunk only (width preserved).
        """
        if reset_cache:
            self.clear_cache()

        B, C, H, W = x.shape

        if use_cache:
            self._ensure_cache(x)
            if self.cache_len > 0 and self._cache is not None:
                padded = torch.cat([self._cache, x], dim=-1)
            elif self.cache_len > 0:
                padded = torch.cat([x.new_zeros(B, C, H, self.cache_len), x], dim=-1)
            else:
                padded = x
        else:
            if self.cache_len > 0:
                padded = torch.cat([x.new_zeros(B, C, H, self.cache_len), x], dim=-1)
            else:
                padded = x

        y = self.conv(padded)  # -> (B, C_out, H_out, W)

        if use_cache and self.cache_len > 0:
            new_cache = self._update_cache(self._cache, x)
            self._cache = new_cache.detach() if detach_cache_grad else new_cache

        return y

    @torch.no_grad()
    def forward_full(self, x):
        """
        Offline causal width-only conv: zero-left-pad width by cache_len; height vanilla.
        Does not touch internal cache.
        """
        B, C, H, W = x.shape
        if self.cache_len > 0:
            padded = torch.cat([x.new_zeros(B, C, H, self.cache_len), x], dim=-1)
        else:
            padded = x
        return self.conv(padded)


# --------------------------
# Minimal streaming example
# --------------------------
if __name__ == "__main__":
    torch.manual_seed(0)

    B, C_in, C_out = 2, 3, 4
    H, W_total = 6, 73

    kh, kw = 3, 5
    dh, dw = 1, 2          # dilation along width = 2  -> cache_len = (5-1)*2 = 8
    stride_h = 2           # vanilla vertical stride
    padding_h = 1          # keep height roughly "same" for kh=3, stride_h=2

    x = torch.randn(B, C_in, H, W_total)

    layer = CausalConv2dCached(
        C_in, C_out,
        kernel_size=(kh, kw),
        dilation=(dh, dw),
        stride_h=stride_h,
        padding_h=padding_h,
        bias=True,
    )
    layer.eval()

    # Offline ground truth (width-causal, height-vanilla)
    y_offline = layer.forward_full(x)

    # Stream over arbitrary width chunks (height is full each time)
    chunk_sizes = [7, 4, 13, 1, 9, 16, 8, 15]  # sums to 73
    assert sum(chunk_sizes) == W_total

    layer.clear_cache()
    outs = []
    start = 0
    for n in chunk_sizes:
        chunk = x[:, :, :, start:start+n]         # (B, C_in, H, n)
        y_chunk = layer(chunk, use_cache=True)    # (B, C_out, H_out, n)
        outs.append(y_chunk)
        start += n

    y_stream = torch.cat(outs, dim=-1)

    max_abs_err = (y_offline - y_stream).abs().max().item()
    print(f"Max absolute error (stream vs offline): {max_abs_err:.6g}")
    assert torch.allclose(y_offline, y_stream, atol=1e-6), "Streaming != offline!"

    # Optional: strict single-column streaming
    layer.clear_cache()
    outs_cols = []
    for t in range(W_total):
        y_t = layer(x[:, :, :, t:t+1], use_cache=True)   # (B, C_out, H_out, 1)
        outs_cols.append(y_t)
    y_cols = torch.cat(outs_cols, dim=-1)
    print(f"Per-column equals offline: {torch.allclose(y_cols, y_offline, atol=1e-6)}")
import torch
import torch.nn as nn

class CausalConv2dCached(nn.Module):
    """
    Causal Conv2d with streaming cache *only along width*.
    Height dimension is handled by vanilla convolution (no caching, non-causal).

    Args:
      in_channels, out_channels: as usual
      kernel_size: int or (kh, kw)
      dilation: int or (dh, dw)
      stride_h: vertical stride (height). Horizontal (width) stride is fixed to 1 for streaming.
      padding_h: vertical padding (height). Width padding is handled manually for causality.
      groups, bias: as in nn.Conv2d

    Shapes:
      x: (B, C_in, H, W_chunk)
      out: (B, C_out, H_out, W_chunk)   # width is preserved; height depends on kh/stride_h/padding_h/dilation_h

    Notes:
      - Causality & cache are applied only along width.
      - Width stride is fixed to 1. Height stride is configurable.
      - Cache length = (kw - 1) * dw
    """
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        dilation=1,
        stride_h=1,
        padding_h=0,
        groups=1,
        bias=True,
    ):
        super().__init__()

        # Normalize tuples
        if isinstance(kernel_size, int):
            kh = kw = kernel_size
        else:
            kh, kw = kernel_size
        if isinstance(dilation, int):
            dh = dw = dilation
        else:
            dh, dw = dilation

        if kw < 1 or kh < 1:
            raise ValueError("kernel_size must be >= 1")
        if dw < 1 or dh < 1:
            raise ValueError("dilation must be >= 1")
        if stride_h < 1:
            raise ValueError("stride_h must be >= 1")

        self.kh, self.kw = int(kh), int(kw)
        self.dh, self.dw = int(dh), int(dw)
        self.stride_h = int(stride_h)
        self.padding_h = int(padding_h)
        self.groups = int(groups)

        # Cache only for width
        self.cache_len = (self.kw - 1) * self.dw

        # Conv2d with:
        #  - height padding handled by Conv2d (padding_h)
        #  - width padding handled manually (0 here)
        #  - width stride fixed to 1
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(self.kh, self.kw),
            stride=(self.stride_h, 1),
            padding=(self.padding_h, 0),
            dilation=(self.dh, self.dw),
            groups=self.groups,
            bias=bias,
        )

        # Internal width cache: (B, C_in, H, cache_len). Created lazily.
        self.register_buffer("_cache", None, persistent=False)

    # ---------------- Cache helpers ----------------
    def clear_cache(self):
        self._cache = None

    def has_cache(self):
        return (self._cache is not None) and (self.cache_len > 0)

    def _ensure_cache(self, x):
        if self.cache_len == 0:
            self._cache = None
            return
        B, C, H, _ = x.shape
        need_new = (
            self._cache is None
            or self._cache.size(0) != B
            or self._cache.size(1) != C
            or self._cache.size(2) != H
            or self._cache.device != x.device
            or self._cache.dtype != x.dtype
        )
        if need_new:
            self._cache = x.new_zeros(B, C, H, self.cache_len)

    def get_state(self):
        return None if self._cache is None else self._cache.detach().clone()

    def set_state(self, state):
        if state is None:
            self._cache = None
        else:
            if state.dim() != 4 or state.size(-1) != self.cache_len:
                raise ValueError(f"state must have shape (B, C_in, H, {self.cache_len})")
            self._cache = state.clone()

    @torch.no_grad()
    def _update_cache(self, cache, x):
        if self.cache_len == 0:
            return None
        W = x.size(-1)
        if cache is None:
            if W >= self.cache_len:
                return x[..., -self.cache_len:]
            pad = x.new_zeros(x.size(0), x.size(1), x.size(2), self.cache_len - W)
            return torch.cat([pad, x], dim=-1)

        if W >= self.cache_len:
            return x[..., -self.cache_len:]
        return torch.cat([cache[..., W:], x], dim=-1)

    # ---------------- Forward ----------------
    def forward(self, x, *, use_cache=True, reset_cache=False, detach_cache_grad=True):
        """
        x: (B, C_in, H, W_chunk)
        Returns outputs for this chunk only (width preserved).
        """
        if reset_cache:
            self.clear_cache()

        B, C, H, W = x.shape

        if use_cache:
            self._ensure_cache(x)
            if self.cache_len > 0 and self._cache is not None:
                padded = torch.cat([self._cache, x], dim=-1)
            elif self.cache_len > 0:
                padded = torch.cat([x.new_zeros(B, C, H, self.cache_len), x], dim=-1)
            else:
                padded = x
        else:
            if self.cache_len > 0:
                padded = torch.cat([x.new_zeros(B, C, H, self.cache_len), x], dim=-1)
            else:
                padded = x

        y = self.conv(padded)  # -> (B, C_out, H_out, W)

        if use_cache and self.cache_len > 0:
            new_cache = self._update_cache(self._cache, x)
            self._cache = new_cache.detach() if detach_cache_grad else new_cache

        return y

    @torch.no_grad()
    def forward_full(self, x):
        """
        Offline causal width-only conv: zero-left-pad width by cache_len; height vanilla.
        Does not touch internal cache.
        """
        B, C, H, W = x.shape
        if self.cache_len > 0:
            padded = torch.cat([x.new_zeros(B, C, H, self.cache_len), x], dim=-1)
        else:
            padded = x
        return self.conv(padded)


# --------------------------
# Minimal streaming example
# --------------------------
if __name__ == "__main__":
    torch.manual_seed(0)

    B, C_in, C_out = 2, 3, 4
    H, W_total = 6, 73

    kh, kw = 3, 5
    dh, dw = 1, 2          # dilation along width = 2  -> cache_len = (5-1)*2 = 8
    stride_h = 2           # vanilla vertical stride
    padding_h = 1          # keep height roughly "same" for kh=3, stride_h=2

    x = torch.randn(B, C_in, H, W_total)

    layer = CausalConv2dCached(
        C_in, C_out,
        kernel_size=(kh, kw),
        dilation=(dh, dw),
        stride_h=stride_h,
        padding_h=padding_h,
        bias=True,
    )
    layer.eval()

    # Offline ground truth (width-causal, height-vanilla)
    y_offline = layer.forward_full(x)

    # Stream over arbitrary width chunks (height is full each time)
    chunk_sizes = [7, 4, 13, 1, 9, 16, 8, 15]  # sums to 73
    assert sum(chunk_sizes) == W_total

    layer.clear_cache()
    outs = []
    start = 0
    for n in chunk_sizes:
        chunk = x[:, :, :, start:start+n]         # (B, C_in, H, n)
        y_chunk = layer(chunk, use_cache=True)    # (B, C_out, H_out, n)
        outs.append(y_chunk)
        start += n

    y_stream = torch.cat(outs, dim=-1)

    max_abs_err = (y_offline - y_stream).abs().max().item()
    print(f"Max absolute error (stream vs offline): {max_abs_err:.6g}")
    assert torch.allclose(y_offline, y_stream, atol=1e-6), "Streaming != offline!"

    # Optional: strict single-column streaming
    layer.clear_cache()
    outs_cols = []
    for t in range(W_total):
        y_t = layer(x[:, :, :, t:t+1], use_cache=True)   # (B, C_out, H_out, 1)
        outs_cols.append(y_t)
    y_cols = torch.cat(outs_cols, dim=-1)
    print(f"Per-column equals offline: {torch.allclose(y_cols, y_offline, atol=1e-6)}")
