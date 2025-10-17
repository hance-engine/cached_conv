"""
A lightweight, stateful “cached causal conv1d” you can drop into a streaming pipeline. 
It keeps the last (kernel_size-1) * dilation input samples per batch in an internal buffer, so each call only needs the new chunk.

Notes
- The internal cache is per-(batch, channel) and is reallocated automatically if batch size, dtype, or device changes.
- `detach_cache_grad=True` (default) keeps streaming clean for inference and avoids cross-chunk graph growth. If you’re doing chunked training and need gradients to flow across chunk boundaries, pass `detach_cache_grad=False`.
- If you need to checkpoint/resume a stream, use `get_state()` / `set_state()` to export/import the cache.
- For typical streaming setups, keep `stride=1`. If you need strided streaming, you’ll usually pair this with decimation logic in your pipeline (not shown here).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class CausalConv1dCached(nn.Module):
    """
    Causal Conv1d with an internal streaming cache.

    - stride is fixed to 1 (common for streaming).
    - cache length = (kernel_size - 1) * dilation
    - forward(x, use_cache=True) returns the outputs for the *new* chunk only.
      The layer updates its internal cache so the next call can pick up where it left off.

    Shapes:
      x: (B, C_in, T)
      out: (B, C_out, T)
    """
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, bias=True):
        super().__init__()
        if kernel_size < 1:
            raise ValueError("kernel_size must be >= 1")
        if dilation < 1:
            raise ValueError("dilation must be >= 1")

        self.kernel_size = int(kernel_size)
        self.dilation = int(dilation)
        self.cache_len = (self.kernel_size - 1) * self.dilation

        # No padding here; we handle causal padding manually
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=self.kernel_size,
            stride=1,
            padding=0,
            dilation=self.dilation,
            bias=bias,
        )

        # Internal cache (B, C_in, cache_len). Lazily initialized per batch/device/dtype.
        self.register_buffer("_cache", None, persistent=False)

    # ---- Cache helpers ----
    def clear_cache(self):
        """Forget all past context."""
        self._cache = None

    def has_cache(self):
        return (self._cache is not None) and (self.cache_len > 0)

    def _ensure_cache(self, x):
        """Allocate/resize cache to match (B, C_in, cache_len)."""
        if self.cache_len == 0:
            self._cache = None
            return

        B, C_in, _ = x.shape
        needs_new = (
            self._cache is None
            or self._cache.size(0) != B
            or self._cache.size(1) != C_in
            or self._cache.device != x.device
            or self._cache.dtype != x.dtype
        )
        if needs_new:
            self._cache = x.new_zeros(B, C_in, self.cache_len)

    def get_state(self):
        """Return a detached copy of the current cache (or None)."""
        return None if self._cache is None else self._cache.detach().clone()

    def set_state(self, state):
        """Set the cache from an external tensor (B, C_in, cache_len) or None."""
        if state is None:
            self._cache = None
        else:
            if state.dim() != 3 or state.size(-1) != self.cache_len:
                raise ValueError(f"state must have shape (B, C_in, {self.cache_len})")
            self._cache = state.clone()

    # ---- Forward paths ----
    @torch.no_grad()
    def _update_cache(self, cache, x):
        """Return the next cache given previous `cache` and new chunk `x`."""
        if self.cache_len == 0:
            return None
        T = x.size(-1)  # temporal length of the input (i.e., input chunk)
        if cache is None:
            # Initial call: cache is just the last cache_len samples of left-padding(zeros)|x
            if T >= self.cache_len:
                return x[:, :, -self.cache_len:]
            else:
                pad = x.new_zeros(x.size(0), x.size(1), self.cache_len - T)
                return torch.cat([pad, x], dim=-1)

        if T >= self.cache_len:
            # New chunk fully replaces the cache
            return x[:, :, -self.cache_len:]
        else:
            # Shift old cache and append new x
            return torch.cat([cache[:, :, T:], x], dim=-1)

    def forward(self, x, *, use_cache=True, reset_cache=False, detach_cache_grad=True):
        """
        x: (B, C_in, T)
        use_cache:
          - True  => prepend internal cache (or zeros on first call) so output length == T
          - False => do *offline* causal conv: left-pad zeros; cache is NOT updated.
        reset_cache: if True, clear cache before using it (useful to begin a stream).
        detach_cache_grad:
          - True (default) detaches cache updates (no cross-chunk autograd)
          - False allows gradients to flow through caches across calls
        """
        if reset_cache:
            self.clear_cache()

        B, C_in, T = x.shape
        if use_cache:
            self._ensure_cache(x)
            if self.cache_len > 0 and self._cache is not None:
                padded = torch.cat([self._cache, x], dim=-1)
            elif self.cache_len > 0:
                # first call: left pad with zeros
                padded = torch.cat([x.new_zeros(B, C_in, self.cache_len), x], dim=-1)
            else:
                padded = x
        else:
            # Offline/one-shot causal: left pad zeros; do not touch the cache
            if self.cache_len > 0:
                padded = torch.cat([x.new_zeros(B, C_in, self.cache_len), x], dim=-1)
            else:
                padded = x

        y = self.conv(padded)  # => (B, C_out, T)

        # Update cache only when streaming
        if use_cache and self.cache_len > 0:
            new_cache = self._update_cache(self._cache, x)
            # Optionally keep graphs across calls (e.g., chunked training)
            self._cache = new_cache.detach() if detach_cache_grad else new_cache

        return y

    @torch.no_grad()
    def forward_full(self, x):
        """
        Pure offline causal conv (zero-left-padded), without touching internal cache.
        Useful for equivalence checks.
        """
        B, C, T = x.shape
        if self.cache_len > 0:
            padded = torch.cat([x.new_zeros(B, C, self.cache_len), x], dim=-1)
        else:
            padded = x
        return self.conv(padded)


# --------------------------
# Minimal streaming example
# --------------------------
if __name__ == "__main__":
    torch.manual_seed(0)

    B, C_in, C_out = 2, 3, 4
    K, D = 5, 2        # kernel_size=5, dilation=2  -> cache_len = (5-1)*2 = 8
    T_total = 64

    x = torch.randn(B, C_in, T_total)

    layer = CausalConv1dCached(C_in, C_out, kernel_size=K, dilation=D, bias=True)
    layer.eval()  # typical for streaming inference

    # Offline "ground truth" (one shot, zero-left-padded)
    y_offline = layer.forward_full(x)

    # Streaming over uneven chunks
    chunk_sizes = [7, 5, 9, 1, 16, 8, 18]  # sums to 64
    assert sum(chunk_sizes) == T_total

    layer.clear_cache()  # begin a fresh stream
    outs = []
    start = 0
    for n in chunk_sizes:
        chunk = x[:, :, start:start+n]            # (B, C_in, n)
        y_chunk = layer(chunk, use_cache=True)    # (B, C_out, n)
        outs.append(y_chunk)
        start += n

    y_stream = torch.cat(outs, dim=-1)

    # Check equivalence
    max_abs_err = (y_offline - y_stream).abs().max().item()
    print(f"Max absolute error (stream vs offline): {max_abs_err:.6g}")
    assert torch.allclose(y_offline, y_stream, atol=1e-6), "Streaming does not match offline!"

    # Optional: single-step usage (T=1 per call)
    layer.clear_cache()
    outs_step = []
    for t in range(T_total):
        y_t = layer(x[:, :, t:t+1], use_cache=True)  # (B, C_out, 1)
        outs_step.append(y_t)
    y_step = torch.cat(outs_step, dim=-1)
    print(f"Stepwise equals offline: {torch.allclose(y_step, y_offline, atol=1e-6)}")
