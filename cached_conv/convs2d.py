import torch
import torch.nn as nn
import torch.nn.functional as F


MAX_BATCH_SIZE = 64

# ---- Helpers ----

def _normalize_2d_padding(padding):
    """
    Returns per-side pads as:
      ((pad_top, pad_bottom), (pad_left, pad_right))

    Accepted forms:
      - int: p -> ((p,p),(p,p))
      - (ph, pw): symmetric per-dim
      - ((pt, pb), (pl, pr)): per-side
      - (pl, pr, pt, pb): flat per-side in F.pad order
    """
    if isinstance(padding, int):
        return (padding, padding), (padding, padding)

    if isinstance(padding, (tuple, list)):
        if len(padding) == 2 and all(isinstance(v, int) for v in padding):
            ph, pw = padding
            return (ph, ph), (pw, pw)

        if len(padding) == 2 and all(isinstance(v, (tuple, list)) and len(v) == 2 for v in padding):
            (pt, pb), (pl, pr) = padding
            return (int(pt), int(pb)), (int(pl), int(pr))

        if len(padding) == 4 and all(isinstance(v, int) for v in padding):
            pl, pr, pt, pb = padding  # F.pad order
            return (pt, pb), (pl, pr)

    raise ValueError(f"Unsupported 2D padding format: {padding}")


def _fpad_tuple(htb, wlr):
    """
    Convert ((top,bottom),(left,right)) -> (left,right,top,bottom) for F.pad
    """
    (pt, pb), (pl, pr) = htb, wlr
    return (int(pl), int(pr), int(pt), int(pb))


# ---- Cached sequential (width-only delay accounting) ----

class CachedSequential2d(nn.Sequential):
    """
    Sequential wrapper that tracks cumulative delay along width.
    """

    def __init__(self, *args, **kwargs):
        cumulative_delay = kwargs.pop("cumulative_delay", 0)
        stride = kwargs.pop("stride", 1)
        super().__init__(*args, **kwargs)

        if isinstance(stride, (tuple, list)):
            stride_w = int(stride[-1])
        else:
            stride_w = int(stride)

        self.cumulative_delay = int(cumulative_delay) * stride_w

        # Pick up last submodule's cumulative_delay, if present
        last_delay = 0
        for i in range(1, len(self) + 1):
            try:
                last_delay = int(self[-i].cumulative_delay)
                break
            except AttributeError:
                pass
        self.cumulative_delay += last_delay


class Sequential2d(CachedSequential2d):
    pass


# ---- Cached padding (width only) ----

class CachedPadding2d(nn.Module):
    """
    Cached padding along the *width* dimension.
    Replaces left zero-padding with the trailing columns of the previous tensor.

    Input/Output shapes: (B, C, H, W*)
    """

    def __init__(self, padding_w: int, crop: bool = False):
        super().__init__()
        self.initialized = 0
        self.padding = int(padding_w)
        self.crop = bool(crop)

    @torch.jit.unused
    @torch.no_grad()
    def init_cache(self, x):
        b, c, h, _ = x.shape
        pad = torch.zeros(MAX_BATCH_SIZE, c, h, self.padding, dtype=x.dtype, device=x.device)
        self.register_buffer("pad", pad)
        self.initialized = 1

    @torch.no_grad()
    def _maybe_reinit(self, x):
        if not self.initialized:
            self.init_cache(x)
            return
        # Re-init if channel/height/device/dtype changed
        b, c, h, _ = x.shape
        if (
            self.pad.shape[1] != c
            or self.pad.shape[2] != h
            or self.pad.device != x.device
            or self.pad.dtype != x.dtype
        ):
            self.init_cache(x)

    def forward(self, x):
        """
        x: (B, C, H, W)
        """
        if self.padding <= 0:
            return x

        self._maybe_reinit(x)

        b, _, h, _ = x.shape
        # Ensure cached height matches current height (if H changed, reinit_cache already ran)
        cached = self.pad[:b, :, :h]  # (b, c, h, pad_w)

        # Prepend cached trailing columns along width
        x = torch.cat([cached, x], dim=-1)  # (b, c, h, pad_w + W)

        # Update cache with rightmost 'padding' columns of the extended input
        cached.copy_(x[..., -self.padding:])  # (b, c, h, pad_w)

        if self.crop:
            x = x[..., :-self.padding]

        return x


# ---- Cached Conv2d (width-only streaming) ----

class CachedConv2d(nn.Conv2d):
    """
    Conv2d with cached width padding. Height behaves like vanilla conv.

    - Width: uses CachedPadding2d(left=total_w_pad) with future/stride compensation.
    - Height: zero-padding (per-side) applied via F.pad before conv.
    """

    def __init__(self, *args, **kwargs):
        padding_arg = kwargs.get("padding", 0)
        cumulative_delay = int(kwargs.pop("cumulative_delay", 0))

        # Disable internal padding; we handle both dims ourselves
        kwargs["padding"] = 0
        super().__init__(*args, **kwargs)

        (pt, pb), (pl, pr) = _normalize_2d_padding(padding_arg)
        total_w_pad = int(pl + pr)
        r_pad_w = int(pr)

        # Stride/dilation are tuples
        stride_w = int(self.stride[-1])

        # Align to stride (future compensation), along width only
        stride_delay = (stride_w - ((r_pad_w + cumulative_delay) % stride_w)) % stride_w
        self.cumulative_delay = (r_pad_w + stride_delay + cumulative_delay) // stride_w

        self.downsampling_delay = CachedPadding2d(stride_delay, crop=True)
        self.cache = CachedPadding2d(total_w_pad)

        # Store height pads for F.pad
        self._h_pad = (int(pt), int(pb))

    def forward(self, x):
        # Width-only delays/caching
        x = self.downsampling_delay(x)
        x = self.cache(x)

        # Height-only zero pad (no caching)
        pt, pb = self._h_pad
        if pt or pb:
            x = F.pad(x, (0, 0, pt, pb))  # (left,right,top,bottom): width pads are 0 here

        return F.conv2d(
            x,
            self.weight,
            self.bias,
            self.stride,
            0,  # padding already handled
            self.dilation,
            self.groups,
        )


# ---- Cached ConvTranspose2d (width-only streaming) ----

class CachedConvTranspose2d(nn.ConvTranspose2d):
    """
    ConvTranspose2d with cached overlap-add along width only.
    Height uses vanilla transposed-conv padding; width is handled manually.
    """

    def __init__(self, *args, **kwargs):
        cd = int(kwargs.pop("cumulative_delay", 0))
        super().__init__(*args, **kwargs)

        self.initialized = 0
        # cumulative delay measured along width
        stride_w = int(self.stride[-1])
        pad_h, pad_w = int(self.padding[0]), int(self.padding[1])
        self.cumulative_delay = pad_w + cd * stride_w

    @torch.jit.unused
    @torch.no_grad()
    def init_cache(self, x):
        # x is post-conv_transpose2d output (B, C_out, H, W)
        b, c, h, _ = x.shape
        pad_w = 2 * int(self.padding[1])
        cache = torch.zeros(MAX_BATCH_SIZE, c, h, pad_w, dtype=x.dtype, device=x.device)
        self.register_buffer("cache", cache)
        self.initialized = 1

    @torch.no_grad()
    def _maybe_reinit(self, x):
        if not self.initialized:
            self.init_cache(x)
            return
        b, c, h, _ = x.shape
        if (
            self.cache.shape[1] != c
            or self.cache.shape[2] != h
            or self.cache.device != x.device
            or self.cache.dtype != x.dtype
        ):
            self.init_cache(x)

    def forward(self, x):
        # Keep height padding vanilla (self.padding[0]); width padding handled manually
        pad_h = int(self.padding[0])

        y = F.conv_transpose2d(
            x,
            self.weight,
            None,  # defer bias
            self.stride,
            (pad_h, 0),  # height pad as usual, width pad disabled (0)
            self.output_padding,
            self.groups,
            self.dilation,
        )

        width_overlap = 2 * int(self.padding[1])
        if width_overlap > 0:
            self._maybe_reinit(y)
            b, _, h, _ = y.shape

            # Overlap-add using cached tail from previous chunk
            y[..., :width_overlap] += self.cache[:b, :, :h]
            # Update cache with current rightmost overlap slice
            self.cache[:b, :, :h].copy_(y[..., -width_overlap:])
            # Crop the trailing overlap from the output
            y = y[..., :-width_overlap]

        # Add bias if present
        if self.bias is not None:
            y = y + self.bias.view(1, -1, 1, 1)
        return y


# ---- Vanilla wrappers (no caching; keep interface parity) ----

class ConvTranspose2d(nn.ConvTranspose2d):
    def __init__(self, *args, **kwargs) -> None:
        kwargs.pop("cumulative_delay", 0)
        super().__init__(*args, **kwargs)
        self.cumulative_delay = 0


class Conv2d(nn.Conv2d):
    """
    Conv2d that supports per-side padding via F.pad.
    """
    def __init__(self, *args, **kwargs):
        padding_arg = kwargs.get("padding", 0)
        kwargs.pop("cumulative_delay", 0)
        # Disable internal padding; apply explicit F.pad for full per-side control
        kwargs["padding"] = 0
        super().__init__(*args, **kwargs)
        (pt, pb), (pl, pr) = _normalize_2d_padding(padding_arg)
        # F.pad expects (left,right,top,bottom)
        self._pad = (int(pl), int(pr), int(pt), int(pb))
        self.cumulative_delay = 0

    def forward(self, x):
        pl, pr, pt, pb = self._pad
        if pl or pr or pt or pb:
            x = F.pad(x, self._pad)
        return F.conv2d(
            x,
            self.weight,
            self.bias,
            self.stride,
            0,
            self.dilation,
            self.groups,
        )


# ---- Branch utilities (width-delay alignment across branches) ----

class AlignBranches2d(nn.Module):
    """
    Align branches by compensating for different cumulative width-delays.
    Applies CachedPadding2d (crop=True) on the input for each branch.
    Height is unchanged.
    """

    def __init__(self, *branches, delays=None, cumulative_delay=0, stride=1):
        super().__init__()
        self.branches = nn.ModuleList(branches)

        if delays is None:
            delays = [getattr(b, "cumulative_delay", 0) for b in self.branches]

        max_delay = int(max(delays))

        self.paddings = nn.ModuleList([
            CachedPadding2d(int(max_delay - d), crop=True) for d in delays
        ])

        if isinstance(stride, (tuple, list)):
            stride_w = int(stride[-1])
        else:
            stride_w = int(stride)

        self.cumulative_delay = int(cumulative_delay) * stride_w + max_delay

    def forward(self, x):
        outs = []
        for branch, pad in zip(self.branches, self.paddings):
            delayed_x = pad(x)
            outs.append(branch(delayed_x))
        return outs


class Branches2d(nn.Module):
    """
    Run multiple branches without alignment; track cumulative width-delay = max(branch delays).
    """

    def __init__(self, *branches, delays=None, cumulative_delay=0, stride=1):
        super().__init__()
        self.branches = nn.ModuleList(branches)

        if delays is None:
            delays = [getattr(b, "cumulative_delay", 0) for b in self.branches]

        max_delay = int(max(delays))

        if isinstance(stride, (tuple, list)):
            stride_w = int(stride[-1])
        else:
            stride_w = int(stride)

        self.cumulative_delay = int(cumulative_delay) * stride_w + max_delay

    def forward(self, x):
        return [branch(x) for branch in self.branches]
