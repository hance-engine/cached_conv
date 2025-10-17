from warnings import warn

import torch

# from .convs1d import AlignBranches as _AlignBranches1d
# from .convs1d import (BranchesCachedConv2d1d, CachedConvTranspose1d,
#                     CachedPadding1d, CachedSequential, Sequential)

# 1d
from .convs1d import AlignBranches as _AlignBranches1d
from .convs1d import Branches as _Branches1d
from .convs1d import CachedConv1d
from .convs1d import CachedConvTranspose1d
from .convs1d import CachedPadding1d
from .convs1d import CachedSequential
from .convs1d import Sequential

from .convs1d import Conv1d as _Conv1d
from .convs1d import ConvTranspose1d as _ConvTranspose1d

# 2d
from .convs2d import AlignBranches2d as _AlignBranches2d
from .convs2d import Branches2d as _Branches2d
from .convs2d import CachedConv2d
from .convs2d import CachedConvTranspose2d
from .convs2d import CachedPadding2d

from .convs2d import Conv2d as _Conv2d
from .convs2d import ConvTranspose2d as _ConvTranspose2d


# GLOBAL VARIABLES
USE_BUFFER_CONV = False


def chunk_process(f, x, N):
    x = torch.split(x, x.shape[-1] // N, -1)
    y = torch.cat([f(_x) for _x in x], -1)
    return y


def use_buffer_conv(state: bool):
    warn(
        "use_buffer_conv is deprecated, use use_cached_conv instead",
        DeprecationWarning,
        stacklevel=2,
    )
    use_cached_conv(state)


def use_cached_conv(state: bool):
    global USE_BUFFER_CONV
    USE_BUFFER_CONV = state


def Conv1d(*args, **kwargs):
    if USE_BUFFER_CONV:
        return CachedConv1d(*args, **kwargs)
    else:
        return _Conv1d(*args, **kwargs)

def Conv2d(*args, **kwargs):
    if USE_BUFFER_CONV:
        return CachedConv2d(*args, **kwargs)
    else:
        return _Conv2d(*args, **kwargs)

def ConvTranspose1d(*args, **kwargs):
    if USE_BUFFER_CONV:
        return CachedConvTranspose1d(*args, **kwargs)
    else:
        return _ConvTranspose1d(*args, **kwargs)

def ConvTranspose2d(*args, **kwargs):
    if USE_BUFFER_CONV:
        return CachedConvTranspose2d(*args, **kwargs)
    else:
        return _ConvTranspose2d(*args, **kwargs)


def AlignBranches1d(*args, **kwargs):
    if USE_BUFFER_CONV:
        return _AlignBranches1d(*args, **kwargs)
    else:
        return _Branches1d(*args, **kwargs)


def test_equal(model_constructor, input_tensor, crop=True):
    initial_state = USE_BUFFER_CONV

    use_cached_conv(False)
    model = model_constructor()
    use_cached_conv(True)
    cmodel = model_constructor()

    for p1, p2 in zip(model.parameters(), cmodel.parameters()):
        p2.data.copy_(p1.data)

    y = model(input_tensor)[..., :-cmodel.cumulative_delay]
    cy = chunk_process(lambda x: cmodel(x), input_tensor,
                       4)[..., cmodel.cumulative_delay:]

    if crop:
        cd = cmodel.cumulative_delay
        y = y[..., cd:-cd]
        cy = cy[..., cd:-cd]

    use_cached_conv(initial_state)
    return torch.allclose(y, cy, 1e-4, 1e-4)
