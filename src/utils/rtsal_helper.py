import torch
import random
import math
import torch.nn.functional as F

### Values taken from https://github.com/PiotrDabkowski/pytorch-saliency/blob/master/sal/utils/mask.py ###
def generate_alternative_image(image):
    cuda = torch.cuda.is_available()

    options = ["blur", "color"]
    method = random.choice(options)

    if method == "blur":
        alternative_image = imsmooth(image, sigma=10)
    elif method == "color":
        n, c, _, _ = image.size()
        color_range = 0.66
        noise = torch.zeros_like(image).normal_(0.11)
        if cuda:
            alternative_image = noise + torch.Tensor(n, c, 1, 1).cuda().uniform_(-color_range/2., color_range/2.)
        else:
            alternative_image = noise + torch.Tensor(n, c, 1, 1).uniform_(-color_range/2., color_range/2.)
    else:
        raise Exception("Unknown option for generating alternative image")

    return alternative_image

### Taken from https://github.com/facebookresearch/TorchRay/blob/6a198ee61d229360a3def590410378d2ed6f1f06/torchray/utils.py ###
def imsmooth(tensor,
             sigma,
             stride=1,
             padding=0,
             padding_mode='constant',
             padding_value=0):
    r"""Apply a 2D Gaussian filter to a tensor.
    The 2D filter itself is implementing by separating the 2D convolution in
    two 1D convolutions, first along the vertical direction and then along
    the horizontal one. Each 1D Gaussian kernel is given by:
    .. math::
        f_i \propto \exp\left(-\frac{1}{2} \frac{i^2}{\sigma^2} \right),
            ~~~ i \in \{-W,\dots,W\},
            ~~~ W = \lceil 4\sigma \rceil.
    This kernel is normalized to sum to one exactly. Given the latter, the
    function calls `torch.nn.functional.conv2d`
    to perform the actual convolution. Various padding parameters and the
    stride are passed to the latter.
    Args:
        tensor (:class:`torch.Tensor`): :math:`N\times C\times H\times W`
            image tensor.
        sigma (float): standard deviation of the Gaussian kernel.
        stride (int, optional): subsampling factor. Default: ``1``.
        padding (int, optional): extra padding. Default: ``0``.
        padding_mode (str, optional): ``'constant'``, ``'reflect'`` or
            ``'replicate'``. Default: ``'constant'``.
        padding_value (float, optional): constant value for the `constant`
            padding mode. Default: ``0``.
    Returns:
        :class:`torch.Tensor`: :math:`N\times C\times H\times W` tensor with
        the smoothed images.
    """

    EPSILON_DOUBLE = torch.tensor(2.220446049250313e-16, dtype=torch.float64)
    EPSILON_SINGLE = torch.tensor(1.19209290E-07, dtype=torch.float32)
    SQRT_TWO_DOUBLE = torch.tensor(math.sqrt(2), dtype=torch.float32)
    SQRT_TWO_SINGLE = SQRT_TWO_DOUBLE.to(torch.float32)

    assert sigma >= 0
    width = math.ceil(4 * sigma)
    filt = (torch.arange(-width,
                         width + 1,
                         dtype=torch.float32,
                         device=tensor.device) /
            (SQRT_TWO_SINGLE * sigma + EPSILON_SINGLE))
    filt = torch.exp(-filt * filt)
    filt /= torch.sum(filt)
    num_channels = tensor.shape[1]
    width = width + padding
    if padding_mode == 'constant' and padding_value == 0:
        other_padding = width
        x = tensor
    else:
        # pad: (before, after) pairs starting from last dimension backward
        x = F.pad(tensor,
                  (width, width, width, width),
                  mode=padding_mode,
                  value=padding_value)
        other_padding = 0
        padding = 0
    x = F.conv2d(x,
                 filt.reshape((1, 1, -1, 1)).expand(num_channels, -1, -1, -1),
                 padding=(other_padding, padding),
                 stride=(stride, 1),
                 groups=num_channels)
    x = F.conv2d(x,
                 filt.reshape((1, 1, 1, -1)).expand(num_channels, -1, -1, -1),
                 padding=(padding, other_padding),
                 stride=(1, stride),
                 groups=num_channels)
    return x
