"""
Losses
"""
from utils import *


def check_shape(x, y):
    assert x.shape == y.shape, 'shape of tensors must be the same!'
    assert x.stride() == y.stride(), 'strides of tensors must be the same!'
    assert x.ndim == y.ndim == 5, 'inputs are expected to have 5D ([N, C, T, H, W])'


def mse(x, y):
    """
    Compute the per-frame MSE loss
    """
    check_shape(x, y)
    N, C, T, H, W = x.shape
    x = x.permute(0, 2, 3, 4, 1).contiguous().view(N, T, H * W * C)
    y = y.permute(0, 2, 3, 4, 1).contiguous().view(N, T, H * W * C)
    return F.mse_loss(x, y, reduction='none').mean(dim=2)


def l1(x, y):
    """
    Compute the per-frame L1 loss
    """
    check_shape(x, y)
    N, C, T, H, W = x.shape
    x = x.permute(0, 2, 3, 4, 1).contiguous().view(N, T, H * W * C)
    y = y.permute(0, 2, 3, 4, 1).contiguous().view(N, T, H * W * C)
    return F.l1_loss(x, y, reduction='none').mean(dim=2)


def psnr(x, y, v_max=1.):
    """
    Compute the per-frame PSNR
    """
    return 10 * torch.log10((v_max ** 2) / (mse(x, y) + 1e-9))


def ssim(x, y, v_max=1., win_size=11):
    """
    Compute the per-frame SSIM
    """
    check_shape(x, y)
    N, C, T, H, W = x.shape
    x = x.permute(0, 2, 1, 3, 4).contiguous().view(N * T, C, H, W)
    y = y.permute(0, 2, 1, 3, 4).contiguous().view(N * T, C, H, W)
    return pytorch_msssim.ssim(x, y, v_max, win_size=win_size, size_average=False).view(N, T)


def ms_ssim(x, y, v_max=1., win_size=11):
    """
    Compute the per-frame MS-SSIM
    """
    check_shape(x, y)
    N, C, T, H, W = x.shape
    x = x.permute(0, 2, 1, 3, 4).contiguous().view(N * T, C, H, W)
    y = y.permute(0, 2, 1, 3, 4).contiguous().view(N * T, C, H, W)
    return pytorch_msssim.ms_ssim(x, y, v_max, win_size=win_size, size_average=False).view(N, T)


def laplacian_l1(x, y, k=20):
    """
    Compute mask loss
    """
    # resize y to fit the shape of x
    N, C, T, H, W = y.shape
    y_reshaped = y.permute(0, 2, 1, 3, 4).contiguous().view(N * T, C, H, W)
    if x.shape[-2:] != y.shape[-2:]:
        y_reshaped = F.interpolate(y_reshaped, size=x.shape[-2:], mode='bilinear', align_corners=False)             # [N*T, C, H', W']
    N, C, T, H, W = x.shape
    x_reshaped = x.permute(0, 2, 1, 3, 4).contiguous().view(N * T, C, H, W)

    # Laplacian kernel
    kernel = torch.tensor([[-1., -1., -1.],
                                        [-1., 8., -1.],
                                        [-1., -1., -1.]], dtype=x.dtype, device=x.device)
    kernel = kernel.view(1, 1, 3, 3).repeat(C, 1, 1, 1)
    y_lap = F.conv2d(y_reshaped, kernel, padding='same', groups=C)
    y_lap = torch.tanh(torch.abs(y_lap) * k)

    return F.l1_loss(x_reshaped, y_lap.detach(), reduction='none').view(N, T, -1).mean(dim=2)


def laplacian_mse(x, y, k=20):
    """
    Compute mask loss
    """
    # resize y to fit the shape of x
    N, C, T, H, W = y.shape
    if x.shape[-2:] != y.shape[-2:]:
        y_reshaped = y.permute(0, 2, 1, 3, 4).contiguous().view(N * T, C, H, W)
        y_reshaped = F.interpolate(y_reshaped, size=x.shape[-2:], mode='bilinear', align_corners=False)             # [N*T, C, H', W']
    N, C, T, H, W = x.shape
    x_reshaped = x.permute(0, 2, 1, 3, 4).contiguous().view(N * T, C, H, W)

    # Laplacian kernel
    kernel = torch.tensor([[-1., -1., -1.],
                                        [-1., 8., -1.],
                                        [-1., -1., -1.]], dtype=x.dtype, device=x.device)
    kernel = kernel.view(1, 1, 3, 3).repeat(C, 1, 1, 1)
    y_lap = F.conv2d(y_reshaped, kernel, padding='same', groups=C)
    y_lap = torch.sigmoid(y_lap * k)

    return F.mse_loss(x_reshaped, y_lap.detach(), reduction='none').view(N, T, -1).mean(dim=2)


def fft_l1(x, y):
    x_fft = torch.fft.fftn(x, dim=[2,3,4])
    y_fft = torch.fft.fftn(y, dim=[2,3,4])
    return l1(x_fft, y_fft)


def fft_mse(x, y):
    x_fft = torch.fft.fftn(x, dim=[2,3,4])
    y_fft = torch.fft.fftn(y, dim=[2,3,4])
    return mse(x_fft, y_fft)


def mask_loss(x, y, model, loss_type='l1'):
    N, C, T, H, W = y.shape
    y = y.permute(0, 2, 1, 3, 4).contiguous().view(N*T, C, H, W)

    # resize y to fit the shape of x
    h, w = x.shape[-2:]
    if x.shape[-2:] != y.shape[-2:]:
        y = F.interpolate(y, size=(h, w), mode='bilinear', align_corners=True)
    y = model(y)
    y = y.view(N, T, C, h, w).permute(0, 2, 1, 3, 4)
    return _compute_loss(loss_type, x, y.detach())


class Gauss_model(nn.Module):
    def __init__(self, kernel_size, sigma, k=20, abs=False):
        super().__init__()
        self.model_gauss = torchvision.transforms.GaussianBlur(kernel_size, sigma)
        self.k = k
        self.abs = abs

    def forward(self, x):
        out = x - self.model_gauss(x)
        if self.abs:
            out = torch.abs(torch.tanh(out * self.k))
        else:
            out = torch.tanh(out * self.k)
        out = torch.clamp(out, 0, 1)
        return out


def compute_loss(name, x, y, model=None):
    assert x.ndim == 5 and y.ndim == 5, 'inputs are expected to have 5D ([N, C, T, H, W])'
    x, y = x.float(), y.float()
    if 'luma' in name:
        return _compute_loss(name.replace('_luma', ''), x[:, [0]], y[:, [0]], model)
    elif 'chroma' in name:
        return _compute_loss(name.replace('_chroma', ''), x[:, 1:], y[:, 1:], model)
    elif 'mask' in name:
        return mask_loss(x, y, model, loss_type=name.replace('_mask', ''))
    else:
        raise NotImplementedError


def _compute_loss(name, x, y, model=None):
    # assert x.ndim == 5 and y.ndim == 5, 'inputs are expected to have 5D ([N, C, T, H, W])'
    # x, y = x.float(), y.float()
    if name == 'base':
        return 0.7 * l1(x, y) + 0.3 * (1. - ms_ssim(x, y, win_size=5))
    elif name == 'mse':
        return mse(x, y)
    elif name == 'l1':
        return l1(x, y)
    elif name == 'ssim':
        return 1. - ssim(x, y)
    elif name == 'ssim_3x3':
        return 1. - ssim(x, y, win_size=3)
    elif name == 'ssim_5x5':
        return 1. - ssim(x, y, win_size=5)
    elif name == 'ssim_7x7':
        return 1. - ssim(x, y, win_size=7)
    elif name == 'ms-ssim':
        return 1. - ms_ssim(x, y)
    elif name == 'ms-ssim_3x3':
        return 1. - ms_ssim(x, y, win_size=3)
    elif name == 'ms-ssim_5x5':
        return 1. - ms_ssim(x, y, win_size=5)
    elif name == 'ms-ssim_7x7':
        return 1. - ms_ssim(x, y, win_size=7)
    elif name == 'laplacian_l1':
        return laplacian_l1(x, y)
    elif name == 'laplacian_mse':
        return laplacian_mse(x, y)
    elif name == 'fft_l1':              # FFT on Y channel only
        return fft_l1(x[:, [0]], y[:, [0]])
    elif name == 'fft_mse':         # FFT on Y channel only
        return fft_mse(x[:, [0]], y[:, [0]])
    else:
        raise ValueError


def compute_metric(name, x, y):
    assert x.ndim == 5 and y.ndim == 5, 'inputs are expected to have 5D ([N, C, T, H, W])'
    x, y = x.float(), y.float()
    if name == 'mse':
        return mse(x, y)
    elif name == 'l1':
        return l1(x, y)
    elif name == 'psnr':
        return psnr(x, y)
    elif name == 'ssim':
        return ssim(x, y)
    elif name == 'ms-ssim':
        return ms_ssim(x, y)
    else:
        raise ValueError


def compute_regularization(name, model):
    raise ValueError

