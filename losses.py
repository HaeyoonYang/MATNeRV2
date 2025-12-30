"""
Losses
"""
from utils import *
from torchvision.models import vgg11, VGG11_Weights, vgg19, VGG19_Weights


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


def compute_loss(name, x, y):
    assert x.ndim == 5 and y.ndim == 5, 'inputs are expected to have 5D ([N, C, T, H, W])'
    x, y = x.float(), y.float()
    if name == 'mse':
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



class Gauss_model(nn.Module):
    def __init__(self, kernel_size, sigma, k=20):
        super().__init__()
        self.model_gauss = torchvision.transforms.GaussianBlur(kernel_size, sigma)
        self.k = k

    def forward(self, x):
        out = x - self.model_gauss(x)
        out = torch.sigmoid(out * self.k)
        return out


def gauss_loss(x, y, gauss_model, loss_type='l1'):
    N, C, T, H, W = x.shape
    x = x.permute(0, 2, 1, 3, 4).view(N*T, C, H, W)
    y = y.permute(0, 2, 1, 3, 4).view(N*T, C, H, W)

    x_gauss = gauss_model(x).view(N, T, C, H, W).permute(0, 2, 1, 3, 4)
    y_gauss = gauss_model(y).view(N, T, C, H, W).permute(0, 2, 1, 3, 4)
    return compute_loss(loss_type, x_gauss, y_gauss)


def mask_loss(x, y, model, loss_type='l1'):
    """
    x: list of masks
    y: GT image
    """
    N, C, T, H, W = y.shape
    y = y.permute(0, 2, 1, 3, 4).view(N*T, C, H, W)
    y_mean = y.mean(dim=1, keepdim=True)

    loss = torch.tensor(0.0, device=y.device)
    for mask in x:
        h, w = mask.shape[-2:]
        if mask.shape[-2:] != y_mean.shape[-2:]:
            y_i = F.interpolate(y_mean, size=(h, w), mode='bilinear', align_corners=True)
        else:
            y_i = y_mean
        y_i = model(y_i).as_strided(size=mask.size(), stride=mask.stride()).detach()
        loss = loss + compute_loss(loss_type, mask, y_i)
    return loss


class VGG_model(nn.Module):
    def __init__(self, vgg_type, depth):
        super().__init__()
        vgg11_layer = [0, 3, 6, 8, 11, 13, 16, 18]
        vgg19_layer = [0, 2, 5, 7, 10, 12, 14, 16, 19, 21, 23, 25, 28, 30, 32, 34]

        self.model = nn.ModuleList()
        if vgg_type == 'vgg11':
            model_vgg = vgg11(weights=VGG11_Weights.DEFAULT)
            self.model.append(model_vgg.features[vgg11_layer[0]])
            for i in range(1, depth):
                self.model.append(
                    model_vgg.features[vgg11_layer[i-1]+1:vgg11_layer[i]+1]
                )
        elif vgg_type == 'vgg19':
            model_vgg = vgg19(weights=VGG19_Weights.DEFAULT)
            self.model.append(model_vgg.features[vgg19_layer[0]])
            for i in range(1, depth):
                self.model.append(
                    model_vgg.features[vgg19_layer[i-1]+1:vgg19_layer[i]+1]
                )

        self.freeze()

    def freeze(self):
        for param in self.model.parameters():
            param.requires_grad_(False)

    def forward(self, x):
        out_list = []
        for layer in self.model:
            x = layer(x)
            out_list.append(x)
        return out_list

def vgg_loss(x, y, model, loss_type='l1'):
    """
    x: output image: [N, C, T, H, W]
    y: GT image: [N, C, T, H, W]
    """
    assert x.shape == y.shape, "The shapes of the tensors doesn't match"
    N, C, T, H, W = x.shape

    x = x.permute(0, 2, 1, 3, 4).view(N*T, C, H, W)
    y = y.permute(0, 2, 1, 3, 4).view(N*T, C, H, W)

    x_out = model(x)            # list of features
    y_out = model(y)            # list of features

    loss = None
    for i in range(len(x_out)):
        x_i = x_out[i].view(N, T, -1, H, W).permute(0, 2, 1, 3, 4)
        y_i = y_out[i].view(N, T, -1, H, W).permute(0, 2, 1, 3, 4)
        loss = compute_loss(loss_type, x_i, y_i) if loss is None else loss + compute_loss(loss_type, x_i, y_i)
    return loss


class Sobel_model(nn.Module):
    def __init__(self):
        super().__init__()

        kernels = [
            [[-1,  0,  1], [-2,  0,  2], [-1,  0,  1]],
            [[ 0,  1,  2], [-1,  0,  1], [-2, -1,  0]],
            [[-1, -2, -1], [ 0,  0,  0], [ 1,  2,  1]],
            [[-2, -1,  0], [-1,  0,  1], [ 0,  1,  2]],
            [[ 1,  0, -1], [ 2,  0, -2], [ 1,  0, -1]],
            [[ 0, -1, -2], [ 1,  0, -1], [ 2,  1,  0]],
            [[ 1,  2,  1], [ 0,  0,  0], [-1, -2, -1]],
            [[ 2,  1,  0], [ 1,  0, -1], [ 0, -1, -2]]
        ]

        self.kernel = torch.cat([torch.tensor(k, dtype=torch.float32).repeat(3, 1, 1).unsqueeze(0) for k in kernels], dim=0)         # [8, 3, 3, 3]

    def forward(self, x):
        filtered = F.conv2d(x, self.kernel.to(x.device), padding='same')
        combined = torch.amax(filtered, dim=1, keepdim=True)            # [N, 1, H, W]
        return combined
    
def sobel_loss(x, y, model, loss_type='l1'):
    """
    x: list of masks
    y: GT image [N, C, T, H, W]
    """
    N, C, T, H, W = y.shape
    y = y.permute(0, 2, 1, 3, 4).view(N*T, C, H, W)
    y_out = model(y)

    loss = None
    for x_i in x:
        if x_i.shape[-2:] != y_out.shape[-2:]:
            scale = (y_out.shape[-2] // x_i.shape[-2], y_out.shape[-1] // x_i.shape[-1])
            y_i = F.max_pool2d(y_out, kernel_size=scale, stride=scale)
        else:
            y_i = y_out
        y_i = y_i.as_strided(size=x_i.size(), stride=x_i.stride())
        loss = compute_loss(loss_type, x_i, y_i) if loss is None else loss + compute_loss(loss_type, x_i, y_i)
    return loss
