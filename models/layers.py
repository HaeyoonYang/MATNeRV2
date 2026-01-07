"""
Shared Layers - all assume 5D inputs
"""
from .utils import *
from timm.models.layers import DropPath


"""
Basic Layers
"""
class Sin(nn.Module):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return torch.sin(input)


class FeatureGrid(nn.Module):
    """
    The module for storing learnable embedding.
    """
    def __init__(self, shape, init_scale):
        super().__init__()
        self.register_parameter(name='weight', param=nn.Parameter(torch.zeros(shape, dtype=torch.float32), requires_grad=True))
        torch.nn.init.uniform_(self.weight, -init_scale, init_scale)

    def forward(self):
        return self.weight #.clone()


class FeatureBuffer(nn.Module):
    """
    The module for storing non-learnable embedding, e.g. for autoencoder.
    """
    def __init__(self, shape):
        super().__init__()
        self.register_parameter(name='weight', param=nn.Parameter(torch.zeros(shape), requires_grad=False))

    def forward(self, idx, x):
        if x is not None:
            output = self.weight[idx] = x.detach().to(self.weight.dtype)
        else:
            output = self.weight[idx]
        return output


class Conv2d(nn.Conv2d):
    def forward(self, input):
        N, T, H, W, _ = input.shape
        x = input.view(N * T, H, W, -1).permute(0, 3, 1, 2)
        x = F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        x = x.permute(0, 2, 3, 1).view(N, T, H, W, -1)
        return x


class Conv3d(nn.Conv3d):
    def forward(self, input):
        x = input.permute(0, 4, 1, 2, 3)
        x = F.conv3d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        x = x.permute(0, 2, 3, 4, 1)
        return x


class AdaptiveAvgPool2d(nn.AdaptiveAvgPool2d):
    def forward(self, input):
        N, T, H, W, C = input.shape
        x = input.view(N * T, H, W, -1).permute(0, 3, 1, 2)         # [N*T, C, H, W]
        x = F.adaptive_avg_pool2d(x, self.output_size)
        x = x.permute(0, 2, 3, 1).view(N, T, self.output_size, self.output_size, C)
        return x

"""
Advanced Layers
"""
class LinearBlock(nn.Module):
    def __init__(self, in_features: int, out_features: int,
                 act='gelu', norm='layernorm', bias: bool = True,
                 norm_first=False) -> None:
        super().__init__()
        self.norm_first = norm_first
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.norm = get_norm(norm)(in_features if self.norm_first else out_features)
        self.act = get_activation(act)()

    def forward(self, input):
        x = self.linear(self.norm(input)) if self.norm_first else self.norm(self.linear(input))
        x = self.act(x)
        return x


class Conv2dBlock(nn.Module):
    def __init__(self, in_features: int, out_features: int, kernel_size=3, 
                 act='gelu', norm='layernorm', bias: bool = True,
                 norm_first=False):
        super().__init__()
        self.norm_first = norm_first
        self.conv = Conv2d(in_channels=in_features, out_channels=out_features, kernel_size=kernel_size, padding='same', bias=bias)
        self.norm = get_norm(norm)(in_features if self.norm_first else out_features)
        self.act = get_activation(act)()

    def forward(self, input):
        x = self.conv(self.norm(input)) if self.norm_first else self.norm(self.conv(input))
        x = self.act(x)
        return x


class BlockBase(nn.Module):
    def __init__(self, in_features, out_features, layerscale_init, droppath):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.layerscale_init = layerscale_init

        # layer scale
        if self.in_features == self.out_features and layerscale_init > 0.:
            self.layerscale = torch.nn.Parameter(self.layerscale_init * torch.ones([self.out_features]), requires_grad=True)
        else:
            self.layerscale = None

        # Stochastic Depth
        if self.in_features == self.out_features and droppath > 0.:
            self.droppath = DropPath(droppath)
        else:
            self.droppath = None

    def extra_repr(self):
        s = 'in_features={in_features}, out_features={out_features}, layerscale_init={layerscale_init}'
        return s.format(**self.__dict__)

    def block_forward(self, input):
        raise NotImplementedError

    def forward(self, input, mask=None):
        x = self.block_forward(input)

        if self.layerscale is not None:
            x = self.layerscale * x

        if self.droppath is not None:
            x = self.droppath(x)

        if mask is not None:
            x = mask * x

        if self.in_features == self.out_features:
            x = x + input

        return x


class MLPBlock(BlockBase):
    def __init__(self, in_features: int, out_features: int, hidden_features: int,
                 act='gelu', norm='layernorm', bias: bool = True,
                 layerscale_init=0., dropout=0., droppath=0.):
        super().__init__(in_features, out_features, layerscale_init, droppath)
        self.norm = get_norm(norm)(in_features)
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = get_activation(act)()
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.dropout2 = nn.Dropout(dropout)

    def block_forward(self, input):
        x = self.norm(input)
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.dropout2(x)
        return x


class ConvNeXtBlock(BlockBase):
    def __init__(self, in_features: int, out_features: int, hidden_features: int, kernel_size=3, 
                 act='gelu', norm='layernorm', bias: bool = True,
                 layerscale_init=0., dropout=0., droppath=0.):
        super().__init__(in_features, out_features, layerscale_init, droppath)
        self.dconv = Conv2d(in_features, in_features, kernel_size, groups=in_features, padding='same', bias=bias)
        self.norm = get_norm(norm)(in_features)
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = get_activation(act)()
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.dropout2 = nn.Dropout(dropout)

    def block_forward(self, input):
        x = self.dconv(input)
        x = self.norm(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.dropout2(x)
        return x


class ConvNeXtBlockLessNorm(BlockBase):
    def __init__(self, in_features: int, out_features: int, hidden_features: int, kernel_size=3, 
                 act='gelu', norm='layernorm', bias: bool = True,
                 layerscale_init=0., dropout=0., droppath=0.):
        super().__init__(in_features, out_features, layerscale_init, droppath)
        self.dconv = Conv2d(in_features, in_features, kernel_size, groups=in_features, padding='same', bias=bias)
        self.norm = get_norm(norm if in_features == out_features else 'none')(in_features)
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = get_activation(act)()
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.dropout2 = nn.Dropout(dropout)
    
    def block_forward(self, input):
        x = self.dconv(input)
        x = self.norm(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.dropout2(x)
        return x


class GridTrilinear3D(nn.Module):
    """
    The module for mapping feature maps to a fixed size with trilinear interpolation.
    """
    def __init__(self, output_size, align_corners=False):
        super().__init__()
        self.output_size = output_size
        self.align_corners = align_corners

    def forward(self, x: torch.Tensor):
        #x = F.interpolate(x, size=self.output_size, mode='trilinear', align_corners=self.align_corners)
        T, H, W, C = x.shape
        assert H == self.output_size[1] and W == self.output_size[2], 'F.interpolate has incorrect results in some cases, so use only temporal scale'
        x = x.view(1, 1, T, H * W * C)
        x = F.interpolate(x, size=(self.output_size[0], H * W * C), mode='bilinear', align_corners=self.align_corners)
        x = x.view(self.output_size + (C,))
        return x


class GridUpsample(nn.Module):
    """
    The module for mapping feature maps to a fixed size with trilinear interpolation.
    """
    def __init__(self, output_size, align_corners=False):
        super().__init__()
        self.output_size = output_size
        self.align_corners = align_corners

    def forward(self, x: torch.Tensor):
        T, H, W, C = x.shape
        # temporal interpolation
        x = x.view(1, 1, T, H * W * C)
        x = F.interpolate(x, size=(self.output_size[0], H * W * C), mode='bilinear', align_corners=self.align_corners)
        x = x.view(self.output_size[0], H, W, C)
        # spatial interpolation
        if (H, W) != self.output_size[1:]:
            x = x.permute(0, 3, 1, 2)           # [T, C, H, W]
            x = F.interpolate(x, size=self.output_size[1:], mode='bilinear', align_corners=self.align_corners)
            x = x.permute(0, 2, 3, 1)           # [T, H, W, C]
        return x


"""
Blocks
"""
class HiNeRVBlock(nn.Module):
    def __init__(self, in_features: int, out_features: int, hidden_features: int,
                           kernel_size=3, act='gelu', norm='layernorm', bias: bool=True,
                           layerscale_init=0., dropout=0., droppath=0., depths=3):
        super().__init__()
        self.blocks = nn.ModuleList()
        for i in range(depths):
            self.blocks.append(ConvNeXtBlock(in_features=in_features if i==0 else out_features, 
                                             out_features=out_features, hidden_features=hidden_features, kernel_size=kernel_size, 
                                             act=act, norm=norm, bias=bias, layerscale_init=layerscale_init, dropout=dropout, droppath=droppath))

    def forward(self, idx: torch.IntTensor, x: torch.Tensor, mask=None):
        for layer in self.blocks:
            x = layer(x, mask)
        return x, None


class MATBlock(nn.Module):
    """
    conv (d) --> MAT (x + mask * t_x) --> conv (d)
    """
    def __init__(self, in_features: int, out_features: int, hidden_features: int,
                           kernel_size=3, act='gelu', norm='layernorm', bias: bool=True,
                           layerscale_init=0., dropout=0., droppath=0., depths=2,
                           mat_dimension=(600, 8)):
        super().__init__()
        # Affine Transform
        in_dim = mat_dimension[1]
        self.gammas = nn.Parameter(nn.init.uniform_(torch.empty(mat_dimension, dtype=torch.float32), -1e-3, 1e-3), requires_grad=True)
        self.betas = nn.Parameter(nn.init.uniform_(torch.empty(mat_dimension, dtype=torch.float32), -1e-3, 1e-3), requires_grad=True)
        self.g_linear = nn.Linear(in_dim, out_features)
        self.b_linear = nn.Linear(in_dim, out_features)

        self.conv1, self.conv2 = [nn.ModuleList() for _ in range(2)]
        for i in range(depths):
            self.conv1.append(ConvNeXtBlock(in_features=in_features if i==0 else out_features, out_features=out_features, hidden_features=hidden_features, kernel_size=kernel_size, act=act, norm=norm, bias=bias, layerscale_init=layerscale_init, dropout=dropout, droppath=droppath))
            self.conv2.append(ConvNeXtBlock(in_features=out_features, out_features=out_features, hidden_features=hidden_features, kernel_size=kernel_size, act=act, norm=norm, bias=bias, layerscale_init=layerscale_init, dropout=dropout, droppath=droppath))

        self.mask_conv = nn.Sequential(
            Conv2d(out_features, 1, kernel_size=3, padding='same'),
            nn.Sigmoid(),
        )

    def forward(self, idx: torch.IntTensor, x: torch.Tensor, mask=None):
        """
        idx: t-index. [N, T]
        """
        for layer1 in self.conv1:
            x = layer1(x, mask)

        # masked affine transform
        m_out = self.mask_conv(x)
        gamma = self.g_linear(self.gammas[idx])             # [N, T, C]
        beta = self.b_linear(self.betas[idx])                       # [N, T, C]
        t_x = x * gamma[:, :, None, None, :] + beta[:, :, None, None, :]

        x = x + t_x * m_out

        for layer2 in self.conv2:
            x = layer2(x, mask)
        return x, m_out


class MATBlock2(nn.Module):
    """
    conv (d) --> MAT (x + mask * t_x) --> conv (1)
    """
    def __init__(self, in_features: int, out_features: int, hidden_features: int,
                           kernel_size=3, act='gelu', norm='layernorm', bias: bool=True,
                           layerscale_init=0., dropout=0., droppath=0., depths=2,
                           mat_dimension=(600, 8)):
        super().__init__()
        # Affine Transform
        in_dim = mat_dimension[1]
        self.gammas = nn.Parameter(nn.init.uniform_(torch.empty(mat_dimension, dtype=torch.float32), -1e-3, 1e-3), requires_grad=True)
        self.betas = nn.Parameter(nn.init.uniform_(torch.empty(mat_dimension, dtype=torch.float32), -1e-3, 1e-3), requires_grad=True)
        self.g_linear = nn.Linear(in_dim, out_features)
        self.b_linear = nn.Linear(in_dim, out_features)

        self.conv1, self.conv2 = [nn.ModuleList() for _ in range(2)]
        for i in range(depths):
            self.conv1.append(ConvNeXtBlock(in_features=in_features if i==0 else out_features, out_features=out_features, hidden_features=hidden_features, kernel_size=kernel_size, act=act, norm=norm, bias=bias, layerscale_init=layerscale_init, dropout=dropout, droppath=droppath))

        self.conv2.append(ConvNeXtBlock(in_features=out_features, out_features=out_features, hidden_features=hidden_features, kernel_size=kernel_size, act=act, norm=norm, bias=bias, layerscale_init=layerscale_init, dropout=dropout, droppath=droppath))

        self.mask_conv = nn.Sequential(
            Conv2d(out_features, 1, kernel_size=3, padding='same'),
            nn.Sigmoid(),
        )

    def forward(self, idx: torch.IntTensor, x: torch.Tensor, mask=None):
        """
        idx: t-index. [N, T]
        """
        for layer1 in self.conv1:
            x = layer1(x, mask)

        # masked affine transform
        m_out = self.mask_conv(x)
        gamma = self.g_linear(self.gammas[idx])             # [N, T, C]
        beta = self.b_linear(self.betas[idx])                       # [N, T, C]
        t_x = x * gamma[:, :, None, None, :] + beta[:, :, None, None, :]

        x = x + t_x * m_out

        for layer2 in self.conv2:
            x = layer2(x, mask)
        return x, m_out


class MATBlock3(nn.Module):
    """
    conv (d) --> MAT (x + mask * t_x) --> conv (d) 
                                                                                --> cross_conv (1): for Chroma path
    """
    def __init__(self, in_features: int, out_features: int, hidden_features: int,
                           kernel_size=3, act='gelu', norm='layernorm', bias: bool=True,
                           layerscale_init=0., dropout=0., droppath=0., depths=2,
                           mat_dimension=(600, 8)):
        super().__init__()
        # Affine Transform
        in_dim = mat_dimension[1]
        self.gammas = nn.Parameter(nn.init.uniform_(torch.empty(mat_dimension, dtype=torch.float32), -1e-3, 1e-3), requires_grad=True)
        self.betas = nn.Parameter(nn.init.uniform_(torch.empty(mat_dimension, dtype=torch.float32), -1e-3, 1e-3), requires_grad=True)
        self.g_linear = nn.Linear(in_dim, out_features)
        self.b_linear = nn.Linear(in_dim, out_features)

        self.conv1, self.conv2 = [nn.ModuleList() for _ in range(2)]
        for i in range(depths):
            self.conv1.append(ConvNeXtBlock(in_features=in_features if i==0 else out_features, out_features=out_features, hidden_features=hidden_features, kernel_size=kernel_size, act=act, norm=norm, bias=bias, layerscale_init=layerscale_init, dropout=dropout, droppath=droppath))
            self.conv2.append(ConvNeXtBlock(in_features=out_features, out_features=out_features, hidden_features=hidden_features, kernel_size=kernel_size, act=act, norm=norm, bias=bias, layerscale_init=layerscale_init, dropout=dropout, droppath=droppath))

        self.mask_conv = nn.Sequential(
            Conv2d(out_features, out_features, kernel_size=3, padding='same', groups=out_features),
            nn.ReLU(inplace=True),
            Conv2d(out_features, 1, kernel_size=1, padding='same'),
            nn.Sigmoid(),
        )
        self._initialize_maskconv()

        self.cross_conv = Conv2d(out_features, 1, kernel_size=1, padding='same')

    def _initialize_maskconv(self):
        laplacian = torch.tensor([[-1., -1., -1.],
                                            [-1., 8., -1.],
                                            [-1., -1., -1.]], dtype=torch.float32)
        with torch.no_grad():
            for i in range(self.mask_conv[0].weight.shape[0]):
                self.mask_conv[0].weight[i, 0, :, :] = laplacian
            
            nn.init.constant_(self.mask_conv[2].weight, 1.0 / self.mask_conv[2].in_channels)

    def forward(self, idx: torch.IntTensor, x: torch.Tensor, mask=None):
        """
        idx: t-index. [N, T]
        """
        for layer1 in self.conv1:
            x = layer1(x, mask)

        # masked affine transform
        m_out = self.mask_conv(x)
        gamma = self.g_linear(self.gammas[idx])             # [N, T, C]
        beta = self.b_linear(self.betas[idx])                       # [N, T, C]
        t_x = x * gamma[:, :, None, None, :] + beta[:, :, None, None, :]

        x = x + t_x * m_out

        for layer2 in self.conv2:
            x = layer2(x, mask)

        cross_lc = self.cross_conv(x)
        return x, m_out, cross_lc


class MATBlock4(nn.Module):
    """
    conv (d) --> rMAT (x + (1+mask) * t_x) --> conv (d) 
    """
    def __init__(self, in_features: int, out_features: int, hidden_features: int,
                           kernel_size=3, act='gelu', norm='layernorm', bias: bool=True,
                           layerscale_init=0., dropout=0., droppath=0., depths=2,
                           mat_dimension=(600, 8)):
        super().__init__()
        # Affine Transform
        in_dim = mat_dimension[1]
        self.gammas = nn.Parameter(nn.init.uniform_(torch.empty(mat_dimension, dtype=torch.float32), -1e-3, 1e-3), requires_grad=True)
        self.betas = nn.Parameter(nn.init.uniform_(torch.empty(mat_dimension, dtype=torch.float32), -1e-3, 1e-3), requires_grad=True)
        self.g_linear = nn.Linear(in_dim, out_features)
        self.b_linear = nn.Linear(in_dim, out_features)

        self.conv1, self.conv2 = [nn.ModuleList() for _ in range(2)]
        for i in range(depths):
            self.conv1.append(ConvNeXtBlock(in_features=in_features if i==0 else out_features, out_features=out_features, hidden_features=hidden_features, kernel_size=kernel_size, act=act, norm=norm, bias=bias, layerscale_init=layerscale_init, dropout=dropout, droppath=droppath))
            self.conv2.append(ConvNeXtBlock(in_features=out_features, out_features=out_features, hidden_features=hidden_features, kernel_size=kernel_size, act=act, norm=norm, bias=bias, layerscale_init=layerscale_init, dropout=dropout, droppath=droppath))

        self.mask_conv = nn.Sequential(
            Conv2d(out_features, out_features, kernel_size=3, padding='same', groups=out_features),
            nn.ReLU(inplace=True),
            Conv2d(out_features, 1, kernel_size=1, padding='same'),
            nn.Sigmoid(),
        )
        self._initialize_maskconv()

    def _initialize_maskconv(self):
        laplacian = torch.tensor([[-1., -1., -1.],
                                            [-1., 8., -1.],
                                            [-1., -1., -1.]], dtype=torch.float32)
        with torch.no_grad():
            for i in range(self.mask_conv[0].weight.shape[0]):
                self.mask_conv[0].weight[i, 0, :, :] = laplacian
            
            nn.init.constant_(self.mask_conv[2].weight, 1.0 / self.mask_conv[2].in_channels)

    def forward(self, idx: torch.IntTensor, x: torch.Tensor, mask=None):
        """
        idx: t-index. [N, T]
        """
        for layer1 in self.conv1:
            x = layer1(x, mask)

        # masked affine transform
        m_out = self.mask_conv(x)
        gamma = self.g_linear(self.gammas[idx])             # [N, T, C]
        beta = self.b_linear(self.betas[idx])                       # [N, T, C]
        t_x = x * gamma[:, :, None, None, :] + beta[:, :, None, None, :]

        x = x + t_x * (1 + m_out)

        for layer2 in self.conv2:
            x = layer2(x, mask)

        return x, m_out


class MATBlock5(nn.Module):
    """
    conv (d) --> rMAT (x + (1+mask) * t_x) --> conv (d) 
    mask conv: curr_mask = sigmoid(upsampled_prev_mask + mask_conv(cat([upsampled_prev_mask, x])))
    """
    def __init__(self, in_features: int, out_features: int, hidden_features: int,
                           kernel_size=3, act='gelu', norm='layernorm', bias: bool=True,
                           layerscale_init=0., dropout=0., droppath=0., depths=2,
                           mat_dimension=(600, 8), first=False):
        super().__init__()
        # Affine Transform
        in_dim = mat_dimension[1]
        self.gammas = nn.Parameter(nn.init.uniform_(torch.empty(mat_dimension, dtype=torch.float32), -1e-3, 1e-3), requires_grad=True)
        self.betas = nn.Parameter(nn.init.uniform_(torch.empty(mat_dimension, dtype=torch.float32), -1e-3, 1e-3), requires_grad=True)
        self.g_linear = nn.Linear(in_dim, out_features)
        self.b_linear = nn.Linear(in_dim, out_features)

        self.conv1, self.conv2 = [nn.ModuleList() for _ in range(2)]
        for i in range(depths):
            self.conv1.append(ConvNeXtBlock(in_features=in_features if i==0 else out_features, out_features=out_features, hidden_features=hidden_features, kernel_size=kernel_size, act=act, norm=norm, bias=bias, layerscale_init=layerscale_init, dropout=dropout, droppath=droppath))
            self.conv2.append(ConvNeXtBlock(in_features=out_features, out_features=out_features, hidden_features=hidden_features, kernel_size=kernel_size, act=act, norm=norm, bias=bias, layerscale_init=layerscale_init, dropout=dropout, droppath=droppath))

        if first:
            mask_features = out_features
        else:
            mask_features = out_features + 1
        self.mask_conv = nn.Sequential(
            Conv2d(mask_features, mask_features, kernel_size=3, padding='same', groups=mask_features),
            nn.ReLU(inplace=True),
            Conv2d(mask_features, 1, kernel_size=1, padding='same'),
        )
        self._initialize_maskconv()

    def _initialize_maskconv(self):
        laplacian = torch.tensor([[-1., -1., -1.],
                                            [-1., 8., -1.],
                                            [-1., -1., -1.]], dtype=torch.float32)
        with torch.no_grad():
            for i in range(self.mask_conv[0].weight.shape[0]):
                self.mask_conv[0].weight[i, 0, :, :] = laplacian
            
            nn.init.constant_(self.mask_conv[2].weight, 1.0 / self.mask_conv[2].in_channels)

    def forward(self, idx: torch.IntTensor, x: torch.Tensor, mask=None, m_in=None):
        """
        idx: t-index. [N, T]
        """
        for layer1 in self.conv1:
            x = layer1(x, mask)

        # masked affine transform
        if m_in is not None:
            # TODO
            m_in_upsampled = F.interpolate(m_in.permute(0, 4, 1, 2, 3), size=x.shape[2:4], mode='bilinear', align_corners=False).permute(0, 2, 3, 1)

        m_out = self.mask_conv(x)



        gamma = self.g_linear(self.gammas[idx])             # [N, T, C]
        beta = self.b_linear(self.betas[idx])                       # [N, T, C]
        t_x = x * gamma[:, :, None, None, :] + beta[:, :, None, None, :]

        x = x + t_x * (1 + m_out)

        for layer2 in self.conv2:
            x = layer2(x, mask)

        return x, m_out


class ATBlock(nn.Module):
    """
    conv (d) --> AT (w/o skip con) --> conv (d)
    """
    def __init__(self, in_features: int, out_features: int, hidden_features: int,
                           kernel_size=3, act='gelu', norm='layernorm', bias: bool=True,
                           layerscale_init=0., dropout=0., droppath=0., depths=2,
                           mat_dimension=(600, 8)):
        super().__init__()
        # Affine Transform
        in_dim = mat_dimension[1]
        self.gammas = nn.Parameter(nn.init.uniform_(torch.empty(mat_dimension, dtype=torch.float32), -1e-3, 1e-3), requires_grad=True)
        self.betas = nn.Parameter(nn.init.uniform_(torch.empty(mat_dimension, dtype=torch.float32), -1e-3, 1e-3), requires_grad=True)
        self.g_linear = nn.Linear(in_dim, out_features)
        self.b_linear = nn.Linear(in_dim, out_features)

        self.conv1, self.conv2 = [nn.ModuleList() for _ in range(2)]
        for i in range(depths):
            self.conv1.append(ConvNeXtBlock(in_features=in_features if i==0 else out_features, out_features=out_features, hidden_features=hidden_features, kernel_size=kernel_size, act=act, norm=norm, bias=bias, layerscale_init=layerscale_init, dropout=dropout, droppath=droppath))
            self.conv2.append(ConvNeXtBlock(in_features=out_features, out_features=out_features, hidden_features=hidden_features, kernel_size=kernel_size, act=act, norm=norm, bias=bias, layerscale_init=layerscale_init, dropout=dropout, droppath=droppath))

    def forward(self, idx: torch.IntTensor, x: torch.Tensor, mask=None):
        """
        idx: t-index. [N, T]
        """
        for layer1 in self.conv1:
            x = layer1(x, mask)

        # affine transform
        gamma = self.g_linear(self.gammas[idx])             # [N, T, C]
        beta = self.b_linear(self.betas[idx])                       # [N, T, C]
        x = x * gamma[:, :, None, None, :] + beta[:, :, None, None, :]

        for layer2 in self.conv2:
            x = layer2(x, mask)
        return x, None


class MMBlock(nn.Module):
    """
    conv (d) --> MM (x * mask) --> conv (d)
    """
    def __init__(self, in_features: int, out_features: int, hidden_features: int,
                           kernel_size=3, act='gelu', norm='layernorm', bias: bool=True,
                           layerscale_init=0., dropout=0., droppath=0., depths=2):
        super().__init__()

        self.conv1, self.conv2 = [nn.ModuleList() for _ in range(2)]
        for i in range(depths):
            self.conv1.append(ConvNeXtBlock(in_features=in_features if i==0 else out_features, out_features=out_features, hidden_features=hidden_features, kernel_size=kernel_size, act=act, norm=norm, bias=bias, layerscale_init=layerscale_init, dropout=dropout, droppath=droppath))
            self.conv2.append(ConvNeXtBlock(in_features=out_features, out_features=out_features, hidden_features=hidden_features, kernel_size=kernel_size, act=act, norm=norm, bias=bias, layerscale_init=layerscale_init, dropout=dropout, droppath=droppath))

        self.mask_conv = nn.Sequential(
            Conv2d(out_features, out_features, kernel_size=3, padding='same', groups=out_features),
            nn.GELU(),
            Conv2d(out_features, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, idx: torch.IntTensor, x: torch.Tensor, mask=None):
        """
        idx: t-index. [N, T]
        """
        for layer1 in self.conv1:
            x = layer1(x, mask)

        # multiply masking
        m_out = self.mask_conv(x)                           # [N, T, H, W, 1]
        x = x * m_out

        for layer2 in self.conv2:
            x = layer2(x, mask)
        return x, m_out


class MABlock(nn.Module):
    """
    conv (d) --> MA (x + x * mask) --> conv (d)
    """
    def __init__(self, in_features: int, out_features: int, hidden_features: int,
                           kernel_size=3, act='gelu', norm='layernorm', bias: bool=True,
                           layerscale_init=0., dropout=0., droppath=0., depths=2):
        super().__init__()

        self.conv1, self.conv2 = [nn.ModuleList() for _ in range(2)]
        for i in range(depths):
            self.conv1.append(ConvNeXtBlock(in_features=in_features if i==0 else out_features, out_features=out_features, hidden_features=hidden_features, kernel_size=kernel_size, act=act, norm=norm, bias=bias, layerscale_init=layerscale_init, dropout=dropout, droppath=droppath))
            self.conv2.append(ConvNeXtBlock(in_features=out_features, out_features=out_features, hidden_features=hidden_features, kernel_size=kernel_size, act=act, norm=norm, bias=bias, layerscale_init=layerscale_init, dropout=dropout, droppath=droppath))

        self.mask_conv = nn.Sequential(
            Conv2d(out_features, out_features, kernel_size=3, padding='same', groups=out_features),
            nn.GELU(),
            Conv2d(out_features, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, idx: torch.IntTensor, x: torch.Tensor, mask=None):
        """
        idx: t-index. [N, T]
        """
        for layer1 in self.conv1:
            x = layer1(x, mask)

        # multiply masking
        m_out = self.mask_conv(x)                           # [N, T, H, W, 1]
        x = x + x * m_out

        for layer2 in self.conv2:
            x = layer2(x, mask)
        return x, m_out


class CAM(nn.Module):
    def __init__(self, channels, reduction_ratio=4):
        super().__init__()

        self.maxpool = nn.AdaptiveMaxPool2d((1, 1))
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.mlp = nn.Sequential(
            nn.Linear(channels, channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(channels // reduction_ratio, channels)
        )

    def forward(self, x):
        """
        x: [N, T, H, W, C]
        """
        N, T, H, W, C = x.shape
        x = x.view(N * T, H, W, C).permute(0, 3, 1, 2)

        x_maxpool = self.maxpool(x).view(x.shape[0], -1)             # [NT, C]
        x_avgpool = self.avgpool(x).view(x.shape[0], -1)             # [NT, C]
        channel_attn = self.mlp(x_maxpool) + self.mlp(x_avgpool)
        channel_attn = channel_attn.unsqueeze(2).unsqueeze(3)           # [NT, C, 1, 1]
        scale = F.sigmoid(channel_attn)
        out = (x * scale).permute(0, 2, 3, 1).view(N, T, H, W, C)
        return out


class SAM(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = Conv2d(2, 1, kernel_size=7, padding='same')

    def forward(self, x):
        """
        x: [N, T, H, W, C]
        """
        scale = torch.cat((torch.max(x, -1, keepdim=True)[0], torch.mean(x, -1, keepdim=True)), dim=-1)
        scale = F.sigmoid(self.conv(scale))
        return x * scale


class SCAMBlock(nn.Module):
    """
    conv layer --> spatial attn (SAM) --> conv layer --> channel attn (CAM) --> conv block
    """
    def __init__(self, in_features: int, out_features: int,hidden_features: int,
                           kernel_size=3, act='gelu', norm='layernorm', bias: bool=True,
                           layerscale_init=0., dropout=0., droppath=0., depths=1):
        super().__init__()
        self.conv1 = ConvNeXtBlock(in_features=in_features, out_features=out_features, hidden_features=hidden_features, 
                                   kernel_size=kernel_size, act=act, norm=norm, bias=bias, layerscale_init=layerscale_init,
                                   dropout=dropout, droppath=droppath)
        self.conv2 = ConvNeXtBlock(in_features=out_features, out_features=out_features, hidden_features=hidden_features,
                                   kernel_size=kernel_size, act=act, norm=norm, bias=bias, layerscale_init=layerscale_init,
                                   dropout=dropout, droppath=droppath)
        
        self.conv_out = nn.ModuleList()
        for i in range(depths):
            self.conv_out.append(ConvNeXtBlock(in_features=out_features, out_features=out_features, hidden_features=hidden_features, kernel_size=kernel_size, act=act, norm=norm, bias=bias, layerscale_init=layerscale_init, dropout=dropout, droppath=droppath))

        # spatial & channel attention
        self.spatial_attn = SAM()
        self.channel_attn = CAM(out_features)

    def forward(self, idx: torch.IntTensor, x: torch.Tensor, mask=None):
        # conv & spatial attention
        x = self.conv1(x, mask)
        x = self.spatial_attn(x)

        # conv & channel attention
        x = self.conv2(x, mask)
        x = self.channel_attn(x)

        # conv out
        for layer in self.conv_out:
            x = layer(x, mask)
        return x, None


class CSAMBlock(nn.Module):
    """
    conv layer --> channel attn (CAM) --> conv layer --> spatial attn (SAM) --> conv block
    """
    def __init__(self, in_features: int, out_features: int,hidden_features: int,
                           kernel_size=3, act='gelu', norm='layernorm', bias: bool=True,
                           layerscale_init=0., dropout=0., droppath=0., depths=1):
        super().__init__()
        self.conv1 = ConvNeXtBlock(in_features=in_features, out_features=out_features, hidden_features=hidden_features, 
                                   kernel_size=kernel_size, act=act, norm=norm, bias=bias, layerscale_init=layerscale_init,
                                   dropout=dropout, droppath=droppath)
        self.conv2 = ConvNeXtBlock(in_features=out_features, out_features=out_features, hidden_features=hidden_features,
                                   kernel_size=kernel_size, act=act, norm=norm, bias=bias, layerscale_init=layerscale_init,
                                   dropout=dropout, droppath=droppath)
        
        self.conv_out = nn.ModuleList()
        for i in range(depths):
            self.conv_out.append(ConvNeXtBlock(in_features=out_features, out_features=out_features, hidden_features=hidden_features, kernel_size=kernel_size, act=act, norm=norm, bias=bias, layerscale_init=layerscale_init, dropout=dropout, droppath=droppath))

        # spatial & channel attention
        self.spatial_attn = SAM()
        self.channel_attn = CAM(out_features)

    def forward(self, idx: torch.IntTensor, x: torch.Tensor, mask=None):
        # conv & channel attention
        x = self.conv1(x, mask)
        x = self.channel_attn(x)

        # conv & spatial attention
        x = self.conv2(x, mask)
        x = self.spatial_attn(x)

        # conv out
        for layer in self.conv_out:
            x = layer(x, mask)
        return x, None


class MATBlock_front(nn.Module):
    def __init__(self, in_features: int, out_features: int, hidden_features: int,
                           kernel_size=3, act='gelu', norm='layernorm', bias: bool=True,
                           layerscale_init=0., dropout=0., droppath=0., depths=2,
                           mat_dimension=(600, 8)):
        super().__init__()
        # Affine Transform
        in_dim = mat_dimension[1]
        self.gammas = nn.Parameter(nn.init.uniform_(torch.empty(mat_dimension, dtype=torch.float32), -1e-3, 1e-3), requires_grad=True)
        self.betas = nn.Parameter(nn.init.uniform_(torch.empty(mat_dimension, dtype=torch.float32), -1e-3, 1e-3), requires_grad=True)
        self.g_linear = nn.Linear(in_dim, in_features)
        self.b_linear = nn.Linear(in_dim, in_features)

        self.conv1 = nn.ModuleList()
        for i in range(depths):
            self.conv1.append(ConvNeXtBlock(in_features=in_features if i==0 else out_features, out_features=out_features, hidden_features=hidden_features, kernel_size=kernel_size, act=act, norm=norm, bias=bias, layerscale_init=layerscale_init, dropout=dropout, droppath=droppath))

        self.mask_conv = nn.Sequential(
            Conv2d(in_features, 1, kernel_size=3, padding='same'),
            nn.Sigmoid(),
        )

    def forward(self, idx: torch.IntTensor, x: torch.Tensor, mask=None):
        """
        idx: t-index. [N, T]
        """
        # masked affine transform
        m_out = self.mask_conv(x)
        gamma = self.g_linear(self.gammas[idx])             # [N, T, C]
        beta = self.b_linear(self.betas[idx])                       # [N, T, C]
        t_x = x * gamma[:, :, None, None, :] + beta[:, :, None, None, :]

        x = x + t_x * (1 + m_out)

        for layer in self.conv1:
            x = layer(x, mask)
        return x, m_out


class MATBlock_back(nn.Module):
    def __init__(self, in_features: int, out_features: int, hidden_features: int,
                           kernel_size=3, act='gelu', norm='layernorm', bias: bool=True,
                           layerscale_init=0., dropout=0., droppath=0., depths=2,
                           mat_dimension=(600, 8)):
        super().__init__()
        # Affine Transform
        in_dim = mat_dimension[1]
        self.gammas = nn.Parameter(nn.init.uniform_(torch.empty(mat_dimension, dtype=torch.float32), -1e-3, 1e-3), requires_grad=True)
        self.betas = nn.Parameter(nn.init.uniform_(torch.empty(mat_dimension, dtype=torch.float32), -1e-3, 1e-3), requires_grad=True)
        self.g_linear = nn.Linear(in_dim, out_features)
        self.b_linear = nn.Linear(in_dim, out_features)

        self.conv1 = nn.ModuleList()
        for i in range(depths):
            self.conv1.append(ConvNeXtBlock(in_features=in_features if i==0 else out_features, out_features=out_features, hidden_features=hidden_features, kernel_size=kernel_size, act=act, norm=norm, bias=bias, layerscale_init=layerscale_init, dropout=dropout, droppath=droppath))

        self.mask_conv = nn.Sequential(
            Conv2d(out_features, 1, kernel_size=3, padding='same'),
            nn.Sigmoid(),
        )

    def forward(self, idx: torch.IntTensor, x: torch.Tensor, mask=None):
        """
        idx: t-index. [N, T]
        """
        for layer1 in self.conv1:
            x = layer1(x, mask)

        # masked affine transform
        m_out = self.mask_conv(x)
        gamma = self.g_linear(self.gammas[idx])             # [N, T, C]
        beta = self.b_linear(self.betas[idx])                       # [N, T, C]
        t_x = x * gamma[:, :, None, None, :] + beta[:, :, None, None, :]

        x = x + t_x * (1 + m_out)
        return x, m_out



"""
Utils
"""
def get_norm(norm, **kwargs):
    if norm == "none":
        return nn.Identity
    elif norm == "layernorm":
        return partial(nn.LayerNorm, eps=1e-6, **kwargs)
    elif norm == "layernorm-no-affine":
        return partial(nn.LayerNorm, elementwise_affine=False, eps=1e-6, **kwargs)
    else:
        raise NotImplementedError


def get_activation(activation):
    if activation == "none":
        return nn.Identity
    elif activation == "relu":
        return nn.ReLU
    elif activation == "relu6":
        return nn.ReLU6
    elif activation == "leaky_relu":
        return nn.LeakyReLU
    elif activation == "gelu":
        return nn.GELU
    elif activation == "gelu_fast":
        return partial(nn.GELU, approximate='tanh')
    elif activation == "tanh":
        return nn.Tanh
    elif activation == "sigmoid":
        return nn.Sigmoid
    elif activation == "softplus":
        return nn.Softplus
    elif activation == "sin":
        return Sin
    else:
        raise NotImplementedError


def get_block(type, **kwargs):
    if type == 'identity':
        return torch.nn.Identity()
    elif type == 'linear_stem':
        return LinearBlock(in_features=kwargs['C1'], out_features=kwargs['C2'],
                           act=kwargs['act'], norm=kwargs['norm'], norm_first=False, bias=kwargs['bias'])
    elif type == 'conv_stem':
        return Conv2dBlock(in_features=kwargs['C1'], out_features=kwargs['C2'], kernel_size=kwargs['kernel_size'],
                           act=kwargs['act'], norm=kwargs['norm'], norm_first=False, bias=kwargs['bias'])
    elif type == 'linear_head':
        return LinearBlock(in_features=kwargs['C1'], out_features=kwargs['C2'],
                           act=kwargs['act'], norm=kwargs['norm'], norm_first=True, bias=kwargs['bias'])
    elif type == 'mlp':
        return MLPBlock(in_features=kwargs['C1'], out_features=kwargs['C2'], hidden_features=kwargs['Ch'],
                        act=kwargs['act'], norm=kwargs['norm'], bias=kwargs['bias'],
                        layerscale_init=kwargs['layerscale'],
                        dropout=kwargs['dropout'], droppath=kwargs['droppath'])
    elif type == 'convnext':
        return ConvNeXtBlock(in_features=kwargs['C1'], out_features=kwargs['C2'], hidden_features=kwargs['Ch'],
                             kernel_size=kwargs['kernel_size'], act=kwargs['act'], norm=kwargs['norm'], bias=kwargs['bias'],
                             layerscale_init=kwargs['layerscale'],
                             dropout=kwargs['dropout'], droppath=kwargs['droppath'])
    elif type == 'convnext-lessnorm':
        return ConvNeXtBlockLessNorm(in_features=kwargs['C1'], out_features=kwargs['C2'], hidden_features=kwargs['Ch'],
                                     kernel_size=kwargs['kernel_size'], act=kwargs['act'], norm=kwargs['norm'], bias=kwargs['bias'],
                                     layerscale_init=kwargs['layerscale'],
                                     dropout=kwargs['dropout'], droppath=kwargs['droppath'])
    elif type == 'hinervblock':
        return HiNeRVBlock(in_features=kwargs['C1'], out_features=kwargs['C2'], hidden_features=kwargs['Ch'],
                           kernel_size=kwargs['kernel_size'], act=kwargs['act'],
                           norm=kwargs['norm'], bias=kwargs['bias'],
                           layerscale_init=kwargs['layerscale'],
                           dropout=kwargs['dropout'], droppath=kwargs['droppath'],
                           depths=kwargs['depths'])
    elif type == 'matblock':
        return MATBlock(in_features=kwargs['C1'], out_features=kwargs['C2'], hidden_features=kwargs['Ch'],
                            kernel_size=kwargs['kernel_size'], act=kwargs['act'],
                            norm=kwargs['norm'], bias=kwargs['bias'],
                            layerscale_init=kwargs['layerscale'],
                            dropout=kwargs['dropout'], droppath=kwargs['droppath'],
                            depths=kwargs['depths'], mat_dimension=kwargs['mat_dim'])
    elif type == 'atblock':
        return ATBlock(in_features=kwargs['C1'], out_features=kwargs['C2'], hidden_features=kwargs['Ch'],
                            kernel_size=kwargs['kernel_size'], act=kwargs['act'],
                            norm=kwargs['norm'], bias=kwargs['bias'],
                            layerscale_init=kwargs['layerscale'],
                            dropout=kwargs['dropout'], droppath=kwargs['droppath'],
                            depths=kwargs['depths'], mat_dimension=kwargs['mat_dim'])
    elif type == 'mmblock':
        return MMBlock(in_features=kwargs['C1'], out_features=kwargs['C2'], hidden_features=kwargs['Ch'],
                            kernel_size=kwargs['kernel_size'], act=kwargs['act'],
                            norm=kwargs['norm'], bias=kwargs['bias'],
                            layerscale_init=kwargs['layerscale'], dropout=kwargs['dropout'], 
                            droppath=kwargs['droppath'], depths=kwargs['depths'])
    elif type == 'mablock':
        return MABlock(in_features=kwargs['C1'], out_features=kwargs['C2'], hidden_features=kwargs['Ch'],
                            kernel_size=kwargs['kernel_size'], act=kwargs['act'],
                            norm=kwargs['norm'], bias=kwargs['bias'],
                            layerscale_init=kwargs['layerscale'], dropout=kwargs['dropout'], 
                            droppath=kwargs['droppath'], depths=kwargs['depths'])
    elif type == 'scamblock':
        return SCAMBlock(in_features=kwargs['C1'], out_features=kwargs['C2'], hidden_features=kwargs['Ch'],
                             kernel_size=kwargs['kernel_size'], act=kwargs['act'],
                             norm=kwargs['norm'], bias=kwargs['bias'],
                             layerscale_init=kwargs['layerscale'], dropout=kwargs['dropout'],
                             droppath=kwargs['droppath'], depths=kwargs['depths'])
    elif type == 'csamblock':
        return CSAMBlock(in_features=kwargs['C1'], out_features=kwargs['C2'], hidden_features=kwargs['Ch'],
                             kernel_size=kwargs['kernel_size'], act=kwargs['act'],
                             norm=kwargs['norm'], bias=kwargs['bias'],
                             layerscale_init=kwargs['layerscale'], dropout=kwargs['dropout'],
                             droppath=kwargs['droppath'], depths=kwargs['depths'])
    elif type == 'matblock_back':
        return MATBlock_back(in_features=kwargs['C1'], out_features=kwargs['C2'], hidden_features=kwargs['Ch'],
                            kernel_size=kwargs['kernel_size'], act=kwargs['act'],
                            norm=kwargs['norm'], bias=kwargs['bias'],
                            layerscale_init=kwargs['layerscale'],
                            dropout=kwargs['dropout'], droppath=kwargs['droppath'],
                            depths=kwargs['depths'], mat_dimension=kwargs['mat_dim'])
    elif type == 'matblock2':
        return MATBlock2(in_features=kwargs['C1'], out_features=kwargs['C2'], hidden_features=kwargs['Ch'],
                            kernel_size=kwargs['kernel_size'], act=kwargs['act'],
                            norm=kwargs['norm'], bias=kwargs['bias'],
                            layerscale_init=kwargs['layerscale'],
                            dropout=kwargs['dropout'], droppath=kwargs['droppath'],
                            depths=kwargs['depths'], mat_dimension=kwargs['mat_dim'])
    elif type == 'matblock_front':
        return MATBlock_front(in_features=kwargs['C1'], out_features=kwargs['C2'], hidden_features=kwargs['Ch'],
                            kernel_size=kwargs['kernel_size'], act=kwargs['act'],
                            norm=kwargs['norm'], bias=kwargs['bias'],
                            layerscale_init=kwargs['layerscale'],
                            dropout=kwargs['dropout'], droppath=kwargs['droppath'],
                            depths=kwargs['depths'], mat_dimension=kwargs['mat_dim'])
    elif type == 'matblock_back':
        return MATBlock_back(in_features=kwargs['C1'], out_features=kwargs['C2'], hidden_features=kwargs['Ch'],
                            kernel_size=kwargs['kernel_size'], act=kwargs['act'],
                            norm=kwargs['norm'], bias=kwargs['bias'],
                            layerscale_init=kwargs['layerscale'],
                            dropout=kwargs['dropout'], droppath=kwargs['droppath'],
                            depths=kwargs['depths'], mat_dimension=kwargs['mat_dim'])
    elif type == 'matblock3':
        return MATBlock3(in_features=kwargs['C1'], out_features=kwargs['C2'], hidden_features=kwargs['Ch'],
                            kernel_size=kwargs['kernel_size'], act=kwargs['act'],
                            norm=kwargs['norm'], bias=kwargs['bias'],
                            layerscale_init=kwargs['layerscale'],
                            dropout=kwargs['dropout'], droppath=kwargs['droppath'],
                            depths=kwargs['depths'], mat_dimension=kwargs['mat_dim'])
    elif type == 'matblock4':
        return MATBlock4(in_features=kwargs['C1'], out_features=kwargs['C2'], hidden_features=kwargs['Ch'],
                            kernel_size=kwargs['kernel_size'], act=kwargs['act'],
                            norm=kwargs['norm'], bias=kwargs['bias'],
                            layerscale_init=kwargs['layerscale'],
                            dropout=kwargs['dropout'], droppath=kwargs['droppath'],
                            depths=kwargs['depths'], mat_dimension=kwargs['mat_dim'])
    else:
        raise NotImplementedError
