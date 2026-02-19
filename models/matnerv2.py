"""
                                    -->  Luma path (Y-path): MAT blocks...
multi-resolution grid --> 
                                    --> Chroma path (CbCr-path): simpler convolutions...

# cross-branch guidance in between two paths
- 1. multiply mask-feature from Luma path to the features in Chroma path
- 2. concat features from Luma & Chroma path and pass through 1x1 convolution and split back
# output
- 1 channel (Y) from Luma path
- 2 channels (CbCr) from chroma path

# Loss
- bigger loss on Luma output?
- calculating PSNR: YCbCr --> RGB and calculate PSNR
"""
from .utils import *
from .layers import *
from .encoding import PositionalEncoder
from .upsample import FastNearestInterpolation, FastTrilinearInterpolation, crop_tensor_nthwc
from .patch_utils import *



def get_encoding_cfg(enc_type, depth, size, scale, **kwargs):

    assert enc_type in ['base', 'upsample']
    assert (enc_type == 'base' and depth == 0) or (enc_type in ['upsample'] and depth >= 0)

    if kwargs['type'] == 'none':
        return {}
    
    # Positional
    Bt, Lt, Bs, Ls = kwargs['pe']
    Lt, Ls = int(Lt), int(Ls)

    # Grid
    if len(kwargs['grid_size']) == 4:
        # Learned/Local learned (full)
        T_grid, H_grid, W_grid, C_grid = kwargs['grid_size']
        if 'grid_depth_scale' in kwargs:
            T_grid_scale, H_grid_scale, W_grid_scale, C_grid_scale = kwargs['grid_depth_scale']
        else:
            T_grid_scale = H_grid_scale = W_grid_scale = C_grid_scale = 1.
    elif len(kwargs['grid_size']) == 2:
        # Local learned (temporal only)
        T_grid, C_grid = kwargs['grid_size']
        H_grid, W_grid = 1, 1
        if 'grid_depth_scale' in kwargs:
            T_grid_scale, C_grid_scale = kwargs['grid_depth_scale']
            H_grid_scale, W_grid_scale = 1, 1
        else:
            T_grid_scale = H_grid_scale = W_grid_scale = C_grid_scale = 1.
    else:
        raise NotImplementedError
    
    T_grid = max(int(T_grid * T_grid_scale ** depth), 1) if T_grid != -1 else size[0]
    H_grid = max(int(H_grid * H_grid_scale ** depth), 1) if H_grid != -1 else size[1]
    W_grid = max(int(W_grid * W_grid_scale ** depth), 1) if W_grid != -1 else size[2]
    C_grid = max(int(C_grid * C_grid_scale ** depth), 1) if C_grid != -1 else size[3]

    if len(kwargs['grid_size']) == 4:
        kwargs['grid_size'] = [T_grid, H_grid, W_grid, C_grid]
    elif len(kwargs['grid_size']) == 2:
        kwargs['grid_size'] = [T_grid, C_grid]
    else:
        raise NotImplementedError
    
    # All configs
    enc_cfg = {
        'type': kwargs['type'],
        # frequency encoding
        'Bt': Bt, 'Lt': Lt, 'Bs': Bs, 'Ls': Ls, 'no_t': kwargs['pe_no_t'],
        # grid encoding
        'grid_size': kwargs['grid_size'], 'grid_level': kwargs['grid_level'], 'grid_level_scale': kwargs['grid_level_scale'],
        'grid_init_scale': kwargs['grid_init_scale'],
        'align_corners': kwargs['align_corners'],
    }

    return enc_cfg


class NeRVEncoding(nn.Module):
    """
    BaseNeRV Encoding, i.e., the feature grid layer.
    """
    def __init__(self, size, channels, **kwargs):
        super().__init__()
        self.size = size
        self.channels = channels

        T, H, W = self.size
        C = self.channels

        # Grids
        self.grids = nn.ParameterList()
        self.grid_expands = nn.ModuleList()

        assert len(kwargs['grid_level_scale_h']) == len(kwargs['grid_level_scale_w'])
        self.grid_level_t = kwargs['grid_level_t']                  # temporal scale levels
        self.grid_level_s = len(kwargs['grid_level_scale_h'])       # spatial scale levels
        self.grid_level = self.grid_level_t * self.grid_level_s
        self.grid_sizes = []

        T_grid, H_grid, W_grid, C_grid = kwargs['grid_size']
        T_scale, C_scale = kwargs['grid_level_scale_t']
        H_scale, W_scale = kwargs['grid_level_scale_h'], kwargs['grid_level_scale_w']

        for i in range(self.grid_level_s):
            H_i, W_i = int(H_grid / H_scale[i]), int(W_grid / W_scale[i])
            for j in range(self.grid_level_t):
                T_i, C_i = int(T_grid / T_scale ** j), int(C_grid / C_scale ** j)
                self.grid_sizes.append((T_i, H_i, W_i, C_i))
                self.grids.append(FeatureGrid((T_i * H_i * W_i, C_i), init_scale=kwargs['grid_init_scale']))
                self.grid_expands.append(GridUpsample((T, H, W)))

    def extra_repr(self):
        s = 'size={size}, channels={channels}, grid_level={grid_level}, grid_sizes={grid_sizes}'
        return s.format(**self.__dict__)
    
    def forward(self, idx: torch.IntTensor, idx_max: tuple[int, int, int], padding: tuple[int, int, int]):
        """
        Inputs:
            idx: patch index tensor with shape [N, 3]
            idx_max: list of 3 ints. Represents the range of patch indexes
            patch_mode: if True, the input is a patch from the full video, and the faster implementation will be used

        Output:
            a tensor with shape [N, T, H, W, C]
        """
        assert idx.ndim == 2 and idx.shape[1] == 3
        assert len(idx_max) == 3

        # Compute the global voxels coordinates
        patch_size = tuple(self.size[d] // idx_max[d] for d in range(3))
        patch_padded_size = tuple(patch_size[d] + 2 * padding[d] for d in range(3))

        px_idx, px_mask = compute_pixel_idx_3d(idx, idx_max, self.size, padding=padding, clipped=True)
        px_idx_flat = (px_idx[0][:, :, None, None] * self.size[1] * self.size[2]
                       + px_idx[1][:, None, :, None] * self.size[2]
                       + px_idx[2][:, None, None, :]).view(-1)
        px_mask_flat = (px_mask[0][:, :, None, None, None]
                        * px_mask[1][:, None, :, None, None]
                        * px_mask[2][:, None, None, :, None]).view(-1, 1)
        
        # Encode
        enc_splits = [self.grid_expands[i](self.grids[i]().view(self.grid_sizes[i])) for i in range(self.grid_level)]
        enc = torch.concat(enc_splits, dim=-1)

        output = (px_mask_flat * torch.index_select(enc.view(self.size[0] * self.size[1] * self.size[2], self.channels), 0, px_idx_flat)).view((idx.shape[0],) + patch_padded_size + (self.channels,))
        assert_shape(output, (idx.shape[0],) + patch_padded_size + (self.channels,))

        return output


class NeRVUpsampler(nn.Module):
    """
    NeRV Upsampler. It combined the upsampling together with the encoding and cropping.
    """
    def __init__(self, **kwargs):
        super().__init__()
        self.in_channels = kwargs['in_channels']
        self.out_channels = kwargs['out_channels']
        self.scale = kwargs['scale']
        self.upsample_type = kwargs['type']
        self.norm = get_norm(kwargs['norm'])(self.in_channels)
        self.act = get_activation(kwargs['act'])()

        # Layer
        if self.upsample_type == 'trilinear':
            self.layer = FastTrilinearInterpolation(kwargs['config'])
        elif self.upsample_type == 'nearest':
            self.layer = FastNearestInterpolation(kwargs['config'])
        elif self.upsample_type == 'conv1x1':
            assert self.scale[0] == 1 and self.scale[1] == self.scale[2]
            self.layer = nn.Sequential(
                nn.Conv2d(self.in_channels, self.out_channels * self.scale[1] * self.scale[2], 1, 1, padding='same'),
                nn.PixelShuffle((self.scale[1]))
            )
        elif self.upsample_type == 'conv3x3':
            assert self.scale[0] == 1 and self.scale[1] == self.scale[2]
            self.layer = nn.Sequential(
                nn.Conv2d(self.in_channels, self.out_channels * self.scale[1] * self.scale[2], 3, 1, padding='same'),
                nn.PixelShuffle((self.scale[1]))
            )
        elif self.upsample_type == 'dconv3x3':
            assert self.scale[0] == 1 and self.scale[1] == self.scale[2]
            self.layer = nn.Sequential(
                nn.Conv2d(self.in_channels, self.in_channels, 3, 1, padding='same', groups=self.in_channels),
                nn.Conv2d(self.in_channels, self.out_channels * self.scale[1] * self.scale[2], 1, 1, padding='same'),
                nn.PixelShuffle((self.scale[1]))
            )
        else:
            raise ValueError

    def extra_repr(self):
        s = 'scale={scale}, upsample_type={upsample_type}'
        return s.format(**self.__dict__)
    
    def forward(self, x: torch.Tensor, idx: torch.IntTensor, idx_max: tuple[int, int, int], size: tuple[int, int, int], scale: tuple[int, int, int], padding : tuple[int, int, int], patch_mode: bool=True, mask: Optional[torch.Tensor]=None):
        """
        During the forward pass, the input tensor will be upscaled by the 'scale' factor, then a cropping will be applied to reduce the size to 'output_size'.

        Inputs:
            - x: input tensor with shape [N, T1, H1, W1, C]
            - idx: patch index tensor with shape [N, 3]
            - idx_max: list of 3 ints. Represents the range of patch indexes.
            - size: list of 3 ints. Represents the size of the full video. It does not have to be the same as the input size, as the input can be a patch from the full video.
            - scale: list of 3 ints. Represents the scale factor. This will be used to compute the output size.
            - padding: list of 3 ints. Represents the padding size. This will be used to compute the output size.
            - patch_mode: if True, the input is a patch from the full video, and the faster implementation will be used.
            - mask: mask tensor with shape [N, T1, H1, W1, C]. If not None, it will be multiplied to the output.

        Output:
            a tensor with shape [N, T2, H2, W2, C]
        """
        assert x.ndim == 5, x.shape
        assert idx.ndim == 2 and idx.shape[1] == 3, idx.shape
        assert len(idx_max) == 3
        assert len(scale) == 3
        assert len(size) == 3
        assert len(padding) == 3

        N, T_in, H_in, W_in, C = x.shape
        T_scale, H_scale, W_scale = int(T_in * scale[0]), int(H_in * scale[1]), int(W_in * scale[2])
        T_out, H_out, W_out = tuple(size[d] // idx_max[d] + 2 * padding[d] for d in range(3))
        assert (T_out - T_in * scale[0]) % 2 == (H_out - H_in * scale[1]) % 2 == (W_out - W_in * scale[2]) % 2 == 0, 'Under this configuration, padding is not symmetric and can cause problems!'

        x = self.norm(x)

        if self.upsample_type in ['trilinear', 'nearest']:
            x = self.layer(x, idx, idx_max, size, scale, padding, patch_mode)
            assert_shape(x, (N, T_out, H_out, W_out, C))
        elif 'conv' in self.upsample_type:
            x = x.view(N * T_in, H_in, W_in, C).permute(0, 3, 1, 2)         # [N*T, C, H, W]
            x = self.layer(x)
            x = x.permute(0, 2, 3, 1).view(N, T_scale, H_scale, W_scale, -1)
            x = crop_tensor_nthwc(x, size=(T_out, H_out, W_out))
            assert_shape(x, (N, T_out, H_out, W_out, x.shape[-1]))
            if mask is not None:
                x = mask * x        # For conv layers, mask should be applied
        else:
            raise NotImplementedError
        
        x = self.act(x)

        return x


class NeRVDecoder(nn.Module):
    """
    NeRV Decoder, i.e., the main model layers
    TODO: divide into Luma & Chroma branch
    """
    def __init__(self, logger, input_size, input_channels, output_size, output_channels, 
                 l_channels, l_channels_reduce, l_channels_reduce_base, l_channels_min, l_depths, l_exps, l_block_cfg, l_upsample_cfg, l_enc_cfg,
                 c_channels, c_channels_reduce, c_channels_reduce_base, c_channels_min, c_depths, c_exps, c_block_cfg, c_upsample_cfg, c_enc_cfg,kernels, scales, paddings, stem_kernels, stem_paddings, stem_cfg, head_cfg):
        super().__init__()
        assert isinstance(input_size, (tuple, list))
        assert isinstance(output_size, (tuple, list))
        assert isinstance(l_depths, (tuple, list))
        assert isinstance(l_exps, (tuple, list))
        assert isinstance(c_depths, (tuple, list))
        assert isinstance(c_exps, (tuple, list))
        assert isinstance(scales, (tuple, list)) and all([isinstance(s, (tuple, list)) for s in scales])
        assert isinstance(paddings, (tuple, list)) and all([isinstance(p, (tuple, list)) for p in paddings])
        assert len(l_depths) == len(l_exps) == len(c_depths) == len(c_exps) == len(kernels) == len(scales) == len(paddings)
        assert all(input_size[d] * math.prod([scale[d] for scale in scales]) == output_size[d] for d in range(3)), \
                f'input_size {input_size} does no match output_size {output_size} and scales {scales}'

        self.input_size = input_size
        self.input_channels = input_channels
        self.ouput_size = output_size
        self.output_channels = output_channels

        self.l_channels = l_channels
        self.l_channels_reduce = l_channels_reduce
        self.l_channels_reduce_base = l_channels_reduce_base
        self.l_channels_min = l_channels_min

        self.c_channels = c_channels
        self.c_channels_reduce = c_channels_reduce
        self.c_channels_reduce_base = c_channels_reduce_base
        self.c_channels_min = c_channels_min
    
        self.l_depths = l_depths
        self.l_exps = l_exps
        self.c_depths = c_depths
        self.c_exps = c_exps
        self.kernels = kernels
        self.scales = [tuple(scales[i]) for i in range(len(scales))]
        self.paddings = [tuple(paddings[i]) for i in range(len(paddings))]

        self.stem_kernels = stem_kernels
        self.stem_paddings = stem_paddings

        self.min_patch_size = tuple(np.prod(np.array(self.scales), axis=0).tolist())

        self.T, self.H, self.W = self.ouput_size

        logger.info(f'NeRV:')

        # Stem
        _l_stem_cfg = cfg_override(stem_cfg, C1=self.input_channels, C2=self.l_channels, Ch=max(self.input_channels, self.l_channels), kernel_size=self.stem_kernels)
        self.l_stem = get_block(**_l_stem_cfg)
        
        _c_stem_cfg = cfg_override(stem_cfg, C1=self.input_channels, C2=self.c_channels, Ch=max(self.input_channels, self.c_channels), kernel_size=self.stem_kernels)
        self.c_stem = get_block(**_c_stem_cfg)

        # NeRV blocks
        self.l_blocks, self.c_blocks = [nn.ModuleList() for _ in range(2)]
        for i, scale in enumerate(self.scales):
            if i == 0:
                T1, H1, W1 = (self.input_size[d] for d in range(3))
                T2, H2, W2 = T1 * scale[0], H1 * scale[1], W1 * scale[2]
                L_C1 = self.l_channels
                L_C2 = max(self.l_channels, self.l_channels_min)
                C_C1 = self.c_channels
                C_C2 = max(self.c_channels, self.c_channels_min)
            else:
                T1, H1, W1 = T2, H2, W2
                T2, H2, W2 = T1 * scale[0], H1 * scale[1], W1 * scale[2]
                L_C1 = L_C2
                C_C1 = C_C2
                L_C2 = max(divide_to_multiple(L_C2, self.l_channels_reduce, self.l_channels_reduce_base), self.l_channels_min)
                C_C2 = max(divide_to_multiple(C_C2, self.c_channels_reduce, self.c_channels_reduce_base), self.c_channels_min)

            logger.info(f'     Stage {i + 1}:  T1 - {T1}  H1 - {H1}  W1 - {W1}  Luma C1 - {L_C1}  Chroma C1 - {C_C1}')
            logger.info(f'                     T2 - {T2}  H2 - {H2}  W2 - {W2}  Luma C2 - {L_C2}  Chroma C2 - {C_C2}')
            logger.info(f'                     Luma Depth - {self.l_depths[i]}  Luma Exp - {self.l_exps[i]}  Chroma Depth - {self.c_depths[i]}  Chroma Exp - {self.c_exps[i]}')
            logger.info(f'                     Kernel - {self.kernels[i]}  Scale - {self.scales[i]}  Padding - {self.paddings[i]}')

            # Block
            self.l_blocks.append(nn.ModuleList())
            self.c_blocks.append(nn.ModuleList())

            # Upsampler
            _l_upsample_cfg = cfg_override(l_upsample_cfg, in_channels=L_C1, out_channels=L_C2, scale=scale)
            self.l_blocks[-1].append(NeRVUpsampler(**_l_upsample_cfg))
            _c_upsample_cfg = cfg_override(c_upsample_cfg, in_channels=C_C1, out_channels=C_C2, scale=scale)
            self.c_blocks[-1].append(NeRVUpsampler(**_c_upsample_cfg))

            # Encoder
            _l_enc_cfg = get_encoding_cfg('upsample', i, size=(T2, H2, W2, L_C1), scale=scale, **l_enc_cfg)
            if _l_enc_cfg['type'] != 'none+none':
                self.l_blocks[-1].append(PositionalEncoder(scale=scale, channels=L_C1, cfg=_l_enc_cfg))
            _c_enc_cfg = get_encoding_cfg('upsample', i, size=(T2, H2, W2, C_C1), scale=scale, **c_enc_cfg)
            if _c_enc_cfg['type'] != 'none+none':
                self.c_blocks[-1].append(PositionalEncoder(scale=scale, channels=C_C1, cfg=_c_enc_cfg))

            # Conv blocks
            _l_block_cfg = cfg_override(l_block_cfg, type=l_block_cfg['types'][i], C1=L_C1, C2=L_C2, Ch=int(L_C2 * self.l_exps[i]), kernel_size=self.kernels[i], depths=self.l_depths[i], mat_dim=(output_size[0], l_block_cfg['mat_chans']))
            self.l_blocks[-1].append(get_block(**_l_block_cfg))
            _c_block_cfg = cfg_override(c_block_cfg, type=c_block_cfg['types'][i], C1=C_C1, C2=C_C2, Ch=int(C_C2 * self.c_exps[i]), kernel_size=self.kernels[i], depths=self.c_depths[i])
            self.c_blocks[-1].append(get_block(**_c_block_cfg))

        # Heads
        logger.info(f'     Output channels: {self.output_channels}')
        _l_head_cfg = cfg_override(head_cfg, C1=L_C2, C2=self.output_channels // 3, Ch=max(L_C2, self.output_channels//3), kernel_size=1)
        self.l_head = get_block(**_l_head_cfg)
        _c_head_cfg = cfg_override(head_cfg, C1=C_C2, C2=self.output_channels // 3 * 2, Ch=max(C_C2, self.output_channels//3*2), kernel_size=1, act='tanh')
        self.c_head = get_block(**_c_head_cfg)

        # initialization
        self.l_stem.apply(self._init_stem_blocks)
        self.c_stem.apply(self._init_stem_blocks)
        self.l_blocks.apply(self._init_stem_blocks)
        self.c_blocks.apply(self._init_stem_blocks)
        self.l_head.apply(self._init_heads)
        self.c_head.apply(self._init_heads)

    def _init_stem_blocks(self, m):
        if isinstance(m, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def _init_heads(self, m):
        if isinstance(m, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def extra_repr(self):
        s = 'input_size={input_size}, input_channels={input_channels}, ouput_size={ouput_size}, output_channels={output_channels}'
        return s.format(**self.__dict__)

    def get_input_padding(self, patch_mode: bool=True):
        return self.stem_paddings if patch_mode else (self.stem_paddings[0], 0, 0)

    def forward(self, x: torch.Tensor, idx: torch.IntTensor, idx_max: tuple[int, int, int], patch_mode: bool=True):
        """
        Inputs:
            - x: input tensor with shape [N, T1, H1, W1, C]
            - idx: patch index tensor with shape [N, 3]
            - idx_max: list of 3 ints. Represents the range of patch indexes
            - patch_mode: if True, the input is a patch from the full video, and the faster implementation will be used

        Output:
            a tensor with shape [N, T2, H2, W2, C]
        """
        assert idx.ndim == 2 and idx.shape[1] == 3
        assert len(idx_max) == 3

        # initial configs
        padding = self.stem_paddings if patch_mode else (0, 0, 0)
        v_size_in = v_size_out = self.input_size
        p_size_in = p_size_out = tuple(v_size_out[d] // idx_max[d] for d in range(3))
        p_size_out_padded = tuple(p_size_out[d] + 2 * padding[d] for d in range(3))

        if patch_mode:
            _, px_mask = compute_pixel_idx_3d(idx, idx_max, v_size_out, padding, clipped=False, return_mask=True)
            px_mask_3d = px_mask[0][:, :, None, None, None] \
                            * px_mask[1][:, None, :, None, None] \
                            * px_mask[2][:, None, None, :, None]
        else:
            px_mask_3d = None

        # Check input
        assert_shape(x, (idx.shape[0],) + p_size_out_padded + (self.input_channels,))

        l_x = self.l_stem(x)
        c_x = self.c_stem(x)
        if px_mask_3d is not None:
            l_x = l_x * px_mask_3d
            c_x = c_x * px_mask_3d

        out_masks = []
        for i, (l_block, c_block) in enumerate(zip(self.l_blocks, self.c_blocks)):
            # update configs
            padding = self.paddings[i] if patch_mode else (self.paddings[0][0], 0, 0)
            scale = self.scales[i]
            v_size_in = v_size_out
            v_size_out = tuple(int(v_size_in[d] * scale[d]) for d in range(3))
            p_size_in = p_size_out
            p_size_in_padded = p_size_out_padded
            p_size_out = tuple(int(p_size_in[d] * scale[d]) for d in range(3))
            p_size_out_padded = tuple(p_size_out[d] + 2 * padding[d] for d in range(3))

            assert all(p_size_in_padded[d] * scale[d] >= p_size_out_padded[d] for d in range(3)), 'the input padding is too small'

            # Compute mask
            # masking is only needed if kernel size > 1
            if patch_mode:
                _, px_mask = compute_pixel_idx_3d(idx, idx_max, v_size_out, padding, clipped=False, return_mask=True)
                px_mask_3d = px_mask[0][:, :, None, None, None] \
                                * px_mask[1][:, None, :, None, None] \
                                * px_mask[2][:, None, None, :, None]
            else:
                px_mask_3d = None

            # Run block
            ## LUMA
            for _, l_layer in enumerate(l_block):
                if isinstance(l_layer, NeRVUpsampler):
                    l_x = l_layer(l_x, idx=idx, idx_max=idx_max,
                              size=v_size_out,
                              scale=scale,
                              padding=padding,
                              patch_mode=patch_mode,
                              mask=px_mask_3d)
                elif isinstance(l_layer, PositionalEncoder):
                    l_x = l_layer(l_x, idx=idx, idx_max=idx_max,
                              size=v_size_out, 
                              scale=scale,
                              padding=padding)
                elif isinstance(l_layer, nn.Identity):
                    l_x = l_layer(l_x)
                else:
                    l_x, mask = l_layer(idx[:, 0:1], l_x, px_mask_3d)
                    if mask is not None:
                        out_masks.append(mask)

            assert_shape(l_x, (idx.shape[0],) + p_size_out_padded + (l_x.shape[-1],))

            ## CHROMA
            for _, c_layer in enumerate(c_block):
                if isinstance(c_layer, NeRVUpsampler):
                    c_x = c_layer(c_x, idx=idx, idx_max=idx_max,
                              size=v_size_out,
                              scale=scale,
                              padding=padding,
                              patch_mode=patch_mode,
                              mask=px_mask_3d)
                elif isinstance(c_layer, PositionalEncoder):
                    c_x = c_layer(c_x, idx=idx, idx_max=idx_max,
                              size=v_size_out, 
                              scale=scale,
                              padding=padding)
                elif isinstance(c_layer, nn.Identity):
                    c_x = c_layer(c_x)
                else:
                    c_x, _ = c_layer(idx[:, 0:1], c_x, px_mask_3d)          # TODO: need to check!

            assert_shape(c_x, (idx.shape[0],) + p_size_out_padded + (c_x.shape[-1],))

            # cross-branch guidance
            if i < len(self.c_blocks) - 1:
                c_x = c_x * out_masks[-1]

        # Head
        l_out = self.l_head(l_x)
        c_out = self.c_head(c_x)
        x = torch.concat([l_out, c_out], dim=-1)
        y = crop_tensor_nthwc(x, p_size_out)
        assert_shape(y, (idx.shape[0],) + p_size_out + (self.output_channels,))

        return y, out_masks


class NeRV(nn.Module):
    """
    NeRV model.
    """
    def __init__(self, encoding, decoder, eval_patch_size):
        super().__init__()

        # Model
        self.encoding = encoding
        self.decoder = decoder
        self.eval_patch_size = eval_patch_size

        # Encoding only account for small amount of parameters, so simply skip them.
        self.no_prune_prefix = ('encoding',) + tuple(k for k, v in self.named_modules() if isinstance(v, PositionalEncoder))
        self.no_quant_prefix = ()

        # The bitstream is the part of data that will be transmitted/stored
        self.bitstream_prefix = ('encoding', 'decoder')

    def forward(self, input):
        """
        Inputs:
            a dictionary with {
                - idx: patch index tensor with shape [N, 3]
                - idx_max: list of 3 ints. Represents the range of patch indexes.
                - video_size: list of 3 ints. Represents the size of the  full video.
                - patch_size: list of 3 ints. Represents the size of the patch.
            }
        Output:
            a tensor with shape [N, C, T, H, W]
        """
        assert all((input['patch_size'][d] % self.decoder.min_patch_size[d] == 0) for d in range(3))
        assert self.eval_patch_size is None or all(input['video_size'][d] % self.eval_patch_size[d] == 0 for d in range(3))

        # config
        if not self.training and self.eval_patch_size is not None:
            # Force using patch mode for evaluation
            idx_max = tuple(input['video_size'][d] // self.eval_patch_size[d] for d in range(3))
            idx = vidx_to_pidx(input['idx'], input['idx_max'], idx_max)
            patch_mode = True
        else:
            # Auto choose frame/patch mode
            idx_max = input['idx_max']
            idx = input['idx']
            patch_mode = False

        # Compute output
        input_padding = self.decoder.get_input_padding(patch_mode=patch_mode)
        output = self.encoding(idx, idx_max, padding=input_padding)
        output, outmask = self.decoder(output, idx, idx_max, patch_mode=patch_mode)

        # Reshape output
        if not self.training and self.eval_patch_size is not None:
            output = patch_to_video(output, input['patch_size'])
        output = output.permute(0, 4, 1, 2, 3).contiguous(memory_format=torch.channels_last_3d)
        outmask = [outm.permute(0, 4, 1, 2, 3).contiguous(memory_format=torch.channels_last_3d) for outm in outmask]

        return output , outmask

    def get_num_parameters(self):
        base_encoding_param = sum([v.numel() for _, v in self.encoding.state_dict().items()])
        decoder_param = sum([v.numel() for _, v in self.decoder.state_dict().items()])
        l_decoder_param = sum([v.numel() for k, v in self.decoder.state_dict().items() if k.startswith('l_')])
        c_decoder_param = sum([v.numel() for k, v in self.decoder.state_dict().items() if k.startswith('c_')])
        upsample_encoding_param = 0
        for block in [self.decoder.l_blocks, self.decoder.c_blocks]:
            for layer in block:
                if isinstance(block, PositionalEncoder):
                    upsample_encoding_param += sum([v.numel() for _, v in layer.state_dict().items()])
        model_param = decoder_param - upsample_encoding_param
        return {
            'All': decoder_param + base_encoding_param,
            'Decoder': decoder_param,
            'Model': model_param,
            'Encoding': base_encoding_param,
            'Upsampling Encoding': upsample_encoding_param,
            'Decoder - Luma branch': l_decoder_param,
            'Decoder - Chroma branch': c_decoder_param,
        }


def build_encoding(args, logger, size, channels):
    # Encoding config
    cfg = {
        'size': size,
        'channels': channels,
        'grid_size': args.base_grid_size,
        'grid_level_t': args.base_grid_level_t,
        'grid_level_scale_t': args.base_grid_level_scale_t,
        'grid_level_scale_h': args.base_grid_level_scale_h,
        'grid_level_scale_w': args.base_grid_level_scale_w,
        'grid_init_scale': args.base_grid_init_scale,
    }

    # Build encoding
    logger.info(f'Building NeRV Encoding with cfg: {cfg}')
    encoding = NeRVEncoding(**cfg)

    return encoding


def build_decoder(args, logger, input_size, input_channels, output_size, output_channels):
    # Network config
    assert len(args.l_depths) == len(args.c_depths) == len(args.l_exps) == len(args.c_exps) == len(args.kernels) == len(args.scales_t) == len(args.scales_hw) == len(args.l_block_types) == len(args.c_block_types)
    cfg = {}
    cfg['input_size'] = input_size
    cfg['input_channels'] = input_channels
    cfg['output_size'] = output_size
    cfg['output_channels'] = output_channels

    cfg['l_channels'] = args.l_channels
    cfg['l_channels_reduce'] = args.l_channels_reduce
    cfg['l_channels_reduce_base'] = args.l_channels_reduce_base
    cfg['l_channels_min'] = args.l_channels_min

    cfg['c_channels'] = args.c_channels
    cfg['c_channels_reduce'] = args.c_channels_reduce
    cfg['c_channels_reduce_base'] = args.c_channels_reduce_base
    cfg['c_channels_min'] = args.c_channels_min

    cfg['l_depths'] = args.l_depths
    cfg['l_exps'] = args.l_exps
    cfg['c_depths'] = args.c_depths
    cfg['c_exps'] = args.c_exps
    cfg['scales'] = [[scale_t, scale_hw, scale_hw] for scale_t, scale_hw in zip(args.scales_t, args.scales_hw)]
    cfg['stem_kernels'] = args.stem_kernels
    cfg['kernels'] = args.kernels

    if tuple(args.paddings) == (-1, -1, -1):
        assert tuple(args.stem_paddings) == (-1, -1, -1), 'both padding must be set/not set at the same time'
        paddings = compute_paddings(output_patchsize=(math.prod(args.scales_t), math.prod(args.scales_hw), math.prod(args.scales_hw)), scales=cfg['scales'], kernel_sizes=tuple((0, k, k) for k in cfg['kernels']), depths=cfg['l_depths'], resize_methods=args.l_upsample_type)
        cfg['stem_paddings'] = tuple(paddings[0][d] + (0 if d == 0 else (cfg['stem_kernels'] - 1) // 2) for d in range(3))
        cfg['paddings'] = paddings[1:]
    else:
        assert tuple(args.stem_paddings) != (-1, -1, -1), 'both padding must be set/not set at the same time'
        assert all(p >= 0 for p in args.stem_paddings) and all(p >= 0 for p in args.paddings)
        cfg['stem_paddings'] = args.stem_paddings
        cfg['paddings'] = [args.paddings for _ in range(len(args.l_depths))]

    # Stem/Blocks/Head/Upsampling/Encoding config
    for prefix in ['l_block', 'c_block', 'l_enc', 'c_enc', 'l_upsample', 'c_upsample', 'stem', 'head']:
        cfg[f'{prefix}_cfg'] = {}
        for k, v in vars(args).items():
            if k.startswith(f'{prefix}_'):
                cfg[f'{prefix}_cfg'][k.replace(f'{prefix}_', f'')] = v

    # Build decoder
    logger.info(f'Building NeRV Decoder with cfg: {cfg}')
    decoder = NeRVDecoder(logger, **cfg)

    return decoder


def build_model(args, logger, input):
    # Set default configurations
    # The first level feature map size
    args.base_size = (
        input['video_size'][0] // int(math.prod(args.scales_t)) if args.base_size[0] == -1 else args.base_size[0],
        input['video_size'][1] // int(math.prod(args.scales_hw)) if args.base_size[1] == -1 else args.base_size[1],
        input['video_size'][2] // int(math.prod(args.scales_hw)) if args.base_size[2] == -1 else args.base_size[2]
    )
    args.base_channels = sum([int(args.base_grid_size[-1] // args.base_grid_level_scale_t[-1] ** i) for i in range(args.base_grid_level_t)] * len(args.base_grid_level_scale_h))

    # The first level grid size
    args.base_grid_size[0] = args.base_grid_size[0] if args.base_grid_size[0] != -1 else args.base_size[0]
    args.base_grid_size[1] = args.base_grid_size[1] if args.base_grid_size[1] != -1 else args.base_size[1]
    args.base_grid_size[2] = args.base_grid_size[2] if args.base_grid_size[2] != -1 else args.base_size[2]
    args.base_grid_size[3] = args.base_grid_size[3] if args.base_grid_size[3] != -1 else 8

    assert args.base_size[0] == input['video_size'][0] // int(math.prod(args.scales_t))
    assert args.base_size[1] == input['video_size'][1] // int(math.prod(args.scales_hw))
    assert args.base_size[2] == input['video_size'][2] // int(math.prod(args.scales_hw))

    # Building encoding and decoder
    encoding = build_encoding(args, logger, args.base_size, args.base_channels)
    decoder = build_decoder(args, logger, args.base_size, args.base_channels, input['video_size'], 3)

    # Building model wrapper
    wrapper_cfg = {}
    wrapper_cfg['eval_patch_size'] = args.eval_patch_size

    logger.info(f'Building NeRV with cfg: {wrapper_cfg}')

    model = NeRV(encoding, decoder, **wrapper_cfg)

    return model


def set_args(parser):
    # NeRV architecture parameters
    group = parser.add_argument_group('Model specific parameters')

    # Model complexity
    group.add_argument(f'--l-channels', type=int, default=256, help='channels of Luma branch')
    group.add_argument(f'--l-channels-reduce', type=float, default=2.0, help='channels reduction factor of Luma branch')
    group.add_argument(f'--l-channels-reduce-base', type=int, default=1, help='the base number which force to be a factor of channels of Luma branch')
    group.add_argument(f'--l-channels-min', type=int, default=1, help='min channels of Luma branch')
    group.add_argument(f'--l-depths', type=int, nargs='+', default=[3, 3, 3, 1], help='depths of Luma branch')
    group.add_argument(f'--l-exps', type=float, nargs='+', default=[4., 4., 4., 1.], help='expansion ratio of Luma branch')

    group.add_argument(f'--c-channels', type=int, default=128, help='channels of Chroma branch')
    group.add_argument(f'--c-channels-reduce', type=float, default=2.0, help='channels reduction factor of Chroma branch')
    group.add_argument(f'--c-channels-reduce-base', type=int, default=1, help='the base number which force to be a factor of channels of Chroma branch')
    group.add_argument(f'--c-channels-min', type=int, default=1, help='min channels of Chroma branch')
    group.add_argument(f'--c-depths', type=int, nargs='+', default=[3, 3, 3, 1], help='depths of Chroma branch')
    group.add_argument(f'--c-exps', type=float, nargs='+', default=[4., 4., 4., 1.], help='expansion ratio of Chroma branch')

    group.add_argument(f'--stem-kernels', type=int, default=1, help='stem kernel sizes of NeRV')
    group.add_argument(f'--kernels', type=int, nargs='+', default=[1, 1, 1, 1], help='kernel sizes of NeRV')
    group.add_argument(f'--scales-t', type=int, nargs='+', default=[1, 1, 1, 1], help='scaling factors in temporal dimension of NeRV')
    group.add_argument(f'--scales-hw', type=int, nargs='+', default=[5, 4, 3, 2], help='scaling factors in spatial dimensions of NeRV')
    group.add_argument(f'--stem-paddings', type=int, nargs='+', default=[0, 1, 1], help='stem paddings of NeRV')
    group.add_argument(f'--paddings', type=int, nargs='+', default=[0, 1, 1], help='paddings of NeRV')

    # Base
    for prefix in ['base']:
        group.add_argument(f'--{prefix}-size', type=int, nargs='+', default=[600, 9, 16], help='encoding size of NeRV')
        group.add_argument(f'--{prefix}-grid-size', type=int, nargs='+', default=[1, 1, 1], help=f'{prefix.title()} grid dimensions (T/H/W). The dimension of highest resolution for the multilevel case.')
        group.add_argument(f'--{prefix}-grid-level-t', type=int, default=2, help=f'{prefix.title()} number of grid levels in temporal dimension')
        group.add_argument(f'--{prefix}-grid-level-scale-t', type=float, nargs='+', default=[2.0, 0.5], help=f'{prefix.title()} grid scaling ratio (T/C) with the grid level.')
        group.add_argument(f'--{prefix}-grid-level-scale-h', type=float, nargs='+', default=[1.0, 1.0, 1.0], help=f'{prefix.title()} grid scaling ratio (H) with the grid level.')
        group.add_argument(f'--{prefix}-grid-level-scale-w', type=float, nargs='+', default=[1.0, 1.0, 1.0], help=f'{prefix.title()} grid scaling ratio (W) with the grid level.')
        group.add_argument(f'--{prefix}-grid-init-scale', type=float, default=1e-3, help=f'{prefix.title()} grid initialization scaling factor.')

    # Blocks
    default_block = {'l-block': 'mlp', 'c-block': 'mlp', 'stem': 'conv_stem', 'head': 'linear_head'}
    default_norm = {'l-block': 'layernorm-no-affine', 'c-block': 'layernorm-no-affine', 'stem': 'none', 'head': 'none'}
    default_act = {'l-block': 'gelu', 'c-block': 'gelu', 'stem': 'none', 'head': 'sigmoid'}
    default_bias = {'l-block': False, 'c-block': False, 'stem': True, 'head': True}
    for prefix in ['l-block', 'c-block', 'stem', 'head']:
        group.add_argument(f'--{prefix}-type', type=str, default=default_block[prefix], help=f'type of {prefix}')
        group.add_argument(f'--{prefix}-norm', type=str, default=default_norm[prefix], help=f'type of normalization for {prefix}s')
        group.add_argument(f'--{prefix}-act', type=str, default=default_act[prefix], help=f'type of activation for {prefix}s')
        group.add_argument(f'--{prefix}-layerscale', type=float, default=0.0, help=f'initial layerscale for {prefix}s (0.: disable)')
        group.add_argument(f'--{prefix}-dropout', type=float, default=0.0, help=f'dropout rate for {prefix}s (0.: disable)')
        group.add_argument(f'--{prefix}-droppath', type=float, default=0.0, help=f'droppath rate for {prefix}s (0.: disable)')
        group.add_argument(f'--{prefix}-bias', type=str_to_bool, default=default_bias[prefix], help=f'Use bias in blocks')
    group.add_argument(f'--l-block-types', type=str, nargs='+', help=f'type of blocks in Luma branch')
    group.add_argument(f'--c-block-types', type=str, nargs='+', help=f'type of blocks in Chroma branch')
    group.add_argument(f'--l-block-mat-chans', type=int, default=2, help=f'number of channels for MAT block in Luma branch')
    group.add_argument(f'--c-block-mat-chans', type=int, default=2, help=f'number of channels for MAT block in Chroma branch')

    # Upsampling Encoders
    for prefix in ['l-enc', 'c-enc']:
        group.add_argument(f'--{prefix}-type', type=str, default='normalized+temp_local_grid', help=f'type of upsampling encoding for {prefix}')
        group.add_argument(f'--{prefix}-align-corners', type=str_to_bool, default=False, help=f'compute upsampling coordinate with align corners for {prefix}')
        group.add_argument(f'--{prefix}-pe', type=float, nargs='+', default=[1.2, 60, 1.2, 60], help=f'Frequency Encoding parameters (Bt/Lt/Bs/Ls) for {prefix}')
        group.add_argument(f'--{prefix}-pe-no-t', type=str_to_bool, default=False, help=f'Do not use temporal dimension for enc encoding for {prefix}')
        group.add_argument(f'--{prefix}-grid-size', type=int, nargs='+', default=[1, 1, 1, 1], help=f'grid dimensions (T/H/W/C) for {prefix}. The dimension of highest resolution.')
        group.add_argument(f'--{prefix}-grid-level', type=int, default=3, help=f'number of grid levels for {prefix}')
        group.add_argument(f'--{prefix}-grid-level-scale', type=float, nargs='+', default=[1.0, 1.0, 1.0, 1.0], help=f'grid scaling ratio (T/H/W/C) with the grid level for {prefix}.')
        group.add_argument(f'--{prefix}-grid-init-scale', type=float, default=1e-3, help=f'grid initialization scaling factor for {prefix}')
        group.add_argument(f'--{prefix}-grid-depth-scale', type=float, nargs='+', default=[1., 1., 1., 1.], help=f'grid scaling ratio (T/H/W/C) with the stage for {prefix}')

    # Upsample layers
    for prefix in ['l-upsample', 'c-upsample']:
        group.add_argument(f'--{prefix}-type', type=str, default='trilinear', help=f'Upsampling method for {prefix}')
        group.add_argument(f'--{prefix}-config', type=str, default='matmul-th-w', help=f'Upsampling method config for {prefix}')
        group.add_argument(f'--{prefix}-norm', type=str, default='layernorm-no-affine', help=f'type of normalization for {prefix}')
        group.add_argument(f'--{prefix}-act', type=str, default='none', help=f'type of activation for {prefix}')

    # Wrapper
    group.add_argument('--eval-patch-size', type=int, nargs='+', default=None, help='patch size during evaluation for NeRV')
