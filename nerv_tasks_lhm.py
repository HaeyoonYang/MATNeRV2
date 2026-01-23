"""
NeRV Task
"""
from utils import *
from losses import compute_loss, compute_metric, Gauss_model_binary



class VideoRegressionTask:
    def __init__(self, args, logger, accelerator, root, training=True, enable_log_eval=False):
        self.root = root
        self.accelerator = accelerator
        self.loss_cfg = ['1.0', args.loss[0]] if len(args.loss) == 1 else args.loss
        if args.mask_loss == []:
            self.mask_loss_cfg = []
        else:
            self.mask_loss_cfg = ['1.0', args.mask_loss[0]] if len(args.mask_loss) == 1 else args.mask_loss
        if args.aux_loss == []:
            self.aux_loss_cfg = []
        else:
            self.aux_loss_cfg = ['1.0', args.aux_loss[0] if len(args.aux_loss) == 1 else args.aux_loss]
        if args.mask_model == 'none':
            self.mask_model = None
        else:
            kernel_size, sigma, k = args.mask_model.split('_')[1:4]
            self.mask_model = Gauss_model_binary(int(kernel_size), float(sigma), float(k), abs=('abs' in args.mask_model)).to(accelerator.device)

        self.metric_cfg = args.train_metric if training else args.eval_metric
        self.training = training
        self.enable_log_eval = enable_log_eval

        logger.info(f'VideoRegressionTask:')
        logger.info(f'     Root: {self.root}')
        logger.info(f'     Losses: {self.loss_cfg}    Mask losses: {self.mask_loss_cfg}    Aux losses: {self.aux_loss_cfg}    Metrics: {self.metric_cfg}')
        logger.info(f'     Training: {self.training}')
        logger.info(f'     Log evaulation: {self.enable_log_eval}')

    def parse_input(self, loader, batch):
        """
        Parse the input to the model during training/evaluation step
        """
        idx, x = batch
        assert idx.ndim == 2, 'idx should have 2 dimensions with shape [N, 3], where each row is the 3D patch coordinate'
        assert x.ndim == 5, 'x should have 5 dimensions with shape [N, C, T, H, W], where each sample is a 3D patch'

        N, C, T, H, W = x.shape
        y = x.permute(0, 2, 1, 3, 4).view(N*T, C, H, W)
        y = y.mean(dim=1, keepdim=True)
        gt_mask = self.mask_model(y)
        gt_mask = gt_mask.view(N, T, 1, H, W).permute(0, 2, 1, 3, 4).detach()

        input = {
            'x': x if self.training else None,
            'idx': idx,
            'idx_max': loader.dataset.num_patches,
            'batch_size': x.shape[0] * loader.dataset.patch_size[0] / (loader.dataset.num_patches[1] * loader.dataset.num_patches[2]),
            'video_size': loader.dataset.video_size,
            'gt_mask': gt_mask if self.training else None,
            'patch_size': loader.dataset.patch_size,
        }

        target = x

        return input, target, gt_mask
    
    def parse_output(self, loader, batch):
        """
        Parse the output from the model during training/evaluation step
        """
        output, outmask = batch
        output = output.contiguous(memory_format=torch.contiguous_format)

        return output, outmask
    
    def compute_loss(self, x, y, loss_cfg, model=None):
        total_loss = None
        for i in range(len(loss_cfg) // 2):
            weight = float(loss_cfg[i * 2])
            loss_type = loss_cfg[i * 2 + 1]
            loss = weight * compute_loss(loss_type, x, y, model)
            total_loss = total_loss + loss if total_loss is not None else loss
        return total_loss
    
    def compute_metrics(self, x, y):
        metrics = {}
        for metric_type in self.metric_cfg:
            metrics[metric_type] = compute_metric(metric_type, x, y)
        return metrics
    
    def step(self, model, loader, batch):
        inputs, targets, gt_masks = self.parse_input(loader, batch)
        outputs, outmasks = self.parse_output(loader, model(inputs))
        if self.training:
            loss = self.compute_loss(outputs, targets, self.loss_cfg)
            m_loss = self.compute_loss(outmasks, gt_masks, self.mask_loss_cfg)
            aux_loss = self.compute_loss(outputs, targets, self.aux_loss_cfg)
            loss = loss + m_loss if m_loss is not None else loss
            loss = loss + aux_loss if aux_loss is not None else loss
        else:
            loss = torch.tensor(0.0, device=outputs.device)
            m_loss = torch.tensor(0.0, device=outputs.device)
            aux_loss = torch.tensor(0.0, device=outputs.device)
        metrics = self.compute_metrics(outputs, targets)
        return inputs, targets, outputs, outmasks, loss, m_loss, metrics
    
    def log_eval(self, dir_name, inputs, outputs, metrics, outmask=None):
        """
        Log the evaluation outputs
        """
        if not self.enable_log_eval:
            return
        
        # Use the first metric
        metric = metrics[self.metric_cfg[0]]

        N, C, T, H, W = outputs.shape
        _, H_img, W_img = inputs['video_size']
        T_patch, H_patch, W_patch = inputs['patch_size']

        assert H_img == H_patch and W_img == W_patch, 'Only full image output is supported'

        idx = inputs['idx'].cpu()
        outputs = outputs.detach().cpu()
        metric = metric.detach().cpu()
        if outmask is not None:
            outmask = outmask.detach().cpu()

        output_dir = os.path.join(self.root, dir_name)

        if self.accelerator.is_main_process:
            os.makedirs(output_dir, exist_ok=True)

        self.accelerator.wait_for_everyone()

        for n in range(N):
            t, h, w = idx[n].numpy()
            patch_idx = 1 + t * T_patch

            for dt in range(T):
                img_id = f'{patch_idx + dt:04d}_{metric[n, dt].numpy():.2f}'
                img = outputs[n, :, dt, :, :].float().cpu()
                torchvision.utils.save_image(img, os.path.join(output_dir, img_id + '.png'), 'png')

                if outmask is not None:
                    mask_id = f'{patch_idx + dt:04d}_mask'
                    mask = outmask[n, :, dt, :, :].float().cpu()
                    torchvision.utils.save_image(mask, os.path.join(output_dir, mask_id + '.png'), 'png')


def set_task_args(parser):
    group = parser.add_argument_group('Task parameters')
    group.add_argument('--loss', default='mse', type=str, nargs='+', help='Loss (default: "mse")')
    group.add_argument('--mask-loss', default=[], type=str, nargs='+', help='Mask Loss (default: [])')
    group.add_argument('--aux-loss', default=[], type=str, nargs='+', help='Auxiliary Loss (default: [])')
    group.add_argument('--mask-model', default='gauss_11_5.0_20', type=str, help='Mask model (default: "gauss_11_5.0_20")')
    group.add_argument('--train-metric', default=['psnr'], type=str, nargs='+', help='Metric (default: "psnr")')
    group.add_argument('--eval-metric', default=['psnr', 'ms_ssim'], type=str, nargs='+', help='Metric (default: "psnr")')
    group.add_argument('--log-eval', type=str_to_bool, default=True, help='Log the output during evaluation.')