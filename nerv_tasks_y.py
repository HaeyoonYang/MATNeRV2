"""
NeRV Task
"""
from utils import *
from losses_y import compute_loss, compute_metric, compute_regularization, Gauss_model2


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
            self.aux_loss_cfg = ['1.0', args.aux_loss[0]] if len(args.aux_loss) == 1 else args.aux_loss
        if args.mask_model == 'none':
            self.mask_model = None
        else:
            kernel_size, sigma = args.mask_model.split('_')[1:3]
            self.mask_model = Gauss_model2(int(kernel_size), float(sigma)).to(accelerator.device)

        self.metric_cfg = args.train_metric if training else args.eval_metric
        self.training = training
        self.enable_log_eval = enable_log_eval

        logger.info(f'VideoRegressionTask:')
        logger.info(f'     Root: {self.root}')
        logger.info(f'     Losses: {self.loss_cfg}    Mask losses: {self.mask_loss_cfg}    Aux losses: {self.aux_loss_cfg}    Metrics: {self.metric_cfg}')
        logger.info(f'     Training: {self.training}')
        logger.info(f'     Log evaluation: {self.enable_log_eval}')

    def parse_input(self, loader, batch):
        """
        Parse the input to the model during training/evaluation step
        """
        idx, x = batch
        assert idx.ndim == 2, 'idx should have 2 dimensions with shape [N, 3], where each row is the 3D patch coordinate'
        assert x.ndim == 5,  'x should have 5 dimensions with shape [N, C, T, H, W], where each sample is a 3D patch'

        x_ycocg = torch.round(x * 255.)
        x_ycocg = rgb_to_ycocg(x_ycocg) / 255.

        input = {
            'x': x if self.training else None,
            'idx': idx, 
            'idx_max': loader.dataset.num_patches,
            'batch_size': x.shape[0] * loader.dataset.patch_size[0] / (loader.dataset.num_patches[1] * loader.dataset.num_patches[2]), # in the number of images
            'video_size': loader.dataset.video_size,
            'patch_size': loader.dataset.patch_size
        }

        target_rgb = x
        target_ycocg = x_ycocg

        return input, target_rgb, target_ycocg

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
            if isinstance(x, list):
                elem_loss = torch.tensor(0.0, device=x.device)
                for j in range(len(x)):
                    elem_loss += compute_loss(loss_type, x[j], y[j] if isinstance(y, list) else y, model)
                loss = weight * elem_loss
            else:
                loss = weight * compute_loss(loss_type, x, y, model)
            total_loss = total_loss + loss if total_loss is not None else loss
        return total_loss

    def compute_metrics(self, x, y):
        metrics = {}
        for metric_type in self.metric_cfg:
            metrics[metric_type] = compute_metric(metric_type, x, y)
        return metrics

    def step(self, model, loader, batch):
        inputs, targets_rgb, targets_ycocg = self.parse_input(loader, batch)
        outputs, outmasks = self.parse_output(loader, model(inputs))
        if self.training:
            loss = self.compute_loss(outputs, targets_ycocg, self.loss_cfg)
            m_loss = self.compute_loss(outmasks[-1] if len(outmasks) > 0 else None, targets_ycocg[:, [0]], self.mask_loss_cfg, self.mask_model)
            aux_loss = self.compute_loss(outputs, targets_ycocg, self.aux_loss_cfg, self.mask_model)
            loss = loss + m_loss if m_loss is not None else loss
            loss = loss + aux_loss if aux_loss is not None else loss
        else:
            loss = torch.tensor(0.0, device=outputs.device)
            m_loss = torch.tensor(0.0, device=outputs.device)

        # compute metrics
        outputs_rgb = torch.round(outputs * 255.)
        outputs_rgb = ycocg_to_rgb(outputs_rgb) / 255.
        metrics = self.compute_metrics(outputs_rgb, targets_rgb)
        return inputs, targets_rgb, outputs_rgb, None if outmasks==[] else outmasks[-1], loss, m_loss, metrics

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