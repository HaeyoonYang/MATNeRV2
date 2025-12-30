"""
NeRV Training Script
"""

from utils import *
from datasets import create_dataset, create_loader, set_dataset_args
from nerv_tasks import VideoRegressionTask, set_task_args
from nerv_compress import *
import sys

from deepspeed.profiling.flops_profiler import get_model_profile

parser = argparse.ArgumentParser()


# Script
group = parser.add_argument_group('Script parameters')

group.add_argument('--compress-folder', default=None, type=str, help='Path to the trained folder for compression')
group.add_argument('--output', default='output/compression', type=str, help='Path to output folder')

group.add_argument('--bitstream', default='', type=str, help='Bitstream path for evaluation.')
group.add_argument('--bitstream-q', default='', type=str, help='Bitstream quantization level for evaluation.')
group.add_argument('--quant-level', default=[8], type=int, nargs='+', help='Quantization level (default: 8)')

group.add_argument('--log-eval', default=False, action='store_true', help='Save copressed images')



def eval_step(args, logger, suffix, epoch, model, loader, task, accelerator, log_output):
    start_time = time.time()
    model.eval()

    accum_loss = None
    accum_metrices = None
    reduced_loss = None
    reduced_metrics = None
    counts = 0
    samples = 0

    for i, batch in enumerate(loader):
        # Eval step
        with torch.no_grad():
            inputs, _, outputs, outmask, loss, metrics = task.step(model, loader, batch)
            mean_loss = loss.mean()
            mean_metrics = {k: v.mean() for k, v in metrics.items()}

        if i == 0:
            accum_loss = mean_loss
            accum_metrices = {k: mean_metrics[k] for k in mean_metrics}
        else:
            accum_loss = accum_loss + mean_loss
            accum_metrices = {k: accum_metrices[k] + mean_metrics[k] for k in mean_metrics}

        counts += 1
        samples += inputs['batch_size'] * accelerator.num_processes

        if (i + 1) % (len(loader) // 4) == 0 or (i + 1) == len(loader):
            # Accumulate loss and metrics globally
            reduced_loss = accelerator.reduce(accum_loss, reduction='mean')
            reduced_metrics = accelerator.reduce(accum_metrices, reduction='mean')
            # Logging loss & metrics
            log_msg = f'Eval' + ('' if not suffix else f' ({suffix})') +  f' - [{i + 1}/{len(loader)}]'
            log_msg = log_msg + f'    img/s: {samples / (time.time() - start_time):.2f}'
            log_msg = log_msg + f'    loss: {reduced_loss.item() / counts:.4f}'
            for k, v in reduced_metrics.items():
                log_msg = log_msg + f'    {k}: {v.item() / counts:.4f}'
            logger.info(log_msg)

        # Logging outputs
        if log_output:
            task.log_eval(dir_name=f'{epoch}' + ('' if not suffix else f'_{suffix.lower()}'), inputs=inputs, outputs=outputs, metrics=metrics, outmask=outmask)

    return reduced_loss.item() / counts, {k: v.item() / counts for k, v in reduced_metrics.items()}



def main():
    start_time = datetime.datetime.now()

    # Set & store args
    args = parser.parse_args()

    # Read yaml file from args.compress_folder
    with open(os.path.join(args.compress_folder, 'args.yaml'), 'r') as fp:
        arg_dict = yaml.load(fp, Loader=yaml.FullLoader)

    for k, v in arg_dict.items():
        if f'--{k.replace("_", "-")}' not in sys.argv:
            if not hasattr(args, k):
                parser.add_argument(f'--{k}', type=type(v), default=v)
            else:
                setattr(args, k, v)

    args = parser.parse_args()

    # Set accelerator, logger, output dir
    accelerator, logger, output_dir = get_accelerator_logger(args)
    logger.info(f'Output dir: {output_dir}')

    # Set seed
    logger.info(f'Set seed: {args.seed}')
    accelerate.utils.set_seed(args.seed, device_specific=True)
    torch.backends.cudnn.deterministic = True   # CuDNN 연산의 결정성 보장
    torch.backends.cudnn.benchmark = False     # 연산 최적화 비활성화 (재현 위해)

    # Task setting
    eval_task = VideoRegressionTask(args, logger, accelerator, root=os.path.join(output_dir, 'eval_output'),
                                    training=False, enable_log_eval=args.log_eval)

    # Optimize steps
    if accelerator.state.dynamo_plugin.backend != accelerate.utils.DynamoBackend.NO:
        # Some precision issues with max-autotune exist, so the default cfg is used here.
        eval_task.step = torch.compile(eval_task.step) #, **accelerator.state.dynamo_plugin.to_kwargs())

    # Datasets & Loaders
    logger.info(f'Create evaluation dataset & loader: {os.path.join(args.dataset, args.dataset_name)}')
    eval_dataset = create_dataset(logger=logger, args=args, training=False)
    eval_loader = create_loader(logger=logger, args=args, training=False, dataset=eval_dataset)

    # Model
    logger.info(f'Create model: {args.model}')
    model = model_cls[args.model].build_model(args, logger, eval_task.parse_input(eval_loader, next(iter(eval_loader)))[0])
    logger.info(f'Model info:')
    logger.info(model)

    # Compute number of parameters & MACs
    logger.info(f'Number of parameters:')
    num_params = model.get_num_parameters()
    for k, v in num_params.items():
        logger.info(f'    {k}: {v / 10**6:.4f}M')

    """
    # Compute MACs
    with torch.no_grad():
        model.eval()
        _, macs, _ = get_model_profile(model=model, 
                                       args=[eval_task.parse_input(eval_loader, next(iter(eval_loader)))[0]],
                                       print_profile=False, detailed=False, warm_up=1, as_string=False)
        macs /= args.eval_batch_size
    logger.info(f'MACs: {macs / 10 ** 9 :.4f}G')
    """

    # Pruning & quanization
    initial_parametrizations(args, logger, model)

    # Place model & loaders
    model, eval_loader = accelerator.prepare(model, eval_loader)

    # Restoring training state
    checkpoint_manager = CheckpointManager(logger, accelerator, os.path.join(output_dir, 'checkpoints'), args.eval_metric[0])
    ckpt_path = os.path.join(args.compress_folder, 'checkpoints/checkpoint_389')
    logger.info(f'Restore from training state: {ckpt_path}')
    epoch = checkpoint_manager.load(ckpt_path) + int(not args.eval_only)

    best_metrics = BestMetricTracker(logger, accelerator, os.path.join(output_dir, 'results'), 
                                    {**{k: ['size', args.eval_metric[0]] for k in ['full', 'pruned', 'qat']},
                                        **{f'Q{quant_level}': ['bpp', args.eval_metric[0]] for quant_level in sorted(args.quant_level, reverse=True)}})
    zeros, total = get_sparsity(model)
    logger.info(f'Number of pruned parameters: {zeros}    total: {total}')
    logger.info(f'Sparsity: {zeros / total:.4f}')

    # _ = eval_step(args, logger, '', epoch, model, eval_loader, eval_task, accelerator, True)

    qat_state_dict = copy.deepcopy(model.state_dict())

    # Restore from bitstream
    if args.bitstream:
        logger.info(f'Restore model weights from bitstream')
        num_bytes = decompress(args, logger, accelerator, model, os.path.join(args.bitstream, 'bitstreams'), int(args.bitstream_q))
        bits_per_pixel = num_bytes * 8 / np.prod(eval_dataset.video_size)

        logger.info(f'Compressed model size: {num_bytes / 10**6:.2f}MB')
        logger.info(f'Bits Per Pixel (BPP): {bits_per_pixel:.4f}')
        
        # Evaluation
        _, metrics = eval_step(args, logger, f'Q{args.bitstream_q}', epoch, model, eval_loader, eval_task, accelerator, args.log_eval)
        best_metrics.update(f'Q{args.bitstream_q}', bits_per_pixel, metrics)
        
        # Complete training
        train_time = datetime.datetime.now() - start_time

    else:
        for quant_level in sorted(args.quant_level, reverse=True):
            # Compress bitstream
            logger.info(f'Compress model weights into bitstream')
            logger.info(f'***  Quant level: {quant_level}bits')
            logger.info(f'***  Sparsity: {zeros / total:.4f}')

            num_bytes = compress_bitstream(args, logger, accelerator, model, os.path.join(output_dir, 'bitstreams'), quant_level)
            bits_per_pixel = num_bytes * 8 / np.prod(eval_dataset.video_size)

            logger.info(f'Compressed model size: {num_bytes / 10**6:.2f}MB')
            logger.info(f'Bits Per Pixel (BPP): {bits_per_pixel:.4f}')

            # Set model weights to zero for ensuring the correctness
            set_zero(model)

            # Decompress bitstream
            logger.info(f'Decompress model weights from bitstream')
            decompress(args, logger, accelerator, model, os.path.join(output_dir, 'bitstreams'), quant_level)

            # Evaluation
            _, metrics = eval_step(args, logger, f'Q{quant_level}', epoch, model, eval_loader, eval_task, accelerator, args.log_eval)
            best_metrics.update(f'Q{quant_level}', bits_per_pixel, metrics)

            # if quant_level == min(args.quant_level): # use the least bitwidth for checkpointing metric
            #     checkpoint_manager.save(epoch, metrics)

            # Restore from the checkpoint
            model.load_state_dict(qat_state_dict)

            # Complete training
            train_time = datetime.datetime.now() - start_time

        logger.info(f'Training completed in: {train_time}')
        logger.info(f'Output are located in: {output_dir}')



if __name__ == '__main__':
    main()