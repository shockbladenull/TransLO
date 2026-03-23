# -*- coding:UTF-8 -*-

import csv
import datetime
import json
import os
import shutil
import sys
import time

import numpy as np
import torch
import torch.distributed as dist
import torch.utils.data
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from configs import translonet_args
from dataset_factory import build_dataset
from kitti_pytorch import points_dataset
from tools.excel_tools import SaveExcel
from tools.euler_tools import quat2mat
from tools.logger_tools import creat_logger, log_print
from tools.oxford_train_eval import run_oxford_detailed_val, should_run_oxford_detailed_val
from tools.tensorboard_tools import (
    log_scalar_group,
    train_global_step,
)
from translo_model import get_loss, translo_model
from utils1.collate_functions import collate_pair


args = translonet_args()
base_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(base_dir)

SOURCE_BACKUP_FILES = (
    'train.py',
    'configs.py',
    'dataset_factory.py',
    'translo_model.py',
    'translo_model_utils.py',
    'conv_util.py',
    'kitti_pytorch.py',
    'tools/oxford_train_eval.py',
)

PROGRESS_UPDATE_INTERVAL = 10
METRIC_FIELDS = (
    'timestamp',
    'phase',
    'epoch',
    'total_epochs',
    'loss',
    'translation_error',
    'rotation_error_deg',
    'lr',
    'epoch_time_sec',
    'avg_data_time_sec',
    'avg_iter_time_sec',
    'samples_per_sec',
    'global_samples',
    'avg_points',
    'gpu_mem_gb',
    'gpu_peak_mem_gb',
    'notes',
)


def parse_requested_gpus():
    if args.multi_gpu is None:
        return []
    return [int(x) for x in args.multi_gpu.split(',') if x]


def is_distributed():
    return getattr(args, 'distributed', False)


def is_main_process():
    return getattr(args, 'rank', 0) == 0


def barrier():
    if is_distributed():
        dist.barrier(device_ids=[args.local_rank])


def unwrap_model(model):
    return model.module if isinstance(model, DistributedDataParallel) else model


def log_message(logger, message):
    if logger is not None:
        log_print(logger, message)


def setup_runtime():
    requested_gpus = parse_requested_gpus()
    if requested_gpus and 'CUDA_VISIBLE_DEVICES' not in os.environ:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.multi_gpu
    world_size = int(os.environ.get('WORLD_SIZE', '1'))

    args.rank = int(os.environ.get('RANK', '0'))
    args.local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    args.world_size = world_size
    args.distributed = world_size > 1

    if args.distributed:
        if requested_gpus and len(requested_gpus) != world_size:
            raise ValueError(
                '--multi_gpu has {} ids, but WORLD_SIZE is {}. Launch DDP with matching processes.'.format(
                    len(requested_gpus), world_size
                )
            )
        torch.cuda.set_device(args.local_rank)
        dist.init_process_group(
            backend='nccl',
            timeout=datetime.timedelta(seconds=args.ddp_timeout_sec),
        )
        args.device = torch.device('cuda', args.local_rank)
    else:
        if len(requested_gpus) > 1:
            raise ValueError(
                'Multi-GPU training now uses DistributedDataParallel. Launch with '
                '`torchrun --nproc_per_node={} train.py ...` instead of plain python.'.format(len(requested_gpus))
            )
        device_id = requested_gpus[0] if requested_gpus else args.gpu
        torch.cuda.set_device(device_id)
        args.device = torch.device('cuda', device_id)

    torch.backends.cudnn.benchmark = True


def cleanup_runtime():
    if is_distributed() and dist.is_initialized():
        dist.destroy_process_group()


def prepare_output_dirs():
    experiment_dir = os.path.join(base_dir, 'experiment')
    if not args.task_name:
        file_dir = os.path.join(
            experiment_dir,
            '{}_{}_{}'.format(
                args.model_name,
                args.train_dataset_type.upper(),
                datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'),
            ),
        )
    else:
        file_dir = os.path.join(experiment_dir, args.task_name)

    eval_dir = os.path.join(file_dir, 'eval')
    log_dir = os.path.join(file_dir, 'logs')
    tensorboard_dir = os.path.join(file_dir, 'tensorboard')
    checkpoints_dir = os.path.join(file_dir, 'checkpoints/translonet')

    if is_main_process():
        for directory in (experiment_dir, file_dir, eval_dir, log_dir, tensorboard_dir, checkpoints_dir):
            os.makedirs(directory, exist_ok=True)
        for filename in SOURCE_BACKUP_FILES:
            backup_path = os.path.join(log_dir, filename)
            os.makedirs(os.path.dirname(backup_path), exist_ok=True)
            shutil.copy2(os.path.join(base_dir, filename), backup_path)

    barrier()
    return file_dir, eval_dir, log_dir, tensorboard_dir, checkpoints_dir


def make_dataloader(dataset, batch_size, shuffle, sampler=None):
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=args.workers,
        collate_fn=collate_pair,
        pin_memory=True,
        worker_init_fn=lambda _: np.random.seed(torch.initial_seed() % (2 ** 32)),
    )


def move_batch_to_device(data):
    pos2, pos1, sample_id, T_gt, T_trans, T_trans_inv, Tr = data
    pos2 = [batch.to(args.device, non_blocking=True) for batch in pos2]
    pos1 = [batch.to(args.device, non_blocking=True) for batch in pos1]
    T_trans = T_trans.to(args.device, dtype=torch.float32, non_blocking=True)
    T_trans_inv = T_trans_inv.to(args.device, dtype=torch.float32, non_blocking=True)
    T_gt = T_gt.to(args.device, dtype=torch.float32, non_blocking=True)
    return pos2, pos1, sample_id, T_gt, T_trans, T_trans_inv, Tr


def quaternion_angle_error_deg(pred_q, gt_q):
    pred_q = pred_q / (torch.norm(pred_q, dim=-1, keepdim=True) + 1e-10)
    gt_q = gt_q / (torch.norm(gt_q, dim=-1, keepdim=True) + 1e-10)
    dot = torch.sum(pred_q * gt_q, dim=-1).abs().clamp(max=1.0)
    return (2.0 * torch.acos(dot)) * (180.0 / np.pi)


def reduce_tensor(tensor, op=dist.ReduceOp.SUM):
    if is_distributed():
        dist.all_reduce(tensor, op=op)
    return tensor


def format_duration(seconds):
    total_seconds = max(int(round(seconds)), 0)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours > 0:
        return '{:d}:{:02d}:{:02d}'.format(hours, minutes, secs)
    return '{:02d}:{:02d}'.format(minutes, secs)


def mean_points_in_batch(point_batches):
    if not point_batches:
        return 0.0
    return float(sum(batch.shape[0] for batch in point_batches) / len(point_batches))


def safe_gpu_memory_stats(device):
    if device.type != 'cuda' or not torch.cuda.is_available():
        return 0.0, 0.0
    device_index = device.index if device.index is not None else torch.cuda.current_device()
    current_mem = torch.cuda.memory_allocated(device_index) / (1024 ** 3)
    peak_mem = torch.cuda.max_memory_allocated(device_index) / (1024 ** 3)
    return current_mem, peak_mem


def init_metric_logs(log_dir):
    if not is_main_process():
        return None

    csv_path = os.path.join(log_dir, 'metrics.csv')
    jsonl_path = os.path.join(log_dir, 'metrics.jsonl')

    if not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0:
        with open(csv_path, 'w', newline='') as handle:
            csv.DictWriter(handle, fieldnames=METRIC_FIELDS).writeheader()

    if not os.path.exists(jsonl_path):
        open(jsonl_path, 'a').close()

    return {
        'csv': csv_path,
        'jsonl': jsonl_path,
    }


def append_metric_record(metric_logs, **record):
    if metric_logs is None or not is_main_process():
        return

    row = {field: record.get(field, '') for field in METRIC_FIELDS}
    with open(metric_logs['csv'], 'a', newline='') as handle:
        csv.DictWriter(handle, fieldnames=METRIC_FIELDS).writerow(row)
    with open(metric_logs['jsonl'], 'a') as handle:
        handle.write(json.dumps(row) + '\n')


def validate_oxford(model, val_loader, epoch, total_epochs, logger, metric_logs=None, lr='', tb_writer=None):
    total_loss = torch.zeros(1, device=args.device, dtype=torch.float64)
    total_trans_error = torch.zeros(1, device=args.device, dtype=torch.float64)
    total_rot_error = torch.zeros(1, device=args.device, dtype=torch.float64)
    total_seen = torch.zeros(1, device=args.device, dtype=torch.float64)
    total_points = torch.zeros(1, device=args.device, dtype=torch.float64)
    total_data_time = torch.zeros(1, device=args.device, dtype=torch.float64)
    total_iter_time = torch.zeros(1, device=args.device, dtype=torch.float64)
    total_steps = torch.zeros(1, device=args.device, dtype=torch.float64)

    model.eval()
    if args.device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(args.device)

    val_start = time.time()
    previous_step_end = val_start
    progress = tqdm(
        val_loader,
        total=len(val_loader),
        smoothing=0.1,
        desc='Val {:03d}/{:03d}'.format(epoch, total_epochs),
        dynamic_ncols=True,
        leave=False,
        disable=not is_main_process(),
    )
    with torch.no_grad():
        for step, data in enumerate(progress, 1):
            iter_start = time.time()
            data_time = iter_start - previous_step_end
            pos2, pos1, _, T_gt, T_trans, T_trans_inv, _ = move_batch_to_device(data)
            l0_q, l0_t, l1_q, l1_t, l2_q, l2_t, l3_q, l3_t, _, q_gt, t_gt, w_x, w_q = model(
                pos2, pos1, T_gt, T_trans, T_trans_inv
            )
            loss = get_loss(l0_q, l0_t, l1_q, l1_t, l2_q, l2_t, l3_q, l3_t, q_gt, t_gt, w_x, w_q)

            t_gt = torch.squeeze(t_gt)
            if t_gt.dim() == 1:
                t_gt = t_gt.unsqueeze(0)
            trans_error = torch.norm(l0_t - t_gt, dim=-1)
            rot_error = quaternion_angle_error_deg(l0_q, q_gt)

            batch_size = l0_q.shape[0]
            iter_end = time.time()
            iter_time = iter_end - iter_start

            total_loss += loss.detach().to(torch.float64) * batch_size
            total_trans_error += trans_error.sum().to(torch.float64)
            total_rot_error += rot_error.sum().to(torch.float64)
            total_seen += batch_size
            total_points += mean_points_in_batch(pos2) * batch_size
            total_data_time += data_time
            total_iter_time += iter_time
            total_steps += 1

            if is_main_process() and (step % PROGRESS_UPDATE_INTERVAL == 0 or step == len(val_loader)):
                denom = total_seen.clamp_min(1.0)
                progress.set_postfix(
                    loss='{:.4f}'.format((total_loss / denom).item()),
                    trans='{:.4f}m'.format((total_trans_error / denom).item()),
                    rot='{:.3f}deg'.format((total_rot_error / denom).item()),
                    iter='{:.2f}s'.format(iter_time),
                )
            previous_step_end = iter_end

    if is_distributed():
        total_loss = reduce_tensor(total_loss)
        total_trans_error = reduce_tensor(total_trans_error)
        total_rot_error = reduce_tensor(total_rot_error)
        total_seen = reduce_tensor(total_seen)
        total_points = reduce_tensor(total_points)
        total_data_time = reduce_tensor(total_data_time)
        total_iter_time = reduce_tensor(total_iter_time)
        total_steps = reduce_tensor(total_steps)

    if total_seen.item() == 0:
        raise RuntimeError('Oxford validation loader is empty')

    val_loss = float((total_loss / total_seen).item())
    val_trans = float((total_trans_error / total_seen).item())
    val_rot = float((total_rot_error / total_seen).item())
    val_time_sec = time.time() - val_start
    avg_data_time_sec = float((total_data_time / total_steps.clamp_min(1.0)).item())
    avg_iter_time_sec = float((total_iter_time / total_steps.clamp_min(1.0)).item())
    avg_points = float((total_points / total_seen.clamp_min(1.0)).item())
    gpu_mem_gb, gpu_peak_mem_gb = safe_gpu_memory_stats(args.device)
    samples_per_sec = float(total_seen.item() / max(val_time_sec, 1e-6))
    log_message(
        logger,
        'EPOCH {:03d}/{:03d} val loss: {:.6f} | trans: {:.6f} m | rot: {:.6f} deg | '
        'time: {} | data: {:.2f}s | iter: {:.2f}s | ips: {:.2f} | pts: {:.1f}k | mem: {:.2f}/{:.2f}G'.format(
            epoch,
            total_epochs,
            val_loss,
            val_trans,
            val_rot,
            format_duration(val_time_sec),
            avg_data_time_sec,
            avg_iter_time_sec,
            samples_per_sec,
            avg_points / 1000.0,
            gpu_mem_gb,
            gpu_peak_mem_gb,
        ),
    )
    append_metric_record(
        metric_logs,
        timestamp=datetime.datetime.now().isoformat(timespec='seconds'),
        phase='val',
        epoch=epoch,
        total_epochs=total_epochs,
        loss=round(val_loss, 6),
        translation_error=round(val_trans, 6),
        rotation_error_deg=round(val_rot, 6),
        lr=lr,
        epoch_time_sec=round(val_time_sec, 3),
        avg_data_time_sec=round(avg_data_time_sec, 4),
        avg_iter_time_sec=round(avg_iter_time_sec, 4),
        samples_per_sec=round(samples_per_sec, 3),
        global_samples=int(total_seen.item()),
        avg_points=round(avg_points, 1),
        gpu_mem_gb=round(gpu_mem_gb, 3),
        gpu_peak_mem_gb=round(gpu_peak_mem_gb, 3),
    )
    if is_main_process():
        log_scalar_group(
            tb_writer,
            'val',
            {
                'loss': val_loss,
                'translation_error': val_trans,
                'rotation_error_deg': val_rot,
                'epoch_time_sec': val_time_sec,
                'avg_data_time_sec': avg_data_time_sec,
                'avg_iter_time_sec': avg_iter_time_sec,
                'samples_per_sec': samples_per_sec,
                'avg_points': avg_points,
                'gpu_mem_gb': gpu_mem_gb,
                'gpu_peak_mem_gb': gpu_peak_mem_gb,
            },
            epoch,
        )
    return {
        'loss': val_loss,
        'translation_error': val_trans,
        'rotation_error_deg': val_rot,
    }


def eval_pose(model, test_list, epoch, log_dir, eval_dir, logger):
    for item in test_list:
        test_dataset = points_dataset(
            is_training=0,
            num_point=args.num_points,
            data_dir_list=[item],
            config=args,
        )
        test_loader = make_dataloader(test_dataset, args.eval_batch_size, shuffle=False)
        line = 0
        total_time = 0.0

        for _, data in tqdm(
            enumerate(test_loader),
            total=len(test_loader),
            smoothing=0.1,
            desc='Eval seq {} e{:03d}'.format(str(item).zfill(2), epoch),
            dynamic_ncols=True,
            leave=False,
            disable=not is_main_process(),
        ):
            pos2, pos1, sample_id, T_gt, T_trans, T_trans_inv, Tr = move_batch_to_device(data)

            model.eval()
            with torch.no_grad():
                start_time = time.time()
                l0_q, l0_t, l1_q, l1_t, l2_q, l2_t, l3_q, l3_t, pc1_output, q_gt, t_gt, w_x, w_q = model(
                    pos2, pos1, T_gt, T_trans, T_trans_inv
                )
                total_time += time.time() - start_time

                pc1 = pc1_output.cpu().numpy()
                pred_q = l0_q.cpu().numpy()
                pred_t = l0_t.cpu().numpy()

                for n0 in range(pc1.shape[0]):
                    cur_Tr = Tr[n0, :, :]

                    qq = pred_q[n0:n0 + 1, :].reshape(4)
                    tt = pred_t[n0:n0 + 1, :].reshape(3, 1)
                    RR = quat2mat(qq)
                    filler = np.expand_dims(np.array([0.0, 0.0, 0.0, 1.0]), axis=0)

                    TT = np.concatenate([np.concatenate([RR, tt], axis=-1), filler], axis=0)
                    TT = np.matmul(cur_Tr, TT)
                    TT = np.matmul(TT, np.linalg.inv(cur_Tr))

                    if line == 0:
                        T_final = TT
                        T = T_final[:3, :].reshape(1, 1, 12)
                        line += 1
                    else:
                        T_final = np.matmul(T_final, TT)
                        T_current = T_final[:3, :].reshape(1, 1, 12)
                        T = np.append(T, T_current, axis=0)

        avg_time = total_time / max(len(test_dataset), 1)
        T = T.reshape(-1, 12)

        fname_txt = os.path.join(log_dir, str(item).zfill(2) + '_pred.npy')
        data_dir = os.path.join(eval_dir, 'translonet_' + str(item).zfill(2))
        os.makedirs(data_dir, exist_ok=True)

        np.save(fname_txt, T)
        shutil.copy2(fname_txt, data_dir)
        os.system(
            sys.executable
            + ' evaluation.py --result_dir '
            + data_dir
            + ' --eva_seqs '
            + str(item).zfill(2)
            + '_pred'
            + ' --epoch '
            + str(epoch)
            + ' --gt_dir pose'
        )
        log_message(logger, 'Seq {} avg inference time: {:.6f}s'.format(str(item).zfill(2), avg_time))


def main():
    global args

    setup_runtime()
    file_dir, eval_dir, log_dir, tensorboard_dir, checkpoints_dir = prepare_output_dirs()
    logger = creat_logger(log_dir, args.model_name) if is_main_process() else None
    metric_logs = init_metric_logs(log_dir)
    total_epochs = max(args.max_epoch - 1, 0)
    tb_writer = SummaryWriter(log_dir=tensorboard_dir) if is_main_process() else None

    try:
        if logger is not None:
            logger.info('----------------------------------------TRAINING----------------------------------')
            logger.info('PARAMETER ...')
            logger.info(args)
            logger.info(
                'Projection profile: %s (H=%d, W=%d, up=%.2f, down=%.2f)',
                args.sensor_profile,
                args.H_input,
                args.W_input,
                args.vertical_view_up,
                args.vertical_view_down,
            )

        model = translo_model(args, args.batch_size, args.H_input, args.W_input, args.is_training).to(args.device)

        if is_distributed():
            model = DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank)
            log_message(
                logger,
                'distributed training world_size={}, local_rank={}, timeout={}s'.format(
                    args.world_size,
                    args.local_rank,
                    args.ddp_timeout_sec,
                ),
            )
        else:
            log_message(logger, 'single gpu is: {}'.format(args.device.index))

        train_dataset = build_dataset('train', args, is_training=1)
        train_sampler = DistributedSampler(train_dataset, shuffle=True) if is_distributed() else None
        train_loader = make_dataloader(train_dataset, args.batch_size, shuffle=True, sampler=train_sampler)
        log_message(logger, 'Train dataset: %s (%d samples)' % (args.train_dataset_type, len(train_dataset)))

        val_loader = None
        excel_eval = None
        if args.val_dataset_type == 'oxford_qe':
            val_dataset = build_dataset('val', args, is_training=0)
            val_sampler = DistributedSampler(val_dataset, shuffle=False) if is_distributed() else None
            val_loader = make_dataloader(val_dataset, args.eval_batch_size, shuffle=False, sampler=val_sampler)
            if is_main_process():
                log_message(logger, 'Validation dataset: %s (%d samples)' % (args.val_dataset_type, len(val_dataset)))
        elif is_main_process():
            if args.val_dataset_type != 'oxford_qe':
                excel_eval = SaveExcel(args.kitti_val_seqs, log_dir)
                log_message(logger, 'Validation dataset: %s (%s)' % (args.val_dataset_type, args.kitti_val_seqs))
        if is_main_process():
            log_message(logger, 'TensorBoard dir: {}'.format(tensorboard_dir))
            log_message(logger, 'KITTI test sequences kept unchanged: {}'.format(args.kitti_test_seqs))

        if args.optimizer == 'SGD':
            optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum)
        elif args.optimizer == 'Adam':
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=args.learning_rate,
                betas=(0.9, 0.999),
                eps=1e-08,
                weight_decay=args.weight_decay,
            )
        else:
            raise ValueError('Unsupported optimizer {}'.format(args.optimizer))

        optimizer.param_groups[0]['initial_lr'] = args.learning_rate
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=args.lr_stepsize,
            gamma=args.lr_gamma,
            last_epoch=-1,
        )

        if args.ckpt is not None:
            checkpoint = torch.load(args.ckpt, map_location=args.device)
            unwrap_model(model).load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['opt_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            init_epoch = checkpoint['epoch']
            log_message(logger, 'load model {}'.format(args.ckpt))
        else:
            init_epoch = 0
            log_message(logger, 'Training from scratch')

        barrier()

        if args.eval_before == 1:
            if val_loader is not None:
                if is_main_process():
                    log_message(logger, 'Epoch {:03d}: running validation before training'.format(init_epoch))
                validate_oxford(
                    unwrap_model(model),
                    val_loader,
                    init_epoch,
                    total_epochs,
                    logger,
                    metric_logs=metric_logs,
                    lr='{:.6e}'.format(optimizer.param_groups[0]['lr']),
                    tb_writer=tb_writer,
                )
            elif is_main_process():
                if val_loader is None:
                    log_message(logger, 'Epoch {:03d}: running KITTI evaluation before training'.format(init_epoch))
                    eval_pose(unwrap_model(model), args.kitti_val_seqs, init_epoch, log_dir, eval_dir, logger)
                    excel_eval.update(eval_dir)
            barrier()

        for epoch in range(init_epoch + 1, args.max_epoch):
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)

            model.train()
            if args.device.type == 'cuda':
                torch.cuda.reset_peak_memory_stats(args.device)

            epoch_loss_sum = 0.0
            epoch_seen = 0.0
            epoch_points_sum = 0.0
            epoch_data_time_sum = 0.0
            epoch_iter_time_sum = 0.0
            epoch_steps = 0.0
            epoch_start = time.time()
            previous_step_end = epoch_start
            progress = tqdm(
                train_loader,
                total=len(train_loader),
                smoothing=0.1,
                desc='Epoch {:03d}/{:03d}'.format(epoch, total_epochs),
                dynamic_ncols=True,
                leave=False,
                disable=not is_main_process(),
            )

            for step, data in enumerate(progress, 1):
                iter_start = time.time()
                data_time = iter_start - previous_step_end
                pos2, pos1, sample_id, T_gt, T_trans, T_trans_inv, Tr = move_batch_to_device(data)
                optimizer.zero_grad()

                l0_q, l0_t, l1_q, l1_t, l2_q, l2_t, l3_q, l3_t, pc1_output, q_gt, t_gt, w_x, w_q = model(
                    pos2, pos1, T_gt, T_trans, T_trans_inv
                )
                loss = get_loss(l0_q, l0_t, l1_q, l1_t, l2_q, l2_t, l3_q, l3_t, q_gt, t_gt, w_x, w_q)
                loss.backward()
                optimizer.step()

                current_batch_size = len(pos2)
                batch_points = mean_points_in_batch(pos2)
                loss_value = float(loss.detach().item())
                iter_end = time.time()
                iter_time = iter_end - iter_start

                epoch_loss_sum += loss_value * current_batch_size
                epoch_seen += current_batch_size
                epoch_points_sum += batch_points * current_batch_size
                epoch_data_time_sum += data_time
                epoch_iter_time_sum += iter_time
                epoch_steps += 1

                if step % PROGRESS_UPDATE_INTERVAL == 0 or step == len(train_loader):
                    step_metrics = reduce_tensor(
                        torch.tensor(
                            [
                                loss_value * current_batch_size,
                                current_batch_size,
                            ],
                            device=args.device,
                            dtype=torch.float64,
                        )
                    )
                    global_step_loss = float(step_metrics[0].item() / max(step_metrics[1].item(), 1.0))
                    if is_main_process():
                        avg_loss = epoch_loss_sum / max(epoch_seen, 1.0)
                        current_mem_gb, peak_mem_gb = safe_gpu_memory_stats(args.device)
                        progress.set_postfix(
                            loss='{:.4f}'.format(loss_value),
                            avg='{:.4f}'.format(avg_loss),
                            lr='{:.2e}'.format(optimizer.param_groups[0]['lr']),
                            data='{:.2f}s'.format(data_time),
                            iter='{:.2f}s'.format(iter_time),
                            ips='{:.1f}'.format((current_batch_size * args.world_size) / max(iter_time, 1e-6)),
                            pts='{:.1f}k'.format(batch_points / 1000.0),
                            mem='{:.2f}/{:.2f}G'.format(current_mem_gb, peak_mem_gb),
                        )
                        log_scalar_group(
                            tb_writer,
                            'train',
                            {
                                'step_loss': global_step_loss,
                                'lr': optimizer.param_groups[0]['lr'],
                            },
                            train_global_step(epoch, step, len(train_loader)),
                        )
                previous_step_end = iter_end

            scheduler.step()
            lr = max(optimizer.param_groups[0]['lr'], args.learning_rate_clip)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            current_mem_gb, peak_mem_gb = safe_gpu_memory_stats(args.device)
            reduced_sums = reduce_tensor(
                torch.tensor(
                    [
                        epoch_loss_sum,
                        epoch_seen,
                        epoch_points_sum,
                        epoch_data_time_sum,
                        epoch_iter_time_sum,
                        epoch_steps,
                    ],
                    device=args.device,
                    dtype=torch.float64,
                )
            )
            reduced_max = reduce_tensor(
                torch.tensor(
                    [
                        time.time() - epoch_start,
                        current_mem_gb,
                        peak_mem_gb,
                    ],
                    device=args.device,
                    dtype=torch.float64,
                ),
                op=dist.ReduceOp.MAX,
            )

            train_loss = float(reduced_sums[0].item() / max(reduced_sums[1].item(), 1.0))
            avg_points = float(reduced_sums[2].item() / max(reduced_sums[1].item(), 1.0))
            avg_data_time_sec = float(reduced_sums[3].item() / max(reduced_sums[5].item(), 1.0))
            avg_iter_time_sec = float(reduced_sums[4].item() / max(reduced_sums[5].item(), 1.0))
            global_samples = int(reduced_sums[1].item())
            epoch_time_sec = float(reduced_max[0].item())
            gpu_mem_gb = float(reduced_max[1].item())
            gpu_peak_mem_gb = float(reduced_max[2].item())
            samples_per_sec = global_samples / max(epoch_time_sec, 1e-6)

            log_message(
                logger,
                'EPOCH {:03d}/{:03d} train loss: {:.6f} | lr: {:.6e} | time: {} | data: {:.2f}s | '
                'iter: {:.2f}s | ips: {:.2f} | pts: {:.1f}k | mem: {:.2f}/{:.2f}G'.format(
                    epoch,
                    total_epochs,
                    train_loss,
                    lr,
                    format_duration(epoch_time_sec),
                    avg_data_time_sec,
                    avg_iter_time_sec,
                    samples_per_sec,
                    avg_points / 1000.0,
                    gpu_mem_gb,
                    gpu_peak_mem_gb,
                ),
            )
            append_metric_record(
                metric_logs,
                timestamp=datetime.datetime.now().isoformat(timespec='seconds'),
                phase='train',
                epoch=epoch,
                total_epochs=total_epochs,
                loss=round(train_loss, 6),
                lr='{:.6e}'.format(lr),
                epoch_time_sec=round(epoch_time_sec, 3),
                avg_data_time_sec=round(avg_data_time_sec, 4),
                avg_iter_time_sec=round(avg_iter_time_sec, 4),
                samples_per_sec=round(samples_per_sec, 3),
                global_samples=global_samples,
                avg_points=round(avg_points, 1),
                gpu_mem_gb=round(gpu_mem_gb, 3),
                gpu_peak_mem_gb=round(gpu_peak_mem_gb, 3),
            )
            if is_main_process():
                w_x_value = float(unwrap_model(model).w_x.detach().item())
                w_q_value = float(unwrap_model(model).w_q.detach().item())
                log_scalar_group(
                    tb_writer,
                    'train',
                    {
                        'epoch_loss': train_loss,
                        'epoch_time_sec': epoch_time_sec,
                        'avg_data_time_sec': avg_data_time_sec,
                        'avg_iter_time_sec': avg_iter_time_sec,
                        'samples_per_sec': samples_per_sec,
                        'avg_points': avg_points,
                        'gpu_mem_gb': gpu_mem_gb,
                        'gpu_peak_mem_gb': gpu_peak_mem_gb,
                        'w_x': w_x_value,
                        'w_q': w_q_value,
                    },
                    epoch,
                )
                # Disable params/grads histogram logging in TensorBoard to keep the UI focused on
                # scalar metrics and Oxford route visualizations.
                # if should_log_histograms(epoch):
                #     log_model_histograms(tb_writer, unwrap_model(model).named_parameters(), epoch)

            if epoch % args.save_eval_interval == 0:
                barrier()
                if is_main_process():
                    model_name = unwrap_model(model).__class__.__name__
                    save_path = os.path.join(
                        checkpoints_dir,
                        '{}_{:03d}_{:04f}.pth.tar'.format(model_name, epoch, train_loss),
                    )
                    torch.save(
                        {
                            'model_state_dict': unwrap_model(model).state_dict(),
                            'opt_state_dict': optimizer.state_dict(),
                            'scheduler': scheduler.state_dict(),
                            'epoch': epoch,
                        },
                        save_path,
                    )
                    log_message(logger, 'Epoch {:03d}: saved checkpoint {}'.format(epoch, os.path.basename(save_path)))

                    if val_loader is not None:
                        log_message(logger, 'Epoch {:03d}: starting Oxford validation'.format(epoch))
                    else:
                        log_message(logger, 'Epoch {:03d}: starting KITTI evaluation'.format(epoch))
                        eval_pose(unwrap_model(model), args.kitti_val_seqs, epoch, log_dir, eval_dir, logger)
                        excel_eval.update(eval_dir)
                if val_loader is not None:
                    validate_oxford(
                        unwrap_model(model),
                        val_loader,
                        epoch,
                        total_epochs,
                        logger,
                        metric_logs=metric_logs,
                        lr='{:.6e}'.format(lr),
                        tb_writer=tb_writer,
                    )
                barrier()

            if should_run_oxford_detailed_val(args, epoch):
                barrier()
                if is_main_process():
                    log_message(logger, 'Epoch {:03d}: starting Oxford detailed validation'.format(epoch))
                    run_oxford_detailed_val(
                        unwrap_model(model),
                        args.device,
                        args,
                        eval_dir,
                        epoch,
                        log_fn=lambda message: log_message(logger, message),
                        show_progress=False,
                        tb_writer=tb_writer,
                    )
                barrier()
    finally:
        if tb_writer is not None:
            tb_writer.flush()
            tb_writer.close()


if __name__ == '__main__':
    try:
        main()
    finally:
        cleanup_runtime()
