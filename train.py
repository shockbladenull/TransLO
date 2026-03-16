# -*- coding:UTF-8 -*-

import datetime
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
from tqdm import tqdm

from configs import translonet_args
from dataset_factory import build_dataset
from kitti_pytorch import points_dataset
from tools.excel_tools import SaveExcel
from tools.euler_tools import quat2mat
from tools.logger_tools import creat_logger, log_print
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
        dist.barrier()


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
        dist.init_process_group(backend='nccl')
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
    checkpoints_dir = os.path.join(file_dir, 'checkpoints/translonet')

    if is_main_process():
        for directory in (experiment_dir, file_dir, eval_dir, log_dir, checkpoints_dir):
            os.makedirs(directory, exist_ok=True)
        for filename in SOURCE_BACKUP_FILES:
            shutil.copy2(os.path.join(base_dir, filename), os.path.join(log_dir, filename))

    barrier()
    return file_dir, eval_dir, log_dir, checkpoints_dir


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


def reduce_epoch_stats(loss_sum, total_seen):
    if is_distributed():
        dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_seen, op=dist.ReduceOp.SUM)
    return float((loss_sum / total_seen.clamp_min(1)).item())


def validate_oxford(model, val_loader, epoch, logger):
    total_loss = torch.zeros(1, device=args.device, dtype=torch.float64)
    total_trans_error = torch.zeros(1, device=args.device, dtype=torch.float64)
    total_rot_error = torch.zeros(1, device=args.device, dtype=torch.float64)
    total_seen = torch.zeros(1, device=args.device, dtype=torch.float64)

    model.eval()
    with torch.no_grad():
        for data in tqdm(val_loader, total=len(val_loader), smoothing=0.9):
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
            total_loss += loss.detach().to(torch.float64) * batch_size
            total_trans_error += trans_error.sum().to(torch.float64)
            total_rot_error += rot_error.sum().to(torch.float64)
            total_seen += batch_size

    if total_seen.item() == 0:
        raise RuntimeError('Oxford validation loader is empty')

    val_loss = float((total_loss / total_seen).item())
    val_trans = float((total_trans_error / total_seen).item())
    val_rot = float((total_rot_error / total_seen).item())
    log_message(
        logger,
        'EPOCH {} val mean loss: {:.6f}, translation error: {:.6f} m, rotation error: {:.6f} deg'.format(
            epoch, val_loss, val_trans, val_rot
        ),
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

        for _, data in tqdm(enumerate(test_loader), total=len(test_loader), smoothing=0.9):
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
    file_dir, eval_dir, log_dir, checkpoints_dir = prepare_output_dirs()
    logger = creat_logger(log_dir, args.model_name) if is_main_process() else None

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
        log_message(logger, 'distributed training world_size={}, local_rank={}'.format(args.world_size, args.local_rank))
    else:
        log_message(logger, 'single gpu is: {}'.format(args.device.index))

    train_dataset = build_dataset('train', args, is_training=1)
    train_sampler = DistributedSampler(train_dataset, shuffle=True) if is_distributed() else None
    train_loader = make_dataloader(train_dataset, args.batch_size, shuffle=True, sampler=train_sampler)
    log_message(logger, 'Train dataset: %s (%d samples)' % (args.train_dataset_type, len(train_dataset)))

    val_loader = None
    excel_eval = None
    if is_main_process():
        if args.val_dataset_type == 'oxford_qe':
            val_dataset = build_dataset('val', args, is_training=0)
            val_loader = make_dataloader(val_dataset, args.eval_batch_size, shuffle=False)
            log_message(logger, 'Validation dataset: %s (%d samples)' % (args.val_dataset_type, len(val_dataset)))
        else:
            excel_eval = SaveExcel(args.kitti_val_seqs, log_dir)
            log_message(logger, 'Validation dataset: %s (%s)' % (args.val_dataset_type, args.kitti_val_seqs))
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
        if is_main_process():
            if val_loader is not None:
                validate_oxford(unwrap_model(model), val_loader, init_epoch, logger)
            else:
                eval_pose(unwrap_model(model), args.kitti_val_seqs, init_epoch, log_dir, eval_dir, logger)
                excel_eval.update(eval_dir)
        barrier()

    for epoch in range(init_epoch + 1, args.max_epoch):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        model.train()
        total_loss = torch.zeros(1, device=args.device, dtype=torch.float64)
        total_seen = torch.zeros(1, device=args.device, dtype=torch.float64)

        for data in tqdm(train_loader, total=len(train_loader), smoothing=0.9, disable=not is_main_process()):
            pos2, pos1, sample_id, T_gt, T_trans, T_trans_inv, Tr = move_batch_to_device(data)
            optimizer.zero_grad()

            l0_q, l0_t, l1_q, l1_t, l2_q, l2_t, l3_q, l3_t, pc1_output, q_gt, t_gt, w_x, w_q = model(
                pos2, pos1, T_gt, T_trans, T_trans_inv
            )
            loss = get_loss(l0_q, l0_t, l1_q, l1_t, l2_q, l2_t, l3_q, l3_t, q_gt, t_gt, w_x, w_q)
            loss.backward()
            optimizer.step()

            current_batch_size = len(pos2)
            total_loss += loss.detach().to(torch.float64) * current_batch_size
            total_seen += current_batch_size

        scheduler.step()
        lr = max(optimizer.param_groups[0]['lr'], args.learning_rate_clip)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        train_loss = reduce_epoch_stats(total_loss, total_seen)
        log_message(logger, 'EPOCH {} train mean loss: {:04f}'.format(epoch, train_loss))

        if epoch % 5 == 0:
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
                log_message(logger, 'Save {}...'.format(model_name))

                if val_loader is not None:
                    validate_oxford(unwrap_model(model), val_loader, epoch, logger)
                else:
                    eval_pose(unwrap_model(model), args.kitti_val_seqs, epoch, log_dir, eval_dir, logger)
                    excel_eval.update(eval_dir)
            barrier()


if __name__ == '__main__':
    try:
        main()
    finally:
        cleanup_runtime()
