# -*- coding:UTF-8 -*-

import datetime
import os
import sys
import time

import numpy as np
import torch
import torch.utils.data
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

"""CREATE DIR"""
base_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(base_dir)
experiment_dir = os.path.join(base_dir, 'experiment')
if not os.path.exists(experiment_dir):
    os.makedirs(experiment_dir)
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
if not os.path.exists(file_dir):
    os.makedirs(file_dir)
eval_dir = os.path.join(file_dir, 'eval')
if not os.path.exists(eval_dir):
    os.makedirs(eval_dir)
log_dir = os.path.join(file_dir, 'logs')
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
checkpoints_dir = os.path.join(file_dir, 'checkpoints/translonet')
if not os.path.exists(checkpoints_dir):
    os.makedirs(checkpoints_dir)

for filename in (
    'train.py',
    'configs.py',
    'dataset_factory.py',
    'translo_model.py',
    'translo_model_utils.py',
    'conv_util.py',
    'kitti_pytorch.py',
):
    os.system('cp %s %s' % (filename, log_dir))


def make_dataloader(dataset, batch_size, shuffle):
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=args.workers,
        collate_fn=collate_pair,
        pin_memory=True,
        worker_init_fn=lambda x: np.random.seed((torch.initial_seed()) % (2 ** 32)),
    )


def move_batch_to_cuda(data):
    pos2, pos1, sample_id, T_gt, T_trans, T_trans_inv, Tr = data
    pos2 = [batch.cuda() for batch in pos2]
    pos1 = [batch.cuda() for batch in pos1]
    T_trans = T_trans.cuda().to(torch.float32)
    T_trans_inv = T_trans_inv.cuda().to(torch.float32)
    T_gt = T_gt.cuda().to(torch.float32)
    return pos2, pos1, sample_id, T_gt, T_trans, T_trans_inv, Tr


def quaternion_angle_error_deg(pred_q, gt_q):
    pred_q = pred_q / (torch.norm(pred_q, dim=-1, keepdim=True) + 1e-10)
    gt_q = gt_q / (torch.norm(gt_q, dim=-1, keepdim=True) + 1e-10)
    dot = torch.sum(pred_q * gt_q, dim=-1).abs().clamp(max=1.0)
    return (2.0 * torch.acos(dot)) * (180.0 / np.pi)


def validate_oxford(model, val_loader, epoch, logger):
    total_loss = 0.0
    total_trans_error = 0.0
    total_rot_error = 0.0
    total_seen = 0

    model = model.eval()
    with torch.no_grad():
        for data in tqdm(val_loader, total=len(val_loader), smoothing=0.9):
            pos2, pos1, _, T_gt, T_trans, T_trans_inv, _ = move_batch_to_cuda(data)
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
            total_loss += loss.item() * batch_size
            total_trans_error += trans_error.sum().item()
            total_rot_error += rot_error.sum().item()
            total_seen += batch_size

    if total_seen == 0:
        raise RuntimeError('Oxford validation loader is empty')

    val_loss = total_loss / total_seen
    val_trans = total_trans_error / total_seen
    val_rot = total_rot_error / total_seen
    log_print(
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


def main():

    global args

    logger = creat_logger(log_dir, args.model_name)
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

    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

    model = translo_model(args, args.batch_size, args.H_input, args.W_input, args.is_training)

    train_dataset = build_dataset('train', args, is_training=1)
    train_loader = make_dataloader(train_dataset, args.batch_size, shuffle=True)
    logger.info('Train dataset: %s (%d samples)', args.train_dataset_type, len(train_dataset))

    val_loader = None
    excel_eval = None
    if args.val_dataset_type == 'oxford_qe':
        val_dataset = build_dataset('val', args, is_training=0)
        val_loader = make_dataloader(val_dataset, args.eval_batch_size, shuffle=False)
        logger.info('Validation dataset: %s (%d samples)', args.val_dataset_type, len(val_dataset))
    else:
        excel_eval = SaveExcel(args.kitti_val_seqs, log_dir)
        logger.info('Validation dataset: %s (%s)', args.val_dataset_type, args.kitti_val_seqs)

    logger.info('KITTI test sequences kept unchanged: %s', args.kitti_test_seqs)

    if args.multi_gpu is not None:
        device_ids = [int(x) for x in args.multi_gpu.split(',')]
        torch.backends.cudnn.benchmark = True
        model = torch.nn.DataParallel(model, device_ids=device_ids)
        model.cuda(device_ids[0])
        log_print(logger, 'multi gpu are:' + str(args.multi_gpu))
    else:
        torch.backends.cudnn.benchmark = True
        torch.cuda.set_device(args.gpu)
        model.cuda()
        log_print(logger, 'just one gpu is:' + str(args.gpu))

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
        checkpoint = torch.load(args.ckpt)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['opt_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        init_epoch = checkpoint['epoch']
        log_print(logger, 'load model {}'.format(args.ckpt))
    else:
        init_epoch = 0
        log_print(logger, 'Training from scratch')

    if args.eval_before == 1:
        if val_loader is not None:
            validate_oxford(model, val_loader, init_epoch, logger)
        else:
            eval_pose(model, args.kitti_val_seqs, init_epoch)
            excel_eval.update(eval_dir)

    for epoch in range(init_epoch + 1, args.max_epoch):
        total_loss = 0
        total_seen = 0
        optimizer.zero_grad()

        for i, data in tqdm(enumerate(train_loader, 0), total=len(train_loader), smoothing=0.9):
            torch.cuda.synchronize()
            pos2, pos1, sample_id, T_gt, T_trans, T_trans_inv, Tr = move_batch_to_cuda(data)
            torch.cuda.synchronize()
            model = model.train()
            torch.cuda.synchronize()
            l0_q, l0_t, l1_q, l1_t, l2_q, l2_t, l3_q, l3_t, pc1_ouput, q_gt, t_gt, w_x, w_q = model(
                pos2, pos1, T_gt, T_trans, T_trans_inv
            )
            loss = get_loss(l0_q, l0_t, l1_q, l1_t, l2_q, l2_t, l3_q, l3_t, q_gt, t_gt, w_x, w_q)
            torch.cuda.synchronize()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            torch.cuda.synchronize()

            current_batch_size = len(pos2)
            if args.multi_gpu is not None:
                total_loss += loss.mean().cpu().data * current_batch_size
            else:
                total_loss += loss.cpu().data * current_batch_size
            total_seen += current_batch_size

        scheduler.step()
        lr = max(optimizer.param_groups[0]['lr'], args.learning_rate_clip)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        train_loss = total_loss / total_seen
        log_print(logger, 'EPOCH {} train mean loss: {:04f}'.format(epoch, float(train_loss)))

        if epoch % 5 == 0:
            save_path = os.path.join(
                checkpoints_dir,
                '{}_{:03d}_{:04f}.pth.tar'.format(model.__class__.__name__, epoch, float(train_loss)),
            )
            torch.save(
                {
                    'model_state_dict': model.module.state_dict() if args.multi_gpu else model.state_dict(),
                    'opt_state_dict': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'epoch': epoch,
                },
                save_path,
            )
            log_print(logger, 'Save {}...'.format(model.__class__.__name__))

            if val_loader is not None:
                validate_oxford(model, val_loader, epoch, logger)
            else:
                eval_pose(model, args.kitti_val_seqs, epoch)
                excel_eval.update(eval_dir)


def eval_pose(model, test_list, epoch):
    for item in test_list:
        test_dataset = points_dataset(
            is_training=0,
            num_point=args.num_points,
            data_dir_list=[item],
            config=args,
        )
        test_loader = make_dataloader(test_dataset, args.eval_batch_size, shuffle=False)
        line = 0

        total_time = 0

        for batch_id, data in tqdm(enumerate(test_loader), total=len(test_loader), smoothing=0.9):
            torch.cuda.synchronize()
            pos2, pos1, sample_id, T_gt, T_trans, T_trans_inv, Tr = move_batch_to_cuda(data)
            torch.cuda.synchronize()

            model = model.eval()

            with torch.no_grad():
                torch.cuda.synchronize()
                start_time = time.time()

                l0_q, l0_t, l1_q, l1_t, l2_q, l2_t, l3_q, l3_t, pc1_ouput, q_gt, t_gt, w_x, w_q = model(
                    pos2, pos1, T_gt, T_trans, T_trans_inv
                )

                torch.cuda.synchronize()
                total_time += (time.time() - start_time)

                pc1_sample_2048 = pc1_ouput.cpu()
                l0_q = l0_q.cpu()
                l0_t = l0_t.cpu()
                pc1 = pc1_sample_2048.numpy()
                pred_q = l0_q.numpy()
                pred_t = l0_t.numpy()

                for n0 in range(pc1.shape[0]):

                    cur_Tr = Tr[n0, :, :]

                    qq = pred_q[n0:n0 + 1, :]
                    qq = qq.reshape(4)
                    tt = pred_t[n0:n0 + 1, :]
                    tt = tt.reshape(3, 1)
                    RR = quat2mat(qq)
                    filler = np.array([0.0, 0.0, 0.0, 1.0])
                    filler = np.expand_dims(filler, axis=0)

                    TT = np.concatenate([np.concatenate([RR, tt], axis=-1), filler], axis=0)

                    TT = np.matmul(cur_Tr, TT)
                    TT = np.matmul(TT, np.linalg.inv(cur_Tr))

                    if line == 0:
                        T_final = TT
                        T = T_final[:3, :]
                        T = T.reshape(1, 1, 12)
                        line += 1
                    else:
                        T_final = np.matmul(T_final, TT)
                        T_current = T_final[:3, :]
                        T_current = T_current.reshape(1, 1, 12)
                        T = np.append(T, T_current, axis=0)

        avg_time = total_time / max(len(test_dataset), 1)

        T = T.reshape(-1, 12)

        fname_txt = os.path.join(log_dir, str(item).zfill(2) + '_pred.npy')
        data_dir = os.path.join(eval_dir, 'translonet_' + str(item).zfill(2))
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)

        np.save(fname_txt, T)
        os.system('cp %s %s' % (fname_txt, data_dir))
        os.system(
            'python evaluation.py --result_dir '
            + data_dir
            + ' --eva_seqs '
            + str(item).zfill(2)
            + '_pred'
            + ' --epoch '
            + str(epoch)
        )
    return 0


if __name__ == '__main__':
    main()
