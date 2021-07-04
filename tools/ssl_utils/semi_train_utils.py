import glob
import os
import torch
import tqdm
from torch.nn.utils import clip_grad_norm_
from .sess import sess
from .pseudo_label import pseudo_label
from .iou_match_3d import iou_match_3d
from .se_ssd import se_ssd

semi_learning_methods = {
    'SESS': sess,
    'Pseudo-Label': pseudo_label,
    '3DIoUMatch': iou_match_3d,
    'SE_SSD': se_ssd,
}

def train_ssl_one_epoch(teacher_model, student_model, optimizer, labeled_loader, unlabeled_loader, epoch_id, lr_scheduler, accumulated_iter, ssl_cfg,
                        rank, tbar, total_it_each_epoch, labeled_loader_iter, unlabeled_loader_iter, tb_log=None, leave_pbar=False, dist=False):

    if rank == 0:
        pbar = tqdm.tqdm(total=total_it_each_epoch, leave=leave_pbar, desc='train', dynamic_ncols=True)

    for cur_it in range(total_it_each_epoch):
        try:
            ud_teacher_batch_dict, ud_student_batch_dict = next(unlabeled_loader_iter)
        except StopIteration:
            unlabeled_loader_iter = iter(unlabeled_loader)
            ud_teacher_batch_dict, ud_student_batch_dict = next(unlabeled_loader_iter)

        try:
            ld_teacher_batch_dict, ld_student_batch_dict = next(labeled_loader_iter)
        except StopIteration:
            labeled_loader_iter = iter(labeled_loader)
            ld_teacher_batch_dict, ld_student_batch_dict = next(labeled_loader_iter)

        #lr_scheduler.step(accumulated_iter)

        try:
            cur_lr = float(optimizer.lr)
        except:
            cur_lr = optimizer.param_groups[0]['lr']

        if tb_log is not None:
            tb_log.add_scalar('meta_data/learning_rate', cur_lr, accumulated_iter)

        optimizer.zero_grad()


        loss, tb_dict, disp_dict = semi_learning_methods[ssl_cfg.NAME](
            teacher_model, student_model,
            ld_teacher_batch_dict, ld_student_batch_dict,
            ud_teacher_batch_dict, ud_student_batch_dict,
            ssl_cfg, epoch_id, dist
        )
        loss.backward()

        clip_grad_norm_(student_model.parameters(), ssl_cfg.STUDENT.GRAD_NORM_CLIP)
        optimizer.step()
        lr_scheduler.step(accumulated_iter)

        accumulated_iter += 1
        disp_dict.update({'loss': loss.item(), 'lr': cur_lr})

        # EMA Teacher
        if ssl_cfg.TEACHER.NUM_ITERS_PER_UPDATE != -1:
            ema_rampup_start, ema_start = ssl_cfg.TEACHER.EMA_EPOCH
            assert ema_rampup_start <= ema_start
            if epoch_id < ema_rampup_start:
                pass
            elif (epoch_id >= ema_rampup_start) and (epoch_id < ema_start):
                if accumulated_iter % ssl_cfg.TEACHER.NUM_ITERS_PER_UPDATE == 0:
                    if dist:
                        #if rank == 0:
                        update_ema_variables(student_model.module.onepass, teacher_model.module.onepass, ssl_cfg.TEACHER.RAMPUP_EMA_MOMENTUM, accumulated_iter)
                    else:
                        update_ema_variables(student_model, teacher_model, ssl_cfg.TEACHER.RAMPUP_EMA_MOMENTUM, accumulated_iter)
            elif epoch_id >= ema_start:
                if accumulated_iter % ssl_cfg.TEACHER.NUM_ITERS_PER_UPDATE == 0:
                    if dist:
                        #if rank == 0:
                        update_ema_variables_with_fixed_momentum(student_model.module.onepass, teacher_model.module.onepass, ssl_cfg.TEACHER.EMA_MOMENTUM)
                    else:
                        update_ema_variables_with_fixed_momentum(student_model, teacher_model, ssl_cfg.TEACHER.EMA_MOMENTUM)
            else:
                raise Exception('Impossible condition for EMA update')

        # log to console and tensorboard
        if rank == 0:
            pbar.update()
            pbar.set_postfix(dict(total_it=accumulated_iter))
            tbar.set_postfix(disp_dict)
            tbar.refresh()

            if tb_log is not None:
                tb_log.add_scalar('train/loss', loss, accumulated_iter)
                tb_log.add_scalar('meta_data/learning_rate', cur_lr, accumulated_iter)
                for key, val in tb_dict.items():
                    tb_log.add_scalar('train/' + key, val, accumulated_iter)
    if rank == 0:
        pbar.close()
    return accumulated_iter

def train_ssl_model(teacher_model, student_model, student_optimizer, labeled_loader, unlabeled_loader,
                    lr_scheduler, ssl_cfg,
                    start_epoch, total_epochs, start_iter, rank, tb_log, ckpt_save_dir,
                    labeled_sampler, unlabeled_sampler,
                    lr_warmup_scheduler=None, ckpt_save_interval=1, max_ckpt_save_num=50,
                    merge_all_iters_to_one_epoch=False, dist=False):
    accumulated_iter = start_iter
    with tqdm.trange(start_epoch, total_epochs, desc='epochs', dynamic_ncols=True, leave=(rank == 0)) as tbar:
        total_it_each_epoch = len(labeled_loader) # total iterations set to labeled set
        assert merge_all_iters_to_one_epoch is False
        labeled_loader_iter = iter(labeled_loader)
        unlabeled_loader_iter = iter(unlabeled_loader)

        for cur_epoch in tbar:
            if labeled_sampler is not None:
                labeled_sampler.set_epoch(cur_epoch)
            if unlabeled_sampler is not None:
                unlabeled_sampler.set_epoch(cur_epoch)
            # train one epoch
            if lr_warmup_scheduler is not None and cur_epoch < ssl_cfg.STUDENT.WARMUP_EPOCH:
                cur_scheduler = lr_warmup_scheduler
            else:
                cur_scheduler = lr_scheduler
            accumulated_iter = train_ssl_one_epoch(
                teacher_model = teacher_model,
                student_model = student_model,
                optimizer = student_optimizer,
                labeled_loader = labeled_loader,
                unlabeled_loader = unlabeled_loader,
                epoch_id = cur_epoch,
                lr_scheduler=cur_scheduler,
                accumulated_iter=accumulated_iter, ssl_cfg=ssl_cfg,
                rank=rank, tbar=tbar, tb_log=tb_log,
                leave_pbar=(cur_epoch + 1 == total_epochs),
                total_it_each_epoch=total_it_each_epoch,
                labeled_loader_iter=labeled_loader_iter,
                unlabeled_loader_iter=unlabeled_loader_iter,
                dist = dist
            )

            # save trained model
            trained_epoch = cur_epoch + 1
            if trained_epoch % ckpt_save_interval == 0 and rank == 0:

                student_ckpt_name = ckpt_save_dir / 'student' / ('checkpoint_epoch_%d' % trained_epoch)

                if dist:
                    save_checkpoint(
                        checkpoint_state(student_model.module.onepass, student_optimizer, trained_epoch, accumulated_iter), filename=student_ckpt_name,
                    )
                else:
                    save_checkpoint(
                        checkpoint_state(student_model, student_optimizer, trained_epoch, accumulated_iter), filename=student_ckpt_name,
                    )

                teacher_ckpt_name = ckpt_save_dir / 'teacher'/ ('checkpoint_epoch_%d' % trained_epoch)

                if dist:
                    save_checkpoint(
                        checkpoint_state(teacher_model.module.onepass, student_optimizer, trained_epoch, accumulated_iter), filename=teacher_ckpt_name,
                    )
                else:
                    save_checkpoint(
                        checkpoint_state(teacher_model, student_optimizer, trained_epoch, accumulated_iter), filename=teacher_ckpt_name,
                    )

def model_state_to_cpu(model_state):
    model_state_cpu = type(model_state)()  # ordered dict
    for key, val in model_state.items():
        model_state_cpu[key] = val.cpu()
    return model_state_cpu

def checkpoint_state(model=None, optimizer=None, epoch=None, it=None):
    optim_state = optimizer.state_dict() if optimizer is not None else None
    if model is not None:
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model_state = model_state_to_cpu(model.module.state_dict())
        else:
            model_state = model.state_dict()
    else:
        model_state = None

    try:
        import pcdet
        version = 'pcdet+' + pcdet.__version__
    except:
        version = 'none'

    return {'epoch': epoch, 'it': it, 'model_state': model_state, 'optimizer_state': optim_state, 'version': version}

def save_checkpoint(state, filename='checkpoint'):
    if False and 'optimizer_state' in state:
        optimizer_state = state['optimizer_state']
        state.pop('optimizer_state', None)
        optimizer_filename = '{}_optim.pth'.format(filename)
        torch.save({'optimizer_state': optimizer_state}, optimizer_filename)

    filename = '{}.pth'.format(filename)
    torch.save(state, filename)

def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 2), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)
        """
        if param.requires_grad:
            ema_param.data.mul_(alpha).add_(1 - alpha, param.data)
        else:
            ema_param.data.mul_(0).add_(1, param.data)
        """


def update_ema_variables_with_fixed_momentum(model, ema_model, alpha):
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)
        """
        if param.requires_grad:
            ema_param.data.mul_(alpha).add_(1 - alpha, param.data)
        else:
            ema_param.data.mul_(0).add_(1, param.data)
        """