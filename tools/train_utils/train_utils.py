import glob
import os

import torch
import tqdm
from torch.nn.utils import clip_grad_norm_
from torch.cuda.amp import autocast, GradScaler
import time
import datetime
from pcdet.utils import common_utils
from pcdet.datasets import build_dataloader
from tools.val_utils import val_utils


def train_one_epoch(model, optimizer, train_loader, model_func, lr_scheduler, accumulated_iter, optim_cfg,
                    rank, tbar, total_it_each_epoch, dataloader_iter, tb_log=None, leave_pbar=False):
    if total_it_each_epoch == len(train_loader):
        dataloader_iter = iter(train_loader)
    if rank == 0:
        pbar = tqdm.tqdm(total=total_it_each_epoch, leave=leave_pbar, desc='train', dynamic_ncols=True)

    for cur_it in range(total_it_each_epoch):
        try:
            batch = next(dataloader_iter)
        except StopIteration:
            dataloader_iter = iter(train_loader)
            batch = next(dataloader_iter)

        lr_scheduler.step(accumulated_iter)

        try:
            cur_lr = float(optimizer.lr)
        except:
            cur_lr = optimizer.param_groups[0]['lr']
        if tb_log is not None:
            tb_log.add_scalar('meta_data/learning_rate', cur_lr, accumulated_iter)
        model.train()
        optimizer.zero_grad()

        ## add by shl
        # scaler = GradScaler()  # 创建一个尺度管理器
        #
        # # Train Step 1: Forward pass, get loss
        # with autocast():
        #     # """开启混合精度模式，只进行前向传播"""
        #     loss, tb_dict, disp_dict = model_func(model, batch)
        #
        # # Train Step 2: Backward pass, get gradient
        # scaler.scale(loss).backward()
        # # """使用尺度管理器进行调整"""
        # scaler.unscale_(optimizer)
        # clip_grad_norm_(model.parameters(), optim_cfg.GRAD_NORM_CLIP)
        #
        # # Train Step 3: Optimize params
        # scaler.step(optimizer)
        # scaler.update()


        loss, tb_dict, disp_dict = model_func(model, batch)
        loss.backward()
        clip_grad_norm_(model.parameters(), optim_cfg.GRAD_NORM_CLIP)
        optimizer.step()
        ## add end

        accumulated_iter += 1
        disp_dict.update({'loss': loss.item(), 'lr': cur_lr})

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


def train_model(model, optimizer, train_loader, model_func, lr_scheduler, optim_cfg,
                start_epoch, total_epochs, start_iter, rank, tb_log, ckpt_save_dir, train_sampler=None,
                lr_warmup_scheduler=None, ckpt_save_interval=1, max_ckpt_save_num=50,
                merge_all_iters_to_one_epoch=False, cfg=None, args_para=None, logger=None, test_loader=None):
    accumulated_iter = start_iter
    max_f2score = -1
    map_epoch_dict = {}
    max_ckpt_save_num = 5
    with tqdm.trange(start_epoch, total_epochs, desc='epochs', dynamic_ncols=True, leave=(rank == 0)) as tbar:
        total_it_each_epoch = len(train_loader)
        if merge_all_iters_to_one_epoch:
            assert hasattr(train_loader.dataset, 'merge_all_iters_to_one_epoch')
            train_loader.dataset.merge_all_iters_to_one_epoch(merge=True, epochs=total_epochs)
            total_it_each_epoch = len(train_loader) // max(total_epochs, 1)
        dataloader_iter = iter(train_loader)
        for cur_epoch in tbar:
            if train_sampler is not None:
                train_sampler.set_epoch(cur_epoch)

            if lr_warmup_scheduler is not None and cur_epoch < optim_cfg.WARMUP_EPOCH:
                cur_scheduler = lr_warmup_scheduler
            else:
                cur_scheduler = lr_scheduler
            accumulated_iter = train_one_epoch(
                model, optimizer, train_loader, model_func,
                lr_scheduler=cur_scheduler,
                accumulated_iter=accumulated_iter, optim_cfg=optim_cfg,
                rank=rank, tbar=tbar, tb_log=tb_log,
                leave_pbar=(cur_epoch + 1 == total_epochs),
                total_it_each_epoch=total_it_each_epoch,
                dataloader_iter=dataloader_iter
            )

            # add by shl
            # evaluate model in val dataset before saving.
            dist_train = False
            output_dir = cfg.ROOT_DIR / 'output' / cfg.EXP_GROUP_PATH / cfg.TAG / 'val'
            output_dir.mkdir(parents=True, exist_ok=True)
            logger.info('**********************VAL! Start evaluation %s/%s**********************' %
                        (cfg.EXP_GROUP_PATH, cfg.TAG))

            eval_output_dir = output_dir / 'eval' / 'eval_with_train'
            eval_output_dir.mkdir(parents=True, exist_ok=True)
            # start evaluation
            model.val = True
            model.eval()
            ret_dict, f2score = val_utils.val_one_epoch(
                cfg, model, test_loader, cur_epoch + 1, logger, dist_test=dist_train,
                result_dir=eval_output_dir, save_to_file=args_para.save_to_file, val=True, tb_log=tb_log, rank=rank,
                accumulated_iter=accumulated_iter - 1
            )
            if rank == 0:
                print('The f2-score of model epoch%d is %f' % (cur_epoch + 1, f2score))
            logger.info('**********************End evaluation %s/%s(%s)**********************' %
                        (cfg.EXP_GROUP_PATH, cfg.TAG, args_para.extra_tag))
            # end add

            # add by shl
            # save trained model
            trained_epoch = cur_epoch + 1
            if trained_epoch % ckpt_save_interval == 0 and rank == 0:
                ckpt_list = glob.glob(str(ckpt_save_dir / 'checkpoint_epoch_*.pth'))
                ckpt_name = ckpt_save_dir / ('checkpoint_epoch_%d' % trained_epoch)
                if trained_epoch == 80:
                    print('+' * 100)
                    print('trained_epoch', trained_epoch)
                    save_checkpoint(
                        checkpoint_state(model, optimizer, trained_epoch, accumulated_iter), filename=ckpt_name,
                    )
                    print('#' * 200)
                    print(map_epoch_dict)
                if trained_epoch == total_epochs:
                    print('+' * 100)
                    print('trained_epoch', trained_epoch)
                    save_checkpoint(
                        checkpoint_state(model, optimizer, trained_epoch, accumulated_iter), filename=ckpt_name,
                    )
                    map_epoch_dict['checkpoint_epoch_%d.pth' % trained_epoch] = f2score
                    map_epoch_dict = dict(sorted(map_epoch_dict.items(), key=lambda kv:(kv[1], kv[0])))
                    print('#' * 200)
                    print(map_epoch_dict)
                elif ckpt_list.__len__() < max_ckpt_save_num:
                    save_checkpoint(
                        checkpoint_state(model, optimizer, trained_epoch, accumulated_iter), filename=ckpt_name,
                    )
                    map_epoch_dict['checkpoint_epoch_%d.pth' % trained_epoch] = f2score
                    map_epoch_dict = dict(sorted(map_epoch_dict.items(), key=lambda kv:(kv[1], kv[0])))
                    print('#' * 200)
                    print(map_epoch_dict)
                else:
                    if f2score > map_epoch_dict[list(map_epoch_dict.keys())[0]]:
                        print('$' * 200)
                        print(f2score, map_epoch_dict)
                        os.remove(ckpt_save_dir / list(map_epoch_dict.keys())[0])
                        map_epoch_dict.pop(list(map_epoch_dict.keys())[0])
                        map_epoch_dict['checkpoint_epoch_%d.pth' % trained_epoch] = f2score
                        map_epoch_dict = dict(sorted(map_epoch_dict.items(), key=lambda kv: (kv[1], kv[0])))
                        save_checkpoint(
                            checkpoint_state(model, optimizer, trained_epoch, accumulated_iter), filename=ckpt_name,
                        )
                        print(map_epoch_dict)
                    else:
                        print('Small f2score, map_epoch_dict=', map_epoch_dict)
                final_map = map_epoch_dict.get(list(map_epoch_dict.keys())[-1], 0)
                if tb_log is not None:
                    tb_log.add_scalar('f2score', final_map, trained_epoch)
            model.val = False
            model.train()

            # end add


            # # # save trained model
            # trained_epoch = cur_epoch + 1
            # if trained_epoch % ckpt_save_interval == 0 and rank == 0:
            #     ckpt_list = glob.glob(str(ckpt_save_dir / 'checkpoint_epoch_*.pth'))
            #     ckpt_list.sort(key=os.path.getmtime)
            #     if ckpt_list.__len__() >= max_ckpt_save_num:
            #         for cur_file_idx in range(0, len(ckpt_list) - max_ckpt_save_num + 1):
            #             os.remove(ckpt_list[cur_file_idx])
            #
            #     ckpt_name = ckpt_save_dir / ('checkpoint_epoch_%d' % trained_epoch)
            #     save_checkpoint(
            #         checkpoint_state(model, optimizer, trained_epoch, accumulated_iter), filename=ckpt_name,
            #     )

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
    torch.save(state, filename, _use_new_zipfile_serialization=False)
