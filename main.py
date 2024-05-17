# python imports
import argparse
import os
import time
from pprint import pprint

import torch
import torch.nn as nn
import torch.utils.data
# for visualization
from torch.utils.tensorboard import SummaryWriter

# our code
from libs.core import load_config
from libs.datasets import make_dataset, make_data_loader
from libs.modeling import make_meta_arch
from libs.utils import (train_one_epoch, eval_one_epoch, 
                        save_checkpoint, make_optimizer, make_scheduler,
                        fix_random_seed, ModelEma, SegEval)


################################################################################
def main(args):
    """main function that handles training / inference"""
    """1. setup parameters / folders"""
    # parse args
    args.start_epoch = 0
    if os.path.isfile(args.config):
        cfg = load_config(args.config)
    else:
        raise ValueError("Config file does not exist.")
    pprint(cfg)

    # prep for output folder (based on time stamp)
    if not os.path.exists(cfg['output_folder']):
        os.mkdir(cfg['output_folder'])
    cfg_filename = os.path.basename(args.config).replace('.yaml', '')
    if len(args.output) == 0:
        ckpt_folder = os.path.join(
            cfg['output_folder'], cfg_filename) # debug的时候，没指定output，直接用文件名做目录
        # ts = datetime.datetime.fromtimestamp(int(time.time()))
        # ckpt_folder = os.path.join(
        #     cfg['output_folder'], cfg_filename + '_' + str(ts).replace(" ", "_"))
    else:
        ckpt_folder = os.path.join(
            cfg['output_folder'], cfg_filename + '_' + str(args.output))
    if not os.path.exists(ckpt_folder):
        os.mkdir(ckpt_folder)
    # tensorboard writer
    tb_writer = SummaryWriter(os.path.join(ckpt_folder, 'logs'))

    # fix the random seeds (this will fix everything)
    rng_generator = fix_random_seed(cfg['init_rand_seed'], include_cuda=True)

    # re-scale learning rate / # workers based on number of GPUs
    cfg['opt']["learning_rate"] *= len(cfg['devices'])
    cfg['loader']['num_workers'] *= len(cfg['devices'])

    """2. create dataset / dataloader"""
    train_dataset = make_dataset(
        cfg['dataset_name'], True, cfg['train_split'], **cfg['dataset']
    )
    val_dataset = make_dataset(
        cfg['dataset_name'], False, cfg['val_split'], **cfg['dataset']
    )
    # update cfg based on dataset attributes (fix to epic-kitchens)
    train_db_vars = train_dataset.get_attributes()
    cfg['model']['train_cfg']['head_empty_cls'] = train_db_vars['empty_label_ids']

    # data loaders
    train_loader = make_data_loader(
        # train_dataset, True, None, **cfg['loader'])
        train_dataset, True, rng_generator, **cfg['loader'])
    val_loader = make_data_loader(
        val_dataset, False, None, 1, cfg['loader']['num_workers']
    )

    """3. create model, optimizer, and scheduler"""
    # model
    model = make_meta_arch(cfg['model_name'], **cfg['model'])
    

    # not ideal for multi GPU training, ok for now
    model = nn.DataParallel(model, device_ids=cfg['devices'])
    # optimizer
    optimizer = make_optimizer(model, cfg['opt'])
    # schedule
    num_iters_per_epoch = len(train_loader)
    scheduler = make_scheduler(optimizer, cfg['opt'], num_iters_per_epoch)

    # enable model EMA
    print("Using model EMA ...")
    model_ema = ModelEma(model)

    """4. Resume from model / Misc"""
    # resume from a checkpoint?
    if args.resume:
        if os.path.isfile(args.resume):
            # load ckpt, reset epoch / best rmse
            checkpoint = torch.load(args.resume,
                map_location = lambda storage, loc: storage.cuda(
                    cfg['devices'][0]))
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            model_ema.module.load_state_dict(checkpoint['state_dict_ema'])
            # also load the optimizer / scheduler if necessary
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            print("=> loaded checkpoint '{:s}' (epoch {:d}".format(
                args.resume, checkpoint['epoch']
            ))
            del checkpoint
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
            return

    # save the current config
    with open(os.path.join(ckpt_folder, 'config.txt'), 'w') as fid:
        pprint(cfg, stream=fid)
        fid.flush()

    """4. training / validation loop"""
    print("\nStart training model {:s} ...".format(cfg['model_name']))

    # start training
    max_epochs = cfg['opt'].get(
        'early_stop_epochs',
        cfg['opt']['epochs'] + cfg['opt']['warmup_epochs']
    )

    # set up evaluator
    det_eval, output_file = None, None
    det_eval = SegEval(cfg['overlaps'])
    output_file = os.path.join(ckpt_folder, 'eval_results.pkl')
    log_file = os.path.join(ckpt_folder, 'log.txt')
    with open(log_file, 'w') as fid:
        fid.truncate(0)

    max_acc = 0
    max_acc_epoch = 0
    max_edit = 0
    max_edit_epoch = 0
    max_f1 = 0
    max_f1_epoch = 0
    for epoch in range(args.start_epoch, max_epochs):
        # train for one epoch
        train_one_epoch(
            train_loader,
            model,
            optimizer,
            scheduler,
            epoch,
            model_ema = model_ema,
            clip_grad_l2norm = cfg['train_cfg']['clip_grad_l2norm'],
            tb_writer=tb_writer,
            print_freq=args.print_freq,
        )

        # eval and save ckpt once in a while
        if (
            # ((epoch + 1) == max_epochs) or 
            (epoch + 1) >=1 and
            ((args.ckpt_freq > 0) and ((epoch + 1) % args.ckpt_freq == 0))
        ):  
            log_content = f"Epoch {epoch+1} done. Testing and Saving model ..."
            print(log_content)
            with open(log_file, 'a') as fid:
                fid.write(log_content + '\n')
                fid.flush()

            start = time.time()
            eval_results = eval_one_epoch(
                val_loader,
                model_ema.module,
                epoch,
                evaluator=det_eval,
                output_file=output_file,
                tb_writer=tb_writer
            )
            end = time.time()
            log_content = f"All done! Total time: {end - start:0.2f} sec"
            print(log_content)
            with open(log_file, 'a') as fid:
                fid.write(log_content + '\n')
                fid.flush()
            log_content = f"Acc:{eval_results['Acc']:1f}  Edit:{eval_results['Edit']:1f}  F1_0.1: {eval_results['F1_0.1']:1f}  F1_0.25: {eval_results['F1_0.25']:1f}  F1_0.5: {eval_results['F1_0.5']:1f}"
            print(log_content)
            with open(log_file, 'a') as fid:
                fid.write(log_content + '\n')
                fid.flush()

            # save checkpoint
            if eval_results['Acc'] > max_acc: 
                max_acc_epoch = epoch + 1
                max_acc = max(max_acc, eval_results['Acc'])
                log_content = f"max acc epoch{epoch+1}"
                print(log_content)
                with open(log_file, 'a') as fid:
                    fid.write(log_content + '\n')
                    fid.flush()
                save_states = {
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }
                save_states['state_dict_ema'] = model_ema.module.state_dict()
                save_checkpoint(
                    save_states,
                    False,
                    file_folder=ckpt_folder,
                    file_name=f'max_acc.pth.tar'
                )
            
            if eval_results['Edit'] > max_edit:
                max_edit_epoch = epoch + 1
                max_edit = max(max_edit, eval_results['Edit'])
                log_content = f"max edit epoch{epoch+1}"
                print(log_content)
                with open(log_file, 'a') as fid:
                    fid.write(log_content + '\n')
                    fid.flush()
                save_states = {
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }
                save_states['state_dict_ema'] = model_ema.module.state_dict()
                save_checkpoint(
                    save_states,
                    False,
                    file_folder=ckpt_folder,
                    file_name=f'max_edit.pth.tar'
                )

            if eval_results['F1_0.1'] > max_f1:
                max_f1_epoch = epoch + 1
                max_f1 = max(max_f1, eval_results['F1_0.1'])
                log_content = f"max f1 epoch{epoch+1} \n"
                print(log_content)
                with open(log_file, 'a') as fid:
                    fid.write(log_content + '\n')
                    fid.flush()
                save_states = {
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }
                save_states['state_dict_ema'] = model_ema.module.state_dict()
                save_checkpoint(
                    save_states,
                    False,
                    file_folder=ckpt_folder,
                    file_name=f'max_f1.pth.tar'
                )
    log_content = f"max acc epoch{max_acc_epoch}; max edit epoch{max_edit_epoch}; max f1 epoch{max_f1_epoch} \n"
    print(log_content)
    with open(log_file, 'a') as fid:   
        fid.write(log_content + '\n')
        fid.flush()
    # wrap up
    tb_writer.close()
    print("All done!")
    return

################################################################################
if __name__ == '__main__':
    """Entry Point"""
    # the arg parser
    parser = argparse.ArgumentParser(
      description='Train a transformer for action segmentation.')
    parser.add_argument('config', metavar='DIR',
                        help='path to a config file')
    parser.add_argument('-p', '--print-freq', default=5, type=int,
                        help='print frequency (default: 5 iterations)')
    parser.add_argument('-c', '--ckpt-freq', default=1, type=int,
                        help='checkpoint frequency (default: every 5 epochs)')
    parser.add_argument('--output', default='', type=str,
                        help='name of exp folder (default: none)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to a checkpoint (default: none)')
    args = parser.parse_args()
    main(args)
