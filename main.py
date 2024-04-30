import argparse
import os, json
import random
import shutil
import time, glob, copy
import os
import time
import torch
import socket
import argparse
import subprocess
import math
from tqdm import tqdm
import warnings
import numpy as np
import torch, pdb
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import matplotlib.pyplot as plt
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import argparse, pdb
import numpy as np
from torch import autograd
from torch.optim import Adam, SGD, AdamW
from typing import Union, List, Optional, Sequence, Dict, Iterator, Tuple, Callable
import pdb
from utils import  AverageMeter, ProgressMeter, init_ema_model, update_ema_model
#from dataloader import CIFAR10, GaussianBlur, TwoCropDataLoader, apply_affine
import builtins
import torchvision.utils as vutils
from PIL import Image
import torchvision
#import decoder_fit2
import tqdm
#from diffusion_extractor import StableDiffusion
#from ssim import MS_SSIM
#from encoder import MoCo
from utils import GuassianDiffusion, sample_N_images, ImageFolderLMDB, MIT_Dataset
from unet import UNet, UNetSmall

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

#from sklearn.metrics import average_precision_score
warnings.filterwarnings("ignore")

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('-j', '--workers', default=16, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--img_size', default=224, type=int,
                    help='img size')
parser.add_argument('--batch_iter', default=48, type=int,
                    help='img size')
parser.add_argument('--temp1', default=1.0, type=float,
                    help='parameter_1')
parser.add_argument('--temp2', default=1.0, type=float,
                    help='parameter_2')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--save_freq', default=5, type=int,
                     help='print frequency (default: 10)')
parser.add_argument('--resume', action = 'store_true',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--world-size', default=-1, type=int,
                        help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='env://', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--setting', default='0_0_0', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--local_rank', default=-1, type=int,
                    help='local rank for distributed training')
parser.add_argument('--local-rank', default=-1, type=int,
                    help='local rank for distributed training')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')

# ae params
parser.add_argument('--huggingface_token', type=str, default='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', help='huggingface API token to load stable diffusion')
parser.add_argument('--train_steps', type=int, default=1300, help='number of optimization steps')
parser.add_argument('--avg_steps', type=int, default=4, help='number of optimization steps')
# model args
parser.add_argument('--symmetric_matrix', type=int, default=0, choices=[0,1], help='1 symmetrical normalized randomwalk matrix,\
        0 for conventional attention matrix')
parser.add_argument('--vit_patch_size', type=int, default=8, choices=[8,16], help = 'the patch size in DINO VIT')
parser.add_argument('--vit_model_arch', type=str, default='vit_base', choices=['vit_base', 'vit_small'], help = 'the model arch in DINO VIT')
parser.add_argument('--vit_resize_img_size', type = int, default = 480, help = 'the smaller edge of input image for VIT')
# t sampling
parser.add_argument('--noise_schedule', type=str, default='random', choices=['random', 'constant', 'increasing', 'decreasing', 'cyclic'], help='noise schedule to use for training')
parser.add_argument('--noise_min_t', type=int, default=0, help='minimum t to use in diffusion model')
parser.add_argument('--noise_max_t', type=int, default=500, help='maximum t to use in diffusion model')
parser.add_argument('--noise_periods', type=float, default=1, help='periods for cyclic noise schedule')
parser.add_argument('--noise_sampling', action='store_true', default=True, help='if true, sample noise random uniformly from below maximum value defined by schedule')
# temp scheduling
parser.add_argument('--num_of_eig', type=int, default=5, help='number of eigenvector to compute')
parser.add_argument('--eig_loss_weight', type=float, default=2, help='weight ratio for primary objective')
parser.add_argument('--ortho_loss_weight', type=float, default=2, help='weight ratio for orthogonal regularization')
# attn buffer
parser.add_argument('--use_buffer_prob', type=float, default=None, help='chance to use buffer')
parser.add_argument('--attn_buffer_size', type=int, default=5, help='attn buffer size')
# optimizer param
parser.add_argument("--data_type", type=str)  # adam/lamb
parser.add_argument("--learning_rate", type=float, default=1e-3)
parser.add_argument("--batch_size", type=int, default=256)
parser.add_argument("--weight_decay", type=float, default=0)

# training params
parser.add_argument("--gpus", type=int, default=1)

# datamodule params
parser.add_argument("--data_path", type=str, default=".")
parser.add_argument("--dataset", type=str, default="cifar10")  # cifar10, stl10, imagenet

# transforms param

args = parser.parse_args()


best_acc1 = 0
EPS = 1e-20
def pca_feat(feat, pca_dim = 3):
    device = feat.device
    N,C,H,W = feat.shape
    feat = feat.reshape(N,C,-1).permute(0,2,1)
    pca_dim = min(min(pca_dim, feat.shape[1]), feat.shape[2])
    [u,s,v] = torch.pca_lowrank(feat, pca_dim, niter = 2)
    #v = multi_grid_low_rank(feat, pca_dim, niter = 1)
    feat = torch.matmul(feat, v)
    feat = feat.reshape(N,H,W,pca_dim).permute(0,3,1,2).contiguous()
    feat_min = feat.reshape(feat.shape[0], 3, -1).min(dim = -1)[0]
    feat_max = feat.reshape(feat.shape[0], 3, -1).max(dim = -1)[0]
    feat = 255 * (feat - feat_min.reshape(-1,3,1,1)) / (feat_max - feat_min).reshape(-1,3,1,1)
    return feat

def clip_gradients(model):
    norms = []
    max_grad = 0
    max_norm = 0
    for name, p in model.named_parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            max_grad = max(max_grad, p.grad.norm(2).item())
            max_norm = max(max_norm, p.data.norm(2).item())
    return max_grad, max_norm

def main():
    import os
    #torch.backends.cudnn.benchmark=False
    cudnn.deterministic = True
    args = parser.parse_args()
    #assert args.batch_size % args.batch_iter == 0
    if not os.path.exists('visualize'):
        os.system('mkdir visualize')
    if not os.path.exists('checkpoint'):
        os.system('mkdir checkpoint')
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if "WORLD_SIZE" in os.environ:
        args.world_size = int(os.environ["WORLD_SIZE"])
    args.distributed = args.world_size >= 1
    ngpus_per_node = torch.cuda.device_count()

    print('start')
    if args.distributed:
        if args.local_rank != -1: # for torch.distributed.launch
            args.rank = args.local_rank
            args.gpu = args.local_rank
        elif 'SLURM_PROCID' in os.environ: # for slurm scheduler
            args.rank = int(os.environ['SLURM_PROCID'])
            args.gpu = args.rank % torch.cuda.device_count()
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
    args.gpu = args.gpu % torch.cuda.device_count()
    print('world_size', args.world_size)
    print('rank', args.rank)
    # suppress printing if not on master gpu
    if args.rank!=0:
        def print_pass(*args):
            pass
        builtins.print = print_pass

    args.distributed = args.world_size >= 1 or args.multiprocessing_distributed
    main_worker(args.gpu, ngpus_per_node, args)

def gray_to_rgb(feat):
    if feat.shape[0] == 1:
        return feat.expand(3, -1, -1)
    else:
        return feat

def adjust_learning_rate(optimizer, epoch, warmup_epoch):
    """Decay the learning rate based on schedule"""
    #lr = args.learning_rate * 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
    if epoch < warmup_epoch:
        lr = args.learning_rate * (epoch + 1) * 1.0 / warmup_epoch
    else:
        lr = args.learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def groupnorm_32(channel):
    return nn.GroupNorm(32, channel)

def main_worker(gpu, ngpus_per_node, args):
    args = copy.deepcopy(args)
    args.cos = True
    save_folder_path = '''checkpoint/time_cifar10_dataset_lr_{}_batch_size_{}_weight_decay_{}_temp1_{}_temp2_{}'''.replace('\n',' ').replace(' ','').format(
                        args.learning_rate, args.batch_size, args.weight_decay, args.temp1, args.temp2)
    args.save_folder_path = save_folder_path
    args.is_master = args.rank == 0

    model = UNet(in_channels = 4, image_size = 128)
    model.cuda(args.gpu)
    ema_model = UNet(in_channels = 4, image_size = 128)
    ema_model.cuda(args.gpu)
    torch.cuda.set_device(args.gpu)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True,
    broadcast_buffers=False)
    ema_model = torch.nn.parallel.DistributedDataParallel(ema_model, device_ids=[args.gpu], find_unused_parameters=True,
    broadcast_buffers=False)
    init_ema_model(model, ema_model)
    optimizer = AdamW(model.parameters(),
                lr= args.learning_rate, weight_decay = args.weight_decay)
    args.start_epoch = 0
    if args.resume:
        args.resume = '{}/last.pth.{}_{}.tar'.format(save_folder_path, args.temp1, args.temp2)
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            ema_model.load_state_dict(checkpoint['ema_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
            del checkpoint
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    def transform_repeat_channel(data):
        return data.expand(3,-1,-1)

    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
                    mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]
                ),
    ])

    transform_test = transforms.Compose([
        torchvision.transforms.Resize(160),
        torchvision.transforms.CenterCrop((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(
                    mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]
                ),
    ])
    train_dataset = MIT_Dataset('/net/projects/willettlab/roxie62/dataset/mit_multiview', transform_train)
    train_dataset = ImageFolderLMDB('/net/projects/willettlab/roxie62/dataset/mit_multiview_depth/train', train_dataset)
    test_dataset = train_dataset
    train_dataset[0]
    print('NUM of training images: {}'.format(len(train_dataset)))
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle = True, drop_last = True)
        test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset, shuffle = False, drop_last = False)
    else:
        train_sampler = None
        test_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last = True, persistent_workers = True)

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, sampler=None, drop_last = False, persistent_workers = False)

    global_batch_size = (torch.distributed.get_world_size() * args.batch_size)
    train_iters_per_epoch = len(train_dataset) // global_batch_size

    if not args.multiprocessing_distributed or (args.multiprocessing_distributed
            and args.is_master):
        if not os.path.exists(save_folder_path):
            os.system('mkdir -p {}'.format(save_folder_path))

    for epoch in range(args.start_epoch, args.epochs + 2):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        input_scaler = None
        train_D(train_sampler, train_loader, model, optimizer, ema_model, epoch, args, ngpus_per_node)
        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                and args.is_master):
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer' : optimizer.state_dict(),
                'ema_state_dict': ema_model.state_dict(),
            }, False, filename = '{}/last.pth.{}_{}.tar'.format(save_folder_path, args.temp1, args.temp2))
            if epoch % 10 == 0 and epoch != 0:
                save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer' : optimizer.state_dict(),
                'ema_state_dict': ema_model.state_dict(),
                },False , filename = '{}/{}.pth.{}_{}.tar'.format(save_folder_path, epoch, args.temp1, args.temp2))

def print_gradients(model, print_str):
    norms = []
    max_grad = 0
    max_norm = 0
    for name, p in model.named_parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            max_grad = max(max_grad, p.grad.norm())
            max_norm = max(max_norm, p.data.norm(2))
    return max_grad, max_norm

def train_D(train_sampler, train_loader, model, optimizer, ema_model, epoch, args, ngpus_per_node):
    loss_name = [
                'loss', 'ent',
                'GPU Mem', 'Time', 'pe', 'ge']
    moco_loss_meter = [AverageMeter(name, ':6.3f') for name in loss_name]
    progress = ProgressMeter(
        len(train_loader),
        moco_loss_meter,
        prefix="Epoch: [{}]".format(epoch))

    def tensor_to_img(tensor):
        return ((tensor.permute(0,2,3,1) * 0.5 + 0.5) * 255).cpu().data.numpy().astype(np.uint8)

    # switch to train mode
    t0 = time.time()

    def pixel_loss(img1, img2):
        loss = ((img1 - img2) ** 2 + 1e-5).sqrt().mean()
        return loss


    diffusion = GuassianDiffusion(1000, args.gpu)
    epoch_iter = len(train_loader)
    for i, (img1, img2, depth) in enumerate(train_loader):
        img = img1.to(args.gpu)
        depth = depth.to(args.gpu)
        t = torch.randint(diffusion.timesteps, (img.shape[0],), dtype=torch.int64).to(
            args.gpu
        )
        xt, eps = diffusion.sample_from_forward_process(img, t)
        pred_img, _, _, ent = model(xt, depth, t.reshape(-1), return_all = True)
        loss = 10 * F.mse_loss(pred_img, img) + 1e-3 * ent
        optimizer.zero_grad()
        loss.backward()
        ge, pe = clip_gradients(model)
        optimizer.step()
        t1 = time.time()
        update_ema_model(model, ema_model, 0.999)
        #update_ema_model(generator, ema_generator, 0.99)
        for val_id, val in enumerate([loss, ent,
                    torch.cuda.max_memory_allocated() / (1024.0 * 1024.0), t1 - t0, pe, ge
                    ]):
            if not isinstance(val, float) and not isinstance(val, int):
                val = val.item()
            moco_loss_meter[val_id].update(val)
        progress.display(i)
        t0 = time.time()

        if i % 10 == 0:
            with torch.no_grad():
                xt, eps = diffusion.sample_from_forward_process(img, t * 0 + 11)
                pred_img, albedo, shading, ent = ema_model(xt, depth, t.reshape(-1), return_all = True)
                albedo= ((albedo[:25].clamp(-1,1) * 0.5 + 0.5).reshape(5,5, 3, 128, 128).permute(0,3, 1, 4, 2).reshape(640, 640, 3) * 255).cpu().data.numpy().astype(np.uint8)
                shading = ((shading[:25].expand(-1, 3, -1, -1)).reshape(5,5, 3, 128, 128).permute(0,3, 1, 4, 2).reshape(640, 640, 3) * 255).cpu().data.numpy().astype(np.uint8)
                np_img = ((img[:25].clamp(-1,1) * 0.5 + 0.5).reshape(5,5, 3, 128, 128).permute(0,3, 1, 4, 2).reshape(640, 640, 3) * 255).cpu().data.numpy().astype(np.uint8)
                pred_img = ((pred_img[:25].clamp(-1,1) * 0.5 + 0.5).reshape(5,5, 3, 128, 128).permute(0,3, 1, 4, 2).reshape(640, 640, 3) * 255).cpu().data.numpy().astype(np.uint8)
                whilte_space = (np.ones((640, 30, 3)).astype(np.float32) * 255).astype(np.uint8)
                if args.gpu == 0:
                    Image.fromarray(np.concatenate([np_img, whilte_space, pred_img, whilte_space, shading, whilte_space, albedo
                                            ], axis = 1)).save(args.save_folder_path + '/ema_{:05d}_{:07d}.png'.format(epoch, i))

                pred_img, albedo, shading, ent = model(xt, depth, t.reshape(-1), return_all = True)
                albedo= ((albedo[:25].clamp(-1,1) * 0.5 + 0.5).reshape(5,5, 3, 128, 128).permute(0,3, 1, 4, 2).reshape(640, 640, 3) * 255).cpu().data.numpy().astype(np.uint8)
                shading = ((shading[:25].expand(-1, 3, -1, -1)).reshape(5,5, 3, 128, 128).permute(0,3, 1, 4, 2).reshape(640, 640, 3) * 255).cpu().data.numpy().astype(np.uint8)
                np_img = ((img[:25].clamp(-1,1) * 0.5 + 0.5).reshape(5,5, 3, 128, 128).permute(0,3, 1, 4, 2).reshape(640, 640, 3) * 255).cpu().data.numpy().astype(np.uint8)
                pred_img = ((pred_img[:25].clamp(-1,1) * 0.5 + 0.5).reshape(5,5, 3, 128, 128).permute(0,3, 1, 4, 2).reshape(640, 640, 3) * 255).cpu().data.numpy().astype(np.uint8)
                whilte_space = (np.ones((640, 30, 3)).astype(np.float32) * 255).astype(np.uint8)
                if args.gpu == 0:
                    Image.fromarray(np.concatenate([np_img, whilte_space, pred_img, whilte_space, shading, whilte_space, albedo
                                            ], axis = 1)).save(args.save_folder_path + '/sgd_{:05d}_{:07d}.png'.format(epoch, i))
        if i % 500 == 0 and i != 0 and False:
            with torch.no_grad():
                z = model.module.fixed_z.to(args.gpu)[:64]
                sampled_images = sample_N_images(64,
                    model.module, diffusion, z, 100, 64, 3, 128,0,
                    device = args.gpu,
                )
                sampled_images_ema = sample_N_images(64,
                    ema_model.module, diffusion, z, 100, 64, 3, 128,0,
                    device = args.gpu,
                )
            if args.gpu == 0:
                sample_img= ((sampled_images[:25].clamp(-1,1) * 0.5 + 0.5).reshape(5,5, 3, 128, 128).permute(0,3, 1, 4, 2).reshape(640, 640, 3) * 255).cpu().data.numpy().astype(np.uint8)
                sample_ema_img= ((sampled_images_ema[:25].clamp(-1,1) * 0.5 + 0.5).reshape(5,5, 3, 128, 128).permute(0,3, 1, 4, 2).reshape(640, 640, 3) * 255).cpu().data.numpy().astype(np.uint8)
                np_img = ((img[:25].clamp(-1,1) * 0.5 + 0.5).reshape(5,5, 3, 128, 128).permute(0,3, 1, 4, 2).reshape(640, 640, 3) * 255).cpu().data.numpy().astype(np.uint8)
                whilte_space = (np.ones((640, 30, 3)).astype(np.float32) * 255).astype(np.uint8)
                Image.fromarray(np.concatenate([np_img, whilte_space, sample_img, whilte_space, sample_ema_img
                                        ], axis = 1)).save(args.save_folder_path + '/{:05d}_{:07d}.png'.format(epoch, i))

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')

if __name__ == '__main__':
    main()
