import builtins, pdb
import datetime
import os,glob
import time
from collections import defaultdict, deque
from pathlib import Path
import numpy as np
import torch, torchvision
import torch.distributed as dist
from torch import inf
import torch.utils.data as data
import math, lmdb
import matplotlib.pyplot as plt
from tqdm import tqdm
import pyarrow as pa
from easydict import EasyDict
unsqueeze3x = lambda x: x[..., None, None, None]
import torchvision, glob
import torch.utils.data as data
from PIL import ImageFilter, ImageOps
import numpy as np
import random, pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

class MIT_Dataset(data.Dataset):
    def __init__(self, root, img_transform):
        img_folder_list = glob.glob(root + '/*')
        img_folder_list.sort()
        self.img_meta = {'img_root':root, 'img_folder_list':[], 'img_folder_img_num':[], 'img_num_cumsum':[]}
        for img_folder_path in img_folder_list:
            img_list = glob.glob(img_folder_path + '/dir*.jpg')
            if len(img_list) == 0:
                continue
            self.img_meta['img_folder_list'].append(img_folder_path.split('/')[-1])
            self.img_meta['img_folder_img_num'].append(len(img_list))
        self.img_meta['img_num_cumsum'] = np.cumsum(np.array(self.img_meta['img_folder_img_num']))
        self.img_transform = img_transform

    def __len__(self):
        return self.img_meta['img_num_cumsum'][-1]

    def __getitem__(self, index):
        folder_index = np.searchsorted(self.img_meta['img_num_cumsum'], index, side = 'right')
        if folder_index == 0:
            folder_offset = index
        else:
            folder_offset = index - self.img_meta['img_num_cumsum'][folder_index - 1]
        img_list = glob.glob(self.img_meta['img_root'] + '/' + self.img_meta['img_folder_list'][folder_index] + '/dir*.jpg')
        img_list.sort()
        pair_img_folder_offset = np.random.choice(np.where(np.arange(len(img_list)) != folder_offset)[0])
        img1 = Image.open(img_list[folder_offset])
        img2 = Image.open(img_list[pair_img_folder_offset])
        return self.img_transform(img1), self.img_transform(img2)

class ImageFolderLMDB(data.Dataset):
    def __init__(self, db_path, img_dataset):
        self.img_dataset = img_dataset
        db_list = glob.glob(db_path + '/*-lock')
        db_list.sort()
        self.env = []
        self.keys = []
        self.keys_len = []
        for db_path in db_list:
            self.env.append(lmdb.open(db_path.split('-lock')[0], subdir=False,
                                 readonly=True, lock=False,
                                 readahead=False, meminit=False))
            with self.env[-1].begin() as txn:
                keys = list(txn.cursor().iternext(values=False))
                self.keys.append(keys)
                self.keys_len.append(len(keys))

    def data_from_index(self, index):
        cumsum = np.cumsum(np.array(self.keys_len)) - 1
        #cumsum = np.insert(cumsum, 0, 0)
        folder_index = np.searchsorted(cumsum, index)
        if folder_index > 0:
            index = index - cumsum[folder_index - 1] - 1
        with self.env[folder_index].begin(write=False) as txn:
            img_index = self.keys[folder_index][index]
            byteflow = txn.get(img_index)
        depth = pa.deserialize(byteflow)
        return int(img_index), depth

    def __getitem__(self, index):
        img_index, depth = self.data_from_index(index)
        img1, img2 = self.img_dataset[img_index]
        depth = F.interpolate(torch.from_numpy(depth)[None,None], size = (img1.shape[1], img1.shape[2]), mode = 'bilinear')
        img_size = 128
        img_crop1 = torchvision.transforms.CenterCrop(img_size)(torchvision.transforms.Resize(img_size)(img1))
        img_crop2 = torchvision.transforms.CenterCrop(img_size)(torchvision.transforms.Resize(img_size)(img2))
        depth_crop = torchvision.transforms.CenterCrop(img_size)(torchvision.transforms.Resize(img_size)(depth))
        return img_crop1, img_crop2, depth_crop[0]

    def __len__(self):
        return sum(self.keys_len)


@torch.no_grad()
def sample_N_images(
    N,
    model,
    diffusion,
    xT=None,
    sampling_steps=250,
    batch_size=64,
    num_channels=3,
    image_size=32,
    num_classes=None,
    device = 0
):
    samples, labels, num_samples = [], [], 0
    num_processes, group = dist.get_world_size(), dist.group.WORLD
    with tqdm(total=math.ceil(N / (batch_size * num_processes))) as pbar:
        while num_samples < N:
            if xT is None:
                xT = (
                    torch.randn(batch_size, num_channels, image_size, image_size)
                    .float()
                    .to(device)
                )
            y = None
            gen_images = diffusion.sample_from_reverse_process(
                model, xT, sampling_steps, {"y": y}, True
            )
            samples_list = [torch.zeros_like(gen_images) for _ in range(num_processes)]

            dist.all_gather(samples_list, gen_images, group)
            samples.append(torch.cat(samples_list).detach().cpu().numpy())
            num_samples += len(xT) * num_processes
            pbar.update(1)
    return torch.from_numpy(np.concatenate(samples))

class GuassianDiffusion:
    """Gaussian diffusion process with 1) Cosine schedule for beta values (https://arxiv.org/abs/2102.09672)
    2) L_simple training objective from https://arxiv.org/abs/2006.11239.
    """

    def __init__(self, timesteps=1000, device="cuda:0"):
        self.timesteps = timesteps
        self.device = device
        self.alpha_bar_scheduler = (
            lambda t: math.cos((t / self.timesteps + 0.008) / 1.008 * math.pi / 2) ** 2
        )
        self.scalars = self.get_all_scalars(
            self.alpha_bar_scheduler, self.timesteps, self.device
        )

        self.clamp_x0 = lambda x: x.clamp(-1, 1)
        self.get_x0_from_xt_eps = lambda xt, eps, t, scalars: (
            self.clamp_x0(
                1
                / unsqueeze3x(scalars.alpha_bar[t].sqrt())
                * (xt - unsqueeze3x((1 - scalars.alpha_bar[t]).sqrt()) * eps)
            )
        )
        self.get_pred_mean_from_x0_xt = (
            lambda xt, x0, t, scalars: unsqueeze3x(
                (scalars.alpha_bar[t].sqrt() * scalars.beta[t])
                / ((1 - scalars.alpha_bar[t]) * scalars.alpha[t].sqrt())
            )
            * x0
            + unsqueeze3x(
                (scalars.alpha[t] - scalars.alpha_bar[t])
                / ((1 - scalars.alpha_bar[t]) * scalars.alpha[t].sqrt())
            )
            * xt
        )

    def get_all_scalars(self, alpha_bar_scheduler, timesteps, device, betas=None):
        """
        Using alpha_bar_scheduler, get values of all scalars, such as beta, beta_hat, alpha, alpha_hat, etc.
        """
        all_scalars = {}
        if betas is None:
            all_scalars["beta"] = torch.from_numpy(
                np.array(
                    [
                        min(
                            1 - alpha_bar_scheduler(t + 1) / alpha_bar_scheduler(t),
                            0.999,
                        )
                        for t in range(timesteps)
                    ]
                )
            ).to(
                device
            )  # hardcoding beta_max to 0.999
        else:
            all_scalars["beta"] = betas
        all_scalars["beta_log"] = torch.log(all_scalars["beta"])
        all_scalars["alpha"] = 1 - all_scalars["beta"]
        all_scalars["alpha_bar"] = torch.cumprod(all_scalars["alpha"], dim=0)
        all_scalars["beta_tilde"] = (
            all_scalars["beta"][1:]
            * (1 - all_scalars["alpha_bar"][:-1])
            / (1 - all_scalars["alpha_bar"][1:])
        )
        all_scalars["beta_tilde"] = torch.cat(
            [all_scalars["beta_tilde"][0:1], all_scalars["beta_tilde"]]
        )
        all_scalars["beta_tilde_log"] = torch.log(all_scalars["beta_tilde"])
        return EasyDict(dict([(k, v.float()) for (k, v) in all_scalars.items()]))

    def sample_from_forward_process(self, x0, t):
        """Single step of the forward process, where we add noise in the image.
        Note that we will use this paritcular realization of noise vector (eps) in training.
        """
        eps = torch.randn_like(x0)
        xt = (
            unsqueeze3x(self.scalars.alpha_bar[t].sqrt()) * x0
            + unsqueeze3x((1 - self.scalars.alpha_bar[t]).sqrt()) * eps
        )
        return xt.float(), eps

    def sample_from_reverse_process(
        self, model, xT, timesteps=None, model_kwargs={}, ddim=False
    ):
        """Sampling images by iterating over all timesteps.

        model: diffusion model
        xT: Starting noise vector.
        timesteps: Number of sampling steps (can be smaller the default,
            i.e., timesteps in the diffusion process).
        model_kwargs: Additional kwargs for model (using it to feed class label for conditioning)
        ddim: Use ddim sampling (https://arxiv.org/abs/2010.02502). With very small number of
            sampling steps, use ddim sampling for better image quality.

        Return: An image tensor with identical shape as XT.
        """
        model.eval()
        final = xT

        # sub-sampling timesteps for faster sampling
        timesteps = timesteps or self.timesteps
        new_timesteps = np.linspace(
            0, self.timesteps - 1, num=timesteps, endpoint=True, dtype=int
        )
        alpha_bar = self.scalars["alpha_bar"][new_timesteps]
        new_betas = 1 - (
            alpha_bar / torch.nn.functional.pad(alpha_bar, [1, 0], value=1.0)[:-1]
        )
        scalars = self.get_all_scalars(
            self.alpha_bar_scheduler, timesteps, self.device, new_betas
        )

        for i, t in zip(np.arange(timesteps)[::-1], new_timesteps[::-1]):
            with torch.no_grad():
                current_t = torch.tensor([t] * len(final), device=final.device)
                current_sub_t = torch.tensor([i] * len(final), device=final.device)
                pred_x0 = model(final, current_t, **model_kwargs)
                pred_epsilon = (final - unsqueeze3x(self.scalars.alpha_bar[t].sqrt()) * pred_x0) / unsqueeze3x((1 - self.scalars.alpha_bar[t]).sqrt())
                #xt = (
                #    unsqueeze3x(self.scalars.alpha_bar[t].sqrt()) * x0
                #    + unsqueeze3x((1 - self.scalars.alpha_bar[t]).sqrt()) * eps
                #)
                # using xt+x0 to derive mu_t, instead of using xt+eps (former is more stable)
                #pred_x0 = self.get_x0_from_xt_eps(
                #    final, pred_epsilon, current_sub_t, scalars
                #)
                pred_mean = self.get_pred_mean_from_x0_xt(
                    final, pred_x0, current_sub_t, scalars
                )
                if i == 0:
                    final = pred_mean
                else:
                    if ddim:
                        final = (
                            unsqueeze3x(scalars["alpha_bar"][current_sub_t - 1]).sqrt()
                            * pred_x0
                            + (
                                1 - unsqueeze3x(scalars["alpha_bar"][current_sub_t - 1])
                            ).sqrt()
                            * pred_epsilon
                        )
                    else:
                        final = pred_mean + unsqueeze3x(
                            scalars.beta_tilde[current_sub_t].sqrt()
                        ) * torch.randn_like(final)
                final = final.detach()
        return final


def init_ema_model(model, ema_model):
    for param_q, param_k in zip(
        model.parameters(), ema_model.parameters()
    ):
        param_k.data.copy_(param_q.data)  # initialize
        param_k.requires_grad = False  # not update by gradient

    for param_q, param_k in zip(
        model.buffers(), ema_model.buffers()
    ):
        param_k.data.copy_(param_q.data)  # initialize

def update_ema_model(model, ema_model, m):
    for param_q, param_k in zip(
        model.parameters(), ema_model.parameters()):
        param_k.data = param_k.data * m + param_q.data * (1.0 - m)

def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)

def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    builtin_print = builtins.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        force = force or (get_world_size() > 8)
        if is_master or force:
            now = datetime.datetime.now().time()
            builtin_print('[{}] '.format(now), end='')  # print with time stamp
            builtin_print(*args, **kwargs)

    builtins.print = print

def init_distributed_mode(args):
    if args.dist_on_itp:
        args.rank = int(os.environ['OMPI_COMM_WORLD_RANK'])
        args.world_size = int(os.environ['OMPI_COMM_WORLD_SIZE'])
        args.gpu = int(os.environ['OMPI_COMM_WORLD_LOCAL_RANK'])
        args.dist_url = "tcp://%s:%s" % (os.environ['MASTER_ADDR'], os.environ['MASTER_PORT'])
        os.environ['LOCAL_RANK'] = str(args.gpu)
        os.environ['RANK'] = str(args.rank)
        os.environ['WORLD_SIZE'] = str(args.world_size)
        # ["RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT", "LOCAL_RANK"]
    elif 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        setup_for_distributed(is_master=True)  # hack
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}, gpu {}'.format(
        args.rank, args.dist_url, args.gpu), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)

class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if v is None:
                continue
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        log_msg = [
            header,
            '[{0' + space_fmt + '}/{1}]',
            'eta: {eta}',
            '{meters}',
            'time: {time}',
            'data: {data}'
        ]
        if torch.cuda.is_available():
            log_msg.append('max mem: {memory:.0f}')
        log_msg = self.delimiter.join(log_msg)
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.4f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()
        self.raw_val = []

    def reset(self):
        self.raw_val = []

    def update(self, val):
        self.val = val
        self.raw_val.append(val)

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        data = np.array(self.raw_val)
        return fmtstr.format(name = self.name, val = self.val, avg = data.mean())


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

import torch.nn.functional as F
from functools import partial
def compute_jac_spectral_norm(model, data, iter):
    def func_wrapper_for_vmap(data):
        data = data[None,...]
        out = model.module(data)
        return out[0]

    def compute_jvp(x, tangent):
        return torch.func.jvp(partial(func_wrapper_for_vmap), (x,), (tangent,))[1]
    data = data.detach()
    data.requires_grad = True
    out = model(data)
    with torch.no_grad():
        u = torch.randn_like(out)
        for _ in range(iter):
            v = torch.autograd.grad(out, data, u, create_graph = False, retain_graph = True)[0]
            v = F.normalize(v.flatten(1,-1), dim = -1).reshape(data.shape)
            u_raw = torch.func.vmap(compute_jvp)(data, v)
            u = F.normalize(u_raw, dim = -1)

    raw_v = torch.autograd.grad(out, data, u, allow_unused=False,
                           create_graph=True, retain_graph=True,
                           is_grads_batched=False)[0]
    sig_val = (raw_v * v).flatten(1,-1).sum(dim = -1)
    return sig_val
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset


class IndexToImageDataset(Dataset):
    """Wrap a dataset to map indices to images

    In other words, instead of producing (X, y) it produces (idx, X). The label
    y is not relevant for our task.
    """
    def __init__(self, base_dataset):
        self.base = base_dataset

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img, _ = self.base[idx]
        return (idx, img)


def gaussian(x, sigma=1.0):
    return np.exp(-(x**2) / (2*(sigma**2)))


def build_gauss_kernel(
        size=5, sigma=1.0, n_channels=1, device=None):
    """Construct the convolution kernel for a gaussian blur

    See https://en.wikipedia.org/wiki/Gaussian_blur for a definition.
    Overall I first generate a NxNx2 matrix of indices, and then use those to
    calculate the gaussian function on each element. The two dimensional
    Gaussian function is then the product along axis=2.
    Also, in_channels == out_channels == n_channels
    """
    if size % 2 != 1:
        raise ValueError("kernel size must be uneven")
    grid = np.mgrid[range(size), range(size)] - size//2
    kernel = np.prod(gaussian(grid, sigma), axis=0)
    # kernel = np.sum(gaussian(grid, sigma), axis=0)
    kernel /= np.sum(kernel)

    # repeat same kernel for all pictures and all channels
    # Also, conv weight should be (out_channels, in_channels/groups, h, w)
    kernel = np.tile(kernel, (n_channels, 1, 1, 1))
    kernel = torch.from_numpy(kernel).to(torch.float).to(device)
    return kernel


def blur_images(images, kernel):
    """Convolve the gaussian kernel with the given stack of images"""
    _, n_channels, _, _ = images.shape
    _, _, kw, kh = kernel.shape
    imgs_padded = F.pad(images, (kw//2, kh//2, kw//2, kh//2), mode='replicate')
    return F.conv2d(imgs_padded, kernel, groups=n_channels)


def laplacian_pyramid(images, kernel, max_levels=5):
    """Laplacian pyramid of each image

    https://en.wikipedia.org/wiki/Pyramid_(image_processing)#Laplacian_pyramid
    """
    current = images
    pyramid = []

    for level in range(max_levels):
        filtered = blur_images(current, kernel)
        diff = current - filtered
        pyramid.append(diff)
        current = F.avg_pool2d(filtered, 2)
    pyramid.append(current)
    return pyramid


class LapLoss(nn.Module):
    def __init__(self, max_levels=5, kernel_size=5, sigma=1.0):
        super(LapLoss, self).__init__()
        self.max_levels = max_levels
        self.kernel_size = kernel_size
        self.sigma = sigma
        self._gauss_kernel = None

    def forward(self, output, target):
        if (self._gauss_kernel is None
                or self._gauss_kernel.shape[1] != output.shape[1]):
            self._gauss_kernel = build_gauss_kernel(
                n_channels=output.shape[1],
                device=output.device)
        output_pyramid = laplacian_pyramid(
            output, self._gauss_kernel, max_levels=self.max_levels)
        target_pyramid = laplacian_pyramid(
            target, self._gauss_kernel, max_levels=self.max_levels)
        diff_levels = [F.l1_loss(o, t)
                        for o, t in zip(output_pyramid, target_pyramid)]
        loss = sum([2**(-2*j) * diff_levels[j]
                    for j in range(self.max_levels)])
        return loss
