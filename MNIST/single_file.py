import os
import time
import json
import argparse
from datetime import datetime
import torch.nn as nn
import torch.backends.cudnn as cudnn
from pathlib import Path
from functools import partial
import pandas as pd

import utils
from spiking_neuron.TCLIF import TCLIFNode
from load_dataset import load_dataset
from models.fc import ffMnist, fbMnist
from utils import *

from abc import abstractmethod
from typing import Callable
import torch
from spiking_neuron import base


class BaseNode(base.MemoryModule):
    def __init__(self,
                 v_threshold: float = 1.,
                 v_reset: float = 0.,
                 surrogate_function: Callable = None,
                 detach_reset: bool = False,
                 step_mode='s', backend='torch',
                 store_v_seq: bool = False):

        assert isinstance(v_reset, float) or v_reset is None
        assert isinstance(v_threshold, float)
        assert isinstance(detach_reset, bool)
        super().__init__()

        if v_reset is None:
            self.register_memory('v', 0.)
        else:
            self.register_memory('v', v_reset)

        self.v_threshold = v_threshold

        self.v_reset = v_reset
        self.detach_reset = detach_reset
        self.surrogate_function = surrogate_function

        self.step_mode = step_mode
        self.backend = backend

        self.store_v_seq = store_v_seq

    @property
    def store_v_seq(self):
        return self._store_v_seq

    @store_v_seq.setter
    def store_v_seq(self, value: bool):
        self._store_v_seq = value
        if value:
            if not hasattr(self, 'v_seq'):
                self.register_memory('v_seq', None)

    @staticmethod
    @torch.jit.script
    def jit_hard_reset(v: torch.Tensor, spike: torch.Tensor, v_reset: float):
        v = (1. - spike) * v + spike * v_reset

        return v

    @staticmethod
    @torch.jit.script
    def jit_soft_reset(v: torch.Tensor, spike: torch.Tensor, v_threshold: float):
        v = v - spike * v_threshold
        return v

    @abstractmethod
    def neuronal_charge(self, x: torch.Tensor):
        raise NotImplementedError

    def neuronal_fire(self):
        return self.surrogate_function(self.v - self.v_threshold)

    def extra_repr(self):
        return f'v_threshold={self.v_threshold}, v_reset={self.v_reset}, detach_reset={self.detach_reset}, step_mode={self.step_mode}, backend={self.backend}'

    def single_step_forward(self, x: torch.Tensor):
        self.v_float_to_tensor(x)
        self.neuronal_charge(x)
        spike = self.neuronal_fire()
        self.neuronal_reset(spike)
        return spike

    def multi_step_forward(self, x_seq: torch.Tensor):
        T = x_seq.shape[0]
        y_seq = []
        if self.store_v_seq:
            v_seq = []
        for t in range(T):
            y = self.single_step_forward(x_seq[t])
            y_seq.append(y)
            if self.store_v_seq:
                v_seq.append(self.v)

        if self.store_v_seq:
            self.v_seq = torch.stack(v_seq)

        return torch.stack(y_seq)

    def v_float_to_tensor(self, x: torch.Tensor):
        if isinstance(self.v, float):
            v_init = self.v
            self.v = torch.full_like(x.data, v_init)


class TCLIFNode(BaseNode):
    def __init__(self,
                 v_threshold=1.,
                 v_reset=0.,
                 surrogate_function: Callable = None,
                 detach_reset=False,
                 hard_reset=False,
                 step_mode='s',
                 k=2,
                 decay_factor: torch.Tensor = torch.full([1, 2], 0, dtype=torch.float),
                 gamma: float = 0.5):
        super(TCLIFNode, self).__init__(v_threshold, v_reset, surrogate_function, detach_reset, step_mode)
        self.k = k
        for i in range(1, self.k + 1):
            self.register_memory('v' + str(i), 0.)

        self.names = self._memories
        self.hard_reset = hard_reset
        self.gamma = gamma
        self.decay = decay_factor
        self.decay_factor = torch.nn.Parameter(decay_factor)

    @property
    def supported_backends(self):
        if self.step_mode == 's':
            return ('torch',)
        elif self.step_mode == 'm':
            return ('torch', 'cupy')
        else:
            raise ValueError(self.step_mode)

    def neuronal_charge(self, x: torch.Tensor):
        # v1: membrane potential of dendritic compartment
        # v2: membrane potential of somatic compartment
        self.names['v1'] = self.names['v1'] - torch.sigmoid(self.decay_factor[0][0]) * self.names['v2'] + x
        self.names['v2'] = self.names['v2'] + torch.sigmoid(self.decay_factor[0][1]) * self.names['v1']
        self.v = self.names['v2']

    def neuronal_reset(self, spike):
        if self.detach_reset:
            spike_d = spike.detach()
        else:
            spike_d = spike

        if not self.hard_reset:
            # soft reset
            self.names['v1'] = self.jit_soft_reset(self.names['v1'], spike_d, self.gamma)
            self.names['v2'] = self.jit_soft_reset(self.names['v2'], spike_d, self.v_threshold)
        else:
            # hard reset
            for i in range(2, self.k + 1):
                self.names['v' + str(i)] = self.jit_hard_reset(self.names['v' + str(i)], spike_d, self.v_reset)

    def forward(self, x: torch.Tensor):
        return super().single_step_forward(x)

    def extra_repr(self):
        return f"v_threshold={self.v_threshold}, v_reset={self.v_reset}, detach_reset={self.detach_reset}, " \
               f"hard_reset={self.hard_reset}, " \
               f"gamma={self.gamma}, k={self.k}, step_mode={self.step_mode}, backend={self.backend}"


def train(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    model.train()

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        images = images.cuda(non_blocking=True)  # images:[bs, 1, 28, 28]
        target = target.cuda(non_blocking=True)
        # print(type(target[0][0]))

        input_im = images.view(-1, args.time_window, 1)  # input_im:[bs, 784, 1]

        if args.task == 'PSMNIST':
            input_im = input_im[:, perm, :]

        reset_states(model=model)
        output = model(input_im)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)
    logging.info(
        'Train Epoch: [{}/{}], lr: {:.6f}, top1: {:.4f}'.format(epoch, args.epochs, optimizer.param_groups[0]['lr'],
                                                                top1.avg))
    return top1.avg, losses.avg


def validate(val_loader, model, criterion, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            images = images.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

            # img.shape = [N, 1, H, W]

            input_im = images.view(-1, args.time_window, 1)  # input_im:[bs, 784, 1]

            if args.task == 'PSMNIST':
                input_im = input_im[:, perm, :]

            reset_states(model=model)
            output = model(input_im)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

    return top1.avg, top5.avg


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


parser = argparse.ArgumentParser(description='Sequential MNIST/PMNIST')
parser.add_argument('--task', default='SMNIST', type=str, help='SMNIST, PSMNIST')
parser.add_argument('--optim', default='adam', type=str, help='optimizer (default: adam)')  # sgd
parser.add_argument('--results-dir', default='', type=str, metavar='PATH', help='path to cache (default: none)')
parser.add_argument('-p', '--print-freq', default=100, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--seed', default=0, type=int, metavar='N', help='seed')
parser.add_argument('--epochs', default=200, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--lr', '--learning-rate', default=0.0005, type=float, metavar='LR', help='initial learning rate',
                    dest='lr')
parser.add_argument('--schedule', default=[60, 80], nargs='*', type=int,
                    help='learning rate schedule (when to drop lr by 10x); does not take effect if --cos is on')
parser.add_argument('--batch-size', default=256, type=int, metavar='N', help='mini-batch size')
parser.add_argument('--wd', default=0, type=float, metavar='W', help='weight decay')
parser.add_argument("--workers", type=int, default=0)
parser.add_argument('--cos', action='store_true', default=False, help='use cosine lr schedule')

# options for SNNs
parser.add_argument('--time-window', default=784, type=int, help='')
parser.add_argument('--threshold', default=1.0, type=float, help='')
parser.add_argument('--detach-reset', action='store_true', default=False, help='')
parser.add_argument('--hard-reset', action='store_true', default=False, help='')
parser.add_argument('--decay-factor', default=1.0, type=float, help='')
parser.add_argument('--beta1', default=0., type=float, help='')
parser.add_argument('--beta2', default=0., type=float, help='')
parser.add_argument('--gamma', default=0.5, type=float, help='dendritic reset scaling hyper-parameter')
parser.add_argument('--sg', default='gau', type=str, help='sg: triangle, exp, gau, rectangle and sigmoid')
parser.add_argument('--neuron', default='tclif', type=str, help='neuron: tclif, lif, alif and plif')
parser.add_argument('--network', default='ff', type=str, help='network(recurrent or feedforward): fb, ff')
parser.add_argument('--ind', default=1, type=int, help='input dim: 1, 4, 8')

args = parser.parse_args()

perm = torch.randperm(784)

if args.results_dir == '':
    args.results_dir = './cs-' + datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

Path(args.results_dir).mkdir(parents=True, exist_ok=True)
logger = setup_logging(os.path.join(args.results_dir, "log-" + datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + ".txt"))

gpu = torch.device('cuda')
seed_everything(seed=args.seed, is_cuda=True)

torch.backends.cudnn.benchmark = True

# build dataloaders
if args.task == 'SMNIST' or args.task == 'PSMNIST':
    train_loader, test_loader, num_classes = load_dataset(dataset='MNIST', batch_size=args.batch_size,
                                                          dataset_path='/datasets/', is_cuda=True,
                                                          num_workers=args.workers)
    args.time_window = 784
    in_dim = args.ind
else:
    raise NotImplementedError

if args.sg == 'exp':
    from surrogate import SingleExponential as SG
elif args.sg == 'triangle':
    from surrogate import Triangle as SG
elif args.sg == 'rectangle':
    from surrogate import Rectangle as SG
elif args.sg == 'sigmoid':
    from surrogate import sigmoid as SG
elif args.sg == 'gau':
    from surrogate import ActFun_adp as SG
else:
    raise NotImplementedError

node = None
if args.neuron == 'lif':
    node = LIFNode
elif args.neuron == 'tclif':
    node = TCLIFNode
elif args.neuron == 'plif':
    node = ParametricLIFNode
elif args.neuron == 'alif':
    node = ALIF

# initialize the learnable betas
beta = torch.full([1, 2], 0., dtype=torch.float)
beta[0][0] = args.beta1
beta[0][1] = args.beta2
init1 = torch.sigmoid(beta[0][0]).cpu().item()
init2 = torch.sigmoid(beta[0][1]).cpu().item()
print("beta init from {:.2f} and {:.2f}".format(-init1, init2))

spk_params = {"time_window": args.time_window,
              'v_threshold': args.threshold,
              'surrogate_function': SG.apply,
              'hard_reset': False,
              'detach_reset': False,
              'decay_factor': beta,
              'gamma': args.gamma}

spiking_neuron = partial(node,
                         v_threshold=spk_params['v_threshold'],
                         surrogate_function=spk_params['surrogate_function'],
                         hard_reset=spk_params['hard_reset'],
                         detach_reset=spk_params['detach_reset'],
                         decay_factor=spk_params['decay_factor'],
                         gamma=spk_params['gamma'])

if args.task == 'SMNIST':
    if args.network == 'ff':
        model = ffMnist(in_dim=in_dim, spiking_neuron=spiking_neuron).to(gpu)
    elif args.network == 'fb':
        model = fbMnist(in_dim=in_dim, spiking_neuron=spiking_neuron).to(gpu)
elif args.task == 'PSMNIST':
    if args.network == 'ff':
        model = ffMnist(in_dim=in_dim, spiking_neuron=spiking_neuron).to(gpu)
    elif args.network == 'fb':
        model = fbMnist(in_dim=in_dim, spiking_neuron=spiking_neuron).to(gpu)
else:
    raise NotImplementedError

logging.info(str(model))
para = utils.count_parameters(model)

criterion = nn.CrossEntropyLoss().cuda(gpu)

if args.optim == 'sgd':
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.wd, momentum=0.9)
elif args.optim == 'adam':
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
else:
    raise NotImplementedError

# dump args
with open(args.results_dir + '/args.json', 'w') as fid:
    json.dump(args.__dict__, fid, indent=2)
logging.info(str(args))
start_epoch = 0

if args.print_freq > len(train_loader):
    args.print_freq = math.ceil(len(train_loader) // 2)

best_acc = argparse.Namespace(top1=0, top5=0)
# For storing results
train_res = pd.DataFrame()
test_res = pd.DataFrame()
best = 0
for epoch in range(start_epoch, args.epochs):
    flag = False
    adjust_learning_rate(optimizer, epoch, args)

    train_acc, train_loss = train(train_loader, model, criterion, optimizer, epoch, args)

    acc1, acc5 = validate(test_loader, model, criterion, args)
    best_acc.top1 = max(best_acc.top1, acc1)
    best_acc.top5 = max(best_acc.top5, acc5)
    train_res[str(epoch)] = [train_acc.cpu().item(), train_loss]
    test_res[str(epoch)] = [acc1.cpu().item()]

    if acc1.cpu().item() >= best:
        flag = True
        best = acc1.cpu().item()
    print('Test Epoch: [{}/{}], lr: {:.6f}, acc: {:.4f}, best: {:.4f}'.format(epoch, args.epochs,
                                                                              optimizer.param_groups[0]['lr'],
                                                                              acc1, best))

    train_res.to_csv(os.path.join(args.results_dir, 'train_res.csv'), index=True)
    test_res.to_csv(os.path.join(args.results_dir, 'test_res.csv'), index=True)

    save_checkpoint({
        'epoch': epoch + 1,
        'best_acc': best_acc,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }, is_best=flag, dirname=args.results_dir, filename='checkpoint.pth.tar')
