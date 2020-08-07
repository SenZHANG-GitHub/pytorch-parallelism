## This repo is for learning parallelism in pytorch
# Architecture
+ There are two types of parallelism in pytorch
    + Model parallelism
        + Split different modules of a model into differetn gpus
        + Can solve the problem of not enough gpu memory when using large models
        + However, usually slow down training speed
    + Data parallelism
        + Copy the model into multiple gpus 
        + Split data batch into different gpus for parallel computing
        + Faster than single gpu training
+ We will focus on data parallelism here 
    + nn.DataParallel
        + Use one process and multithreading multi-gpus
        + Easy to use, just one line of code and the training data batch will also automatically split
        + However, using only one process may lead to performance overhead caused by GIL of python interpreter
    + nn.DistributedDataParallel
        + Use multiprocessing where a process is created for each GPU
        + Recommended method!
# Useful Links
+ [official doc 1](https://pytorch.org/tutorials/beginner/dist_overview.html): Overview and links to different tutorials
+ [official doc 2](https://pytorch.org/docs/master/generated/torch.nn.parallel.DistributedDataParallel.html#torch.nn.parallel.DistributedDataParallel): API for torch.nn.DistributedDataParallel
+ [official doc 3](https://pytorch.org/docs/master/distributed.html#distributed-basics): Basics of pytorch distributed training
+ [official doc 4](https://pytorch.org/tutorials/intermediate/dist_tuto.html): Good explanation of concepts (scatter, gather, reduce, all-reduce, broadcast, all-gather) and examples of low-level communications
+ [official doc 5](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html): Introduction to distributed data parallel. A good explanation of saving and loading ckp.
+ [official demo 1](https://github.com/pytorch/examples/blob/master/imagenet/main.py): Elegant example of DistributedDataParallel usage in imagenet that use ``mp.spawn()`` to init parallelism
+ [3rd party demo 1](https://fyubang.com/2019/07/23/distributed-training3/): Elegant example that use torch.distributed.launch to init parallelism
+ [3rd party demo 2](https://yangkky.github.io/2019/07/08/distributed-pytorch-tutorial.html): An example that use ``mp.spawn()`` to init parallelism 

# How to Do Distributed Data Parallelism in pytorch
+ You should be good after reading the above tutorials : )

## Initialization of DistributedDataParralelism using mp.spawn()
+ We put the usage of each component in the comments, where key modifications are:
    + (1) ``mp.spawn()``
    + (2) Correcting ``gpu, batch_size, workers, world_size`` and ``rank``
    + (3) ``torch.utils.data.distrbuted.init_process_group()``
    + (4) model: ``torch.DistributedDataParallel``
    + (5) train_sampler: ``torch.utils.data.distributed.DistributedSampler``
        + Need to move shuffle into dataset class rather than in loader
    + (6) ``train_sampler.set_epoch(epoch)``
    + (7) Only save ckp when ``args.rank % ngpus_per_node == 0``
+ Training script:
    + **Single node, multiple GPUs**
        + ``python main.py --dist-url 'tcp://127.0.0.1:FREEPORT' --world-size 1 --rank 0 --dist-backend 'nccl' --multiprocessing-distributed``
    + **Multiple nodes**
        + Node 0: ``python main.py --dist-url 'tcp://IP_OF_NODE0:FREEPORT' --world-size 2 --rank 0 --dist-backend 'nccl' --multiprocess-distributed``
        + Node 1: ``python main.py --dist-url 'tcp://IP_OF_NODE0:FREEPORT' --world-size 2 --rank 1 --dist-backend 'nccl' --multiprocess-distributed``
```python
import torch.nn as nn
import torch.distributed as dist
import torch.utils.data.distributed
import torch.multiprocessing as mp

parser.add_argument('-j', '--workers', default=4, type=int)
parser.add_argument('-b', '--batch-size', default=256, type=int, help='this is the total batch size of all gpus')
parser.add_argument('--world-size', default=-1, type=int, help='number of nodes for distributed training -> will be adjusted by * ngpus_per_node in code')
parser.add_argument('--rank', default=-1, type=int, help='node rank for distributed training -> (1) should be different for each machine (2) will be adjusted by args.rank * ngpu_per_node + gpu in code')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str, help='url used to set up distributed training')

# whether to use seed training -> warning!: Need to turn on the CUDNN deterministic setting, which can slow down the trainign considerably and we may see unexpected behavior when restarting from checkpoints!
if args.seed is not None:
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    cudnn.deterministic = True
    warning.warn('You have chose to seed training. This will turn on the CUDNN deterministic setting, which ca slow down your training considerably! You may see unexpected behavior when restarting from checkpoints')

# ngpus_per_node will be used for determining correct sub-batch-size for each single-gpu process 
# we can use CUDA_VISIBLE_DEVICES to choose gpus to use
ngpus_per_node = torch.cuda.device_count()
if args.multiprocessing_distributed:
    # assume there are multiple nodes (machines) as well, the total gpu used (world size) should be adjusted
    args.world_size = ngpus_per_node * args.world_size 
    mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
else:
    main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    # gpu will be specified automatically by mp.spawn() which range from [0, ngpus_per_node-1]
    if args.dist_url == 'env://' and args.rank== -1:
        args.rank=int(os.environ['rank'])
    if args.multiprocessing_distributed:
        # if we use single machine: args.rank is the rank of gpus
        # if we use multiple machines: args.rank is the rank of machines (nodes)! And we need to adjust args.rank to be the rank of gpus!
        args.rank = args.rank * ngpus_per_node + gpu
    
    ## Important: How to set up init_process_group
    # if init_method='env://': we are using system env variables like os.environ['MASTER_ADDR'] and os.environ['MASTER_PORT'].
    # We can also directly specify the url for init_method, e.g. 'tcp://224.66.41.62:23456'
    # if we use torch.distributed.launch instead of mp.spawn(), torch.distributed.launch will automatically specify the env variables for us!
    # init_method can also be a shared filename, and the system will also do the remaining 
    dist.init_process_group(backend='nccl', init_method=args.dist_url, world_size=args.world_size, rank=args.rank)

    # we need to use a scaled batch_size and workers for each sub-process to ensure the total num unchanged
    args.batch_size = int(args.batch_size / ngpus_per_node)
    args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)

    # set up model for DistributedDataParallelism
    args.gpu = gpu
    torch.cuda.set_device(args.gpu)
    model.cuda(args.gpu)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[[args.gpu])

    # resume from a checkpoint and run from current args.gpu
    loc = 'cuda:{}'.format(args.gpu)
    ckp = torch.load(args.resume, map_location=loc)
    model.load_state_dict(ckp['state_dict']
    optimizer.load_state_dict(ckp['optimizer'])

    # use cudnn benchmark to accelerate the code
    cudnn.benchmark = True

    train_dataset = datasets.ImageFolder(
        traindir, 
        transforms.Compose([
            transforms.RandomResizedCrop(224), 
            transforms.RandomHorizontalFlip(), 
            transforms.ToTensor(), 
            normalize,
        ]))
    val_dataset = dataset.ImageFoler(
        valdir,
        transforms.Compose([
            transforms.Resize(256), 
            transforms.CenterCrop224(), 
            transforms.ToTensor(), 
            normalize,
        ]))


    # we need to split dataset for each process using DistributedSampler
    # batch_size and num_workers should be adjusted for each sub-process
    # shuffle should be False if we use train_sampler (we need to move the shuffle function into dataset init!)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_loader = torch.utils.data.Dataloader(
        train_dataset, batch_size=args.batch_size, num_workers=args.workers
        shuffle=(train_sampler is None), pin_memory=True,
        sampler=train_sampler)

    # ? we dont use DisbutributedSampler for val_loader
    val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=args.batch_size, num_workers=args.workers,
            shuffle=False, pin_memory=True)

    for epoch in range(args.start_epoch, args.epochs):
        # set epoch for train_sampler
        train_sampler.set_epoch(epoch)
        train(...)
        validate(...)
        
    if not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank % ngpus_per_node == 0):
        save_ckp(...)


# validate(...) is similar except that we use model.eval() and with torch.no_grad():
def train(...):
    model.train()
    for i, (images, target) in enumerate(train_loader):
        images = images.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)
        
        ...


def adjust_learning_rate(optimizer, epoch, args):
    """Sets lr to the initial lr decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
def save_ckp('state, is_best, filename='ckp.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')

```
## Initialization of DistributedDataParralelism using torch.distributed.launch
+ we can run from cmd line using ``-m torch.distributed.launch``
+ e.g. ``CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 main.py``
+ torch.distributed.launch will distribute a local_rank which can be accessed by torch.distributed.get_rank()
```python
import torch
import torch.nn as nn
from torch.utils.data.distributed import DistributedSampler

# torch.distributed.launch will automatically specify rank for us
local_rank = torch.distributed.get_rank()
torch.cuda.set_device(local_rank)
device = torch.device('cuda', local_rank)

dataset = RandomDataset(...)
# here batch_size is for single-gpu, total batch size will be ngpu * this batch_size
rand_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        sampler=DistributedSampler(dataset))
model = Model(...)
# move model to specified gpu first before using DistributedDataParallel
model.to(device)
model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[local_rank],
        output_device=local_rank)

# write remaining codes as normal

```
+ The above usage usage is for single-node multi-process distributed training
+ Below we list a more complete list of potential usage
+ **Single-Node multi-process distributed training**
    + ``python -m torch.distributed.launch --nproc_per_node=NUM_GPUS_YOU_HAVE main.py``
    + master addr and port will be automatically distributed in this local machine
+ **Multi-Node multi-process distributed training: (e.g. two nodes)**
    + For node 1
        + ``python -m torch.distributed.launch --nproc_per_node=NUM_GPUS_YOU_HAVE --nnodes=2 --node_rank=0 --master_addr="192.168.1.1" --master_port=1234 main.py``
    + For node 2
        + ``python -m torch.distributed.launch --nproc_per_node=NUM_GPUS_YOU_HAVE --nnodes=2 --node_rank=1 --master_addr="192.168.1.1" --master_port=1234 main.py``
+ To look up what optional arguments this module offers
    + ``python -m torch.distributed.launch --help``
+ **Important Notices for torch.distributed.launch**
    + we can use ``os.environ['LOCAL_RANK']`` to get local_rank if we launch the script with ``--use_env=True`` 
    + we can also use argparse to get the local rank
    + ```python
      import argparse
      parser.argparse.ArgumentParser()
      parser.add_argument('--local_rank', type=int)
      args = parser.parse_args()
      
      # Method 1:
      torch.cuda.set_device(args.local_rank)
      
      # Method 2:
      with torch.cuda.device(args.local_rank):
          # Code block
      ```
    + we need to call ``torch.distributed.init_process_group`` at the beginning to start the distributed backend. And here the init_method must be ``'env://'``, which is the only supported init_method by this module
    + local_rank is not globally unique: it is only unique per process on a machine.










































