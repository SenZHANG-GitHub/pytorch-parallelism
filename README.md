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
+ [official demo 1](https://github.com/pytorch/examples/blob/master/imagenet/main.py): Elegant example of DistributedDataParallel usage in imagenet that use ``mp.spawn()`` to init parallelism
+ [3rd party demo 1](https://fyubang.com/2019/07/23/distributed-training3/): Elegant example that use torch.distributed.launch to init parallelism
+ [3rd party demo 2](https://yangkky.github.io/2019/07/08/distributed-pytorch-tutorial.html): An example that use ``mp.spawn()`` to init parallelism 

# How to Do Distributed Data Parallelism in pytorch
+ You should be good after reading the above tutorials : )
---
**Initialization of DistributedDataParralelism using mp.spawn()**
```python
import torch.nn as nn
import torch.distributed as dist
import torch.utils.data.distributed
import torch.multiprocessing as mp

# whether to use seed training -> warning!: Need to turn on the CUDNN deterministic setting, which can slow down the trainign considerably and we may see unexpected behavior when restarting from checkpoints!
if args.seed is not None:
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    cudnn.deterministic = True

# ngpus_per_node will be used for determining correct sub-batch-size for each single-gpu process 
# we can use CUDA_VISIBLE_DEVICES to choose gpus to use
ngpus_per_node = torch.cuda.device_count()
if args.multiprocessing_distributed:
    # assume there are multiple nodes (machines) as well, the total gpu used (world size) should be adjusted
    args.world_size = ngpus_per_node * args.world_size 
    mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))

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
    dist.init_process_group(backend='nccl', init_method='env://', world_size=args.world_size, rank=args.rank)

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

```







































