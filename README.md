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
+ [official demo 1](https://github.com/pytorch/examples/blob/master/imagenet/main.py): Elegant example of DistributedDataParallel usage in imagenet that use ``mp.spawn()`` to init parallelism
+ [3rd party demo 1](https://fyubang.com/2019/07/23/distributed-training3/) 
