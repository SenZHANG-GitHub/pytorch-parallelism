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
