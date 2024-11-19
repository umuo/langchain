# Ray

ray提供了强大的并行计算的能力，并且支持分布式计算。

ray的分布式计算基于Actor模型，每个actor是一个独立的进程，可以独立运行，并且可以进行远程调用。

ray可以轻松地调度数以千计的任务，并通过分布式计算来加速任务的执行

ray可以将一个大任务拆分成多个子任务，然后将这些子任务分配给不同的actor，从而实现并行计算。

这里的actor 指的是一个独立的进程，而不是一个线程。也可以指不同机器上的进程。



使用 @ray.remote 装饰器将这个函数变成远程任务。
通过 ray.get(futures) 等待所有任务完成并获取它们的结果。
Ray 会在集群中并行处理这些任务，提高计算效率。


ray启动的时候，如果是本地模式，会默认启动一个主节点，
进程名是gcs_server(global control store server)，
其他ray进程是worker进程，进程名是raylet(raylet)。

我们可以单独启动一个主节点
启动命令：`ray start --head --port=63790`

启动worker节点：`ray start --address=192.168.44.130:63790`

window 操作系统作为worker节点：
```bash
set RAY_ENABLE_WINDOWS_OR_OSX_CLUSTER=1
ray start --address=192.168.44.130:63790
```

如果不指定端口，默认是6379。
> 在 Ray 中，您不需要单独安装 Redis，因为 Ray 自带了一个内置的 Redis 实现。Ray 的 gcs_server（Global Control Store Server）使用这个内置的 Redis 作为其后端存储来管理集群的状态和元数据。





