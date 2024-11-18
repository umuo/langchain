# --coding: utf-8--
import ray
import time
import os

# 初始化Ray集群配置, 本地启动
ray.init(
    num_cpus=4,  # 设置CPU核心数量为4
    num_gpus=1,  # 设置GPU数量为1
    namespace="test_01",  # 设置命名空间为test_01
    ignore_reinit_error=True,  # 如果ray已经启动了，则忽略错误，避免重复初始化
    dashboard_host="0.0.0.0",
    dashboard_port=8265
)

@ray.remote
class Counter:
    def __init__(self):
        self.count = 0

    def increment(self):
        self.count += 1
        return self.count

    def server_msg(self, msg):
        print("进程1接收到消息：", msg)
        return msg + " <------from server"

if __name__ == '__main__':
    counter = Counter.remote()
    # 进程1进行5次增量
    result = 0
    for _ in range(5):
        result = counter.increment.remote()
    print("result: ", result)
    pid = os.getpid()
    print("current pid: ", pid)
    print(" server is started ! ")
    while True:
        time.sleep(1)
