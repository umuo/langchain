# --coding: utf-8--
import ray
from ray_server import Counter

ray.init(address="auto", ignore_reinit_error=True)

counter = Counter.remote()

server_ret_msg = counter.server_msg.remote("发送一个测试消息")
print(ray.get(server_ret_msg))

