# --coding: utf-8--
import ray
import time
import os
import netifaces  # 确保安装了这个库


def get_ip_address():
    # 获取所有网络接口的 IP 地址
    ipv4_addresses = []
    interfaces = netifaces.interfaces()
    
    for interface in interfaces:
        addresses = netifaces.ifaddresses(interface)
        if netifaces.AF_INET in addresses:
            for link in addresses[netifaces.AF_INET]:
                ip_address = link['addr']
                # 排除回环地址和 Docker 的 IP 地址（通常以 172 开头）
                # if not ip_address.startswith("127.0") and not ip_address.startswith("172."):
                if ip_address.startswith("192.168.44"):
                    ipv4_addresses.append(ip_address)

    return ipv4_addresses


# 启动 Ray 集群
ray.init(
    address="192.168.44.130:63790",
    _node_ip_address=get_ip_address()[0],  # 指定本机 IP
    ignore_reinit_error=True,
    runtime_env={"env_vars": {"RAY_DEDUP_LOGS": "0"}},
    # _temp_dir=r"E:\DeskTop\results\1902436747",
)

# 定义一个远程任务，模拟耗时计算
@ray.remote
def process_data(data):
    ip_addresses = get_ip_address()
    # 选择第一个有效的 IP 地址
    ip_address = ip_addresses[0] if ip_addresses else "No valid IP found"
    print(f"Task '{data}' is being processed on machine with IP: {ip_address}")
    time.sleep(1)  # 模拟处理时间
    return f"Processed {data}"

pid = os.getpid()
print("current pid: ", pid)

# 创建任务
data = ["task_1", "task_2", "task_3", "task_4", "task_5", "task_6"]
futures = [process_data.remote(d) for d in data]

print("Waiting for tasks to complete...")
# 等待并获取结果
results = ray.get(futures)

# 输出结果
for result in results:
    print(result)

# 关闭 Ray 集群
ray.shutdown()

"""
2024-11-18 18:58:31,279	INFO worker.py:1634 -- Connecting to existing Ray cluster at address: 192.168.44.130:63790...
2024-11-18 18:58:31,292	INFO worker.py:1819 -- Connected to Ray cluster.
current pid:  15807
Waiting for tasks to complete...
(process_data pid=15842) Task 'task_1' is being processed on machine with IP: 192.168.44.131
Processed task_1
Processed task_2
Processed task_3
Processed task_4
Processed task_5
Processed task_6
(process_data pid=42320, ip=192.168.44.130) Task 'task_6' is being processed on machine with IP: 192.168.44.130 [repeated 9x across cluster] (Ray deduplicates logs by default. Set RAY_DEDUP_LOGS=0 to disable log deduplication, or see https://docs.ray.io/en/master/ray-observability/user-guides/configure-logging.html#log-deduplication for more options.)

打印两个节点，证明是两个节点共同完成了整个作业。
"""
