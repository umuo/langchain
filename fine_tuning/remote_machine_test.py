from itertools import count
from re import sub
import modal
import time
import subprocess
import threading
import secrets
import os 
"""
现在设置好 token，才可运行此代码。

"""

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git")
    .pip_install("jupyterlab")
    .pip_install("fastapi[standard]")
    .run_commands("git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git")
    .run_commands("cd LLaMA-Factory && pip install -e .[torch,metrics]")
    # .run_commands("llamafactory-cli webui --server_port 8080")
    # .env({"HALT_AND_CATCH_FIRE": 0})
    # .run_commands("git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git")
)

# 使用 Dockerfile 构建镜像
# image = modal.Image.from_dockerfile("Dockerfile")
app = modal.App("llamafactory", image=image)


# 默认是cpu，可以指定gpu运行  T4 -> 16g    L4 -> 24g
@app.function(gpu=modal.gpu.L4(count=1))
def square(x):
    print("This code is running on a remote machine!")
    resp = subprocess.check_call(["nvidia-smi"])
    print(resp)
    return x ** 2

@app.function(gpu=modal.gpu.L4(count=1), image=image, mounts=[modal.Mount.from_local_dir("./test", remote_path="/root/LLaMA-Factory")])
def install():
    resp = subprocess.run(
        ["pip", "install", "-e", ".[torch,metrics]"],
        cwd="/root/LLaMA-Factory/LLaMA-Factory",
        check=True,
    )
    print(resp)
    # resp = subprocess.run(
    #     ["llamafactory-cli", "webui"],
    #     cwd="/root/LLaMA-Factory/LLaMA-Factory",
    #     check=True,
    # )
    # print(resp)
    pass

@app.function(gpu=modal.gpu.L4(count=1), image=image)
# @modal.web_endpoint(method="GET") # 暴露 Web UI
def web_ui():
    def print_output(pipe, is_stderr=False):
        for line in iter(pipe.readline, b""):
            if is_stderr:
                print(f"STDERR: {line.decode().strip()}")
            else:
                print(f"STDOUT: {line.decode().strip()}")
    # 执行 llamafactory-cli webui 命令
    # --server_port 指定端口，需要与 @stub.web_endpoint() 对应
    # --server_name 指定服务器名称，0.0.0.0 允许外部访问

    with modal.forward(8080) as tunnel:
        print(f"tunnel.url        = {tunnel.url}")
        print(f"tunnel.tls_socket = {tunnel.tls_socket}")
        process = subprocess.Popen(
            [
                "llamafactory-cli",
                "webui",
                "--server_port",
                "8080",
                # "--server_name",
                # "0.0.0.0",
                # "--share", # 如果需要生成公网链接，加上这个参数
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        print("子进程已启动，主程序继续执行...")
        stdout_thread = threading.Thread(target=print_output, args=(process.stdout,))
        stderr_thread = threading.Thread(target=print_output, args=(process.stderr, True))

        stdout_thread.start()
        stderr_thread.start()

        while process.poll() is None:
            print("子进程仍在运行...")
            time.sleep(10)

        print("子进程执行完毕")
        print(f"返回值: {process.returncode}")

    # process = modal.container_app.spawn_sandbox(
    #     "llamafactory-cli",
    #     "webui",
    #     "--server_port",
    #     "8080",
    #     "--server_name",
    #     "0.0.0.0",
    #     "--share", # 如果需要生成公网链接，加上这个参数
    #     mounts=[modal.Mount.from_local_dir("./test", remote_path="/root/LLaMA-Factory/LLaMA-Factory")],
    #     timeout=86400, # 超时时间，根据你的任务调整
    # )
    # for log in process.logs():
    #     print(log.decode(), end="")
    return "Web UI started"

@app.function()
@modal.web_endpoint(method="GET")
def http_test(x: int):
    return {"result": x}
    pass

@app.function(image=image)
def run_jupyter():
    token = secrets.token_urlsafe(13)
    with modal.forward(8888) as tunnel:
        url = tunnel.url + "/?token=" + token
        print(f"Starting Jupyter at {url}")
        subprocess.run(
            [
                "jupyter",
                "lab",
                "--no-browser",
                "--allow-root",
                "--ip=0.0.0.0",
                "--port=8888",
                "--LabApp.allow_origin='*'",
                "--LabApp.allow_remote_access=1",
            ],
            env={**os.environ, "JUPYTER_TOKEN": token, "SHELL": "/bin/bash"},
            stderr=subprocess.DEVNULL,
        )

@app.local_entrypoint()
def main():
    # print("the square is ", square.remote(42))
    # install.remote()
    # web_ui.remote()
    run_jupyter.remote()
    pass


