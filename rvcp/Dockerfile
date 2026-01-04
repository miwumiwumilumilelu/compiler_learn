# 使用 Ubuntu 基础镜像
FROM ubuntu:latest

# 更新源并安装 RISC-V 工具链和 QEMU
# 为了防止 apt 交互询问，加上 DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
    gcc-riscv64-linux-gnu \
    qemu-user

# 设置工作目录
WORKDIR /app