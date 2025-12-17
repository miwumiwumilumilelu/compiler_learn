import os
import subprocess
import shutil
import sys
from pathlib import Path

# ================= 配置区域 =================
COMPILER = "g++" # 或者 clang++
CFLAGS = ["-std=c++17", "-g", "-I.", "-Wall"]
BUILD_DIR = "build"
TARGET_NAME = "compiler"
# ===========================================

def build():
    project_root = Path(__file__).parent.absolute()
    build_path = project_root / BUILD_DIR
    target_path = build_path / TARGET_NAME

    if os.name == 'nt':
        target_path = target_path.with_suffix(".exe")

    print(f"🔧 项目根目录: {project_root}")
    
    # 清理旧构建
    if build_path.exists():
        shutil.rmtree(build_path)
    build_path.mkdir(parents=True, exist_ok=True)

    source_files = []
    print("\n🔍 扫描源文件...")
    
    # 遍历所有 .cpp 文件
    for file_path in project_root.rglob("*.cpp"):
        # 1. 排除 build 目录
        if BUILD_DIR in file_path.parts:
            continue
            
        # 2. 【关键修改】排除所有以 test_ 开头的文件
        # 假设测试文件都叫 test_xxx.cpp
        if file_path.name.startswith("test_"):
            print(f"   🚫 跳过测试文件: {file_path.name}")
            continue

        source_files.append(str(file_path))
        print(f"   ✅ 添加编译: {os.path.relpath(file_path, project_root)}")

    if not source_files:
        print("❌ 错误: 未找到任何源文件！")
        sys.exit(1)

    cmd = [COMPILER] + CFLAGS + source_files + ["-o", str(target_path)]

    print(f"\n🚀 开始编译 ({len(source_files)} 个文件)...")
    try:
        subprocess.run(cmd, check=True)
        print("\n✅ 编译成功！")
        if os.name == 'nt':
            print(f"👉 运行: {build_path}\\{TARGET_NAME}")
        else:
            print(f"👉 运行: ./{BUILD_DIR}/{TARGET_NAME}")

    except subprocess.CalledProcessError:
        print("\n❌ 编译失败，请解决上述代码错误。")
        sys.exit(1)

if __name__ == "__main__":
    build()