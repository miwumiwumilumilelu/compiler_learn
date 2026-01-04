import os
import subprocess
import shutil
import sys
from pathlib import Path

# ================= 配置区域 =================
COMPILER = "g++" 
CFLAGS = ["-std=c++17", "-g", "-I.", "-Wall", "-fsanitize=address,undefined"]
BUILD_DIR = "build"
TARGET_NAME = "compiler"
DOCKER_IMAGE = "compiler-env" 
# ===========================================

# 一个极简的 SysY 运行时库
RUNTIME_C_CONTENT = """
#include <stdio.h>
#include <sys/time.h>

void putint(int a) {
    printf("%d", a);
}

void putch(int a) {
    printf("%c", a);
}

void putfloat(float a) {
    printf("%f", a);
}

int getint() {
    int t;
    scanf("%d", &t);
    return t;
}

int getch() {
    char c;
    scanf("%c", &c);
    return (int)c;
}

int getfloat() {
    float t;
    scanf("%f", &t);
    return (int)t; // 简化的转换
}

void _sysy_starttime(int lineno) {
    // 简单实现：打印日志或什么都不做
}

void _sysy_stoptime(int lineno) {
    // 简单实现
}
"""

def build():
    project_root = Path(__file__).parent.absolute()
    build_path = project_root / BUILD_DIR
    target_path = build_path / TARGET_NAME

    if os.name == 'nt':
        target_path = target_path.with_suffix(".exe")

    print(f"🔧 项目根目录: {project_root}")
    
    if build_path.exists():
        pass 
    build_path.mkdir(parents=True, exist_ok=True)

    # 1. 创建 runtime.c
    runtime_c_path = build_path / "runtime.c"
    with open(runtime_c_path, "w") as f:
        f.write(RUNTIME_C_CONTENT)
    print(f"📄 生成运行时库源码: {runtime_c_path}")

    source_files = []
    for file_path in project_root.rglob("*.cpp"):
        if BUILD_DIR in file_path.parts: continue
        if file_path.name.startswith("test_"): continue
        source_files.append(str(file_path))

    if not source_files:
        print("❌ 错误: 未找到任何源文件！")
        sys.exit(1)

    cmd = [COMPILER] + CFLAGS + source_files + ["-o", str(target_path)]

    print(f"🚀 正在编译编译器...")
    try:
        subprocess.run(cmd, check=True)
        print("✅ 编译器构建成功！")
    except subprocess.CalledProcessError:
        print("\n❌ 编译失败。")
        sys.exit(1)
    
    return project_root, target_path, runtime_c_path

def run_tests(project_root, compiler_path, runtime_c_path):
    print("\n🧪 开始自动化测试流水线...")
    
    test_dir = project_root / "test" / "custom"
    if not test_dir.exists():
        print(f"❌ 未找到测试目录: {test_dir}")
        return

    test_cases = sorted(list(test_dir.glob("*.manbin")))
    if not test_cases:
        print("⚠️  未找到 .manbin 测试文件")
        return

    success_count = 0
    
    # 在 Docker 里的路径
    rel_runtime = runtime_c_path.relative_to(project_root)

    for manbin_file in test_cases:
        case_name = manbin_file.name
        asm_file = manbin_file.with_suffix(".s")     
        exe_file = manbin_file.with_suffix("")       
        out_file = manbin_file.with_suffix(".out")   

        print(f"   👉 测试: {case_name}", end="", flush=True)

        try:
            # 1. 生成汇编
            with open(asm_file, "w") as f:
                subprocess.run([str(compiler_path), str(manbin_file)], stdout=f, check=True)
            
            rel_asm = asm_file.relative_to(project_root)
            rel_exe = exe_file.relative_to(project_root)
            
            # 2. 编译 (链接 runtime.c)
            docker_gcc_cmd = [
                "docker", "run", "--rm",
                "-v", f"{project_root}:/app",
                "-w", "/app",
                DOCKER_IMAGE,
                "riscv64-linux-gnu-gcc", 
                str(rel_asm), str(rel_runtime), # <--- 链接 runtime.c
                "-o", str(rel_exe), "-static"
            ]
            
            subprocess.run(docker_gcc_cmd, check=True, capture_output=True)

            # 3. 运行
            docker_qemu_cmd = [
                "docker", "run", "--rm",
                "-v", f"{project_root}:/app",
                "-w", "/app",
                DOCKER_IMAGE,
                "qemu-riscv64", str(rel_exe)
            ]
            
            result = subprocess.run(docker_qemu_cmd, check=False, capture_output=True)
            actual_result = result.returncode

            with open(out_file, "w") as f:
                f.write(str(actual_result))
            
            print(f" -> 结果: {actual_result} (已写入 .out)")
            success_count += 1

        except subprocess.CalledProcessError as e:
            print(f" -> ❌ 失败")
            if e.stderr:
                print(f"\n[错误日志]:\n{e.stderr.decode('utf-8')}")
        except Exception as e:
            print(f" -> ❌ 脚本错误: {e}")

    print(f"\n🎉 测试完成: {success_count}/{len(test_cases)} 个用例已处理。")

if __name__ == "__main__":
    root, compiler, runtime = build()
    run_tests(root, compiler, runtime)