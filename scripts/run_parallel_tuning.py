#!/usr/bin/env python3
"""
并行超参数调优脚本

同时启动4个训练实验，每个使用不同的超参数配置。
支持单GPU（时间分片）或多GPU（并行）。

Usage:
    # 单GPU（顺序执行，但可以异步启动监控进度）:
    python scripts/run_parallel_tuning.py --gpu 0

    # 多GPU（真正并行，每个实验占用1个GPU）:
    python scripts/run_parallel_tuning.py --multi-gpu 0,1,2,3

    # 不指定GPU（自动检测）:
    python scripts/run_parallel_tuning.py
"""

import subprocess
import argparse
import time
import signal
import sys
import os
from pathlib import Path
from datetime import datetime
import torch

# 4个配置文件
CONFIGS = [
    'configs/tune1_loss_weights.yaml',
    'configs/tune2_lr_embed.yaml',
    'configs/tune3_depth_dropout.yaml',
    'configs/tune4_hybrid.yaml'
]

# 进程列表（用于cleanup）
processes = []


def signal_handler(sig, frame):
    """处理Ctrl+C信号，终止所有子进程"""
    print("\n\n⚠️  收到中断信号，正在终止所有训练进程...")
    for i, proc in enumerate(processes):
        if proc.poll() is None:  # 进程还在运行
            print(f"  终止进程 {i+1}...")
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
    print("✓ 所有进程已终止")
    sys.exit(0)


def run_single_gpu(gpu_id):
    """
    单GPU模式：顺序执行4个实验

    虽然是顺序，但可以让它们在后台运行并监控进度
    """
    print("="*80)
    print("单GPU模式：顺序执行4个实验")
    print(f"使用GPU: {gpu_id}")
    print("="*80)

    # 确保log目录存在
    Path('log').mkdir(exist_ok=True)

    start_time = datetime.now()

    for i, config in enumerate(CONFIGS, 1):
        config_name = Path(config).stem
        log_file = f"log/tune_parallel_{config_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

        print(f"\n[{i}/4] 启动实验: {config_name}")
        print(f"  配置文件: {config}")
        print(f"  日志文件: {log_file}")

        # 构建命令
        cmd = [
            'python', 'scripts/train_model_fast.py',
            '--config', config
        ]

        # 设置环境变量
        env = {
            **dict(os.environ),
            'CUDA_VISIBLE_DEVICES': str(gpu_id)
        }

        # 启动进程
        with open(log_file, 'w') as log_f:
            proc = subprocess.Popen(
                cmd,
                stdout=log_f,
                stderr=subprocess.STDOUT,
                env=env
            )
            processes.append(proc)

        print(f"  PID: {proc.pid}")
        print(f"  等待完成...")

        # 等待进程完成
        try:
            proc.wait()
            if proc.returncode == 0:
                print(f"  ✓ 实验 {config_name} 完成!")
            else:
                print(f"  ✗ 实验 {config_name} 失败! (返回码: {proc.returncode})")
        except KeyboardInterrupt:
            signal_handler(signal.SIGINT, None)

    end_time = datetime.now()
    elapsed = end_time - start_time

    print("\n" + "="*80)
    print("✓ 所有实验完成!")
    print(f"总耗时: {elapsed}")
    print("="*80)


def run_multi_gpu(gpu_ids):
    """
    多GPU模式：并行执行4个实验

    每个实验占用1个GPU，真正并行运行
    支持非连续GPU IDs（例如：0,5,6,7）
    """
    import os

    print("="*80)
    print("多GPU模式：并行执行4个实验")
    print(f"使用GPU: {gpu_ids}")
    print("="*80)

    if len(gpu_ids) < len(CONFIGS):
        print(f"⚠️  警告: GPU数量({len(gpu_ids)})少于实验数量({len(CONFIGS)})")
        print("   部分实验将共享GPU")

    # 确保log目录存在
    Path('log').mkdir(exist_ok=True)

    start_time = datetime.now()

    # 启动所有实验
    for i, config in enumerate(CONFIGS):
        gpu_id = gpu_ids[i % len(gpu_ids)]  # 循环分配GPU
        config_name = Path(config).stem
        log_file = f"log/tune_parallel_{config_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

        print(f"\n[{i+1}/4] 启动实验: {config_name}")
        print(f"  配置文件: {config}")
        print(f"  GPU: {gpu_id}")
        print(f"  日志文件: {log_file}")

        # 构建命令
        cmd = [
            'python', 'scripts/train_model_fast.py',
            '--config', config
        ]

        # 设置环境变量（指定GPU）
        env = {
            **dict(os.environ),
            'CUDA_VISIBLE_DEVICES': str(gpu_id)
        }

        # 启动进程（后台）
        with open(log_file, 'w') as log_f:
            proc = subprocess.Popen(
                cmd,
                stdout=log_f,
                stderr=subprocess.STDOUT,
                env=env
            )
            processes.append(proc)

        print(f"  PID: {proc.pid}")
        time.sleep(2)  # 避免同时启动导致资源冲突

    print("\n" + "="*80)
    print("所有实验已启动，正在后台运行...")
    print("="*80)

    # 监控进程状态
    print("\n监控进程状态 (Ctrl+C 终止所有进程):")

    try:
        while True:
            all_done = True
            for i, proc in enumerate(processes):
                config_name = Path(CONFIGS[i]).stem
                if proc.poll() is None:
                    print(f"  [{i+1}/4] {config_name}: 运行中 (PID: {proc.pid})")
                    all_done = False
                else:
                    if proc.returncode == 0:
                        print(f"  [{i+1}/4] {config_name}: ✓ 完成")
                    else:
                        print(f"  [{i+1}/4] {config_name}: ✗ 失败 (返回码: {proc.returncode})")

            if all_done:
                break

            print()
            time.sleep(30)  # 每30秒更新一次

    except KeyboardInterrupt:
        signal_handler(signal.SIGINT, None)

    end_time = datetime.now()
    elapsed = end_time - start_time

    print("\n" + "="*80)
    print("✓ 所有实验完成!")
    print(f"总耗时: {elapsed}")
    print("="*80)

    # 统计结果
    print("\n实验结果总结:")
    for i, proc in enumerate(processes):
        config_name = Path(CONFIGS[i]).stem
        if proc.returncode == 0:
            print(f"  [{i+1}/4] {config_name}: ✓ 成功")
        else:
            print(f"  [{i+1}/4] {config_name}: ✗ 失败 (返回码: {proc.returncode})")


def main():
    parser = argparse.ArgumentParser(description='并行超参数调优')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--gpu', type=int, default=None,
                       help='单GPU模式，指定GPU ID（顺序执行）')
    group.add_argument('--multi-gpu', type=str, default=None,
                       help='多GPU模式，逗号分隔的GPU IDs（并行执行），例如: 0,1,2,3')

    args = parser.parse_args()

    # 注册信号处理器
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # 检查GPU可用性
    if not torch.cuda.is_available():
        print("⚠️  警告: CUDA不可用，将使用CPU训练（会非常慢）")
        run_single_gpu('cpu')
        return

    num_gpus = torch.cuda.device_count()
    print(f"检测到 {num_gpus} 个GPU")

    # 决定运行模式
    if args.multi_gpu:
        # 多GPU模式
        gpu_ids = [int(x.strip()) for x in args.multi_gpu.split(',')]

        # 验证GPU IDs
        invalid_gpus = [gid for gid in gpu_ids if gid >= num_gpus]
        if invalid_gpus:
            print(f"❌ 错误: GPU IDs {invalid_gpus} 不存在（可用GPU: 0-{num_gpus-1}）")
            sys.exit(1)

        run_multi_gpu(gpu_ids)

    elif args.gpu is not None:
        # 单GPU模式
        if args.gpu >= num_gpus:
            print(f"❌ 错误: GPU {args.gpu} 不存在（可用GPU: 0-{num_gpus-1}）")
            sys.exit(1)

        run_single_gpu(args.gpu)

    else:
        # 自动模式：根据GPU数量决定
        if num_gpus >= 4:
            print("自动模式: 检测到>=4个GPU，使用多GPU并行模式")
            run_multi_gpu(list(range(min(4, num_gpus))))
        elif num_gpus >= 2:
            print(f"自动模式: 检测到{num_gpus}个GPU，使用多GPU并行模式")
            run_multi_gpu(list(range(num_gpus)))
        else:
            print("自动模式: 只有1个GPU，使用单GPU顺序模式")
            run_single_gpu(0)


if __name__ == '__main__':
    main()
