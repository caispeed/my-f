import os, sys, pathlib

sys.path.insert(0, os.path.dirname(pathlib.Path(__file__).parent.absolute()))

import torch
import numpy as np
import os
from copy import copy
import json
from tqdm import tqdm  # ✅ 新增：导入进度条库
from torch.utils.data import Subset  # ✅ 新增这行
from utils.io_tools import load_yaml, seed_everything, convert_list_to_tuple
from utils.visual_utils import plot_relative_error_distribution, plot_loss_backward
from utils.model_utils import load_data, load_model
from torch.optim.lr_scheduler import ReduceLROnPlateau
from utils.backward_utils import *

with open("./config/clamp_bounds.json", "r") as f:
    CLAMP_BOUNDS_BY_TOPOLOGY = convert_list_to_tuple(json.load(f))

YELLOW = "\033[93m"
RESET = "\033[0m"
GREEN = "\033[92m"
RED = "\033[91m"
BLUE = "\033[94m"
MAGENTA = "\033[95m"
CYAN = "\033[96m"
WHITE = "\033[97m"
seed_everything(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def invert_performance_to_params(model, graph_data, idx, param_templates,
                                 results_dict, global_perf_dict, scalers=None,
                                 lr=1e-2, num_steps=500, verbose=False):
    model.eval()
    device = next(model.parameters()).device
    graph_data = copy(graph_data).to(device)

    x_params = initialize_param_vector(CLAMP_BOUNDS_BY_TOPOLOGY, graph_data, device)
    optimizer = torch.optim.Adam([x_params], lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=50)

    best_loss = float("inf")
    best_params = None
    best_param_perf = None
    early_stop_counter = 0
    loss_history = []

    # ✅ 修改 1：添加 mininterval=1.0，强制它最快 1 秒刷新一次，防止刷屏
    # ✅ 修改 2：去掉 leave=False (可选)，如果你想保留每个电路的进度条就删掉 leave=False，想跑完就消失就保留
    progress_bar = tqdm(range(num_steps),
                        desc=f"Optimizing {graph_data.circuit_type}",
                        leave=False,
                        mininterval=1.0)

    for step in progress_bar:
        optimizer.zero_grad()
        loss, out, _ = run_optimization_step(model, graph_data, x_params, param_templates)
        loss.backward()
        optimizer.step()
        loss_history.append(loss.item())
        scheduler.step(loss.item())

        clamp_params(CLAMP_BOUNDS_BY_TOPOLOGY, x_params, graph_data)

        # ✅ 修改 3：彻底注释掉这行，不再显示跳动的 loss 值
        # progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})

        if loss.item() < best_loss - 1e-6:
            best_loss = loss.item()
            best_params = x_params.clone().detach()
            best_param_perf = out.clone().detach()
            early_stop_counter = 0
        else:
            early_stop_counter += 1

        if early_stop_counter >= 200:
            break

    best_perf_pred, rel_error, area, success = log_final_info(
        graph_data, best_params, best_param_perf, best_loss, scalers, idx, results_dict, global_perf_dict
    )

    return best_params.detach(), best_perf_pred, best_loss, rel_error, success


def main():
    # 初始化记录变量
    relative_errors = []
    best_losses = []
    results_dict = {}
    success_dict = {}
    all_dict = {}
    finished = []

    # 1. 加载数据
    test_dataset, scalers, global_perf_dict = load_data(loader=False, heldout=False)
    model = load_model(device)

    print(f"\n{CYAN}=== Starting Backward Evaluation (Limit: 2000 Samples) ==={RESET}")

    # ✅【关键修改 1】只取前 2000 个样本（如果不够2000就跑全部）
    TARGET_COUNT = 2000
    if len(test_dataset) > TARGET_COUNT:
        # 使用 Subset 创建一个只包含前 2000 个索引的“虚拟数据集”
        # 它是懒加载的，不会报错，也不会占用额外内存
        test_subset = Subset(test_dataset, range(TARGET_COUNT))
    else:
        test_subset = test_dataset

    # ✅【关键修改 2】添加总进度条
    for i, sample in enumerate(tqdm(test_subset, desc="Total Progress", unit="sample", dynamic_ncols=True)):
        sample = copy(sample)

        # 如果这个类型的电路已经“毕业”了（满500个），就跳过，省时间
        if sample.circuit_type in finished:
            continue

        # 运行反向设计（调用你之前优化过的函数）
        optimized_params, pred_perf, best_loss, rel_error, success = invert_performance_to_params(
            model=model,
            graph_data=sample,
            idx=i,
            global_perf_dict=global_perf_dict,
            results_dict=results_dict,
            scalers=scalers,
            param_templates=load_yaml('./dataset/param_templates.yaml'),
            lr=1e-6,
            num_steps=1000,
            verbose=False
        )

        # 统计结果
        if success:
            success_dict[sample.circuit_type] = success_dict.get(sample.circuit_type, 0) + 1
            all_dict[sample.circuit_type] = all_dict.get(sample.circuit_type, 0) + 1
            best_losses.append(best_loss)
            relative_errors.append(rel_error)

            # 简单的成功日志
            # print(f"{GREEN}[SUCCESS]{RESET} Sample {i:<4} | {sample.circuit_type:<8} | Loss: {best_loss:.4f}")

            # 检查是否凑够 500 个
            df = results_dict.get(sample.circuit_type)
            if df is not None and len(df) == 500 and sample.circuit_type not in finished:
                finished.append(sample.circuit_type)
                # ✅【关键修改 3】简化文件名，防止 'topologies' 未定义报错
                save_name = f"./results/circuits/{sample.circuit_type}_full_500.csv"
                df.to_csv(save_name, index=False)
                print(f"\n{YELLOW}>>> 🎉 Collected 500 successful samples for {sample.circuit_type}! Saved to {save_name}{RESET}")
        else:
            success_dict[sample.circuit_type] = success_dict.get(sample.circuit_type, 0)
            all_dict[sample.circuit_type] = all_dict.get(sample.circuit_type, 0) + 1
            # print(f"{RED}[FAILURE]{RESET} Sample {i:<4} | {sample.circuit_type:<8} | Loss: {best_loss:.4f}")

    # 循环结束后的总结
    print(f"\n{CYAN}=== Evaluation Finished! Saving all remaining data... ==={RESET}")

    # ✅【关键修改 4】兜底保存逻辑：保存所有跑出来的结果，不管够不够 500 个
    for circuit_type, df in results_dict.items():
        # 如果这个电路还没保存过（不在 finished 里），或者你想覆盖保存
        if circuit_type not in finished and df is not None and len(df) > 0:
            save_name = f"./results/circuits/{circuit_type}_partial_{len(df)}.csv"
            df.to_csv(save_name, index=False)
            print(f"  💾 Saved partial results for {circuit_type}: {len(df)} samples -> {save_name}")

    # 打印最终统计
    print(f"\n{GREEN}=== Final Statistics (Success Rate) ==={RESET}")
    for c_type, total in all_dict.items():
        succ = success_dict.get(c_type, 0)
        rate = (succ / total * 100) if total > 0 else 0
        print(f"  • {c_type:<10}: {succ}/{total} ({rate:.2f}%)")

if __name__ == "__main__":
    main()