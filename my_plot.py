import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 设置结果路径 (你刚才生成的文件夹)
results_dir = "./results/circuits"

all_errors = []

# 1. 遍历所有生成的 CSV 文件
if not os.path.exists(results_dir):
    print("❌ 找不到结果文件夹！")
    exit()

for filename in os.listdir(results_dir):
    if filename.endswith(".csv"):
        filepath = os.path.join(results_dir, filename)
        try:
            df = pd.read_csv(filepath)
            # 读取 'rel_error_pred' (AI 预测的相对误差)
            if 'rel_error_pred' in df.columns:
                # 过滤掉极端的失败值 (比如 > 1000% 的)
                valid_errors = df['rel_error_pred'][df['rel_error_pred'] < 100]
                all_errors.extend(valid_errors.tolist())
                print(f"✅ 加载 {filename}: {len(valid_errors)} 个有效样本")
        except Exception as e:
            print(f"⚠️ 读取 {filename} 失败: {e}")

# 2. 画直方图
if len(all_errors) == 0:
    print("❌ 没有找到任何误差数据，请检查 CSV 文件是否生成。")
else:
    plt.figure(figsize=(10, 6))
    plt.hist(all_errors, bins=50, color='skyblue', edgecolor='black', alpha=0.7)

    # 统计信息
    median_err = np.median(all_errors)
    mean_err = np.mean(all_errors)

    plt.axvline(median_err, color='red', linestyle='dashed', linewidth=1, label=f'Median: {median_err:.2f}%')
    plt.axvline(mean_err, color='green', linestyle='dashed', linewidth=1, label=f'Mean: {mean_err:.2f}%')

    plt.title("AI Prediction Relative Error Distribution")
    plt.xlabel("Relative Error (%)")
    plt.ylabel("Count")
    plt.legend()
    plt.grid(axis='y', alpha=0.5)

    # 保存图片
    os.makedirs("./results", exist_ok=True)
    plt.savefig("./results/my_error_plot.png")
    print(f"\n🎉 图表已生成！请查看 ./results/my_error_plot.png")
    plt.show()