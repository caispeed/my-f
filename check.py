import pandas as pd
import shutil
import os

# 设定你要批量处理的根目录，通常是存放所有原始数据的文件夹
ROOT_DIR = "dataset"


def clean_single_file(csv_path):
    """处理单个 CSV 文件的清洗逻辑"""
    if csv_path.endswith("_backup.csv"):
        return

    backup_path = csv_path.replace(".csv", "_backup.csv")

    print(f"\n📂 正在扫描: {csv_path}")

    try:
        # 有些文件可能为空或格式损坏，用 try-except 保护一下
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"    ❌ 读取失败或文件为空，跳过。原因: {e}")
        return

    before_len = len(df)

    # 找出所有被污染成字符串 (object) 的列
    str_cols = df.select_dtypes(include=['object']).columns.tolist()

    if not str_cols:
        print("    ✅ 数据集非常干净，无需清洗！")
        return

    print(f"     抓到错误列: {str_cols}")
    print(f"     正在备份至: {backup_path}")
    shutil.copy(csv_path, backup_path)

    print("     正在清理...")
    for col in str_cols:
        # 强行转成数字，遇到纯字母变成 NaN
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # 剔除所有包含 NaN 的脏数据行
    df = df.dropna()
    after_len = len(df)

    # 覆盖保存回原路径
    df.to_csv(csv_path, index=False)

    print(f"    ✨ 清洗完毕！清除脏数据: {before_len - after_len} 行，剩余纯净数据: {after_len} 行。")


def batch_clean():
    print(f"=== 🚀 开始批量扫描 [{ROOT_DIR}] 目录下的所有 CSV 文件 ===")

    csv_files = []
    # os.walk 会递归遍历 ROOT_DIR 下的所有主目录和子目录
    for root, dirs, files in os.walk(ROOT_DIR):
        for file in files:
            # 只找 .csv 文件，且跳过备份文件
            if file.endswith(".csv") and not file.endswith("_backup.csv"):
                # 把路径拼起来，比如 dataset/LNA/CGLNA/dataset.csv
                full_path = os.path.join(root, file)
                csv_files.append(full_path)

    if not csv_files:
        print("❌ 没有找到任何 CSV 文件，请检查 ROOT_DIR 路径是否正确！")
        return

    print(f"[*] 🎯 共定位到 {len(csv_files)} 个 CSV 文件，开始逐一排雷...\n")

    # 挨个清洗
    for file in csv_files:
        clean_single_file(file)

    print("\n=== 🎉 批量数据清洗任务全部完成！ ===")


if __name__ == "__main__":
    batch_clean()