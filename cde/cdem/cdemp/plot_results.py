import re
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix
import os

# ================= 配置区域 =================
# 请将此处改为您服务器上实际的日志文件名
LOG_FILE_PATH = 'cem_results.txt' 
# 图片保存路径
SAVE_DIR = './plots'
# ===========================================

# 创建保存目录
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

def parse_log_file(file_path):
    """解析日志文件，提取真实标签和预测标签"""
    print(f"正在读取文件: {file_path} ...")
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 使用分隔符切分样本
    blocks = content.split('---------------------------------------------------------------')
    
    y_true = []
    y_pred = []
    
    for block in blocks:
        # 提取 Ground Truth (真实情感)
        gt_match = re.search(r'Emotion:\s*([a-z]+)', block)
        # 提取 Predictions (预测情感)，通常取第一个作为 Top-1 预测
        pred_match = re.search(r'Pred Emotions:\s*([a-z]+)', block)
        
        if gt_match and pred_match:
            y_true.append(gt_match.group(1).strip())
            y_pred.append(pred_match.group(1).strip())
            
    print(f"成功提取样本数: {len(y_true)}")
    return y_true, y_pred

def plot_accuracy_distribution(y_true, y_pred, emotions_list):
    """绘制 32 类情感准确率分布图"""
    print("正在绘制准确率分布图...")
    
    # 计算每一类的准确率
    acc_dict = {}
    for emo in emotions_list:
        indices = [i for i, x in enumerate(y_true) if x == emo]
        total = len(indices)
        correct = sum([1 for i in indices if y_pred[i] == emo])
        acc = correct / total if total > 0 else 0.0
        acc_dict[emo] = acc * 100  # 转换为百分比

    # 按准确率从高到低排序
    sorted_items = sorted(acc_dict.items(), key=lambda x: x[1], reverse=True)
    emotions_sorted = [x[0] for x in sorted_items]
    acc_sorted = [x[1] for x in sorted_items]

    # 绘图设置
    plt.figure(figsize=(15, 8))
    # 使用光谱配色方案，分数高的颜色亮
    norm = plt.Normalize(min(acc_sorted), max(acc_sorted))
    colors = plt.cm.viridis(norm(acc_sorted))
    
    bars = plt.bar(emotions_sorted, acc_sorted, color=colors, edgecolor='black', alpha=0.9)
    
    # 添加数值标签
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                 f'{height:.1f}', ha='center', va='bottom', fontsize=9, rotation=0)

    plt.xlabel('Emotion Categories', fontsize=14, fontweight='bold')
    plt.ylabel('Accuracy (%)', fontsize=14, fontweight='bold')
    plt.title('Per-Class Emotion Accuracy Distribution (Real Data)', fontsize=16, fontweight='bold')
    plt.xticks(rotation=90, fontsize=11)
    plt.grid(axis='y', linestyle='--', alpha=0.4)
    plt.tight_layout()
    
    save_path = os.path.join(SAVE_DIR, 'emotion_accuracy_distribution.png')
    plt.savefig(save_path, dpi=300)
    print(f"准确率图已保存至: {save_path}")

def plot_confusion_matrix(y_true, y_pred, emotions_list):
    """绘制混淆矩阵热力图"""
    print("正在绘制混淆矩阵...")
    
    # 计算混淆矩阵 (归一化到 0-1 之间，显示比例)
    cm = confusion_matrix(y_true, y_pred, labels=emotions_list, normalize='true')

    plt.figure(figsize=(22, 20))
    # 使用 Blues 色系，深色代表数值大
    sns.heatmap(cm, annot=False, cmap='Blues', 
                xticklabels=emotions_list, yticklabels=emotions_list,
                cbar_kws={'label': 'Proportion'})
    
    plt.xlabel('Predicted Label', fontsize=18, fontweight='bold')
    plt.ylabel('True Label', fontsize=18, fontweight='bold')
    plt.title('Confusion Matrix of Emotion Decoding', fontsize=22, fontweight='bold')
    plt.xticks(rotation=90, fontsize=12)
    plt.yticks(rotation=0, fontsize=12)
    plt.tight_layout()
    
    save_path = os.path.join(SAVE_DIR, 'emotion_confusion_matrix.png')
    plt.savefig(save_path, dpi=300)
    print(f"混淆矩阵已保存至: {save_path}")

if __name__ == "__main__":
    # 1. 检查依赖
    try:
        import matplotlib
        import seaborn
        import sklearn
    except ImportError as e:
        print("缺少必要的库，请先运行: pip install matplotlib seaborn scikit-learn")
        exit()

    # 2. 解析数据
    if not os.path.exists(LOG_FILE_PATH):
        print(f"错误: 找不到文件 {LOG_FILE_PATH}，请修改脚本中的 LOG_FILE_PATH 变量。")
    else:
        y_true, y_pred = parse_log_file(LOG_FILE_PATH)
        
        # 获取所有唯一的标签列表（排序后作为坐标轴）
        unique_emotions = sorted(list(set(y_true)))
        print(f"共检测到 {len(unique_emotions)} 类情感。")

        # 3. 绘图
        # 设置全局字体，防止中文乱码（如果是英文论文保持默认即可）
        plt.rcParams.update({'font.size': 12, 'font.family': 'serif'})
        
        plot_accuracy_distribution(y_true, y_pred, unique_emotions)
        plot_confusion_matrix(y_true, y_pred, unique_emotions)
        
        print("\n所有绘图任务完成！")