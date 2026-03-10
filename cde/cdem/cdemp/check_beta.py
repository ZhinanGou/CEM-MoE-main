import torch
import os
import sys

# ================= 配置区域 =================
# 请将下面的路径修改为您实际保存的模型文件路径
# 通常在 save/ 文件夹下，文件名类似 CEM_迭代次数_PPL.tar
model_path = "save/cCEM_21999_42.4063/CEM_21999_42.4063" 
# ===========================================

def load_and_check_beta():
    if not os.path.exists(model_path):
        print(f"错误：找不到文件 {model_path}")
        print("请检查路径是否正确，或使用绝对路径。")
        return

    print(f"正在加载模型：{model_path} ...")
    
    try:
        # 加载模型文件 (map_location='cpu' 保证即使没有 GPU 也能读取)
        checkpoint = torch.load(model_path, map_location='cpu')
        
        # 1. 检查是否包含 'model' key (这是 model1.py save_model 函数定义的结构)
        if 'model' in checkpoint:
            model_state = checkpoint['model']
            
            # 2. 查找 beta 参数
            if 'beta' in model_state:
                beta_tensor = model_state['beta']
                beta_value = beta_tensor.item() # 转为 python float
                
                # 3. 计算经过 Sigmoid 后的真实权重
                # 注意：代码中 beta 定义为 nn.Parameter(torch.tensor(0.5))
                # 并在 forward 中使用了 self.beta * x ...
                # 等等！回看您的代码：
                # self.beta = nn.Parameter(torch.tensor(0.5)) 
                # z = self.beta * x + (1 - self.beta) * y
                # 在您的代码中，self.beta 直接就是权重值本身（没有经过 sigmoid）
                # 如果您按照论文里写的加了 Sigmoid，这里就要用 torch.sigmoid(beta_value)
                # 按照您上传的 model1 (1).py 原始内容：它是直接使用的，没有 Sigmoid。
                
                print("-" * 30)
                print(f"【读取成功】")
                print(f"模型文件: {os.path.basename(model_path)}")
                print(f"Beta 参数值: {beta_value:.6f}")
                print("-" * 30)
                
                # 简单的分析打印
                if beta_value > 0.5:
                    print(f"分析: Beta ({beta_value:.4f}) > 0.5，模型更倾向于【对话时间/长度】(Time Feature)。")
                else:
                    print(f"分析: Beta ({beta_value:.4f}) < 0.5，模型更倾向于【交互轮次】(Turn Feature)。")
                    
            else:
                print("错误：在模型参数中未找到 'beta'。请确认该模型是否是包含 Learnable Beta 的版本。")
        else:
            # 可能是直接保存的 state_dict 而不是字典
            if 'beta' in checkpoint:
                 print(f"Beta 参数值: {checkpoint['beta'].item():.6f}")
            else:
                 print("错误：未识别的模型文件结构。")

    except Exception as e:
        print(f"读取过程中发生错误: {e}")

if __name__ == "__main__":
    load_and_check_beta()