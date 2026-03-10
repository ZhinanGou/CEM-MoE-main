# test_size.py
import torch
from src.models.CEM.model1 import CEM  # 对应你的 model1 (1).py
from src.utils import config
from src.utils.data.loader import prepare_data_seq
from src.models.common import count_parameters # 直接用你项目自带的统计函数

def get_model_size():
    print("正在初始化环境和词表...")
    # 1. 模拟数据加载获取 vocab 和 decoder_number (必须和训练时一致)
    # 这步会读取你的数据集，请确保数据路径正确
    _, _, _, vocab, dec_num = prepare_data_seq(batch_size=config.batch_size)
    
    print("正在加载 CEM-MoE 模型...")
    # 2. 初始化模型
    # model1.py 里的类名虽然叫 CEM，但注释显示它是 MoE 版本
    model = CEM(
        vocab,
        decoder_number=dec_num,
        is_eval=True
    )
    
    # 3. 使用你项目自带的工具函数统计
    params = count_parameters(model)
    
    print("\n" + "="*30)
    print(f"模型名称: CEM-MoE (Ours)")
    print(f"总参数量 (Raw): {params}")
    print(f"总参数量 (M): {params / 1e6:.2f} M")
    print("="*30)

if __name__ == "__main__":
    get_model_size()