import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchaudio
from torch.utils.data import Dataset, DataLoader

class FullReferenceHead(nn.Module):
    """
    完全参考（Full Reference, FR）头，用于计算基于完全参考信号的输出。
    输入是测试信号和参考信号的嵌入（embedding），输出是它们之间的匹配度。
    """
    def __init__(self, embedding_dim=200):
        """
        初始化完全参考头的结构，包含两个全连接层。
        :param embedding_dim: 嵌入向量的维度（默认为200）
        """
        super().__init__()
        # 第一个全连接层，将输入的embedding维度（2 * embedding_dim）映射到64维
        self.fc1 = nn.Linear(embedding_dim * 2, 64)
        
        # ReLU激活函数
        self.relu = nn.ReLU()
        
        # 第二个全连接层，将64维映射到1维，输出FR得分
        self.fc2 = nn.Linear(64, 1)
        
    def forward(self, e_i, e_j):
        """
        前向传播函数：计算完全参考得分。
        :param e_i: 测试信号的嵌入（embedding）
        :param e_j: 参考信号的嵌入（embedding）
        :return: 完全参考得分
        """
        # 将测试信号和参考信号的嵌入在维度1（即特征维度）上进行拼接
        concat = torch.cat([e_i, e_j], dim=1)
        
        # 通过第一个全连接层进行处理
        out = self.fc1(concat)
        
        # 通过ReLU激活函数
        out = self.relu(out)
        
        # 通过第二个全连接层得到最终输出
        out = self.fc2(out)
        
        # 返回结果，并去掉多余的维度（压缩维度1）
        return out.squeeze()  # 输出一个标量
