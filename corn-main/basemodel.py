import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchaudio
from torch.utils.data import Dataset, DataLoader

class ResidualBlock(nn.Module):
    """
    残差块（Residual Block）定义。包含了卷积层、批归一化、ReLU激活函数和残差连接。
    """
    def __init__(self, in_channels=256, mid_channels=512):
        """
        初始化残差块，定义三个卷积层以及与残差连接相关的参数。
        :param in_channels: 输入通道数（默认256）
        :param mid_channels: 中间层通道数（默认512）
        """
        super().__init__()
        # 批归一化和激活函数
        self.bn1 = nn.BatchNorm1d(in_channels)
        self.relu = nn.ReLU()
        
        # 第一个卷积层
        self.conv1 = nn.Conv1d(in_channels, mid_channels, kernel_size=1)
        self.bn2 = nn.BatchNorm1d(mid_channels)
        
        # 第二个卷积层
        self.conv2 = nn.Conv1d(mid_channels, mid_channels, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(mid_channels)
        
        # 第三个卷积层
        self.conv3 = nn.Conv1d(mid_channels, in_channels, kernel_size=1)
        
        # 残差连接的参数，初始化为6，表示在残差连接中对输入和输出进行加权
        self.a = nn.Parameter(torch.ones(in_channels) * 6)
        self.sigmoid = nn.Sigmoid()  # Sigmoid函数用于将参数映射到[0,1]区间
    
    def forward(self, x):
        """
        前向传播函数。
        :param x: 输入张量
        :return: 残差块的输出
        """
        identity = x  # 保留输入张量用于残差连接
        
        # 第一阶段：批归一化 -> ReLU -> 卷积
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        
        # 第二阶段：批归一化 -> ReLU -> 卷积
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        
        # 第三阶段：批归一化 -> ReLU -> 卷积
        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)
        
        # 残差连接的线性加权（利用Sigmoid函数得到权重）
        weight = self.sigmoid(self.a).view(1, -1, 1)  # 权重在[0,1]之间
        out = weight * identity + (1 - weight) * out  # 加权残差连接
        
        return out

class BlurPool(nn.Module):
    """
    模型中的一个模糊池化层，使用高斯核进行池化操作，用于下采样。
    """
    def __init__(self, channels, stride=2):
        """
        初始化模糊池化层，使用高斯核来进行下采样。
        :param channels: 输入通道数
        :param stride: 步长（默认为2）
        """
        super().__init__()
        self.stride = stride
        
        # 创建一个高斯核，大小为5，形状为[1, 1, 5]，然后复制成输入通道数的大小
        kernel = torch.tensor([0.0625, 0.25, 0.375, 0.25, 0.0625], dtype=torch.float32)
        kernel = kernel.view(1, 1, -1).repeat(channels, 1, 1)
        self.register_buffer('kernel', kernel)  # 将核作为缓冲区注册
        self.pad = nn.ReflectionPad1d(2)  # 反射填充，用于处理边缘
        
    def forward(self, x):
        """
        前向传播函数，执行模糊池化操作。
        :param x: 输入张量
        :return: 下采样后的张量
        """
        x = self.pad(x)  # 对输入进行填充
        x = F.conv1d(x, self.kernel, stride=self.stride, groups=x.shape[1])  # 使用高斯核进行卷积下采样
        return x

class BaseModel(nn.Module):
    """
    基础模型类，包含多个卷积块、池化层、残差块以及全连接层。
    """
    def __init__(self):
        """
        初始化模型，定义网络的各个层：
        1. 首先通过卷积和池化进行特征提取
        2. 然后通过残差块进一步抽取特征
        3. 最后使用MLP进行进一步的计算
        """
        super().__init__()
        # 可学习的μ-law参数
        self.mu = nn.Parameter(torch.tensor(4.0))
        
        # 第一个卷积池化块
        self.conv1 = nn.Conv1d(1, 128, kernel_size=4, stride=2)
        self.bn1 = nn.BatchNorm1d(128)
        self.relu1 = nn.ReLU()
        self.blurpool1 = BlurPool(128, stride=2)
        
        # 第二个卷积池化块
        self.conv2 = nn.Conv1d(128, 256, kernel_size=4, stride=2)
        self.bn2 = nn.BatchNorm1d(256)
        self.relu2 = nn.ReLU()
        self.blurpool2 = BlurPool(256, stride=2)
        
        # 残差块
        self.residual1 = ResidualBlock(256, 512)
        self.residual2 = ResidualBlock(256, 512)
        self.residual3 = ResidualBlock(256, 512)
        
        # MLP层
        self.bn_mlp = nn.BatchNorm1d(512)
        self.fc1 = nn.Linear(512, 1024)
        self.bn_fc1 = nn.BatchNorm1d(1024)
        self.relu_fc1 = nn.ReLU()
        self.fc2 = nn.Linear(1024, 200)
        self.bn_fc2 = nn.BatchNorm1d(200)
        
    def mu_law(self, x):
        """
        实现μ-law压缩，无量化。
        :param x: 输入信号
        :return: 经过μ-law变换后的信号
        """
        mu = torch.abs(self.mu) + 1e-4  # 保证μ是正数
        return torch.sign(x) * torch.log(1 + mu * torch.abs(x)) / torch.log(1 + mu)
        
    def forward(self, x):
        """
        前向传播函数：
        1. 先应用μ-law压缩
        2. 经过卷积池化层、残差块处理
        3. 最后通过MLP得到嵌入向量
        :param x: 输入音频张量
        :return: 模型的嵌入向量
        """
        # 应用μ-law压缩
        x = self.mu_law(x)
        
        # 第一个卷积池化块
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.blurpool1(x)
        
        # 第二个卷积池化块
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.blurpool2(x)
        
        # 通过残差块进行处理
        x = self.residual1(x)
        x = self.residual2(x)
        x = self.residual3(x)
        
        # 计算时域统计量（每个通道的均值和标准差）
        mean = torch.mean(x, dim=2, keepdim=True)
        std = torch.std(x, dim=2, keepdim=True)
        stats = torch.cat([mean, std], dim=1)  # 将均值和标准差拼接在一起
        stats = stats.squeeze(2)  # 去掉最后一个维度
        
        # 执行批归一化
        stats = self.bn_mlp(stats)
        
        # MLP层处理
        stats = self.fc1(stats)
        stats = self.bn_fc1(stats)
        stats = self.relu_fc1(stats)
        embedding = self.fc2(stats)
        embedding = self.bn_fc2(embedding)
        
        return embedding
