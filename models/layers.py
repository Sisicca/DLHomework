import numpy as np

class Layer:
    """神经网络层的基类"""
    def __init__(self):
        self.params = {}
        self.grads = {}
    
    def forward(self, x):
        raise NotImplementedError
    
    def backward(self, dout):
        raise NotImplementedError


class FullyConnected(Layer):
    """全连接层（线性层）"""
    def __init__(self, input_dim, output_dim, weight_scale=1e-3):
        """
        初始化全连接层
        
        参数:
            input_dim: 输入维度
            output_dim: 输出维度
            weight_scale: 权重初始化的缩放因子
        """
        super().__init__()
        
        # 使用高斯分布初始化权重
        self.params['W'] = weight_scale * np.random.randn(input_dim, output_dim)
        # 初始化偏置为0
        self.params['b'] = np.zeros(output_dim)
        
        # 保存输入数据，用于反向传播
        self.x = None
    
    def forward(self, x):
        """
        前向传播
        
        参数:
            x: 输入数据，形状为(batch_size, input_dim)
            
        返回:
            输出数据，形状为(batch_size, output_dim)
        """
        # 保存输入，用于反向传播
        self.x = x
        
        # 前向传播计算
        out = np.dot(x, self.params['W']) + self.params['b']
        
        return out
    
    def backward(self, dout):
        """
        反向传播
        
        参数:
            dout: 输出梯度，形状为(batch_size, output_dim)
            
        返回:
            输入梯度，形状为(batch_size, input_dim)
        """
        # 计算权重和偏置的梯度
        self.grads['W'] = np.dot(self.x.T, dout)
        self.grads['b'] = np.sum(dout, axis=0)
        
        # 计算输入梯度
        dx = np.dot(dout, self.params['W'].T)
        
        return dx


class Conv2D(Layer):
    """二维卷积层"""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, weight_scale=1e-3):
        """
        初始化卷积层
        
        参数:
            in_channels: 输入通道数
            out_channels: 输出通道数（卷积核数量）
            kernel_size: 卷积核大小（假设是正方形）
            stride: 步长
            padding: 填充大小
            weight_scale: 权重初始化的缩放因子
        """
        super().__init__()
        
        # 卷积参数
        self.stride = stride
        self.padding = padding
        self.kernel_size = kernel_size
        
        # 初始化权重和偏置
        self.params['W'] = weight_scale * np.random.randn(out_channels, in_channels, kernel_size, kernel_size)
        self.params['b'] = np.zeros(out_channels)
        
        # 保存中间结果，用于反向传播
        self.x = None
        self.x_padded = None
    
    def forward(self, x):
        """
        前向传播
        
        参数:
            x: 输入数据，形状为(batch_size, in_channels, height, width)
            
        返回:
            输出数据，形状为(batch_size, out_channels, out_height, out_width)
        """
        # 获取输入尺寸
        N, C, H, W = x.shape
        
        # 获取卷积核尺寸
        F, _, HH, WW = self.params['W'].shape
        
        # 计算输出尺寸
        out_height = (H + 2 * self.padding - HH) // self.stride + 1
        out_width = (W + 2 * self.padding - WW) // self.stride + 1
        
        # 初始化输出数组
        out = np.zeros((N, F, out_height, out_width))
        
        # 保存输入，用于反向传播
        self.x = x
        
        # 填充输入
        if self.padding > 0:
            self.x_padded = np.pad(x, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)), 
                                   mode='constant')
        else:
            self.x_padded = x
        
        # 执行卷积操作
        for n in range(N):  # 遍历每个样本
            for f in range(F):  # 遍历每个卷积核
                for i in range(0, out_height):
                    for j in range(0, out_width):
                        # 提取当前窗口
                        h_start = i * self.stride
                        h_end = h_start + HH
                        w_start = j * self.stride
                        w_end = w_start + WW
                        
                        # 卷积操作（逐元素相乘并求和）
                        window = self.x_padded[n, :, h_start:h_end, w_start:w_end]
                        out[n, f, i, j] = np.sum(window * self.params['W'][f]) + self.params['b'][f]
        
        return out
    
    def backward(self, dout):
        """
        反向传播
        
        参数:
            dout: 输出梯度，形状为(batch_size, out_channels, out_height, out_width)
            
        返回:
            输入梯度，形状为(batch_size, in_channels, height, width)
        """
        # 获取各种尺寸
        N, F, out_height, out_width = dout.shape
        _, C, H, W = self.x.shape
        _, _, HH, WW = self.params['W'].shape
        
        # 初始化梯度
        dW = np.zeros_like(self.params['W'])
        db = np.zeros_like(self.params['b'])
        dx_padded = np.zeros_like(self.x_padded)
        
        # 计算偏置梯度
        db = np.sum(dout, axis=(0, 2, 3))
        
        # 计算权重梯度和输入梯度
        for n in range(N):
            for f in range(F):
                for i in range(out_height):
                    for j in range(out_width):
                        # 计算窗口位置
                        h_start = i * self.stride
                        h_end = h_start + HH
                        w_start = j * self.stride
                        w_end = w_start + WW
                        
                        # 更新权重梯度
                        window = self.x_padded[n, :, h_start:h_end, w_start:w_end]
                        dW[f] += window * dout[n, f, i, j]
                        
                        # 更新输入梯度
                        dx_padded[n, :, h_start:h_end, w_start:w_end] += self.params['W'][f] * dout[n, f, i, j]
        
        # 保存梯度
        self.grads['W'] = dW
        self.grads['b'] = db
        
        # 如果有填充，需要去除填充部分
        if self.padding > 0:
            dx = dx_padded[:, :, self.padding:-self.padding, self.padding:-self.padding]
        else:
            dx = dx_padded
        
        return dx


class MaxPool2D(Layer):
    """最大池化层"""
    def __init__(self, pool_size=2, stride=2):
        """
        初始化最大池化层
        
        参数:
            pool_size: 池化窗口大小
            stride: 步长
        """
        super().__init__()
        
        self.pool_size = pool_size
        self.stride = stride
        
        # 保存中间结果，用于反向传播
        self.x = None
        self.max_indices = None
    
    def forward(self, x):
        """
        前向传播
        
        参数:
            x: 输入数据，形状为(batch_size, channels, height, width)
            
        返回:
            输出数据，形状为(batch_size, channels, out_height, out_width)
        """
        # 获取输入尺寸
        N, C, H, W = x.shape
        
        # 计算输出尺寸
        out_height = (H - self.pool_size) // self.stride + 1
        out_width = (W - self.pool_size) // self.stride + 1
        
        # 初始化输出数组
        out = np.zeros((N, C, out_height, out_width))
        
        # 保存最大值索引，用于反向传播
        self.max_indices = np.zeros((N, C, out_height, out_width, 2), dtype=int)
        
        # 保存输入，用于反向传播
        self.x = x
        
        # 执行最大池化操作
        for n in range(N):
            for c in range(C):
                for i in range(out_height):
                    for j in range(out_width):
                        # 提取当前窗口
                        h_start = i * self.stride
                        h_end = h_start + self.pool_size
                        w_start = j * self.stride
                        w_end = w_start + self.pool_size
                        
                        window = x[n, c, h_start:h_end, w_start:w_end]
                        
                        # 找到最大值
                        max_val = np.max(window)
                        out[n, c, i, j] = max_val
                        
                        # 保存最大值的位置
                        max_idx = np.argmax(window)
                        max_h, max_w = np.unravel_index(max_idx, (self.pool_size, self.pool_size))
                        self.max_indices[n, c, i, j] = [max_h, max_w]
        
        return out
    
    def backward(self, dout):
        """
        反向传播
        
        参数:
            dout: 输出梯度，形状为(batch_size, channels, out_height, out_width)
            
        返回:
            输入梯度，形状为(batch_size, channels, height, width)
        """
        # 获取各种尺寸
        N, C, out_height, out_width = dout.shape
        _, _, H, W = self.x.shape
        
        # 初始化输入梯度
        dx = np.zeros_like(self.x)
        
        # 反向传播最大池化
        for n in range(N):
            for c in range(C):
                for i in range(out_height):
                    for j in range(out_width):
                        # 计算窗口位置
                        h_start = i * self.stride
                        h_end = h_start + self.pool_size
                        w_start = j * self.stride
                        w_end = w_start + self.pool_size
                        
                        # 获取最大值位置
                        max_h, max_w = self.max_indices[n, c, i, j]
                        
                        # 将梯度传递给最大值位置
                        dx[n, c, h_start + max_h, w_start + max_w] += dout[n, c, i, j]
        
        return dx


class Flatten(Layer):
    """展平层，将多维输入展平为二维输出"""
    def __init__(self):
        super().__init__()
        self.x_shape = None
    
    def forward(self, x):
        """
        前向传播
        
        参数:
            x: 输入数据，形状为(batch_size, channels, height, width)或其他多维形状
            
        返回:
            展平后的输出数据，形状为(batch_size, features)
        """
        # 保存输入形状，用于反向传播
        self.x_shape = x.shape
        
        # 展平操作，保持第一维不变
        out = x.reshape(x.shape[0], -1)
        
        return out
    
    def backward(self, dout):
        """
        反向传播
        
        参数:
            dout: 输出梯度，形状为(batch_size, features)
            
        返回:
            输入梯度，形状与原始输入形状相同
        """
        # 恢复原始输入形状
        dx = dout.reshape(self.x_shape)
        
        return dx
