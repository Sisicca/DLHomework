import numpy as np
from models.layers import Conv2D, MaxPool2D, Flatten, FullyConnected
from utils.activations import get_activation

class CNN:
    """三层卷积神经网络分类器"""
    def __init__(self, input_shape=(3, 32, 32), num_filters=[32, 64], kernel_sizes=[5, 3], 
                 hidden_dim=128, num_classes=10, activation='relu', weight_scale=1e-3):
        """
        初始化三层CNN模型
        
        参数:
            input_shape: 输入形状，默认为CIFAR-10的(3, 32, 32)
            num_filters: 卷积层滤波器数量列表
            kernel_sizes: 卷积核大小列表
            hidden_dim: 全连接隐藏层维度
            num_classes: 分类数量，默认为CIFAR-10的10个类别
            activation: 激活函数类型，'relu', 'sigmoid', 或 'tanh'
            weight_scale: 权重初始化的缩放因子
        """
        self.params = {}
        self.grads = {}
        self.layers = []
        self.activations = []
        
        # 确保层数正确
        if len(num_filters) != 2 or len(kernel_sizes) != 2:
            raise ValueError("CNN必须有2个卷积层")
        
        # 获取输入维度
        C, H, W = input_shape
        
        # 第一卷积层
        self.layers.append(Conv2D(C, num_filters[0], kernel_sizes[0], stride=1, padding=2, weight_scale=weight_scale))
        self.activations.append(get_activation(activation))
        
        # 第一池化层
        self.layers.append(MaxPool2D(pool_size=2, stride=2))
        self.activations.append(None)  # 池化层没有激活函数
        
        # 计算第一池化层输出尺寸（假设padding=2，stride=1）
        H1 = (H + 2*2 - kernel_sizes[0]) // 1 + 1
        W1 = (W + 2*2 - kernel_sizes[0]) // 1 + 1
        
        # 池化后的尺寸
        H1_pool = (H1 - 2) // 2 + 1
        W1_pool = (W1 - 2) // 2 + 1
        
        # 第二卷积层
        self.layers.append(Conv2D(num_filters[0], num_filters[1], kernel_sizes[1], stride=1, padding=1, weight_scale=weight_scale))
        self.activations.append(get_activation(activation))
        
        # 第二池化层
        self.layers.append(MaxPool2D(pool_size=2, stride=2))
        self.activations.append(None)  # 池化层没有激活函数
        
        # 计算第二池化层输出尺寸（假设padding=1，stride=1）
        H2 = (H1_pool + 2*1 - kernel_sizes[1]) // 1 + 1
        W2 = (W1_pool + 2*1 - kernel_sizes[1]) // 1 + 1
        
        # 池化后的尺寸
        H2_pool = (H2 - 2) // 2 + 1
        W2_pool = (W2 - 2) // 2 + 1
        
        # 展平层
        self.layers.append(Flatten())
        self.activations.append(None)  # 展平层没有激活函数
        
        # 计算展平后的维度
        flatten_dim = num_filters[1] * H2_pool * W2_pool
        
        # 全连接隐藏层
        self.layers.append(FullyConnected(flatten_dim, hidden_dim, weight_scale))
        self.activations.append(get_activation(activation))
        
        # 输出层
        self.layers.append(FullyConnected(hidden_dim, num_classes, weight_scale))
        
        # Softmax输出（在损失函数中实现）
        self.activations.append(get_activation('softmax'))
    
    def forward(self, X):
        """
        前向传播
        
        参数:
            X: 输入数据，形状为(batch_size, channels, height, width)
            
        返回:
            输出预测，形状为(batch_size, num_classes)
        """
        # 逐层前向传播
        out = X
        
        for i in range(len(self.layers)):
            # 层的前向传播
            out = self.layers[i].forward(out)
            
            # 如果有激活函数，应用它
            if self.activations[i] is not None:
                out = self.activations[i].forward(out)
        
        return out
    
    def backward(self, dout):
        """
        反向传播
        
        参数:
            dout: 输出层梯度
            
        返回:
            输入梯度
        """
        # 逆序执行反向传播
        for i in reversed(range(len(self.layers))):
            # 如果有激活函数，先计算激活函数的反向传播
            if self.activations[i] is not None:
                dout = self.activations[i].backward(dout)
            
            # 层的反向传播
            dout = self.layers[i].backward(dout)
        
        return dout
    
    def get_weights(self):
        """
        获取模型的所有权重参数
        
        返回:
            权重列表
        """
        weights = []
        for layer in self.layers:
            if hasattr(layer, 'params') and 'W' in layer.params:
                weights.append(layer.params['W'])
        return weights
    
    def get_params_and_grads(self):
        """
        获取模型的所有参数和梯度
        
        返回:
            params: 参数字典
            grads: 梯度字典
        """
        params = {}
        grads = {}
        
        for i, layer in enumerate(self.layers):
            if hasattr(layer, 'params'):
                for param_name, param in layer.params.items():
                    params[f"layer{i}_{param_name}"] = param
                    grads[f"layer{i}_{param_name}"] = layer.grads[param_name]
        
        return params, grads
    
    def predict(self, X):
        """
        预测类别
        
        参数:
            X: 输入数据，形状为(batch_size, channels, height, width)
            
        返回:
            预测的类别索引，形状为(batch_size,)
        """
        probs = self.forward(X)
        return np.argmax(probs, axis=1)
    
    def save(self, filename):
        """
        保存模型参数
        
        参数:
            filename: 保存文件名
        """
        params, _ = self.get_params_and_grads()
        np.save(filename, params)
    
    def load(self, filename):
        """
        加载模型参数
        
        参数:
            filename: 加载文件名
        """
        params = np.load(filename, allow_pickle=True).item()
        
        for i, layer in enumerate(self.layers):
            if hasattr(layer, 'params'):
                for param_name in layer.params:
                    key = f"layer{i}_{param_name}"
                    if key in params:
                        layer.params[param_name] = params[key]
