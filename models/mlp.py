import numpy as np
from models.layers import FullyConnected
from utils.activations import get_activation

class MLP:
    """三层多层感知机神经网络分类器"""
    def __init__(self, input_dim=3072, hidden_dims=[512, 128], num_classes=10, 
                 activation='relu', weight_scale=1e-3):
        """
        初始化三层MLP模型
        
        参数:
            input_dim: 输入维度，默认为CIFAR-10的3*32*32=3072
            hidden_dims: 隐藏层维度列表，长度为1或2
            num_classes: 分类数量，默认为CIFAR-10的10个类别
            activation: 激活函数类型，'relu', 'sigmoid', 或 'tanh'
            weight_scale: 权重初始化的缩放因子
        """
        self.params = {}
        self.grads = {}
        self.layers = []
        self.activations = []
        
        # 确保hidden_dims正确
        if len(hidden_dims) not in [1, 2]:
            raise ValueError("MLP必须有1到2个隐藏层")
        
        # 第一层：输入层到第一个隐藏层
        self.layers.append(FullyConnected(input_dim, hidden_dims[0], weight_scale))
        self.activations.append(get_activation(activation))
        
        # 如果有第二个隐藏层
        if len(hidden_dims) == 2:
            self.layers.append(FullyConnected(hidden_dims[0], hidden_dims[1], weight_scale))
            self.activations.append(get_activation(activation))
            last_hidden_dim = hidden_dims[1]
        else:
            last_hidden_dim = hidden_dims[0]
        
        # 输出层
        self.layers.append(FullyConnected(last_hidden_dim, num_classes, weight_scale))
        
        # Softmax输出（在损失函数中实现）
        self.activations.append(get_activation('softmax'))
    
    def forward(self, X):
        """
        前向传播
        
        参数:
            X: 输入数据，形状为(batch_size, input_dim)
            
        返回:
            输出预测，形状为(batch_size, num_classes)
        """
        # 逐层前向传播
        out = X
        
        for i in range(len(self.layers)):
            # 线性层
            out = self.layers[i].forward(out)
            
            # 激活函数
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
            # 激活函数反向传播
            dout = self.activations[i].backward(dout)
            
            # 线性层反向传播
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
            for param_name, param in layer.params.items():
                params[f"layer{i}_{param_name}"] = param
                grads[f"layer{i}_{param_name}"] = layer.grads[param_name]
        
        return params, grads
    
    def predict(self, X):
        """
        预测类别
        
        参数:
            X: 输入数据，形状为(batch_size, input_dim)
            
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
            for param_name in layer.params:
                key = f"layer{i}_{param_name}"
                if key in params:
                    layer.params[param_name] = params[key]
