import numpy as np

class Activation:
    """激活函数基类"""
    def forward(self, x):
        raise NotImplementedError
    
    def backward(self, dout):
        raise NotImplementedError


class ReLU(Activation):
    """ReLU激活函数: f(x) = max(0, x)"""
    def forward(self, x):
        self.x = x
        return np.maximum(0, x)
    
    def backward(self, dout):
        dx = dout.copy()
        dx[self.x <= 0] = 0
        return dx


class Sigmoid(Activation):
    """Sigmoid激活函数: f(x) = 1 / (1 + exp(-x))"""
    def forward(self, x):
        self.out = 1 / (1 + np.exp(-x))
        return self.out
    
    def backward(self, dout):
        return dout * self.out * (1 - self.out)


class Tanh(Activation):
    """Tanh激活函数: f(x) = tanh(x)"""
    def forward(self, x):
        self.out = np.tanh(x)
        return self.out
    
    def backward(self, dout):
        return dout * (1 - self.out ** 2)


class Softmax(Activation):
    """Softmax激活函数，通常用于多分类问题的输出层"""
    def forward(self, x):
        # 为了数值稳定性，减去每一行的最大值
        x_shifted = x - np.max(x, axis=1, keepdims=True)
        exp_x = np.exp(x_shifted)
        self.out = exp_x / np.sum(exp_x, axis=1, keepdims=True)
        return self.out
    
    def backward(self, dout):
        # 通常与交叉熵损失一起使用，所以这里返回恒等变换
        # 实际梯度会在交叉熵损失中计算
        return dout


def get_activation(activation_name):
    """根据名称获取激活函数实例"""
    activation_map = {
        'relu': ReLU(),
        'sigmoid': Sigmoid(),
        'tanh': Tanh(),
        'softmax': Softmax()
    }
    
    if activation_name.lower() not in activation_map:
        raise ValueError(f"不支持的激活函数: {activation_name}")
    
    return activation_map[activation_name.lower()]

