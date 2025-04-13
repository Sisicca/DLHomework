import numpy as np

class Optimizer:
    """优化器基类"""
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate
    
    def update(self, params, grads):
        raise NotImplementedError


class SGD(Optimizer):
    """随机梯度下降优化器
    
    参数更新规则: w = w - learning_rate * dw
    """
    def __init__(self, learning_rate=0.01, momentum=0.0):
        """
        初始化SGD优化器
        
        参数:
            learning_rate: 学习率
            momentum: 动量系数，用于加速训练
        """
        super().__init__(learning_rate)
        self.momentum = momentum
        self.velocities = {}
    
    def update(self, params, grads):
        """
        更新参数
        
        参数:
            params: 需要更新的参数字典
            grads: 参数的梯度字典
        """
        for key in params:
            # 初始化速度
            if key not in self.velocities:
                self.velocities[key] = np.zeros_like(params[key])
            
            # 使用动量更新速度
            self.velocities[key] = self.momentum * self.velocities[key] - self.learning_rate * grads[key]
            
            # 更新参数
            params[key] += self.velocities[key]


class Adam(Optimizer):
    """Adam优化器 (Adaptive Moment Estimation)
    
    结合了动量和RMSProp的思想，自适应调整每个参数的学习率
    """
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        """
        初始化Adam优化器
        
        参数:
            learning_rate: 学习率
            beta1: 一阶矩估计的指数衰减率
            beta2: 二阶矩估计的指数衰减率
            epsilon: 防止除零的小常数
        """
        super().__init__(learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = {}  # 一阶矩估计
        self.v = {}  # 二阶矩估计
        self.t = 0   # 时间步
    
    def update(self, params, grads):
        """
        更新参数
        
        参数:
            params: 需要更新的参数字典
            grads: 参数的梯度字典
        """
        self.t += 1
        
        for key in params:
            # 初始化一阶和二阶矩估计
            if key not in self.m:
                self.m[key] = np.zeros_like(params[key])
                self.v[key] = np.zeros_like(params[key])
            
            # 更新一阶矩估计
            self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * grads[key]
            
            # 更新二阶矩估计
            self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * (grads[key] ** 2)
            
            # 计算偏差修正后的一阶和二阶矩估计
            m_corrected = self.m[key] / (1 - self.beta1 ** self.t)
            v_corrected = self.v[key] / (1 - self.beta2 ** self.t)
            
            # 更新参数
            params[key] -= self.learning_rate * m_corrected / (np.sqrt(v_corrected) + self.epsilon)


def get_optimizer(optimizer_name, **kwargs):
    """
    根据名称获取优化器实例
    
    参数:
        optimizer_name: 优化器名称，'sgd'或'adam'
        **kwargs: 传递给优化器构造函数的关键字参数
        
    返回:
        优化器实例
    """
    optimizer_map = {
        'sgd': SGD,
        'adam': Adam
    }
    
    if optimizer_name.lower() not in optimizer_map:
        raise ValueError(f"不支持的优化器: {optimizer_name}")
    
    return optimizer_map[optimizer_name.lower()](**kwargs)
