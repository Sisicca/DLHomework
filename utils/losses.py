import numpy as np

class Loss:
    """损失函数基类"""
    def forward(self, y_pred, y_true):
        raise NotImplementedError
    
    def backward(self):
        raise NotImplementedError


class CrossEntropyLoss(Loss):
    """交叉熵损失函数，通常与Softmax激活函数一起使用
    
    公式: L = -sum(y_true * log(y_pred))
    其中y_true是one-hot编码的标签，y_pred是模型预测的概率分布
    """
    def forward(self, y_pred, y_true):
        """
        计算交叉熵损失
        
        参数:
            y_pred: 预测概率, 形状(batch_size, num_classes)
            y_true: 真实标签，可以是one-hot编码或类别索引
            
        返回:
            交叉熵损失值
        """
        self.batch_size = y_pred.shape[0]
        
        # 处理非one-hot编码的标签
        if y_true.ndim == 1 or y_true.shape[1] == 1:
            y_true_one_hot = np.zeros_like(y_pred)
            if y_true.ndim == 1:
                y_true_one_hot[np.arange(self.batch_size), y_true] = 1
            else:
                y_true_one_hot[np.arange(self.batch_size), y_true.flatten()] = 1
            self.y_true = y_true_one_hot
        else:
            self.y_true = y_true
        
        # 添加一个小的常数epsilon，防止log(0)
        epsilon = 1e-15
        y_pred_clipped = np.clip(y_pred, epsilon, 1 - epsilon)
        self.y_pred = y_pred_clipped
        
        # 计算交叉熵损失
        # 注意：这里直接使用真实标签的one-hot编码，只考虑正确类别的负对数概率
        log_probs = -np.log(y_pred_clipped)
        loss = np.sum(self.y_true * log_probs) / self.batch_size
        
        return loss
    
    def backward(self):
        """
        计算交叉熵损失的梯度
        
        注意：如果输入已经经过softmax处理，则梯度简化为(y_pred - y_true)
        
        返回:
            损失函数关于预测值的梯度, 形状(batch_size, num_classes)
        """
        # 交叉熵与softmax的组合梯度
        dout = (self.y_pred - self.y_true) / self.batch_size
        return dout


class L2Regularization:
    """L2正则化，用于防止过拟合
    
    公式: L_reg = 0.5 * lambda * sum(w^2)
    """
    def __init__(self, reg_strength=0.0):
        """
        初始化L2正则化
        
        参数:
            reg_strength: 正则化强度，lambda值
        """
        self.reg_strength = reg_strength
    
    def forward(self, weights):
        """
        计算L2正则化损失
        
        参数:
            weights: 模型权重列表
            
        返回:
            L2正则化损失值
        """
        reg_loss = 0.0
        if self.reg_strength > 0:
            for w in weights:
                reg_loss += 0.5 * self.reg_strength * np.sum(w ** 2)
        return reg_loss
    
    def backward(self, weights):
        """
        计算L2正则化损失的梯度
        
        参数:
            weights: 模型权重列表
            
        返回:
            每个权重矩阵的梯度字典，键为权重索引，值为对应梯度
        """
        grads = {}
        if self.reg_strength > 0:
            for i, w in enumerate(weights):
                # 对应的梯度为lambda * w，保持与权重相同的形状
                grads[i] = self.reg_strength * w
                
        return grads
