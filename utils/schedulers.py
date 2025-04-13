import numpy as np

class LRScheduler:
    """学习率调度器基类"""
    def __init__(self, optimizer):
        self.optimizer = optimizer
        self.initial_lr = optimizer.learning_rate
        self.current_lr = optimizer.learning_rate
    
    def step(self, epoch=None):
        """更新学习率
        
        参数:
            epoch: 当前迭代轮数
        """
        raise NotImplementedError
    
    def get_lr(self):
        """获取当前学习率"""
        return self.current_lr


class ConstantLR(LRScheduler):
    """常数学习率"""
    def step(self, epoch=None):
        """更新学习率
        
        参数:
            epoch: 当前迭代轮数
        """
        pass  # 不更新学习率


class LinearDecayLR(LRScheduler):
    """线性衰减学习率调度器
    
    学习率随着训练轮数线性衰减，从初始学习率降到最小学习率
    """
    def __init__(self, optimizer, total_epochs, min_lr=0.0):
        """
        初始化线性衰减学习率调度器
        
        参数:
            optimizer: 优化器实例
            total_epochs: 总训练轮数
            min_lr: 最小学习率
        """
        super().__init__(optimizer)
        self.total_epochs = total_epochs
        self.min_lr = min_lr
    
    def step(self, epoch):
        """更新学习率
        
        参数:
            epoch: 当前迭代轮数
        """
        # 计算线性衰减后的学习率
        decay_factor = max(0, 1 - epoch / self.total_epochs)
        self.current_lr = self.min_lr + (self.initial_lr - self.min_lr) * decay_factor
        
        # 更新优化器的学习率
        self.optimizer.learning_rate = self.current_lr


class CosineDecayLR(LRScheduler):
    """余弦衰减学习率调度器
    
    学习率按余弦函数从初始值衰减到最小值
    """
    def __init__(self, optimizer, total_epochs, min_lr=0.0):
        """
        初始化余弦衰减学习率调度器
        
        参数:
            optimizer: 优化器实例
            total_epochs: 总训练轮数
            min_lr: 最小学习率
        """
        super().__init__(optimizer)
        self.total_epochs = total_epochs
        self.min_lr = min_lr
    
    def step(self, epoch):
        """更新学习率
        
        参数:
            epoch: 当前迭代轮数
        """
        # 计算余弦衰减后的学习率
        cosine_decay = 0.5 * (1 + np.cos(np.pi * epoch / self.total_epochs))
        self.current_lr = self.min_lr + (self.initial_lr - self.min_lr) * cosine_decay
        
        # 更新优化器的学习率
        self.optimizer.learning_rate = self.current_lr


class WarmupLR(LRScheduler):
    """Warmup学习率调度器
    
    先从很小的学习率线性增加到初始学习率，然后使用另一个调度器继续
    """
    def __init__(self, optimizer, warmup_epochs, after_scheduler=None):
        """
        初始化Warmup学习率调度器
        
        参数:
            optimizer: 优化器实例
            warmup_epochs: 预热训练的轮数
            after_scheduler: 预热后使用的调度器
        """
        super().__init__(optimizer)
        self.warmup_epochs = warmup_epochs
        self.after_scheduler = after_scheduler
        self.min_lr = self.initial_lr * 0.1  # 预热起始学习率为初始学习率的10%
    
    def step(self, epoch):
        """更新学习率
        
        参数:
            epoch: 当前迭代轮数
        """
        if epoch < self.warmup_epochs:
            # 线性预热阶段
            alpha = epoch / self.warmup_epochs
            self.current_lr = self.min_lr + alpha * (self.initial_lr - self.min_lr)
        else:
            # 使用预热后的调度器
            if self.after_scheduler is not None:
                self.after_scheduler.step(epoch - self.warmup_epochs)
                self.current_lr = self.after_scheduler.get_lr()
        
        # 更新优化器的学习率
        self.optimizer.learning_rate = self.current_lr


def get_scheduler(scheduler_name, optimizer, **kwargs):
    """
    根据名称获取学习率调度器实例
    
    参数:
        scheduler_name: 调度器名称，'constant', 'linear', 'cosine', 'warmup'
        optimizer: 优化器实例
        **kwargs: 传递给调度器构造函数的关键字参数
        
    返回:
        调度器实例
    """
    scheduler_map = {
        'constant': ConstantLR,
        'linear': LinearDecayLR,
        'cosine': CosineDecayLR,
        'warmup': WarmupLR
    }
    
    if scheduler_name.lower() not in scheduler_map:
        raise ValueError(f"不支持的学习率调度器: {scheduler_name}")
    
    if scheduler_name.lower() == 'warmup' and 'after_scheduler' not in kwargs:
        # 如果是warmup但没有指定after_scheduler，默认使用余弦衰减
        total_epochs = kwargs.get('total_epochs', 100)
        min_lr = kwargs.get('min_lr', 0.0)
        warmup_epochs = kwargs.get('warmup_epochs', 5)
        
        after_scheduler = CosineDecayLR(optimizer, total_epochs - warmup_epochs, min_lr)
        return WarmupLR(optimizer, warmup_epochs, after_scheduler)
    
    # 根据调度器类型提取相应的参数
    if scheduler_name.lower() == 'constant':
        return ConstantLR(optimizer)
    elif scheduler_name.lower() == 'linear':
        total_epochs = kwargs.get('total_epochs', 100)
        min_lr = kwargs.get('min_lr', 0.0)
        return LinearDecayLR(optimizer, total_epochs, min_lr)
    elif scheduler_name.lower() == 'cosine':
        total_epochs = kwargs.get('total_epochs', 100)
        min_lr = kwargs.get('min_lr', 0.0)
        return CosineDecayLR(optimizer, total_epochs, min_lr)
    elif scheduler_name.lower() == 'warmup':
        warmup_epochs = kwargs.get('warmup_epochs', 5)
        after_scheduler = kwargs.get('after_scheduler')
        return WarmupLR(optimizer, warmup_epochs, after_scheduler)
