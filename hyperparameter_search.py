import numpy as np
import os
import argparse
import json
import time
import copy
from itertools import product
import matplotlib.pyplot as plt
from utils.data_utils import load_cifar10, preprocess_data, DataLoader
from utils.losses import CrossEntropyLoss, L2Regularization
from utils.optimizers import get_optimizer
from utils.schedulers import get_scheduler
from models.mlp import MLP
from models.cnn import CNN

def compute_accuracy(y_pred, y_true):
    """
    计算分类准确率
    
    参数:
        y_pred: 模型预测，形状为(batch_size, num_classes)
        y_true: 真实标签，形状为(batch_size,)
        
    返回:
        准确率
    """
    pred_classes = np.argmax(y_pred, axis=1)
    return np.mean(pred_classes == y_true)

def evaluate_model(model, data_loader, criterion, l2_reg):
    """
    评估模型性能
    
    参数:
        model: 模型实例
        data_loader: 数据加载器
        criterion: 损失函数
        l2_reg: L2正则化
        
    返回:
        loss: 平均损失
        accuracy: 平均准确率
    """
    loss = 0.0
    accuracy = 0.0
    
    for X_batch, y_batch in data_loader:
        # 前向传播
        y_pred = model.forward(X_batch)
        
        # 计算损失
        batch_loss = criterion.forward(y_pred, y_batch)
        
        # 添加L2正则化
        reg_loss = l2_reg.forward(model.get_weights())
        total_loss = batch_loss + reg_loss
        
        # 计算准确率
        batch_accuracy = compute_accuracy(y_pred, y_batch)
        
        # 累计损失和准确率
        loss += total_loss
        accuracy += batch_accuracy
    
    # 计算平均损失和准确率
    loss /= len(data_loader)
    accuracy /= len(data_loader)
    
    return loss, accuracy

def train_model(model, train_loader, val_loader, optimizer, scheduler, criterion, l2_reg, num_epochs, early_stopping=10):
    """
    训练模型
    
    参数:
        model: 模型实例
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        optimizer: 优化器
        scheduler: 学习率调度器
        criterion: 损失函数
        l2_reg: L2正则化
        num_epochs: 训练轮数
        early_stopping: 早停耐心值
        
    返回:
        best_val_accuracy: 最佳验证准确率
        history: 训练历史记录
    """
    # 训练历史记录
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'lr': []
    }
    
    # 用于早停的变量
    best_val_loss = float('inf')
    best_val_accuracy = 0.0
    best_model_params = None
    patience_counter = 0
    
    # 训练循环
    for epoch in range(num_epochs):
        # 更新学习率
        scheduler.step(epoch)
        current_lr = scheduler.get_lr()
        history['lr'].append(current_lr)
        
        # 训练阶段
        model.train = True
        train_loss = 0.0
        train_acc = 0.0
        
        # 遍历训练批次
        for X_batch, y_batch in train_loader:
            # 前向传播
            y_pred = model.forward(X_batch)
            
            # 计算损失
            loss = criterion.forward(y_pred, y_batch)
            
            # 添加L2正则化
            reg_loss = l2_reg.forward(model.get_weights())
            total_loss = loss + reg_loss
            
            # 计算准确率
            accuracy = compute_accuracy(y_pred, y_batch)
            
            # 反向传播
            dout = criterion.backward()
            model.backward(dout)
            
            # 添加L2正则化梯度
            params, grads = model.get_params_and_grads()
            reg_grads = l2_reg.backward(model.get_weights())
            
            # 更新梯度（添加正则化梯度）
            for i, layer in enumerate(model.layers):
                if hasattr(layer, 'params') and 'W' in layer.params:
                    layer.grads['W'] += reg_grads.get(i, 0)
            
            # 优化器更新参数
            optimizer.update(params, grads)
            
            # 累计损失和准确率
            train_loss += total_loss
            train_acc += accuracy
        
        # 计算训练集平均损失和准确率
        train_loss /= len(train_loader)
        train_acc /= len(train_loader)
        
        # 验证阶段
        model.train = False
        val_loss, val_acc = evaluate_model(model, val_loader, criterion, l2_reg)
        
        # 保存训练历史
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # 打印epoch结果
        if (epoch + 1) % 5 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}] - "
                f"Train Loss: {train_loss:.4f} - Train Acc: {train_acc:.4f} - "
                f"Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.4f}")
        
        # 保存最佳模型
        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            best_val_loss = val_loss
            patience_counter = 0
            # 保存最佳模型参数
            best_model_params = copy.deepcopy(model.get_params_and_grads()[0])
        else:
            patience_counter += 1
        
        # 早停
        if early_stopping > 0 and patience_counter >= early_stopping:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    # 恢复最佳模型参数
    if best_model_params is not None:
        for i, layer in enumerate(model.layers):
            if hasattr(layer, 'params'):
                for param_name in layer.params:
                    key = f"layer{i}_{param_name}"
                    if key in best_model_params:
                        layer.params[param_name] = best_model_params[key]
    
    return best_val_accuracy, history

def hyperparameter_search(args):
    """
    执行超参数搜索
    
    参数:
        args: 命令行参数
    """
    # 设置随机种子
    np.random.seed(args.seed)
    
    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)
    
    # 加载数据集
    print("正在加载CIFAR-10数据集...")
    X_train, y_train, X_test, y_test, class_names = load_cifar10(args.data_dir)
    
    # 预处理数据
    print("正在预处理数据...")
    # MLP需要展平图像，CNN保持原始形状
    flatten = (args.model == 'mlp')
    X_train, y_train, X_val, y_val, X_test, y_test = preprocess_data(
        X_train, y_train, X_test, y_test, validation_split=args.val_split, flatten=flatten)
    
    print(f"数据形状 - 训练集: {X_train.shape}, 验证集: {X_val.shape}, 测试集: {X_test.shape}")
    
    # 创建数据加载器
    train_loader = DataLoader(X_train, y_train, batch_size=args.batch_size)
    val_loader = DataLoader(X_val, y_val, batch_size=args.batch_size)
    
    # 定义要搜索的超参数空间
    if args.model == 'mlp':
        hyperparams = {
            'hidden_dims': [
                [128, 64],
                [256, 128],
                [512, 128],
                [1024, 256]
            ],
            'learning_rate': [0.01, 0.001, 0.0001],
            'weight_scale': [1e-2, 1e-3, 1e-4],
            'reg_strength': [0.0, 1e-4, 1e-3, 1e-2],
            'activation': ['relu', 'sigmoid', 'tanh']
        }
    else:  # CNN
        hyperparams = {
            'num_filters': [
                [16, 32],
                [32, 64],
                [64, 128]
            ],
            'hidden_dim': [64, 128, 256],
            'learning_rate': [0.01, 0.001, 0.0001],
            'weight_scale': [1e-2, 1e-3, 1e-4],
            'reg_strength': [0.0, 1e-4, 1e-3],
            'activation': ['relu', 'sigmoid', 'tanh']
        }
    
    # 确定搜索空间大小
    param_combinations = []
    for param_values in product(*hyperparams.values()):
        param_dict = dict(zip(hyperparams.keys(), param_values))
        param_combinations.append(param_dict)
    
    print(f"超参数搜索空间大小: {len(param_combinations)}")
    
    # 创建损失函数
    criterion = CrossEntropyLoss()
    
    # 存储结果
    results = []
    
    for i, params in enumerate(param_combinations):
        print(f"\n超参数组合 {i+1}/{len(param_combinations)}:")
        print(json.dumps(params, indent=2))
        
        # 创建模型
        if args.model == 'mlp':
            model = MLP(
                input_dim=X_train.shape[1],
                hidden_dims=params['hidden_dims'],
                num_classes=10,
                activation=params['activation'],
                weight_scale=params['weight_scale']
            )
        else:  # CNN
            model = CNN(
                input_shape=(3, 32, 32),
                num_filters=params['num_filters'],
                kernel_sizes=[5, 3],  # 固定卷积核大小
                hidden_dim=params['hidden_dim'],
                num_classes=10,
                activation=params['activation'],
                weight_scale=params['weight_scale']
            )
        
        # 创建L2正则化
        l2_reg = L2Regularization(reg_strength=params['reg_strength'])
        
        # 创建优化器 - 修复参数传递问题
        if args.optimizer.lower() == 'sgd':
            optimizer = get_optimizer(args.optimizer, learning_rate=params['learning_rate'], 
                                   momentum=args.momentum)
        else:  # adam
            optimizer = get_optimizer(args.optimizer, learning_rate=params['learning_rate'], 
                                   beta1=args.beta1, beta2=args.beta2)
        
        # 创建学习率调度器
        scheduler = get_scheduler(args.scheduler, optimizer, total_epochs=args.search_epochs, 
                                min_lr=args.min_lr, warmup_epochs=args.warmup_epochs)
        
        # 训练模型
        start_time = time.time()
        best_val_accuracy, history = train_model(
            model, train_loader, val_loader, optimizer, scheduler, criterion, l2_reg,
            args.search_epochs, early_stopping=args.patience
        )
        training_time = time.time() - start_time
        
        # 存储结果
        result = {
            'params': params,
            'best_val_accuracy': float(best_val_accuracy),
            'training_time': training_time,
            'final_train_loss': float(history['train_loss'][-1]),
            'final_train_acc': float(history['train_acc'][-1]),
            'final_val_loss': float(history['val_loss'][-1]),
            'final_val_acc': float(history['val_acc'][-1]),
        }
        results.append(result)
        
        print(f"验证准确率: {best_val_accuracy:.4f}, 训练时间: {training_time:.2f}秒")
    
    # 按验证准确率排序结果
    results.sort(key=lambda x: x['best_val_accuracy'], reverse=True)
    
    # 打印最佳超参数
    print("\n最佳超参数组合:")
    print(json.dumps(results[0]['params'], indent=2))
    print(f"验证准确率: {results[0]['best_val_accuracy']:.4f}")
    
    # 保存所有结果
    results_path = os.path.join(args.save_dir, f"{args.model}_hyperparam_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n超参数搜索结果已保存到: {results_path}")
    
    # 绘制超参数影响图
    plot_hyperparam_effects(args.model, results, args.save_dir)
    
    return results[0]['params']

def plot_hyperparam_effects(model_type, results, save_dir):
    """
    绘制超参数对验证准确率的影响
    
    参数:
        model_type: 模型类型
        results: 超参数搜索结果
        save_dir: 保存目录
    """
    # 提取参数和准确率
    params_to_plot = ['learning_rate', 'reg_strength', 'weight_scale']
    
    # 每个参数的所有不同值
    param_values = {}
    for param in params_to_plot:
        param_values[param] = sorted(list(set([r['params'][param] for r in results])))
    
    # 绘制超参数影响
    fig, axes = plt.subplots(1, len(params_to_plot), figsize=(15, 5))
    
    for i, param in enumerate(params_to_plot):
        # 按参数值分组计算平均准确率
        avg_acc = []
        for value in param_values[param]:
            matching_results = [r['best_val_accuracy'] for r in results if r['params'][param] == value]
            avg_acc.append(sum(matching_results) / len(matching_results))
        
        # 绘制参数影响图
        axes[i].plot(range(len(param_values[param])), avg_acc, 'o-')
        axes[i].set_xlabel(param)
        axes[i].set_ylabel('平均验证准确率')
        axes[i].set_title(f'{param}对准确率的影响')
        axes[i].set_xticks(range(len(param_values[param])))
        axes[i].set_xticklabels([str(v) for v in param_values[param]])
        
        # 添加最佳值标记
        best_idx = avg_acc.index(max(avg_acc))
        axes[i].scatter([best_idx], [avg_acc[best_idx]], color='red', s=100, zorder=5)
        axes[i].text(best_idx, avg_acc[best_idx], f'最佳: {param_values[param][best_idx]}', 
                    fontsize=10, ha='center')
    
    plt.tight_layout()
    
    # 保存图像
    plot_path = os.path.join(save_dir, f"{model_type}_hyperparam_effects.png")
    plt.savefig(plot_path)
    print(f"超参数影响图已保存到: {plot_path}")

def parse_args():
    parser = argparse.ArgumentParser(description='CIFAR-10分类器超参数搜索')
    
    # 数据参数
    parser.add_argument('--data_dir', type=str, default='cifar-10-batches-py', help='数据集目录')
    parser.add_argument('--val_split', type=float, default=0.1, help='验证集比例')
    parser.add_argument('--batch_size', type=int, default=128, help='批大小')
    
    # 模型参数
    parser.add_argument('--model', type=str, default='mlp', choices=['mlp', 'cnn'], help='模型类型')
    
    # 训练参数
    parser.add_argument('--search_epochs', type=int, default=20, help='搜索过程中的训练轮数')
    parser.add_argument('--optimizer', type=str, default='sgd', choices=['sgd', 'adam'], help='优化器')
    parser.add_argument('--momentum', type=float, default=0.9, help='SGD动量')
    parser.add_argument('--beta1', type=float, default=0.9, help='Adam一阶矩估计的指数衰减率')
    parser.add_argument('--beta2', type=float, default=0.999, help='Adam二阶矩估计的指数衰减率')
    parser.add_argument('--scheduler', type=str, default='constant', 
                       choices=['constant', 'linear', 'cosine', 'warmup'], help='学习率调度器')
    parser.add_argument('--min_lr', type=float, default=1e-4, help='最小学习率')
    parser.add_argument('--warmup_epochs', type=int, default=5, help='预热训练的轮数')
    
    # 其他参数
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--save_dir', type=str, default='results', help='保存结果的目录')
    parser.add_argument('--patience', type=int, default=5, help='早停耐心值，0表示不使用早停')
    
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    best_params = hyperparameter_search(args)
