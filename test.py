import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
import time
from utils.data_utils import load_cifar10, preprocess_data, DataLoader, visualize_samples
from models.mlp import MLP
from models.cnn import CNN

def test(args):
    """
    在测试集上评估模型
    
    参数:
        args: 命令行参数
    """
    # 设置随机种子
    np.random.seed(args.seed)
    
    # 加载数据集
    print("正在加载CIFAR-10数据集...")
    X_train, y_train, X_test, y_test, class_names = load_cifar10(args.data_dir)
    
    # 预处理数据
    print("正在预处理数据...")
    # MLP需要展平图像，CNN保持原始形状
    flatten = (args.model == 'mlp')
    X_train, y_train, X_val, y_val, X_test, y_test = preprocess_data(
        X_train, y_train, X_test, y_test, validation_split=args.val_split, flatten=flatten)
    
    print(f"测试集形状: {X_test.shape}")
    
    # 创建数据加载器
    test_loader = DataLoader(X_test, y_test, batch_size=args.batch_size, shuffle=False)
    
    # 创建模型
    if args.model == 'mlp':
        model = MLP(
            input_dim=X_test.shape[1],
            hidden_dims=args.hidden_dims,
            num_classes=10,
            activation=args.activation,
            weight_scale=args.weight_scale
        )
    elif args.model == 'cnn':
        model = CNN(
            input_shape=(3, 32, 32),
            num_filters=args.num_filters,
            kernel_sizes=args.kernel_sizes,
            hidden_dim=args.hidden_dims[-1],
            num_classes=10,
            activation=args.activation,
            weight_scale=args.weight_scale
        )
    else:
        raise ValueError(f"不支持的模型类型: {args.model}")
    
    # 加载模型参数
    model_path = os.path.join(args.save_dir, f"{args.model}_best.npy")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}")
    
    print(f"加载模型参数: {model_path}")
    model.load(model_path)
    
    # 在测试集上评估
    print("在测试集上评估模型...")
    test_accuracy = 0.0
    class_correct = np.zeros(10)
    class_total = np.zeros(10)
    all_preds = []
    all_labels = []
    
    start_time = time.time()
    
    # 遍历测试批次
    for X_batch, y_batch in test_loader:
        # 前向传播
        y_pred = model.forward(X_batch)
        
        # 计算预测类别
        pred_classes = np.argmax(y_pred, axis=1)
        
        # 保存预测和标签
        all_preds.append(pred_classes)
        all_labels.append(y_batch)
        
        # 累计每个类别的正确预测数量
        for i in range(len(y_batch)):
            label = y_batch[i]
            class_total[label] += 1
            if pred_classes[i] == label:
                class_correct[label] += 1
    
    # 合并所有批次的预测和标签
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    
    # 计算总体准确率
    test_accuracy = np.mean(all_preds == all_labels)
    
    # 计算每个类别的准确率
    class_accuracy = class_correct / class_total
    
    # 打印结果
    print(f"\n测试集准确率: {test_accuracy:.4f}")
    print("\n每个类别的准确率:")
    for i in range(10):
        print(f"{class_names[i]}: {class_accuracy[i]:.4f}")
    
    test_time = time.time() - start_time
    print(f"\n测试耗时: {test_time:.2f}秒")
    
    # 混淆矩阵
    conf_matrix = np.zeros((10, 10), dtype=int)
    for i in range(len(all_labels)):
        conf_matrix[all_labels[i], all_preds[i]] += 1
    
    # 创建可视化文件夹
    vis_dir = os.path.join(args.save_dir, 'visualizations')
    os.makedirs(vis_dir, exist_ok=True)
    
    # 绘制混淆矩阵
    plt.figure(figsize=(10, 8))
    plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(10)
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    
    # 添加数字标签
    thresh = conf_matrix.max() / 2.
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            plt.text(j, i, format(conf_matrix[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if conf_matrix[i, j] > thresh else "black")
    
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    # 保存混淆矩阵
    conf_matrix_path = os.path.join(vis_dir, f"{args.model}_confusion_matrix.png")
    plt.savefig(conf_matrix_path)
    print(f"\nConfusion matrix saved to: {conf_matrix_path}")
    plt.show()
    
    # 可视化一些错误分类的样本
    if args.visualize:
        print("\nVisualizing misclassified samples...")
        # 找出错误分类的样本
        error_indices = np.where(all_preds != all_labels)[0]
        
        if len(error_indices) > 0:
            # 选择一些错误样本
            num_samples = min(20, len(error_indices))
            selected_indices = np.random.choice(error_indices, num_samples, replace=False)
            
            # 提取样本
            if flatten:
                # 对于MLP，需要重塑回图像形状
                selected_samples = X_test[selected_indices].reshape(-1, 3, 32, 32)
            else:
                selected_samples = X_test[selected_indices]
            
            # 获取真实标签和预测标签
            true_labels = all_labels[selected_indices]
            pred_labels = all_preds[selected_indices]
            
            # 调整图像通道顺序为(N, 32, 32, 3)，以便matplotlib显示
            selected_samples = np.transpose(selected_samples, (0, 2, 3, 1))
            
            # 创建图像网格
            fig, axes = plt.subplots(4, 5, figsize=(15, 12))
            axes = axes.flatten()
            
            for i in range(num_samples):
                # 计算均值和标准差用于反归一化
                mean = np.mean(selected_samples[i])
                std = np.std(selected_samples[i])
                
                # 可视化图像
                axes[i].imshow(selected_samples[i] * std + mean)
                axes[i].set_title(f"True: {class_names[true_labels[i]]}\nPred: {class_names[pred_labels[i]]}")
                axes[i].axis('off')
            
            plt.tight_layout()
            
            # 保存错误样本图像
            error_path = os.path.join(vis_dir, f"{args.model}_misclassified_samples.png")
            plt.savefig(error_path)
            print(f"Misclassified samples saved to: {error_path}")
            plt.show()
            
            # 可视化每个类别的一些测试样本
            print("\nVisualizing test samples for each class...")
            sample_per_class = 8
            fig, axes = plt.subplots(10, sample_per_class, figsize=(20, 15))
            
            for cls in range(10):
                # 找到这个类别的所有样本
                cls_indices = np.where(all_labels == cls)[0]
                if len(cls_indices) == 0:
                    continue
                
                # 随机选择一些样本
                cls_sample_indices = np.random.choice(cls_indices, 
                                                    min(sample_per_class, len(cls_indices)), 
                                                    replace=False)
                
                # 提取样本
                if flatten:
                    cls_samples = X_test[cls_sample_indices].reshape(-1, 3, 32, 32)
                else:
                    cls_samples = X_test[cls_sample_indices]
                
                # 调整通道顺序
                cls_samples = np.transpose(cls_samples, (0, 2, 3, 1))
                
                # 可视化
                for j, sample_idx in enumerate(range(min(sample_per_class, len(cls_indices)))):
                    if j < len(cls_samples):
                        # 计算均值和标准差用于反归一化
                        mean = np.mean(cls_samples[j])
                        std = np.std(cls_samples[j])
                        
                        # 显示图像
                        axes[cls, j].imshow(cls_samples[j] * std + mean)
                        correct = (all_preds[cls_sample_indices[j]] == cls)
                        color = "green" if correct else "red"
                        axes[cls, j].set_title(f"{class_names[cls]}", color=color, fontsize=8)
                        axes[cls, j].axis('off')
            
            plt.tight_layout()
            samples_path = os.path.join(vis_dir, f"{args.model}_class_samples.png")
            plt.savefig(samples_path)
            print(f"Test samples visualization saved to: {samples_path}")
            plt.show()
    
    print(f"\nAll visualizations saved to: {vis_dir}")
    return test_accuracy, class_accuracy

def parse_args():
    parser = argparse.ArgumentParser(description='测试CIFAR-10分类器')
    
    # 数据参数
    parser.add_argument('--data_dir', type=str, default='cifar-10-batches-py', help='数据集目录')
    parser.add_argument('--val_split', type=float, default=0.1, help='验证集比例')
    parser.add_argument('--batch_size', type=int, default=128, help='批大小')
    
    # 模型参数
    parser.add_argument('--model', type=str, default='mlp', choices=['mlp', 'cnn'], help='模型类型')
    parser.add_argument('--hidden_dims', type=int, nargs='+', default=[512, 128], help='隐藏层维度')
    parser.add_argument('--num_filters', type=int, nargs='+', default=[32, 64], help='CNN滤波器数量')
    parser.add_argument('--kernel_sizes', type=int, nargs='+', default=[5, 3], help='CNN卷积核大小')
    parser.add_argument('--activation', type=str, default='relu', help='激活函数')
    parser.add_argument('--weight_scale', type=float, default=1e-3, help='权重初始化缩放因子')
    
    # 其他参数
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--save_dir', type=str, default='results', help='保存结果的目录')
    parser.add_argument('--visualize', action='store_true', help='是否可视化错误样本')
    
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    test(args)
