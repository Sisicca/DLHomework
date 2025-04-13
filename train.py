import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
import time
import pickle
from utils.data_utils import load_cifar10, preprocess_data, DataLoader, plot_training_history, visualize_weights
from utils.losses import CrossEntropyLoss, L2Regularization
from utils.optimizers import get_optimizer
from utils.schedulers import get_scheduler
from models.mlp import MLP
from models.cnn import CNN

def compute_accuracy(y_pred, y_true):
    """
    Compute classification accuracy
    
    Args:
        y_pred: Model predictions, shape (batch_size, num_classes)
        y_true: True labels, shape (batch_size,)
        
    Returns:
        Accuracy
    """
    pred_classes = np.argmax(y_pred, axis=1)
    return np.mean(pred_classes == y_true)

def train(args):
    """
    Train the model
    
    Args:
        args: Command line arguments
    """
    # Set random seed
    np.random.seed(args.seed)
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Load dataset
    print("Loading CIFAR-10 dataset...")
    X_train, y_train, X_test, y_test, class_names = load_cifar10(args.data_dir)
    
    # Preprocess data
    print("Preprocessing data...")
    # MLP requires flattened images, CNN keeps original shape
    flatten = (args.model == 'mlp')
    X_train, y_train, X_val, y_val, X_test, y_test = preprocess_data(
        X_train, y_train, X_test, y_test, validation_split=args.val_split, flatten=flatten)
    
    print(f"Data shapes - Training: {X_train.shape}, Validation: {X_val.shape}, Test: {X_test.shape}")
    
    # Create data loaders
    train_loader = DataLoader(X_train, y_train, batch_size=args.batch_size)
    val_loader = DataLoader(X_val, y_val, batch_size=args.batch_size)
    
    # Create model
    if args.model == 'mlp':
        model = MLP(
            input_dim=X_train.shape[1],
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
        raise ValueError(f"Unsupported model type: {args.model}")
    
    # Create loss function
    criterion = CrossEntropyLoss()
    
    # Create L2 regularization
    l2_reg = L2Regularization(reg_strength=args.reg_strength)
    
    # Create optimizer
    if args.optimizer.lower() == 'sgd':
        optimizer = get_optimizer(args.optimizer, learning_rate=args.learning_rate, 
                               momentum=args.momentum)
    else:  # adam
        optimizer = get_optimizer(args.optimizer, learning_rate=args.learning_rate, 
                               beta1=args.beta1, beta2=args.beta2)
    
    # Create learning rate scheduler
    scheduler = get_scheduler(args.scheduler, optimizer, total_epochs=args.num_epochs, 
                            min_lr=args.min_lr, warmup_epochs=args.warmup_epochs)
    
    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'lr': []
    }
    
    # Variables for early stopping
    best_val_acc = 0.0
    best_epoch = 0
    patience_counter = 0
    best_model = None
    
    # Create visualization directory
    vis_dir = os.path.join(args.save_dir, 'visualizations')
    os.makedirs(vis_dir, exist_ok=True)
    
    # Training loop
    print(f"Start training {args.model.upper()} model...")
    for epoch in range(args.num_epochs):
        # Update learning rate
        scheduler.step(epoch)
        current_lr = scheduler.get_lr()
        history['lr'].append(current_lr)
        
        # Training phase
        model.train = True
        train_loss = 0.0
        train_acc = 0.0
        
        start_time = time.time()
        
        # Iterate through training batches
        for batch_idx, (X_batch, y_batch) in enumerate(train_loader):
            # Forward pass
            y_pred = model.forward(X_batch)
            
            # Calculate loss
            loss = criterion.forward(y_pred, y_batch)
            
            # Add L2 regularization
            reg_loss = l2_reg.forward(model.get_weights())
            total_loss = loss + reg_loss
            
            # Calculate accuracy
            accuracy = compute_accuracy(y_pred, y_batch)
            
            # Backward pass
            dout = criterion.backward()
            model.backward(dout)
            
            # Add L2 regularization gradients
            params, grads = model.get_params_and_grads()
            reg_grads = l2_reg.backward(model.get_weights())
            
            # 修复正则化梯度应用方式
            reg_idx = 0  # 用于跟踪正则化梯度的索引
            for layer in model.layers:
                if hasattr(layer, 'params') and 'W' in layer.params:
                    # 确保使用的正则化梯度形状与层权重形状匹配
                    if reg_idx in reg_grads:
                        # 获取该层对应的正则化梯度
                        reg_grad = reg_grads[reg_idx]
                        # 确保形状一致
                        if reg_grad.shape == layer.grads['W'].shape:
                            layer.grads['W'] += reg_grad
                        else:
                            print(f"警告: 跳过形状不匹配的正则化梯度，层形状: {layer.grads['W'].shape}, 梯度形状: {reg_grad.shape}")
                    reg_idx += 1
            
            # Optimizer updates parameters
            optimizer.update(params, grads)
            
            # Accumulate loss and accuracy
            train_loss += total_loss
            train_acc += accuracy
            
            # Print batch progress
            if (batch_idx + 1) % 50 == 0:
                print(f"Epoch [{epoch+1}/{args.num_epochs}] Batch [{batch_idx+1}/{len(train_loader)}] "
                      f"Loss: {total_loss:.4f} Acc: {accuracy:.4f} LR: {current_lr:.6f}")
        
        # Calculate average training loss and accuracy
        train_loss /= len(train_loader)
        train_acc /= len(train_loader)
        
        # Validation phase
        model.train = False
        val_loss = 0.0
        val_acc = 0.0
        
        # Iterate through validation batches
        for X_batch, y_batch in val_loader:
            # Forward pass
            y_pred = model.forward(X_batch)
            
            # Calculate loss
            loss = criterion.forward(y_pred, y_batch)
            
            # Add L2 regularization
            reg_loss = l2_reg.forward(model.get_weights())
            total_loss = loss + reg_loss
            
            # Calculate accuracy
            accuracy = compute_accuracy(y_pred, y_batch)
            
            # Accumulate loss and accuracy
            val_loss += total_loss
            val_acc += accuracy
        
        # Calculate average validation loss and accuracy
        val_loss /= len(val_loader)
        val_acc /= len(val_loader)
        
        # Save training history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # Print epoch results
        epoch_time = time.time() - start_time
        print(f"Epoch [{epoch+1}/{args.num_epochs}] - {epoch_time:.2f}s - "
              f"Train Loss: {train_loss:.4f} - Train Acc: {train_acc:.4f} - "
              f"Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.4f}")
        
        # Save best model based on validation accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            patience_counter = 0
            
            # Save model parameters
            model_path = os.path.join(args.save_dir, f"{args.model}_best.npy")
            model.save(model_path)
            print(f"Best model saved to {model_path}")
            
            # Keep a copy of the best model
            best_model = model
            
            # Save best model to pickle file (for easier loading)
            best_model_path = os.path.join(args.save_dir, "best_model.pkl")
            with open(best_model_path, 'wb') as f:
                pickle.dump(model, f)
            print(f"Best model saved to {best_model_path}")
        else:
            patience_counter += 1
        
        # Early stopping
        if args.patience > 0 and patience_counter >= args.patience:
            print(f"Early stopping at epoch {epoch+1}, best epoch: {best_epoch+1}")
            break
    
    # Training completed
    print(f"Training completed! Best validation accuracy: {best_val_acc:.4f}, best epoch: {best_epoch+1}")
    
    # Save training history data as pickle
    history_data_path = os.path.join(args.save_dir, "training_history.pkl")
    with open(history_data_path, 'wb') as f:
        pickle.dump(history, f)
    print(f"Training history data saved to: {history_data_path}")
    
    # Plot and save training history
    print("Plotting training history...")
    history_path = os.path.join(vis_dir, f"{args.model}_training_history.png")
    plot_training_history(history, save_path=history_path)
    
    # Visualize model weights and parameters
    print("Visualizing model weights and parameters...")
    weights = model.get_weights()
    
    # Visualize first layer weights (applicable to both MLP and CNN)
    # if len(weights) > 0:
    #     first_layer_path = os.path.join(vis_dir, "first_layer_weights.png")
    #     visualize_weights(weights[0], class_names, 
    #                      title=f"{args.model.upper()} First Layer Weights", 
    #                      save_path=first_layer_path)
    
    # # For MLP model, visualize the last layer weights
    # if args.model == 'mlp' and len(weights) > 2:
    #     last_layer_path = os.path.join(vis_dir, "last_layer_weights.png")
    #     visualize_weights(weights[-2], class_names, 
    #                      title=f"{args.model.upper()} Output Layer Weights", 
    #                      save_path=last_layer_path)
    
    # Visualize weight distributions
    plt.figure(figsize=(15, 10))
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Create a grid for weight distributions
    n_weights = len(weights)
    cols = 3
    rows = (n_weights + cols - 1) // cols
    
    for i, w in enumerate(weights):
        if w.ndim > 0:  # Skip empty weights
            plt.subplot(rows, cols, i+1)
            
            # Flatten weights for histogram
            flat_weights = w.flatten()
            
            # 移除异常值以更好地显示分布
            q_low, q_high = np.percentile(flat_weights, [1, 99])
            filtered_weights = flat_weights[(flat_weights >= q_low) & (flat_weights <= q_high)]
            
            # Plot histogram with better visibility
            plt.hist(filtered_weights, bins=50, alpha=0.8, color='#4285F4', 
                    edgecolor='black', linewidth=0.5)
            plt.title(f"Layer {i+1} Weight Distribution", fontsize=12)
            plt.xlabel("Weight Value", fontsize=10)
            plt.ylabel("Count", fontsize=10)
            
            # 显示均值和标准差
            mean = np.mean(flat_weights)
            std = np.std(flat_weights)
            plt.axvline(mean, color='red', linestyle='dashed', linewidth=1)
            plt.text(0.95, 0.95, f'Mean: {mean:.6f}\nStd: {std:.6f}', 
                    transform=plt.gca().transAxes, fontsize=8,
                    verticalalignment='top', horizontalalignment='right')
            
            plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    weight_dist_path = os.path.join(vis_dir, "weight_distributions.png")
    plt.savefig(weight_dist_path, dpi=300, bbox_inches='tight')
    print(f"Weight distributions saved to: {weight_dist_path}")
    plt.close()
    
    # For CNN model, visualize all convolutional filters
    if args.model == 'cnn':
        # Visualize first convolutional layer filters
        for i, w in enumerate(weights):
            if w.ndim == 4:  # Convolutional layer weights
                conv_path = os.path.join(vis_dir, f"conv{i+1}_filters.png")
                visualize_weights(w, class_names, 
                                title=f"{args.model.upper()} Conv{i+1} Filters", 
                                save_path=conv_path)
        
        # Visualize fully connected layer weights if present
        for i, w in enumerate(weights):
            if w.ndim == 2 and w.shape[0] > 10:  # Fully connected layer (not output layer)
                fc_path = os.path.join(vis_dir, "fc_weights.png")
                plt.figure(figsize=(12, 8))
                
                # 增强对比度的可视化方法
                vmin, vmax = np.percentile(w, [1, 99])  # 使用百分位数避免异常值影响
                plt.imshow(w, aspect='auto', cmap='plasma', vmin=vmin, vmax=vmax)
                
                plt.colorbar()
                plt.title(f"{args.model.upper()} Fully Connected Layer Weights", fontsize=14)
                plt.xlabel("Input Features", fontsize=12)
                plt.ylabel("Output Features", fontsize=12)
                plt.savefig(fc_path, dpi=300, bbox_inches='tight')
                plt.close()
                print(f"FC layer weights visualization saved to: {fc_path}")
                break
    
    print(f"All visualizations saved to: {vis_dir}")
    
    return model, history

def parse_args():
    parser = argparse.ArgumentParser(description='Train CIFAR-10 Classifier')
    
    # Data parameters
    parser.add_argument('--data_dir', type=str, default='cifar-10-batches-py', help='Dataset directory')
    parser.add_argument('--val_split', type=float, default=0.1, help='Validation set ratio')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    
    # Model parameters
    parser.add_argument('--model', type=str, default='mlp', choices=['mlp', 'cnn'], help='Model type')
    parser.add_argument('--hidden_dims', type=int, nargs='+', default=[512, 128], help='Hidden layer dimensions')
    parser.add_argument('--num_filters', type=int, nargs='+', default=[32, 64], help='Number of CNN filters')
    parser.add_argument('--kernel_sizes', type=int, nargs='+', default=[5, 3], help='CNN kernel sizes')
    parser.add_argument('--activation', type=str, default='relu', choices=['relu', 'sigmoid', 'tanh'], help='Activation function')
    parser.add_argument('--weight_scale', type=float, default=1e-3, help='Weight initialization scale factor')
    
    # Training parameters
    parser.add_argument('--num_epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--optimizer', type=str, default='sgd', choices=['sgd', 'adam'], help='Optimizer')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='Initial learning rate')
    parser.add_argument('--min_lr', type=float, default=1e-4, help='Minimum learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum')
    parser.add_argument('--beta1', type=float, default=0.9, help='Adam beta1 parameter')
    parser.add_argument('--beta2', type=float, default=0.999, help='Adam beta2 parameter')
    parser.add_argument('--scheduler', type=str, default='constant', 
                       choices=['constant', 'linear', 'cosine', 'warmup'], help='Learning rate scheduler')
    parser.add_argument('--warmup_epochs', type=int, default=5, help='Number of warmup epochs')
    parser.add_argument('--reg_strength', type=float, default=1e-4, help='L2 regularization strength')
    
    # Other parameters
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--save_dir', type=str, default='results', help='Directory to save results')
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience, 0 means no early stopping')
    
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    train(args)
