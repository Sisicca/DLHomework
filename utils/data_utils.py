import numpy as np
import pickle
import os
import matplotlib.pyplot as plt

def unpickle(file):
    """
    Load CIFAR-10 dataset file
    
    Args:
        file: Path to CIFAR-10 data file
        
    Returns:
        Loaded data dictionary
    """
    with open(file, 'rb') as fo:
        data = pickle.load(fo, encoding='bytes')
    return data


def load_cifar10(data_dir='cifar-10-batches-py'):
    """
    Load CIFAR-10 dataset
    
    Args:
        data_dir: CIFAR-10 dataset directory
        
    Returns:
        X_train: Training images data, shape (50000, 3, 32, 32)
        y_train: Training labels, shape (50000,)
        X_test: Test images data, shape (10000, 3, 32, 32)
        y_test: Test labels, shape (10000,)
    """
    # Load training data
    X_train = []
    y_train = []
    
    for i in range(1, 6):
        batch_file = os.path.join(data_dir, f'data_batch_{i}')
        batch_data = unpickle(batch_file)
        
        if i == 1:
            X_train = batch_data[b'data']
            y_train = batch_data[b'labels']
        else:
            X_train = np.vstack((X_train, batch_data[b'data']))
            y_train = np.hstack((y_train, batch_data[b'labels']))
    
    # Load test data
    test_file = os.path.join(data_dir, 'test_batch')
    test_data = unpickle(test_file)
    X_test = test_data[b'data']
    y_test = np.array(test_data[b'labels'])
    
    # Reshape data
    X_train = X_train.reshape(-1, 3, 32, 32)
    X_test = X_test.reshape(-1, 3, 32, 32)
    
    # Get class names
    meta_file = os.path.join(data_dir, 'batches.meta')
    meta_data = unpickle(meta_file)
    class_names = [name.decode('utf-8') for name in meta_data[b'label_names']]
    
    return X_train, y_train, X_test, y_test, class_names


def preprocess_data(X_train, y_train, X_test, y_test, validation_split=0.1, flatten=True, normalize=True):
    """
    Preprocess CIFAR-10 dataset
    
    Args:
        X_train: Training images data
        y_train: Training labels
        X_test: Test images data
        y_test: Test labels
        validation_split: Validation set ratio
        flatten: Whether to flatten image data
        normalize: Whether to normalize data
        
    Returns:
        Processed training, validation and test data
    """
    # Split training and validation sets
    num_train = len(X_train)
    num_val = int(num_train * validation_split)
    num_train -= num_val
    
    # Randomly shuffle data
    indices = np.random.permutation(len(X_train))
    train_idx, val_idx = indices[:num_train], indices[num_train:]
    
    X_val = X_train[val_idx]
    y_val = y_train[val_idx]
    X_train = X_train[train_idx]
    y_train = y_train[train_idx]
    
    # Convert data type to float32
    X_train = X_train.astype(np.float32)
    X_val = X_val.astype(np.float32)
    X_test = X_test.astype(np.float32)
    
    # Normalize
    if normalize:
        # Calculate mean and std for each channel
        mean = np.mean(X_train, axis=(0, 2, 3), keepdims=True)
        std = np.std(X_train, axis=(0, 2, 3), keepdims=True)
        
        X_train = (X_train - mean) / (std + 1e-7)
        X_val = (X_val - mean) / (std + 1e-7)
        X_test = (X_test - mean) / (std + 1e-7)
    
    # Flatten image data for MLP
    if flatten:
        X_train = X_train.reshape(X_train.shape[0], -1)
        X_val = X_val.reshape(X_val.shape[0], -1)
        X_test = X_test.reshape(X_test.shape[0], -1)
    
    return X_train, y_train, X_val, y_val, X_test, y_test


class DataLoader:
    """
    Data loader for batch loading
    """
    def __init__(self, X, y, batch_size=64, shuffle=True):
        """
        Initialize data loader
        
        Args:
            X: Feature data
            y: Label data
            batch_size: Batch size
            shuffle: Whether to shuffle data in each epoch
        """
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.n_samples = X.shape[0]
        self.indices = np.arange(self.n_samples)
        self.idx = 0
    
    def __iter__(self):
        """Iterator initialization"""
        self.idx = 0
        if self.shuffle:
            np.random.shuffle(self.indices)
        return self
    
    def __next__(self):
        """Get next batch"""
        if self.idx >= self.n_samples:
            # All samples have been traversed, raise StopIteration
            raise StopIteration
        
        # Calculate current batch indices
        batch_indices = self.indices[self.idx:min(self.idx + self.batch_size, self.n_samples)]
        self.idx += self.batch_size
        
        # Return current batch data
        return self.X[batch_indices], self.y[batch_indices]
    
    def __len__(self):
        """Calculate number of batches"""
        return (self.n_samples + self.batch_size - 1) // self.batch_size


def visualize_samples(X, y, class_names, num_samples=10, save_path=None):
    """
    Visualize CIFAR-10 image samples
    
    Args:
        X: Image data
        y: Labels
        class_names: List of class names
        num_samples: Number of samples to display per class
        save_path: Path to save the plot (if None, plot will be displayed only)
    """
    # Ensure X has shape (N, 3, 32, 32)
    if X.ndim == 2:
        X = X.reshape(-1, 3, 32, 32)
    
    # Adjust channel order to (N, 32, 32, 3) for matplotlib display
    X = np.transpose(X, (0, 2, 3, 1))
    
    # Select samples from each class
    samples_per_class = []
    for i in range(10):
        indices = np.where(y == i)[0]
        if len(indices) > 0:
            selected_indices = indices[:min(num_samples, len(indices))]
            samples_per_class.append((X[selected_indices], [class_names[i]] * len(selected_indices)))
    
    # Create image grid
    fig, axes = plt.subplots(len(samples_per_class), num_samples, figsize=(15, 15))
    
    for i, (class_samples, class_labels) in enumerate(samples_per_class):
        for j in range(min(num_samples, len(class_samples))):
            ax = axes[i, j]
            ax.imshow(class_samples[j])
            if j == 0:  # Add class name only in the first column
                ax.set_ylabel(class_labels[j], fontsize=12, fontweight='bold')
            ax.set_xticks([])
            ax.set_yticks([])
    
    plt.suptitle("CIFAR-10 Sample Images", fontsize=16, fontweight='bold', y=0.92)
    plt.tight_layout()
    
    # Save plot if save_path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Sample images saved to: {save_path}")
    
    plt.show()


def plot_training_history(history, save_path=None):
    """
    Plot and save training history
    
    Args:
        history: Dictionary containing training metrics 
        save_path: Path to save the plot (if None, plot will be displayed only)
    """
    epochs = range(1, len(history['train_loss']) + 1)
    
    plt.figure(figsize=(16, 6))
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Plot loss curves
    plt.subplot(1, 3, 1)
    plt.plot(epochs, history['train_loss'], 'o-', color='#4285F4', label='Train Loss', linewidth=2, markersize=4)
    plt.plot(epochs, history['val_loss'], 'o-', color='#EA4335', label='Validation Loss', linewidth=2, markersize=4)
    plt.title('Training and Validation Loss', fontsize=14, fontweight='bold')
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # Plot accuracy curves
    plt.subplot(1, 3, 2)
    plt.plot(epochs, history['train_acc'], 'o-', color='#4285F4', label='Train Accuracy', linewidth=2, markersize=4)
    plt.plot(epochs, history['val_acc'], 'o-', color='#EA4335', label='Validation Accuracy', linewidth=2, markersize=4)
    plt.title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # Plot learning rate
    if 'lr' in history:
        plt.subplot(1, 3, 3)
        plt.plot(epochs, history['lr'], 'o-', color='#34A853', linewidth=2, markersize=4)
        plt.title('Learning Rate', fontsize=14, fontweight='bold')
        plt.xlabel('Epochs', fontsize=12)
        plt.ylabel('Learning Rate', fontsize=12)
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot if save_path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training history plot saved to: {save_path}")
    
    plt.show()


def visualize_weights(weights, class_names, title="First Layer Weights Visualization", save_path=None):
    """
    Visualize network weights
    
    Args:
        weights: Weight matrix to visualize
        class_names: List of class names
        title: Plot title
        save_path: Path to save the plot (if None, plot will be displayed only)
    """
    plt.style.use('seaborn-v0_8-white')
    
    # For MLP weights visualization
    if weights.ndim == 2:
        num_classes = min(10, weights.shape[0])
        num_features = weights.shape[1]
        
        if num_features == 3072:  # 32x32x3 flattened image
            # Reshape weights to image format
            weights_images = weights.reshape(weights.shape[0], 3, 32, 32)
            weights_images = np.transpose(weights_images, (0, 2, 3, 1))
            
            # Visualize weights of first neurons
            fig, axes = plt.subplots(2, 5, figsize=(15, 6))
            axes = axes.flatten()
            
            for i in range(num_classes):
                # 增强对比度的归一化方法
                weight_img = weights_images[i]
                
                # 计算数据范围的百分位数，用于消除异常值影响
                vmin, vmax = np.percentile(weight_img, [1, 99])
                
                # 使用更强的对比度
                normalized_img = np.clip((weight_img - vmin) / (vmax - vmin + 1e-10), 0, 1)
                
                # 使用更鲜明的颜色映射
                axes[i].imshow(normalized_img, cmap='viridis')
                axes[i].set_title(f'Class: {class_names[i]}', fontsize=12)
                axes[i].axis('off')
            
            plt.suptitle(title, fontsize=16, fontweight='bold', y=0.95)
            fig.tight_layout()
            
    # For CNN filter visualization
    elif weights.ndim == 4:  # (num_filters, channels, height, width)
        num_filters = min(16, weights.shape[0])
        fig, axes = plt.subplots(4, 4, figsize=(12, 12))
        
        for i in range(num_filters):
            ax = axes[i // 4, i % 4]
            # Average across channels for each filter
            filter_img = np.mean(weights[i], axis=0)
            
            # 增强对比度的归一化方法
            vmin, vmax = np.percentile(filter_img, [1, 99])
            normalized_img = np.clip((filter_img - vmin) / (vmax - vmin + 1e-10), 0, 1)
            
            im = ax.imshow(normalized_img, cmap='plasma')
            ax.set_title(f'Filter {i+1}', fontsize=12)
            ax.axis('off')
        
        # Add a colorbar
        fig.subplots_adjust(right=0.85)
        cbar_ax = fig.add_axes([0.88, 0.15, 0.03, 0.7])
        fig.colorbar(im, cax=cbar_ax)
        
        plt.suptitle(title, fontsize=16, fontweight='bold', y=0.92)
        
    # Save plot if save_path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Weights visualization saved to: {save_path}")
    
    plt.show()
