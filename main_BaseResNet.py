import torch
import torch.nn as nn
import torch.optim as optim
from ResNet18 import create_resnet18_model
from torch.utils.data import DataLoader, Subset, Dataset
from torchvision import transforms, datasets
import numpy as np
import random
import os
import csv
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

##############################
#       超参数配置
##############################
config = {
    # 数据参数
    'data_dir': '/data/coding/AES256_DataSet',
    'img_size': 224,  # ResNet通常使用224x224输入
    'num_workers': 4,
    
    # 数据集划分比例
    'split_ratios': [0.75, 0.15, 0.15],  # 训练集、验证集、测试集比例

    # 训练参数
    'seed': 42,
    'batch_size': 20,
    'num_epochs': 20,
    'learning_rate': 1e-4,
    
    # 预训练模型路径
    'pretrained_model_path': '',  # 为None则重新开始训练模型后测试，若有，则直接使用预训练模型在测试集上测试

    # ResNet18参数
    'use_pretrained': False,  # 是否使用预训练权重
    'num_classes': 20,  # 自动从数据集获取

    # 保存目录
    'save_dir': 'ResNet18'
}

##############################
#       工具函数
##############################
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def compute_mean_std(dataset, batch_size=32, num_workers=4):
    """计算数据集的全局均值和标准差"""
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    total_sum = 0.0
    total_sq_sum = 0.0
    total_pixels = 0
    for images, labels in tqdm(dataloader):
        images = images.float()
        batch_pixels = images.numel()
        total_sum += images.sum().item()
        total_sq_sum += (images ** 2).sum().item()
        total_pixels += batch_pixels
    mean = total_sum / total_pixels
    std = (total_sq_sum / total_pixels - mean ** 2) ** 0.5
    return [mean], [std]

def calculate_metrics(true_labels, pred_labels, average='macro'):
    """计算评估指标"""
    cm = confusion_matrix(true_labels, pred_labels)
    specificity = []
    for i in range(len(cm)):
        tn = cm.sum() - (cm[i, :].sum() + cm[:, i].sum() - cm[i, i])
        fp = cm[:, i].sum() - cm[i, i]
        specificity_i = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        specificity.append(specificity_i)
    specificity_macro = np.mean(specificity)
    return {
        'accuracy': accuracy_score(true_labels, pred_labels),
        'precision': precision_score(true_labels, pred_labels, average=average),
        'recall': recall_score(true_labels, pred_labels, average=average),
        'f1': f1_score(true_labels, pred_labels, average=average),
        'specificity': specificity_macro,
        'confusion_matrix': cm
    }

def test_model(model, test_loader, criterion, device, save_dir):
    """测试模型并保存结果"""
    model.eval()
    test_loss, test_preds, test_labels = 0.0, [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            test_loss += criterion(outputs, labels).item()
            test_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
            test_labels.extend(labels.cpu().numpy())

    test_metrics = calculate_metrics(test_labels, test_preds)

    # 打印测试结果
    print(f"\nTest Loss: {test_loss / len(test_loader):.4f}")
    print(f"Test Accuracy: {test_metrics['accuracy']:.4f}, Precision: {test_metrics['precision']:.4f}, Recall: {test_metrics['recall']:.4f}, F1: {test_metrics['f1']:.4f}, Specificity: {test_metrics['specificity']:.4f}")
    print(f"Test Confusion Matrix:\n{test_metrics['confusion_matrix']}")

    # 保存测试结果
    test_csv_path = os.path.join(save_dir, 'test_metrics.csv')
    with open(test_csv_path, 'w') as f:
        writer = csv.writer(f)
        writer.writerow([
            'test_loss',
            'test_acc', 'test_precision', 'test_recall', 'test_f1', 'test_specificity', 'confusion_matrix'
        ])
        writer.writerow([
            test_loss / len(test_loader),
            test_metrics['accuracy'],
            test_metrics['precision'],
            test_metrics['recall'],
            test_metrics['f1'],
            test_metrics['specificity'],
            str(test_metrics['confusion_matrix'].tolist())
        ])

    print("Test metrics saved to:", test_csv_path)
    return test_metrics

##############################
#       主流程
##############################
if __name__ == "__main__":
    os.makedirs(config['save_dir'], exist_ok=True)
    set_seed(config['seed'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 打印使用的划分比例
    train_ratio, val_ratio, test_ratio = config['split_ratios']
    print(f"使用数据集划分比例: 训练集={train_ratio*100}%, 验证集={val_ratio*100}%, 测试集={test_ratio*100}%")

    # 基础转换
    base_transform = transforms.Compose([
        transforms.Resize((config['img_size'], config['img_size'])),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor()
    ])

    # 创建标准图像数据集
    full_dataset = datasets.ImageFolder(config['data_dir'], transform=base_transform)
    config['num_classes'] = len(full_dataset.class_to_idx)
    print(f"数据集类别数量: {config['num_classes']}")

    ###########################################################
    # 按类别顺序划分数据集
    ###########################################################
    
    # 按类别收集样本索引
    class_indices = {}
    for idx, (_, label) in enumerate(full_dataset):
        if label not in class_indices:
            class_indices[label] = []
        class_indices[label].append(idx)
    
    # 按文件路径排序每个类别的样本（确保顺序一致性）
    for label, indices in class_indices.items():
        # 获取对应样本的文件路径
        paths = [full_dataset.samples[i][0] for i in indices]
        # 按路径排序
        sorted_indices = [idx for _, idx in sorted(zip(paths, indices))]
        class_indices[label] = sorted_indices
    
    # 创建划分容器
    train_idx = []
    val_idx = []
    test_idx = []
    
    # 对每个类别进行划分
    for label, indices in class_indices.items():
        n = len(indices)
        train_end = int(train_ratio * n)
        val_end = train_end + int(val_ratio * n)
        
        # 确保所有样本都被分配
        train_idx.extend(indices[:train_end])
        val_idx.extend(indices[train_end:val_end])
        test_idx.extend(indices[val_end:])
    
    # 打印划分统计信息
    print(f"数据集划分结果: 训练集={len(train_idx)}, 验证集={len(val_idx)}, 测试集={len(test_idx)}")
    print(f"训练集占比: {len(train_idx)/len(full_dataset):.2%}")
    print(f"验证集占比: {len(val_idx)/len(full_dataset):.2%}")
    print(f"测试集占比: {len(test_idx)/len(full_dataset):.2%}")
    
    # 提取子集
    train_subset = Subset(full_dataset, train_idx)
    val_subset = Subset(full_dataset, val_idx)
    test_subset = Subset(full_dataset, test_idx)

    ###########################################################
    # 继续原有流程
    ###########################################################
    
    # 计算训练集统计量
    train_mean, train_std = compute_mean_std(train_subset)

    # 使用训练集统计量创建转换
    train_transform = transforms.Compose([
        transforms.Resize((config['img_size'], config['img_size'])),
        transforms.Grayscale(num_output_channels=1),
        transforms.RandomHorizontalFlip(p=0.5),  # 数据增强
        transforms.RandomRotation(10),  # 数据增强
        transforms.ToTensor(),
        transforms.Normalize(train_mean, train_std)
    ])
    val_transform = transforms.Compose([
        transforms.Resize((config['img_size'], config['img_size'])),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize(train_mean, train_std)
    ])
    test_transform = transforms.Compose([
        transforms.Resize((config['img_size'], config['img_size'])),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize(train_mean, train_std)
    ])

    # 应用预处理
    full_dataset.transform = train_transform
    val_subset.dataset.transform = val_transform
    test_subset.dataset.transform = test_transform

    # 数据加载器
    train_loader = DataLoader(train_subset, batch_size=config['batch_size'],
                              shuffle=True, num_workers=config['num_workers'])
    val_loader = DataLoader(val_subset, batch_size=config['batch_size'],
                            shuffle=False, num_workers=config['num_workers'])
    test_loader = DataLoader(test_subset, batch_size=config['batch_size'],
                             shuffle=False, num_workers=config['num_workers'])

    # 初始化ResNet18模型
    model = create_resnet18_model(
        num_classes=config['num_classes'],
        use_pretrained=config['use_pretrained'],
        in_channels=1  # 灰度图
    ).to(device)
    
    # 打印模型结构
    print("ResNet18模型结构:")
    print(model)
    
    # 定义损失函数
    criterion = nn.CrossEntropyLoss()
    
    # 检查预训练模型路径
    pretrained_path = config['pretrained_model_path']
    if pretrained_path and os.path.exists(pretrained_path):
        print(f"加载预训练模型: {pretrained_path}")
        model.load_state_dict(torch.load(pretrained_path, map_location=device))
        print("开始测试预训练模型...")
        test_model(model, test_loader, criterion, device, config['save_dir'])
        exit(0)  # 测试完成后退出

    # 如果没有提供预训练模型，则进行训练
    print("未提供预训练模型，开始训练新模型...")
    
    # 优化器
    optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=1e-4)

    # 学习率调度器
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
         optimizer, mode='max', factor=0.1, patience=5, verbose=True
    )

    # 初始化记录文件
    csv_path = os.path.join(config['save_dir'], 'metrics.csv')
    if not os.path.exists(csv_path):
        with open(csv_path, 'w') as f:
            writer = csv.writer(f)
            writer.writerow([
                'epoch', 'train_loss', 'val_loss',
                'train_acc', 'train_precision', 'train_recall', 'train_f1',
                'val_acc', 'val_precision', 'val_recall', 'val_f1', 'val_specificity', 'confusion_matrix'
            ])

    # 训练循环
    best_f1 = 0.0

    for epoch in range(config['num_epochs']):
        # 训练步骤
        model.train()
        train_loss, all_preds, all_labels = 0.0, [], []
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}"):
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            all_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        # 验证步骤
        model.eval()
        val_loss, val_preds, val_labels = 0.0, [], []
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                val_loss += criterion(outputs, labels).item()
                val_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
                val_labels.extend(labels.cpu().numpy())

        # 计算指标
        train_metrics = calculate_metrics(all_labels, all_preds)
        val_metrics = calculate_metrics(val_labels, val_preds)

        # 更新学习率调度器
        scheduler.step(val_metrics['f1'])

        # 保存最佳模型
        if val_metrics['f1'] > best_f1:
            best_f1 = val_metrics['f1']
            best_model_path = os.path.join(config['save_dir'], 'best.pth')
            torch.save(model.state_dict(), best_model_path)
            print(f"保存新的最佳模型到: {best_model_path} (F1={val_metrics['f1']:.4f})")

        # 打印结果
        print(f"\nEpoch {epoch + 1}/{config['num_epochs']}")
        print(f"Train Loss: {train_loss / len(train_loader):.4f}, Val Loss: {val_loss / len(val_loader):.4f}")
        print(f"Train Accuracy: {train_metrics['accuracy']:.4f}, Precision: {train_metrics['precision']:.4f}, Recall: {train_metrics['recall']:.4f}, F1: {train_metrics['f1']:.4f}")
        print(f"Val Accuracy: {val_metrics['accuracy']:.4f}, Precision: {val_metrics['precision']:.4f}, Recall: {val_metrics['recall']:.4f}, F1: {val_metrics['f1']:.4f}, Specificity: {val_metrics['specificity']:.4f}")
        print(f"Confusion Matrix:\n{val_metrics['confusion_matrix']}")

        # 写入CSV
        with open(csv_path, 'a') as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch + 1,
                train_loss / len(train_loader),
                val_loss / len(val_loader),
                train_metrics['accuracy'],
                train_metrics['precision'],
                train_metrics['recall'],
                train_metrics['f1'],
                val_metrics['accuracy'],
                val_metrics['precision'],
                val_metrics['recall'],
                val_metrics['f1'],
                val_metrics['specificity'],
                str(val_metrics['confusion_matrix'].tolist())
            ])

    # 训练完成后加载最佳模型进行测试
    print("\n训练完成，加载最佳模型进行测试...")
    model.load_state_dict(torch.load(os.path.join(config['save_dir'], 'best.pth')))
    test_model(model, test_loader, criterion, device, config['save_dir'])

    print("整个流程完成!")