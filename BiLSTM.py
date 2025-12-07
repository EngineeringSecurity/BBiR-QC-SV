import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score
import re
from collections import Counter
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import csv
from datetime import datetime
import shutil

class TweetDataset(Dataset):
    """自定义数据集类"""
    def __init__(self, texts, labels, word2idx, max_len):
        self.texts = texts
        self.labels = labels
        self.word2idx = word2idx
        self.max_len = max_len
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        # 文本转换为索引序列
        tokens = text.split()[:self.max_len]
        indices = [self.word2idx.get(token, self.word2idx['<UNK>']) for token in tokens]
        
        # 填充序列
        if len(indices) < self.max_len:
            indices += [self.word2idx['<PAD>']] * (self.max_len - len(indices))
        else:
            indices = indices[:self.max_len]
        
        return {
            'text': torch.tensor(indices, dtype=torch.long),
            'label': torch.tensor(label, dtype=torch.long)
        }

class BiLSTMClassifier(nn.Module):
    """BiLSTM谣言分类器"""
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, 
                 dropout, pretrained_embeddings=None):
        super(BiLSTMClassifier, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # 如果提供了预训练词向量，加载它们
        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(pretrained_embeddings)
            self.embedding.weight.requires_grad = False  # 是否微调词向量
        
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=n_layers,
                           bidirectional=True, dropout=dropout, batch_first=True)
        
        self.dropout = nn.Dropout(dropout)
        
        # 因为是双向LSTM，所以hidden_dim要乘以2
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        
    def forward(self, text):
        # text shape: [batch_size, seq_len]
        
        embedded = self.embedding(text)  # [batch_size, seq_len, embedding_dim]
        
        # LSTM输出
        lstm_output, (hidden, cell) = self.lstm(embedded)
        # lstm_output shape: [batch_size, seq_len, hidden_dim * 2]
        
        # 获取最后时间步的输出（双向LSTM的拼接）
        # 取前向和后向的最后一个隐藏状态
        hidden_forward = hidden[-2, :, :]  # 前向最后一层
        hidden_backward = hidden[-1, :, :]  # 后向最后一层
        
        # 拼接前向和后向的隐藏状态
        hidden_concat = torch.cat((hidden_forward, hidden_backward), dim=1)
        # hidden_concat shape: [batch_size, hidden_dim * 2]
        
        output = self.fc(self.dropout(hidden_concat))
        
        return output
    
    def extract_features(self, text):
        """提取分类向量（全连接层之前的特征）"""
        # text shape: [batch_size, seq_len]
        
        embedded = self.embedding(text)  # [batch_size, seq_len, embedding_dim]
        
        # LSTM输出
        lstm_output, (hidden, cell) = self.lstm(embedded)
        
        # 获取最后时间步的输出（双向LSTM的拼接）
        hidden_forward = hidden[-2, :, :]  # 前向最后一层
        hidden_backward = hidden[-1, :, :]  # 后向最后一层
        
        # 拼接前向和后向的隐藏状态
        hidden_concat = torch.cat((hidden_forward, hidden_backward), dim=1)
        # hidden_concat shape: [batch_size, hidden_dim * 2]
        
        return hidden_concat

class TextProcessor:
    """文本处理器"""
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.vocab_size = 0
        
    def build_vocab(self, texts, min_freq=2):
        """构建词汇表"""
        word_freq = Counter()
        
        for text in texts:
            # 简单的分词（按空格分割）
            tokens = text.split()
            word_freq.update(tokens)
        
        # 构建词汇表
        self.word2idx = {'<PAD>': 0, '<UNK>': 1}
        idx = 2
        
        for word, freq in word_freq.items():
            if freq >= min_freq:
                self.word2idx[word] = idx
                idx += 1
        
        self.idx2word = {idx: word for word, idx in self.word2idx.items()}
        self.vocab_size = len(self.word2idx)
        
        print(f"词汇表大小: {self.vocab_size}")
        return self.word2idx
    
    def preprocess_text(self, text):
        """预处理文本"""
        # 转换为小写
        text = text.lower()
        
        # 移除URL
        text = re.sub(r'http\S+', '', text)
        
        # 移除特殊字符，但保留基本标点
        text = re.sub(r'[^a-zA-Z0-9\s.,!?]', '', text)
        
        # 合并多个空格
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()

def load_data_from_folders(data_dir):
    """从文件夹加载数据，假设结构为：data_dir/class1/, data_dir/class2/"""
    texts = []
    labels = []
    filenames = []
    
    # 获取所有子文件夹（类别）
    classes = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    class_to_label = {class_name: idx for idx, class_name in enumerate(classes)}
    
    print(f"发现 {len(classes)} 个类别: {class_to_label}")
    
    for class_name in classes:
        class_dir = os.path.join(data_dir, class_name)
        label = class_to_label[class_name]
        
        # 读取该类别下的所有txt文件
        for filename in os.listdir(class_dir):
            if filename.endswith(".txt"):
                file_path = os.path.join(class_dir, filename)
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                
                texts.append(content)
                labels.append(label)
                filenames.append(filename)
    
    print(f"总共加载了 {len(texts)} 个样本")
    for class_name, label in class_to_label.items():
        count = sum(1 for l in labels if l == label)
        print(f"  {class_name}: {count} 个样本")
    
    return texts, labels, filenames, class_to_label

def extract_tweet_texts_from_content(content):
    """从TXT文件内容中提取所有推文文本"""
    tweets_text = []
    lines = content.split('\n')
    
    current_tweet_text = ""
    in_tweet = False
    
    for line in lines:
        if line.startswith('[Tweet'):
            in_tweet = True
            if current_tweet_text:
                tweets_text.append(current_tweet_text)
                current_tweet_text = ""
        elif line.startswith('Text:'):
            if in_tweet:
                text_content = line.replace('Text:', '').strip()
                current_tweet_text += " " + text_content
        elif line.strip() == "" and current_tweet_text:
            # 推文结束
            tweets_text.append(current_tweet_text)
            current_tweet_text = ""
            in_tweet = False
    
    # 添加最后一个推文
    if current_tweet_text:
        tweets_text.append(current_tweet_text)
    
    # 将所有推文文本合并为一个文档
    all_text = " ".join(tweets_text)
    return all_text

def calculate_metrics(preds, labels):
    """计算准确率、精确率、召回率和F1分数"""
    acc = accuracy_score(labels, preds)
    precision = precision_score(labels, preds, average='weighted', zero_division=0)
    recall = recall_score(labels, preds, average='weighted', zero_division=0)
    f1 = f1_score(labels, preds, average='weighted', zero_division=0)
    
    return acc, precision, recall, f1

def train_model(model, train_loader, val_loader, criterion, optimizer, device, epochs=10, results_file="training_results.csv"):
    """训练模型并记录每轮指标"""
    train_losses = []
    val_losses = []
    
    # 创建结果文件并写入表头
    with open(results_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Epoch', 'Train_Acc', 'Train_Precision', 'Train_Recall', 'Train_F1', 
                        'Val_Acc', 'Val_Precision', 'Val_Recall', 'Val_F1', 
                        'Train_Loss', 'Val_Loss', 'Timestamp'])
    
    best_train_f1 = 0.0000  # 改为跟踪训练集上的最佳F1分数
    best_model_state = None
    
    for epoch in range(epochs):
        # 训练阶段
        model.train()
        total_train_loss = 0.0000
        train_preds = []
        train_labels = []
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} - Training"):
            texts = batch['text'].to(device)
            labels = batch['label'].to(device)
            
            optimizer.zero_grad()
            outputs = model(texts)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
            
            # 收集预测结果
            _, preds = torch.max(outputs, 1)
            train_preds.extend(preds.cpu().numpy())
            train_labels.extend(labels.cpu().numpy())
        
        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # 计算训练集指标
        train_acc, train_precision, train_recall, train_f1 = calculate_metrics(train_preds, train_labels)
        
        # 验证阶段
        model.eval()
        total_val_loss = 0.0000
        val_preds = []
        val_labels = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} - Validation"):
                texts = batch['text'].to(device)
                labels = batch['label'].to(device)
                
                outputs = model(texts)
                loss = criterion(outputs, labels)
                total_val_loss += loss.item()
                
                _, preds = torch.max(outputs, 1)
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
        
        avg_val_loss = total_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        # 计算验证集指标
        val_acc, val_precision, val_recall, val_f1 = calculate_metrics(val_preds, val_labels)
        
        # 记录时间戳
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # 输出到控制台 - 所有数值保留四位小数
        print(f"\nEpoch {epoch+1}/{epochs}:")
        print(f"  Train - Loss: {avg_train_loss:.4f}, Acc: {train_acc:.4f}, F1: {train_f1:.4f}")
        print(f"  Val   - Loss: {avg_val_loss:.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f}")
        print(f"  Train - Precision: {train_precision:.4f}, Recall: {train_recall:.4f}")
        print(f"  Val   - Precision: {val_precision:.4f}, Recall: {val_recall:.4f}")
        
        # 保存到CSV文件（保持原始精度）
        with open(results_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([epoch+1, train_acc, train_precision, train_recall, train_f1,
                           val_acc, val_precision, val_recall, val_f1,
                           avg_train_loss, avg_val_loss, timestamp])
        
        # 保存最佳模型（基于训练集F1分数）- 这是主要的修改点
        if train_f1 > best_train_f1:
            best_train_f1 = train_f1
            best_model_state = model.state_dict().copy()
            print(f"  新的最佳训练F1分数: {best_train_f1:.4f}")
    
    # 加载最佳模型
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model, train_losses, val_losses

def evaluate_model(model, test_loader, device, class_names):
    """评估模型在测试集上的表现"""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            texts = batch['text'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(texts)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # 计算各种指标
    acc, precision, recall, f1 = calculate_metrics(all_preds, all_labels)
    
    print(f"\n=== 测试集表现 ===")
    print(f"准确率: {acc:.4f}")
    print(f"精确率: {precision:.4f}")
    print(f"召回率: {recall:.4f}")
    print(f"F1分数: {f1:.4f}")
    print("\n详细分类报告:")
    print(classification_report(all_labels, all_preds, target_names=class_names, digits=4))
    
    # 保存测试结果
    with open("test_results.csv", 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Metric', 'Value'])
        writer.writerow(['Accuracy', f"{acc:.4f}"])
        writer.writerow(['Precision', f"{precision:.4f}"])
        writer.writerow(['Recall', f"{recall:.4f}"])
        writer.writerow(['F1-Score', f"{f1:.4f}"])
        writer.writerow(['Timestamp', datetime.now().strftime("%Y-%m-%d %H:%M:%S")])
    
    return all_preds, all_labels, acc, precision, recall, f1

def plot_training_curves(train_losses, val_losses, results_file="training_results.csv"):
    """绘制训练曲线"""
    # 从CSV文件读取指标
    df = pd.read_csv(results_file)
    
    plt.figure(figsize=(15, 5))
    
    # 损失曲线
    plt.subplot(1, 3, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    
    # 准确率曲线
    plt.subplot(1, 3, 2)
    plt.plot(df['Train_Acc'], label='Train Accuracy')
    plt.plot(df['Val_Acc'], label='Val Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Training and Validation Accuracy')
    
    # F1分数曲线
    plt.subplot(1, 3, 3)
    plt.plot(df['Train_F1'], label='Train F1')
    plt.plot(df['Val_F1'], label='Val F1')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.legend()
    plt.title('Training and Validation F1 Score')
    
    plt.tight_layout()
    plt.savefig('training_curves.png', dpi=300, bbox_inches='tight')
    plt.show()

def extract_features_to_npy(model, device, data_dir, output_dir, word2idx, max_len, batch_size=32):
    """提取所有样本的分类向量并保存为.npy文件，保持原始文件结构"""
    print("开始提取分类向量...")
    
    # 加载数据
    texts, labels, filenames, class_to_label = load_data_from_folders(data_dir)
    
    # 提取推文文本
    processed_texts = []
    for content in texts:
        tweet_text = extract_tweet_texts_from_content(content)
        processed_texts.append(tweet_text)
    
    # 文本预处理
    processor = TextProcessor()
    preprocessed_texts = [processor.preprocess_text(text) for text in processed_texts]
    
    # 创建数据集和数据加载器
    dataset = TweetDataset(preprocessed_texts, labels, word2idx, max_len)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    # 提取特征
    model.eval()
    all_features = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="提取特征"):
            texts_batch = batch['text'].to(device)
            features_batch = model.extract_features(texts_batch)
            all_features.append(features_batch.cpu().numpy())
    
    # 合并所有批次的特征
    all_features = np.vstack(all_features)
    
    # 创建输出目录结构
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    
    # 为每个类别创建子目录
    label_to_class = {v: k for k, v in class_to_label.items()}
    for class_name in class_to_label.keys():
        class_dir = os.path.join(output_dir, class_name)
        os.makedirs(class_dir, exist_ok=True)
    
    # 保存特征向量到对应的类别文件夹
    feature_index = 0
    for i, (filename, label) in enumerate(zip(filenames, labels)):
        class_name = label_to_class[label]
        
        # 生成新的文件名（将.txt替换为.npy）
        new_filename = filename.replace('.txt', '.npy')
        output_path = os.path.join(output_dir, class_name, new_filename)
        
        # 保存特征向量
        np.save(output_path, all_features[feature_index])
        feature_index += 1
    
    print(f"分类向量已保存到: {output_dir}")
    print(f"总共处理了 {len(filenames)} 个文件")

def main():
    # 配置参数
    DATA_DIR = "phemetxt"  # 您的数据文件夹路径
    OUTPUT_FEATURE_DIR = "features"  # 特征向量输出目录
    PRETRAINED_MODEL_PATH = "bilstm_rumor_classifier.pth"  # 预训练模型路径
    BATCH_SIZE = 32
    EMBEDDING_DIM = 500
    HIDDEN_DIM = 256*2
    OUTPUT_DIM = 2  # 二分类
    N_LAYERS = 2
    DROPOUT = 0.5
    MAX_LEN = 600  # 最大序列长度
    EPOCHS = 15
    LEARNING_RATE = 0.001
    RESULTS_FILE = "training_results.csv"
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 检查预训练模型是否存在
    if os.path.exists(PRETRAINED_MODEL_PATH):
        print("发现预训练模型，直接加载...")
        
        # 加载预训练模型
        checkpoint = torch.load(PRETRAINED_MODEL_PATH, map_location=device)
        word2idx = checkpoint['word2idx']
        max_len = checkpoint['max_len']
        class_to_label = checkpoint['class_to_label']
        
        # 创建模型
        model = BiLSTMClassifier(
            vocab_size=len(word2idx),
            embedding_dim=EMBEDDING_DIM,
            hidden_dim=HIDDEN_DIM,
            output_dim=len(class_to_label),
            n_layers=N_LAYERS,
            dropout=DROPOUT
        ).to(device)
        
        # 加载模型权重
        model.load_state_dict(checkpoint['model_state_dict'])
        print("预训练模型加载成功!")
        
    else:
        print("未发现预训练模型，开始训练新模型...")
        
        # 1. 加载数据
        print("加载数据...")
        texts, labels, filenames, class_to_label = load_data_from_folders(DATA_DIR)
        
        # 2. 提取推文文本
        print("提取推文文本...")
        processed_texts = []
        for content in texts:
            tweet_text = extract_tweet_texts_from_content(content)
            processed_texts.append(tweet_text)
        
        # 3. 文本预处理
        print("预处理文本...")
        processor = TextProcessor()
        preprocessed_texts = [processor.preprocess_text(text) for text in processed_texts]
        
        # 4. 构建词汇表
        print("构建词汇表...")
        word2idx = processor.build_vocab(preprocessed_texts, min_freq=2)
        
        # 5. 分割数据集
        print("分割数据集...")
        X_train, X_temp, y_train, y_temp = train_test_split(
            preprocessed_texts, labels, test_size=0.3, random_state=42, stratify=labels
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
        )
        
        print(f"训练集: {len(X_train)}, 验证集: {len(X_val)}, 测试集: {len(X_test)}")
        
        # 6. 创建数据加载器
        train_dataset = TweetDataset(X_train, y_train, word2idx, MAX_LEN)
        val_dataset = TweetDataset(X_val, y_val, word2idx, MAX_LEN)
        test_dataset = TweetDataset(X_test, y_test, word2idx, MAX_LEN)
        
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
        
        # 7. 创建模型
        print("创建模型...")
        model = BiLSTMClassifier(
            vocab_size=processor.vocab_size,
            embedding_dim=EMBEDDING_DIM,
            hidden_dim=HIDDEN_DIM,
            output_dim=OUTPUT_DIM,
            n_layers=N_LAYERS,
            dropout=DROPOUT
        ).to(device)
        
        # 8. 定义损失函数和优化器
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        
        # 9. 训练模型
        print("开始训练...")
        model, train_losses, val_losses = train_model(
            model, train_loader, val_loader, criterion, optimizer, device, EPOCHS, RESULTS_FILE
        )
        
        # 10. 评估模型
        print("评估模型...")
        class_names = list(class_to_label.keys())
        preds, true_labels, test_acc, test_precision, test_recall, test_f1 = evaluate_model(
            model, test_loader, device, class_names
        )
        
        # 11. 绘制训练曲线
        plot_training_curves(train_losses, val_losses, RESULTS_FILE)
        
        # 12. 保存模型
        torch.save({
            'model_state_dict': model.state_dict(),
            'word2idx': word2idx,
            'processor': processor,
            'max_len': MAX_LEN,
            'class_to_label': class_to_label
        }, PRETRAINED_MODEL_PATH)
        
        print(f"模型已保存为 '{PRETRAINED_MODEL_PATH}'")
        
        # 13. 输出最终总结
        print("\n=== 训练总结 ===")
        print(f"最佳测试集准确率: {test_acc:.4f}")
        print(f"最佳测试集F1分数: {test_f1:.4f}")
        print(f"训练结果保存在: {RESULTS_FILE}")
        print(f"测试结果保存在: test_results.csv")
        print(f"训练曲线保存在: training_curves.png")
    
    # 无论是否有预训练模型，都进行特征提取
    print("\n开始提取所有样本的分类向量...")
    extract_features_to_npy(
        model=model,
        device=device,
        data_dir=DATA_DIR,
        output_dir=OUTPUT_FEATURE_DIR,
        word2idx=word2idx,
        max_len=MAX_LEN,
        batch_size=BATCH_SIZE
    )
    
    print("\n=== 处理完成 ===")
    print(f"原始数据目录: {DATA_DIR}")
    print(f"特征向量保存目录: {OUTPUT_FEATURE_DIR}")
    print(f"模型文件: {PRETRAINED_MODEL_PATH}")

if __name__ == "__main__":
    main()