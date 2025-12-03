import torch
import torch.nn as nn
from torchvision.models import resnet18

def create_resnet18_model(num_classes, use_pretrained=False, in_channels=1):
    """
    创建并配置ResNet18模型
    
    参数:
        num_classes (int): 分类类别数量
        use_pretrained (bool): 是否使用预训练权重
        in_channels (int): 输入通道数，默认为1（灰度图）
    
    返回:
        model (nn.Module): 配置好的ResNet18模型
    """
    # 加载预训练或随机初始化的ResNet18
    model = resnet18(pretrained=use_pretrained)
    
    # 修改第一层卷积以适应单通道输入
    original_conv1 = model.conv1
    model.conv1 = nn.Conv2d(
        in_channels, 
        original_conv1.out_channels, 
        kernel_size=original_conv1.kernel_size,
        stride=original_conv1.stride,
        padding=original_conv1.padding,
        bias=original_conv1.bias is not None
    )
    
    # 如果使用预训练权重，需要初始化新的第一层卷积权重
    if use_pretrained:
        # 将预训练的3通道权重转换为单通道
        with torch.no_grad():
            # 对3通道权重取平均值来适应单通道输入
            new_weight = original_conv1.weight.mean(dim=1, keepdim=True)
            model.conv1.weight.copy_(new_weight)
    
    # 修改最后的全连接层以适应我们的类别数
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    
    return model

def print_model_summary(model, input_size=(1, 224, 224)):
    """
    打印模型结构摘要
    
    参数:
        model (nn.Module): 模型实例
        input_size (tuple): 输入尺寸 (channels, height, width)
    """
    try:
        from torchsummary import summary
        summary(model, input_size=input_size)
    except ImportError:
        print("请安装torchsummary: pip install torchsummary")
        print("模型结构:")
        print(model)

# 测试代码
if __name__ == "__main__":
    # 创建模型实例进行测试
    model = create_resnet18_model(
        num_classes=10,
        use_pretrained=False,
        in_channels=1
    )
    
    # 打印模型结构
    print("ResNet18模型结构:")
    print(model)
    
    # 创建模拟输入
    batch_size = 2
    image = torch.randn(batch_size, 1, 224, 224)
    
    # 前向传播测试
    output = model(image)
    
    print(f"\n模型测试:")
    print(f"输入形状: {image.shape}")
    print(f"输出形状: {output.shape}")
    print("模型测试通过！")