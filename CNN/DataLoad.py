import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

class CustomDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_dir, self.images[idx])
        image = Image.open(img_name)

        # 加载标签和 ROI（假设标签和 ROI 存储方式与图像同名但扩展名不同）
        label_name = img_name.replace('.png', '.txt')  # 假设标签文件与图像同名但扩展名为 .txt
        # 这里需要根据您的标签文件格式来解析标签
        label = ...  # 读取并解析标签文件

        # 可选：读取和解析 ROI 数据
        # roi = ...  # 类似地读取并解析 ROI 数据

        if self.transform:
            image = self.transform(image)

        return image, label  # , roi （如果您需要 ROI 数据）

# 使用转换创建数据集
transform = transforms.Compose([
    transforms.Resize((160, 160)),  # 调整图像大小
    transforms.ToTensor()
])

dataset = CustomDataset(image_dir='path/to/your/image_dir', transform=transform)
