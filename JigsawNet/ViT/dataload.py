import os
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm
import joblib

class PieceDataset(Dataset):
    def __init__(self, root_folder, is_train=True):
        self.root_folder = root_folder
        self.is_train = is_train
        self.transform = transforms.Compose([
            transforms.ToTensor(), 
            transforms.Resize((224, 224)), 
            transforms.Normalize([0.5, 0.5, 0.5], 
                                    [0.5, 0.5, 0.5]) 
        ]) 
        self.valid_transform = transforms.Compose([   
            transforms.ToTensor(), 
            transforms.Resize((224, 224)),  
            transforms.Normalize([0.5, 0.5, 0.5], 
                                    [0.5, 0.5, 0.5]) 
        ]) 
        self.data = self.load_data()

    def load_data(self):
        data = []
        image_path_root = os.path.join(self.root_folder, 'image')
        with open(os.path.join(self.root_folder, "target.txt")) as f:
            for i, line in enumerate(f):
                # if i > 10000:
                #     break
                line = line.rstrip()
                label = int(line)
                image_path = os.path.join(image_path_root, f"{i}.png")
                
                data.append((image_path, label))
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_path, label = self.data[idx]
        # try:
        if self.is_train:
            image = Image.open(image_path)
            image = self.transform(image)
        else:
            image = Image.open(image_path)
            image = self.valid_transform(image)
        # except OSError as e:
        #     # 记录错误并返回占位符或其他处理
        #     print(f"加载图像 {image_path} 时出错: {e}")
        #     return None  # 占位符或其他处理
        
        return image, label


def get_train_dataloader(root_folder, batch_size):
     
    train_dataset = PieceDataset(root_folder)
    train_dataloader = DataLoader(train_dataset, num_workers=32, batch_size=batch_size, shuffle=True)
    return train_dataloader

def get_test_dataloader(root_folder, batch_size):
    test_dataset = PieceDataset(root_folder)
    test_dataloader = DataLoader(test_dataset, num_workers=32, batch_size=batch_size, shuffle=False)
    return test_dataloader

if __name__ == '__main__':
    root_folder = '/work/csl/code/piece/dataset/training_dataset'
    train_dataset = PieceDataset(root_folder)
    # joblib.dump(train_dataset, 'dataset_object.pkl')

    # image, label = train_dataset[0]
    # print(image.shape)
    # print(label)