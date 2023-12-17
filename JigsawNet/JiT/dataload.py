import os
from PIL import Image
import random
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
            transforms.Normalize([0.485, 0.456, 0.406], 
                            [0.229, 0.224, 0.225]) 
        ]) 
        
        self.data = self.load_data()
        self.num_samples = len(self.data)

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
        try:
            image = Image.open(image_path)
            image = self.transform(image)
            
        except OSError as e:
            # 记录错误并返回占位符或其他处理
            print(f"加载图像 {image_path} 时出错: {e}")
            return None  # 占位符或其他处理
        return image, label

    def shuffle_data(self):
        random.shuffle(self.data)


def get_train_dataloader(root_folder, batch_size):
     
    train_data = PieceDataset(root_folder)
    train_data.shuffle_data()
    train_dataloader = DataLoader(train_data, num_workers=16, batch_size=batch_size, shuffle=True)
    return train_dataloader

def get_test_dataloader(root_folder, batch_size):
    test_data = PieceDataset(root_folder)
    test_dataloader = DataLoader(test_data, num_workers=16, batch_size=batch_size, shuffle=False)
    return test_dataloader

if __name__ == '__main__':
    root_folder = '/work/csl/code/piece/dataset/training_dataset'
    data = PieceDataset(root_folder)
    # joblib.dump(train_data, 'dataset_object.pkl')
    train_data, valid_data = random_split(data, [train_set_size, valid_set_size])
    image, label = train_dataset[0]
    print(image.shape)
    print(label)