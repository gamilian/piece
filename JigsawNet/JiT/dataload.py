import os
import cv2
import random
import h5py
from torch.utils.data import DataLoader, Dataset, random_split 
from torchvision.datasets import ImageFolder
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
        rois = []
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
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = self.transform(image)

        except OSError as e:
            # 记录错误并返回占位符或其他处理
            print(f"加载图像 {image_path} 时出错: {e}")
            return None  # 占位符或其他处理
        return image, label

    def shuffle_data(self):
        random.shuffle(self.data)




class HDF5Dataset(Dataset):
    def __init__(self, hdf5_file):
        self.hdf5_file = hdf5_file

        with h5py.File(hdf5_file, 'r') as f:
            self.images_group = f['images']
            self.labels_group = f['labels']
            self.num_samples = len(self.images_group)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        image = torch.from_numpy(self.images_group[f'image_{idx}'][:])
        label = torch.from_numpy(self.labels_group[f'label_{idx}'][:])

        return image, label

def get_train_dataloader(root_folder, batch_size, prefetch_factor=4):

    data = PieceDataset(root_folder)
    train_set_size, valid_set_size = int(0.9 * len(data)), len(data) - int(0.9 * len(data))
    train_data, valid_data = random_split(data, [train_set_size, valid_set_size])

    train_dataloader = DataLoader(train_data, num_workers=64, batch_size=batch_size,
                                  shuffle=True, prefetch_factor=prefetch_factor)
    valid_dataloader = DataLoader(valid_data, num_workers=8, batch_size=batch_size,
                                shuffle=False, prefetch_factor=prefetch_factor)
    return train_dataloader, valid_dataloader

def get_test_dataloader(root_folder, batch_size, prefetch_factor=2):
    test_data = PieceDataset(root_folder)
    test_dataloader = DataLoader(test_data, num_workers=8, batch_size=batch_size,
                                 shuffle=False, prefetch_factor=prefetch_factor)
    return test_dataloader


if __name__ == '__main__':
    root_folder = '/data/csl/dataset/jigsaw_dataset/12_17'
    data = PieceDataset(root_folder)
    # joblib.dump(train_data, 'dataset_object.pkl')
    data.shuffle_data()


