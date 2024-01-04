import torch
import os
import cv2
import csv
import random
import numpy as np
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from tqdm import tqdm

def get_coutours(image, bg_color):
    '''
    Get the contours of the image
    '''
    # Set background color threshold
    lower_bg = np.array(bg_color)
    upper_bg = np.array(bg_color)

    # Create a background mask
    bg_mask = cv2.inRange(image, lower_bg, upper_bg)
    # Make the background part pure black and other parts white
    bg_mask = np.where(bg_mask == 255, 0, np.where(bg_mask == 0, 255, bg_mask))

    # Extract edge contours
    contours, _ = cv2.findContours(bg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return contours

def calculate_euclidean_distance(point1, point2):
    '''
    Calculate the Euclidean distance between two points
    '''
    return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

# 使用dp算法近似轮廓
def approx_contours(image, contour, beta=0.0008):
    '''
    Approximate contours using the dp algorithm
    '''
    # # Find the farthest vertex v1 and v2
    # max_distance = 0
    # v1, v2 = 0, 0
    # for i in range(len(contour)):
    #     for j in range(i + 1, len(contour)):
    #         # Calculate the Euclidean distance between two points
    #         distance = calculate_euclidean_distance(contour[i][0], contour[j][0])
    #         if distance > max_distance:
    #             max_distance = distance
    #             v1, v2 = i, j
    # # print(f"v1:{contour[v1][0]}\nv2:{contour[v2][0]}\n")

    # # The best accuracy parameter is about 0.0004
    epsilon = beta * cv2.arcLength(contour, True)

    # if v1 < v2:
    #     # Extract the contour of the v1v2 interval
    #     v1v2_contour = contour[v1:v2 + 1]  
    #     # Extract the contour of the v2v1 interval
    #     v2v1_contour = np.concatenate((contour[v2 + 1:], contour[:v1]))
    # else:
    #     # Extract the contour of the v1v2 interval
    #     v1v2_contour = np.concatenate(contour[v1:], contour[:v2 + 1])
    #     # Extract the contour of the v2v1 interval
    #     v2v1_contour = contour[v2 + 1:v1]
    # # Execute the DP algorithm on v1 to v2
    # dp_v1v2 = cv2.approxPolyDP(v1v2_contour, epsilon, closed=True)  
    # # Execute the DP algorithm on v2 to v1
    # dp_v2v1 = cv2.approxPolyDP(v2v1_contour, epsilon, closed=True)
    # # Combine the two contours  
    # approx = np.concatenate((dp_v1v2, dp_v2v1))

    approx = cv2.approxPolyDP(contour, epsilon, closed=True)
    # Plot Approximate Contour Points
    # print(f'v1到v2曲线近似点数目{len(dp_v1v2)}')
    # for point in dp_v1v2:
    #     cv2.circle(image, tuple(point[0]), 2, (232, 16, 16), -1)
    # print(f'v2到v1曲线近似点数目{len(dp_v2v1)}')
    # for point in dp_v2v1:
    #     cv2.circle(image, tuple(point[0]), 2, (22, 26, 216), -1)
    # # Draw an approximate polygon on an image
    # cv2.polylines(image, [dp_v1v2], isClosed=False, color=(232, 16, 16), thickness=1)
    # cv2.polylines(image, [dp_v2v1], isClosed=False, color=(232, 16, 16), thickness=1)
    return approx

class PieceDataset(Dataset):
    def __init__(self, data_path, is_train=True):
        self.data_path = data_path
        self.is_train = is_train
        self.max_length = 500
        # self.max_length = 3000
        self.sep_token = np.array([[0, 0]])
        self.data = self.load_data()
        
    
    def __len__(self):
        return len(self.data)
    
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        input_tensor = sample['input_tensor']
        label = torch.from_numpy(sample['label']).float()
        true_len = sample['true_len']
        cond = {
            'true_len': true_len
        }
        return input_tensor, label, cond

    def shuffle_data(self):
        random.shuffle(self.data)

    def load_data(self):
        data = []
        with open(self.data_path) as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            for i, row in tqdm(enumerate(reader)):
                # if i >= 15000:
                #     break
                sample = {}
                path1, path2 = row[0], row[1]
                bg_color_str = row[2].split()
                bg_color = [int(x) for x in bg_color_str]
                rotation_angle = float(row[3])
                translation_x, translation_y = float(row[4])/1000, float(row[5])/1000
                image1, image2 = cv2.imread(path1), cv2.imread(path2)
                contour1, contour2 = get_coutours(image1, bg_color)[0], get_coutours(image2, bg_color)[0]
                approx1, approx2 = approx_contours(image1, contour1, 0.001).squeeze(1), approx_contours(image2, contour2, 0.001).squeeze(1)
                # input_tensor = np.concatenate((contour1.squeeze(1), self.sep_token, contour2.squeeze(1)), axis=0)
                input_tensor = np.concatenate((approx1, self.sep_token, approx2), axis=0)
                true_len = len(input_tensor)
                completed_input = np.zeros((self.max_length, 2))
                # 将 input_tensor 的数据复制到零矩阵的开头部分
                completed_input[:true_len, :] = input_tensor
                # true_len1, true_len2 = len(approx1), len(approx2)
                # completed_approx1 = np.zeros((self.max_length, 2))
                # completed_approx1[:true_len1, :] = approx1
                # completed_approx2 = np.zeros((self.max_length, 2))
                # completed_approx2[:true_len2, :] = approx2
                # input_tensor = np.stack((approx1, approx2), axis=0)
          
                sample['input_tensor'] = completed_input.astype(np.float32)
                sample['label'] = np.array([rotation_angle, translation_x, translation_y]).astype(np.float32)
                sample['true_len'] = true_len
                data.append(sample)
        return data

def get_train_dataloader(data_path, batch_size, prefetch_factor=8):

    data = PieceDataset(data_path)
    train_set_size, valid_set_size = int(0.9 * len(data)), len(data) - int(0.9 * len(data))
    train_data, valid_data = random_split(data, [train_set_size, valid_set_size])

    train_dataloader = DataLoader(train_data, num_workers=8, batch_size=batch_size,
                                  shuffle=True, prefetch_factor=prefetch_factor)
    valid_dataloader = DataLoader(valid_data, num_workers=32, batch_size=batch_size,
                                shuffle=False, prefetch_factor=prefetch_factor)
    return train_dataloader, valid_dataloader

    # while True:
    #     yield from train_dataloader


if __name__ == '__main__':
    data_path = '/work/csl/code/piece/dataset/jit_dataset/train.csv'
    train_dataloader, valid_dataloader = get_train_dataloader(data_path, batch_size=16)
    for i, (input_tensor, label, cond) in enumerate(train_dataloader):
        print(i)
        print(input_tensor.shape)
        print(label.shape)
        if i == 10:
            break
        

