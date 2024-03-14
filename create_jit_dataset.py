import glob
import csv
import numpy as np
import cv2
from tqdm import tqdm
import logging
import time
from multiprocessing import Pool, Lock, Value
import os, sys
sys.path.append('/data/csl/code/piece/JigsawNet/JigsawCNN')
from Utils import GtPose

def create_dataset(raw_dataset_path, training_dataset_path):
    global global_idx
    for path in raw_dataset_path:
        gt_pose = GtPose(os.path.join(path, "ground_truth.txt"))
        with open(os.path.join(path, "bg_color.txt"), "r") as f:
            line = f.readline().rstrip().split()
            bg_color = [int(i) for i in line][::-1]

        num_fragments = len(gt_pose.data)
        for i in range(num_fragments):
            for j in range(i + 1, num_fragments):
                path1 = os.path.join(path, f"fragment_{i + 1:04d}.png")
                path2 = os.path.join(path, f"fragment_{j + 1:04d}.png")
                gt_matrix = np.matmul(np.linalg.inv(gt_pose.data[i]), gt_pose.data[j])
                rotation_angle = np.arctan2(gt_matrix[0, 1], gt_matrix[0, 0])
                translate_x, translate_y = gt_matrix[1, 2], gt_matrix[0, 2]
                bg_color_string = " ".join([str(i) for i in bg_color])
                row_list = [path1, path2, bg_color_string, rotation_angle, translate_x, translate_y]
                with open(training_dataset_path, 'a+', newline='') as file:
                    csv_writer = csv.writer(file)
                    csv_writer.writerow(row_list)


if __name__ == '__main__':
    raw_dataset_path = glob.glob(os.path.join("/data/csl/dataset/jigsaw_dataset/raw_dataset", "*"))
    training_dataset_path = '/data/csl/dataset/jigsaw_dataset/12_25/traning_dataset.csv'
    with open(training_dataset_path, "w") as f:
        pass
    create_dataset(raw_dataset_path, training_dataset_path)