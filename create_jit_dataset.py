import glob
import csv
import numpy as np
import cv2
from tqdm import tqdm
import logging
import time
from multiprocessing import Pool, Lock, Value
import os, sys
sys.path.append('/work/csl/code/piece/JigsawNet/JigsawCNN')
from Utils import GtPose
# # 写锁
# write_lock = Lock()


# def process_path(args):
#     global global_idx
#     path, i, j, gt_pose, bg_color, training_dataset_path = args
#     path1 = os.path.join(path, f"fragment_{i + 1:04d}.png")
#     path2 = os.path.join(path, f"fragment_{j + 1:04d}.png")
#     image1 = cv2.imread(path1)
#     image2 = cv2.imread(path2)
#     gt_matrix = np.matmul(np.linalg.inv(gt_pose.data[i]), gt_pose.data[j])
#     rotation_angle = np.arctan2(gt_matrix[0, 1], gt_matrix[0, 0])
#     translate_x, translate_y = gt_matrix[1, 2], gt_matrix[0, 2]
#     row_list = [path1, path2, bg_color, rotation_angle, translate_x, translate_y]
#     with open(training_dataset_path, 'a+', newline='') as file:
#         with write_lock:
#             csv_writer = csv.writer(file)
#             csv_writer.writerow(row_list)



# def create_dataset(raw_dataset_path, training_dataset_path, processes=16):
#     global global_idx
#     args_list = []
#     for path in raw_dataset_path:
#         gt_pose = GtPose(os.path.join(path, "ground_truth.txt"))
#         with open(os.path.join(path, "bg_color.txt"), "r") as f:
#             line = f.readline().rstrip().split()
#             bg_color = [int(i) for i in line][::-1]

#         num_fragments = len(gt_pose.data)
#         for i in range(num_fragments):
#             for j in range(i + 1, num_fragments):
#                 args_list.append((path, i, j, gt_pose, bg_color,
#                                  training_dataset_path))

#     # args_list = [args_list[4]]
#     total_tasks = len(args_list)
#     print(f"create dataset, total_tasks: {total_tasks}")
#     with Pool(processes=processes) as pool:
#         results = list(
#             tqdm(pool.imap(process_path, args_list), total=total_tasks))

#     print(f"create dataset done, total dataset: {total_tasks}")


# if __name__ == '__main__':
#     raw_dataset_path = glob.glob(os.path.join("/work/csl/code/piece/dataset/raw_dataset", "*"))
#     training_dataset_path = '/work/csl/code/piece/dataset/12_24'
#     with open(os.path.join(training_dataset_path, "target.txt"), "w+") as f:
#         pass
#     create_dataset(raw_dataset_path, training_dataset_path, num_works)



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
    raw_dataset_path = glob.glob(os.path.join("/work/csl/code/piece/dataset/raw_dataset", "*"))
    training_dataset_path = '/work/csl/code/piece/dataset/12_25/traning_dataset.csv'
    with open(training_dataset_path, "w") as f:
        pass
    create_dataset(raw_dataset_path, training_dataset_path)