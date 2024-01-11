import glob
import numpy as np
import cv2
from tqdm import tqdm
import logging
import time
from multiprocessing import Pool, Lock, Value
from coutours_process import *
from image_process import *
from util import *
import os, sys
sys.path.append('/work/csl/code/piece/JigsawNet/JigsawCNN')
from Utils import GtPose
from PairwiseAlignment2Image import FusionImage2

# 写锁
write_lock = Lock()
# 进程池全局共享变量
global_idx = Value('i', 0)
t = Value('i', 0)

def process_image_pair(path, i, j, bg_color, training_dataset_path, gt_trans, num_negatives):
    image1 = cv2.imread(os.path.join(path, f"fragment_{i + 1:04d}.png"))
    image2 = cv2.imread(os.path.join(path, f"fragment_{j + 1:04d}.png"))
    # Calculate contour
    contours1, contours2 = get_coutours(
        image1, bg_color), get_coutours(image2, bg_color)
    approx1, approx2 = approx_contours(
        image1, contours1[0], 0.001), approx_contours(image2, contours2[0], 0.001)

    # split contour
    mmp1, segments_si1, = split_contours(contours1[0], approx1)
    mmp2, segments_si2 = split_contours(contours2[0], approx2)
    # search match segments
    match_segments1, match_segments2 = search_match_segments(
        image1, image2, segments_si1, segments_si2)
    # prob_matrix, prb_score = [], []
    r_err_threshold, t_err_threshold = 4, 50
    for k in range(len(match_segments1)):
        if k >= num_negatives:
            break
        matrix = calculate_transform_matrix(
            match_segments1[k], match_segments2[k])
        trans = np.matrix([[matrix[0, 0], matrix[1, 0], matrix[1, 2]],
                           [matrix[0, 1], matrix[1, 1], matrix[0, 2]], [0, 0, 1]])
        item = FusionImage2(image1, image2, trans, bg_color)
        theta_err = np.abs(np.arccos(
            trans[0, 0]) * 180 / 3.1415926 - np.arccos(gt_trans[0, 0]) * 180 / 3.1415926)
        translation_err = np.sqrt(
            (trans[0, 2] - gt_trans[0, 2]) ** 2 + (trans[1, 2] - gt_trans[1, 2]) ** 2)
        if theta_err < r_err_threshold and translation_err < t_err_threshold:
            with global_idx.get_lock():
                image_path = os.path.join(
                    training_dataset_path, "image", f"{global_idx.value}.png")
                global_idx.value += 1
                with open(os.path.join(training_dataset_path, "target.txt"), "a+") as f:
                    f.write(f"1\n")
            cv2.imwrite(image_path, item[0])
        else:
            with global_idx.get_lock():
                image_path = os.path.join(
                    training_dataset_path, "image", f"{global_idx.value}.png")
                global_idx.value += 1
                with open(os.path.join(training_dataset_path, "target.txt"), "a+") as f:
                    f.write(f"0\n")
            cv2.imwrite(image_path, item[0])


def erro_generate(path, i, j, bg_color, training_dataset_path, num_negatives):
    image1 = cv2.imread(os.path.join(path, f"fragment_{i + 1:04d}.png"))
    image2 = cv2.imread(os.path.join(path, f"fragment_{j + 1:04d}.png"))
    # Calculate contour
    contours1, contours2 = get_coutours(
        image1, bg_color), get_coutours(image2, bg_color)
    approx1, approx2 = approx_contours(
        image1, contours1[0], 0.001), approx_contours(image2, contours2[0], 0.001)

    # split contour
    mmp1, segments_si1, = split_contours(contours1[0], approx1)
    mmp2, segments_si2 = split_contours(contours2[0], approx2)
    # search match segments
    match_segments1, match_segments2 = search_match_segments(
        image1, image2, segments_si1, segments_si2)
    # prob_matrix, prb_score = [], []
    for k in range(len(match_segments1)):
        if k >= num_negatives//8:
            break
        matrix = calculate_transform_matrix(
            match_segments1[k], match_segments2[k])
        matrix = [[matrix[0, 0], matrix[1, 0], matrix[1, 2]],
                  [matrix[0, 1], matrix[1, 1], matrix[0, 2]], [0, 0, 1]]
        item = FusionImage2(image1, image2, matrix, bg_color)
        with global_idx.get_lock():
            image_path = os.path.join(
                training_dataset_path, "image", f"{global_idx.value}.png")
            global_idx.value += 1
            with open(os.path.join(training_dataset_path, "target.txt"), "a+") as f:
                f.write(f"0\n")
        cv2.imwrite(image_path, item[0])


def add_uniform_noise_to_rigid_transform_2d(matrix, angle_noise_low, angle_noise_high, translation_noise_low, translation_noise_high):
    '''
    不是标准刚性矩阵
    '''
    if matrix.shape != (3, 3):
        raise ValueError("Input matrix must be 3x3.")

    # 提取旋转矩阵和平移向量
    rotation_matrix = matrix[0:2, 0:2]
    translation_vector = matrix[0:2, 2]

    # 从旋转矩阵计算旋转角度
    rotation_angle = np.arctan2(rotation_matrix[0, 1], rotation_matrix[0, 0])
    # 重新构建带有噪声的旋转矩阵
    if np.random.randn() < 0:
        noisy_rotation_angle = np.random.uniform(
            angle_noise_low, angle_noise_high)
    else:
        noisy_rotation_angle = - \
            np.random.uniform(angle_noise_low, angle_noise_high)
    rotation_angle = rotation_angle + noisy_rotation_angle
    cos_angle = np.cos(rotation_angle)
    sin_angle = np.sin(rotation_angle)
    rotation_matrix = np.array([[cos_angle, sin_angle],
                                [-sin_angle, cos_angle]])
    # 重新构建带有噪声的旋转矩阵
    noisy_translation_vector = np.random.uniform(
        translation_noise_low, translation_noise_high)
    translation_vector = translation_vector + \
        np.array([sin_angle, cos_angle]) * \
        np.array([0, noisy_translation_vector])

    # 构建带有噪声的刚性变换矩阵
    noisy_transform = np.eye(3)
    noisy_transform[0:2, 0:2] = rotation_matrix
    noisy_transform[0:2, 2] = translation_vector
    return noisy_transform


def add_erro_to_rigid_transform_2d(matrix, angle_noise_mean, angle_noise_variance, translation_noise_mean, translation_noise_variance):
    '''
    不是标准刚性矩阵
    '''
    if matrix.shape != (3, 3):
        raise ValueError("Input matrix must be 3x3.")

    # 提取旋转矩阵和平移向量
    rotation_matrix = matrix[0:2, 0:2]
    translation_vector = matrix[0:2, 2]

    # 从旋转矩阵计算旋转角度
    rotation_angle = np.arctan2(rotation_matrix[0, 1], rotation_matrix[0, 0])

    # 向旋转角度和平移向量添加噪声
    # noisy_rotation_angle = rotation_angle + np.random.normal(0, angle_noise_level)
    # noisy_translation_vector = translation_vector + np.random.normal(0, translation_noise_level, size=translation_vector.shape)
    # 双峰分布代替gaussian分布
    # 减少重叠
    if translation_vector[0] > 0:
        flag_x = 1
    else:
        flag_x = -1
    if translation_vector[1] > 0:
        flag_y = 1
    else:
        flag_y = -1
    noisy_vec = np.array([flag_x, flag_y]) * np.random.normal(translation_noise_mean,
                                                              translation_noise_variance, size=translation_vector.shape)
    if np.random.randn() < 0:
        rotation_angle = rotation_angle + \
            np.random.normal(angle_noise_mean, angle_noise_variance)
        translation_vector = translation_vector + noisy_vec
    else:
        rotation_angle = rotation_angle + \
            np.random.normal(-angle_noise_mean, angle_noise_variance)
        translation_vector = translation_vector + noisy_vec

    cos_angle = np.cos(rotation_angle)
    sin_angle = np.sin(rotation_angle)
    rotation_matrix = np.array([[cos_angle, sin_angle],
                                [-sin_angle, cos_angle]])

    # 构建带有噪声的刚性变换矩阵
    noisy_transform = np.eye(3)
    noisy_transform[0:2, 0:2] = rotation_matrix
    noisy_transform[0:2, 2] = translation_vector
    return noisy_transform


def process_path(args):
    global global_idx
    global t
    path, i, j, gt_pose, bg_color, training_dataset_path, num_positives, num_negatives = args

    image1 = cv2.imread(os.path.join(path, f"fragment_{i + 1:04d}.png"))
    image2 = cv2.imread(os.path.join(path, f"fragment_{j + 1:04d}.png"))
    gt_matrix = np.matmul(np.linalg.inv(gt_pose.data[i]), gt_pose.data[j])
    gt_item = FusionImage2(image1, image2, gt_matrix, bg_color)
    if gt_item[1] < 0.0005:
        erro_generate(path, i, j, bg_color, training_dataset_path, num_negatives)
        return

    with global_idx.get_lock():
        image_path = os.path.join(
            training_dataset_path, "image", f"{global_idx.value}.png")
        global_idx.value += 1
        with open(os.path.join(training_dataset_path, "target.txt"), "a+") as f:
            f.write(f"1\n")
    cv2.imwrite(image_path, gt_item[0])

    for _ in range(num_positives//2):
        noise_gt = add_uniform_noise_to_rigid_transform_2d(
            gt_matrix, 0, 0.03, 0, 3)
        noise_item = FusionImage2(image1, image2, noise_gt, bg_color)
        with global_idx.get_lock():
            image_path = os.path.join(
                training_dataset_path, "image", f"{global_idx.value}.png")
            global_idx.value += 1
            with open(os.path.join(training_dataset_path, "target.txt"), "a+") as f:
                f.write(f"1\n")
        cv2.imwrite(image_path, noise_item[0])

    for _ in range(num_positives):
        erro_matrix = add_uniform_noise_to_rigid_transform_2d(
            gt_matrix, 0.03, 0.06, 4, 25)
        erro_item = FusionImage2(image1, image2, erro_matrix, bg_color)
        with global_idx.get_lock():
            image_path = os.path.join(
                training_dataset_path, "image", f"{global_idx.value}.png")
            global_idx.value += 1
            with open(os.path.join(training_dataset_path, "target.txt"), "a+") as f:
                f.write(f"1\n")
        cv2.imwrite(image_path, erro_item[0])

    for _ in range(num_negatives//8):
        erro_matrix = add_uniform_noise_to_rigid_transform_2d(
            gt_matrix, 0.07, 0.12, 20, 50)
        erro_item = FusionImage2(image1, image2, erro_matrix, bg_color)
        with global_idx.get_lock():
            image_path = os.path.join(
                training_dataset_path, "image", f"{global_idx.value}.png")
            global_idx.value += 1
            with open(os.path.join(training_dataset_path, "target.txt"), "a+") as f:
                f.write(f"0\n")
        cv2.imwrite(image_path, erro_item[0])

  
    process_image_pair(path, i, j, bg_color, training_dataset_path, gt_matrix, num_negatives)


def create_dataset(raw_dataset_path, training_dataset_path, num_positives=1, num_negatives=4, processes=16):
    global global_idx
    args_list = []
    for path in raw_dataset_path:
        gt_pose = GtPose(os.path.join(path, "ground_truth.txt"))
        with open(os.path.join(path, "bg_color.txt"), "r") as f:
            line = f.readline().rstrip().split()
            bg_color = [int(i) for i in line][::-1]

        num_fragments = len(gt_pose.data)
        for i in range(num_fragments):
            for j in range(i + 1, num_fragments):
                args_list.append((path, i, j, gt_pose, bg_color,
                                 training_dataset_path, num_positives, num_negatives))

    total_tasks = len(args_list)
    print(f"create dataset, total_tasks: {total_tasks}")
    with Pool(processes=processes) as pool:
        results = list(
            tqdm(pool.imap(process_path, args_list), total=total_tasks))

    print(f"create dataset done, total dataset: {global_idx.value}")


if __name__ == '__main__':
    raw_dataset_path = glob.glob(os.path.join("/work/csl/code/piece/dataset/puzzles", "*"))
    # raw_dataset_path = glob.glob(os.path.join("/work/csl/code/piece/dataset/raw_dataset", "*_*"))
    # raw_dataset_path = glob.glob(os.path.join("/work/csl/code/piece/dataset/", "6*"))
    num_positives = 1000
    num_negatives = 520
    num_works = 64
    # print(raw_dataset_path)
    training_dataset_path = '/work/csl/code/piece/dataset/szp_train_dataset'
    # training_dataset_path = '/work/csl/code/piece/dataset/tmp_test'
    with open(os.path.join(training_dataset_path, "target.txt"), "w+") as f:
        pass
    create_dataset(raw_dataset_path, training_dataset_path,
                   num_positives, num_negatives, num_works)
