import os
import glob
import numpy as np
import cv2
from tqdm import tqdm

from JigsawNet.JigsawCNN.Utils import GtPose
from JigsawNet.JigsawCNN.PairwiseAlignment2Image import FusionImage2
from PairwiseMatching.coutours_process import *
from PairwiseMatching.image_process import calculate_prob
from multiprocessing import Pool, Lock, Value

# 写锁
write_lock = Lock()
# 进程池全局共享变量
global_idx = Value('i', 0)

def add_uniform_noise_to_rigid_transform_2d(matrix, angle_noise_mean, translation_noise_mean):
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
    
    # 减少重叠
    if translation_vector[0] > 0:
        flag_x = 1
    else:
        flag_x = -1
    if translation_vector[1] > 0:
        flag_y = 1
    else:   
        flag_y = -1

    noisy_rotation_angle = rotation_angle + np.random.uniform(-angle_noise_mean, angle_noise_mean)
    noisy_translation_vector = translation_vector + np.array([flag_x, flag_y])*np.random.uniform(0, translation_noise_mean, size=translation_vector.shape)
    # 重新构建带有噪声的旋转矩阵
    cos_angle = np.cos(noisy_rotation_angle)
    sin_angle = np.sin(noisy_rotation_angle)
    noisy_rotation_matrix = np.array([[cos_angle, sin_angle],
                                      [-sin_angle, cos_angle]])

    # 构建带有噪声的刚性变换矩阵
    noisy_transform = np.eye(3)
    noisy_transform[0:2, 0:2] = noisy_rotation_matrix
    noisy_transform[0:2, 2] = noisy_translation_vector
    return noisy_transform

def add_erro_to_rigid_transform_2d(matrix, angle_noise_low, angle_noise_high, translation_noise_low, translation_noise_high):
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
    # noisy_vec = np.array([flag_x, flag_y]) * np.random.normal(translation_noise_mean, translation_noise_variance, size=translation_vector.shape)
    # if np.random.randn() < 0:
    #     noisy_rotation_angle = rotation_angle + np.random.normal(angle_noise_mean, angle_noise_variance)
    #     noisy_translation_vector = translation_vector + noisy_vec
    # else:
    #     noisy_rotation_angle = rotation_angle + np.random.normal(-angle_noise_mean, angle_noise_variance)
    #     noisy_translation_vector = translation_vector + noisy_vec
    if np.random.randn() < 0:
        noisy_rotation_angle = rotation_angle + np.random.uniform(angle_noise_low, angle_noise_high)
    else:
        noisy_rotation_angle = rotation_angle - np.random.uniform(angle_noise_low, angle_noise_high)
    noisy_translation_vector = translation_vector + np.array([flag_x, flag_y])*np.random.uniform(translation_noise_low, translation_noise_high, size=translation_vector.shape)
    # 重新构建带有噪声的旋转矩阵
    cos_angle = np.cos(noisy_rotation_angle)
    sin_angle = np.sin(noisy_rotation_angle)
    noisy_rotation_matrix = np.array([[cos_angle, sin_angle],
                                      [-sin_angle, cos_angle]])

    # 构建带有噪声的刚性变换矩阵
    noisy_transform = np.eye(3)
    noisy_transform[0:2, 0:2] = noisy_rotation_matrix
    noisy_transform[0:2, 2] = noisy_translation_vector
    return noisy_transform

def process_path(args):
    global global_idx
    path, i, j, gt_pose, bg_color, training_dataset_path, num_positives, num_negatives = args
   
    image1 =  cv2.imread(os.path.join(path, f"fragment_{i + 1:04d}.png"))
    image2 =  cv2.imread(os.path.join(path, f"fragment_{j + 1:04d}.png"))
    gt_matrix = np.matmul(np.linalg.inv(gt_pose.data[i]), gt_pose.data[j])
    
    # contours1, contours2 = get_coutours(image1, bg_color), get_coutours(image2, bg_color)
    # approx1, approx2 = approx_contours(image1, contours1[0]), approx_contours(image2, contours2[0])
    # mmp1, segments_si1, = split_contours(contours1[0], approx1)
    # mmp2, segments_si2 = split_contours(contours2[0], approx2)
    # rigid_transform = np.array([[gt_matrix[0, 0], gt_matrix[1, 0], gt_matrix[1, 2]], [gt_matrix[0, 1], gt_matrix[1, 1], gt_matrix[0, 2]], [0, 0, 1]])
    # score = calculate_prob(image1, image2, segments_si1, segments_si2, rigid_transform)
    gt_item = FusionImage2(image1, image2, gt_matrix, bg_color)
    if gt_item[1] < 0.0005:
        return
    
    with global_idx.get_lock():
        current_dataset_path = os.path.join(training_dataset_path, f"{global_idx.value}")
        global_idx.value += 1

    if not os.path.exists(current_dataset_path):
        os.makedirs(current_dataset_path)
    with open(os.path.join(current_dataset_path, "target.txt"), "w+") as f:
        f.write("0 1\n")
        cv2.imwrite(os.path.join(current_dataset_path, "state.png"), gt_item[0])
    with open(os.path.join(current_dataset_path, "roi.txt"), "w+") as f:
        f.write(f"{gt_item[3][0]} {gt_item[3][1]} {gt_item[3][2]} {gt_item[3][3]}\n")

    for _ in range(num_positives - 1):
        noise_gt = add_uniform_noise_to_rigid_transform_2d(gt_matrix, 0.03, 1)
        noise_item = FusionImage2(image1, image2, noise_gt, bg_color)
        with global_idx.get_lock():
            current_dataset_path = os.path.join(training_dataset_path, f"{global_idx.value}")
            global_idx.value += 1
        if not os.path.exists(current_dataset_path):
            os.makedirs(current_dataset_path)
        with open(os.path.join(current_dataset_path, "target.txt"), "w+") as f:
            f.write("0 1\n")
            cv2.imwrite(os.path.join(current_dataset_path, "state.png"), noise_item[0])
        with open(os.path.join(current_dataset_path, "roi.txt"), "w+") as f:
            f.write(f"{gt_item[3][0]} {gt_item[3][1]} {gt_item[3][2]} {gt_item[3][3]}\n")
        
    
    for _ in range(num_negatives):
        erro_matrix = add_erro_to_rigid_transform_2d(gt_matrix, 0.5, 2.7, 3, 15)
        erro_item = FusionImage2(image1, image2, erro_matrix, bg_color)
        with global_idx.get_lock():
            current_dataset_path = os.path.join(training_dataset_path, f"{global_idx.value}")
            global_idx.value += 1
        if not os.path.exists(current_dataset_path):
            os.makedirs(current_dataset_path)
        with open(os.path.join(current_dataset_path, "target.txt"), "w+") as f:
            f.write("1 0\n")
            cv2.imwrite(os.path.join(current_dataset_path, "state.png"), erro_item[0])
        with open(os.path.join(current_dataset_path, "roi.txt"), "w+") as f:
            f.write(f"0 0 1 1\n")

            
def create_dataset(raw_dataset_path, training_dataset_path, num_positives=1, num_negatives=4, processes=16):
    global global_idx
    args_list = []
    for path in raw_dataset_path:
        gt_pose = GtPose(os.path.join(path, "groundTruth.txt"))
        with open (os.path.join(path, "bg_color.txt"), "r") as f:
            line = f.readline().rstrip().split()
            bg_color = [int(i) for i in line][::-1]
            
        num_fragments = len(gt_pose.data)
        for i in range(num_fragments):
            for j in range(i + 1, num_fragments):
                args_list.append((path, i, j, gt_pose, bg_color, training_dataset_path, num_positives, num_negatives))

    total_tasks = len(args_list)
    with Pool(processes=processes) as pool:
        results = list(tqdm(pool.imap(process_path, args_list), total=total_tasks))

    print(f"create dataset done, total dataset: {global_idx.value}")


if __name__ == '__main__':
    raw_dataset_path = glob.glob(os.path.join("dataset", "raw_dataset", "*"))
    # raw_dataset_path = 'dataset/raw_dataset/MIT_ex'
    num_positives = 10
    num_negatives = 20
    num_works = 1
    training_dataset_path = 'dataset/training_dataset/std'
    
    create_dataset(raw_dataset_path, training_dataset_path, num_positives, num_negatives, num_works)