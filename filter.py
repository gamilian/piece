import numpy as np
import cv2
import os
import operator
import logging
import time
from tqdm import tqdm
from PairwiseMatching.coutours_process import *
from PairwiseMatching.image_process import *
from PairwiseMatching.icp import icp
from PairwiseMatching.util import *
from multiprocessing import Pool, Manager


def setup_logging():
    """
    Configure the logging system to output to files and console at the same time
    """
    # Create a logger and set the log level
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    # Create a file handler and set the log level and format
    file_handler = logging.FileHandler("logs/my_log.log")
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(message)s')
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    # Create a stream processor and set log level and format
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)
    stream_formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(message)s')
    stream_handler.setFormatter(stream_formatter)
    logger.addHandler(stream_handler)

class Transform2d:
    def __init__(self, v1=-1, v2=-1, score=-1, transform=np.identity(3)):
        self.frame1 = v1
        self.frame2 = v2
        self.score = score
        self.transform = transform

def get_final_transform_pair(args):
    shared_data, data_path, bg_color, t = args
    v1, v2, score, initial_transform = t.frame1, t.frame2, t.score, t.transform
    initial_transformed = np.array([[initial_transform[0, 0], initial_transform[1, 0], initial_transform[1, 2]],
                                [initial_transform[0, 1], initial_transform[1, 1], initial_transform[0, 2]],
                                [0, 0, 1]])
    image_path1 = os.path.join(data_path, f"fragment_{str(v1 + 1).rjust(4, '0')}.png")
    image_path2 = os.path.join(data_path, f"fragment_{str(v2 + 1).rjust(4, '0')}.png")
    image1 = cv2.imread(image_path1)
    image2 = cv2.imread(image_path2)
    # Calculate contour
    contours1, contours2 = get_coutours(image1, bg_color), get_coutours(image2, bg_color)
    approx1, approx2 = approx_contours(image1, contours1[0], 0.001), approx_contours(image2, contours2[0], 0.001)
    # split contour
    mmp1, segments_si1, = split_contours(contours1[0], approx1)
    mmp2, segments_si2 = split_contours(contours2[0], approx2)
    points1, points2 = contours1[0].squeeze(1), contours2[0].squeeze(1)[::-1]
    lcs1, lcs2 = longest_common_subsequence(points1, points2, initial_transformed, 3)
    cloud1, cloud2 = np.array([points1[i] for i in lcs1]), np.array([points2[i] for i in lcs2])
    cloud2_transformed = np.dot(np.column_stack((cloud2, np.ones(len(cloud2)))), initial_transformed.T)[:, :2]
    final_transform = icp(cloud1, cloud2_transformed)
    final_transform = np.matmul(final_transform, initial_transformed)
    final_score = int(calculate_prob(image1, image2, segments_si1, segments_si2, final_transform, 2))
    final_transform = np.array([[final_transform[0, 0], final_transform[1, 0], final_transform[1, 2]],
                                [final_transform[0, 1], final_transform[1, 1], final_transform[0, 2]],
                                [0, 0, 1]])
    # lcs1, lcs2 = longest_common_continuous_subsequence_circular(points1, points2, final_transform, 4)
    # cloud2 = np.array([points2[i] for i in lcs2])
    # points = cloud2[::-1]
    if final_score < score:
        final_score = score
        final_transform = initial_transform
    if final_score > 80:
    # offset_maxtrix = fusion_image(image1, image2, matrix, bg_color)[2]
    # line = np.dot(np.column_stack((line, np.ones(len(line)))), offset_maxtrix.T)[:, :2]
        shared_data.append(Transform2d(v1, v2, final_score, final_transform))


def get_final_transform(data_path, processes=32):
    t1 = time.time()
    args_list = []
    vit_filter_alignments_file = os.path.join(data_path, "vit_filter_alignments.txt")
    with open(os.path.join(data_path, "bg_color.txt"), "r") as f:
        line = f.readline().rstrip().split()
        bg_color = [int(i) for i in line][::-1]
    with open(vit_filter_alignments_file) as f:
        all_line = [line.rstrip() for line in f]
        for line in all_line:
            if line[0:4] != "Node":
                data_str_list = line.split()
                v1,v2,score, m1,m2,m3,m4,m5,m6,m7,m8,m9 = [t(s) for t,s in zip((int,int, float, float,float,float,float,float,float,float,float,float), data_str_list[0:12])]
                transform = np.array([[m1,m2,m3], [m4,m5,m6], [m7,m8,m9]])
                args_list.append(Transform2d(v1, v2, score, transform))

    logging.info(f"args_list length: {len(args_list)}")
    logging.info("Starting get_final_transform")
    with Manager() as manager:
        shared_data = manager.list()
        args_list = [(shared_data, data_path, bg_color, t) for t in args_list]
        total_tasks = len(args_list)
        with Pool(processes=processes) as pool:
            results = list(tqdm(pool.imap(get_final_transform_pair, args_list), total=total_tasks))
        data = list(shared_data)
    t2 = time.time()
    logging.info(f"get_final_transform Complete complete, cost {t2-t1} seconds")
    max_score_dict = {}
    for item in data:
        key = (item.frame1, item.frame2)
        score = item.score
        # 如果这对 frame1 和 frame2 不存在于字典中，或者当前 score 更高，则更新字典
        if key not in max_score_dict or item.score > max_score_dict[key].score:
            max_score_dict[key] = item

    # filtered_data 现在包含了每对 frame1 和 frame2 中 score 最高的实例
    filtered_data = list(max_score_dict.values())
    alignments_file = os.path.join(data_path, "alignments.txt")
    with open(alignments_file, 'w+') as f:
        for item in filtered_data:
            frame1, frame2, score, trans = item.frame1, item.frame2, item.score, item.transform
            f.write(f"{frame1} {frame2} {score} ")
            f.write(f"{trans[0, 0]} {trans[0, 1]} {trans[0, 2]} {trans[1, 0]} {trans[1, 1]} {trans[1, 2]} 0 0 1\n")

if __name__=='__main__':
    data_path = "JigsawNet/Measure/hand_tear_image"
    num_works = 50
    setup_logging()
    get_final_transform(data_path, num_works)






