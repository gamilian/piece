import cv2
import os
import logging
import time
from multiprocessing import Pool
import config
from coutours_process import *
from image_process import *
from icp import icp
from util import *
from multiprocessing import Lock

write_lock = Lock()


def setup_logging():
    """
    Configure the logging system to output to files and console at the same time
    """
    # Create a logger and set the log level
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    # Create a file handler and set the log level and format
    file_handler = logging.FileHandler(config.log_path)
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

def process_image_pair(args):
    i, j, dataset_path, bg_color = args
    logging.info(f"Loading image {i} and image {j}")
    t1 = time.time()
    image_path1 = os.path.join(dataset_path, f"fragment_{str(i).rjust(4, '0')}.png")
    image_path2 = os.path.join(dataset_path, f"fragment_{str(j).rjust(4, '0')}.png")
    image1 = cv2.imread(image_path1)
    image2 = cv2.imread(image_path2)
    # Calculate contour
    contours1, contours2 = get_coutours(image1, bg_color), get_coutours(image2, bg_color)
    # mu1 = cv2.moments(contours1[0], True)
    # mu2 = cv2.moments(contours2[0], True)
    # mc1 = [mu1['m10'] / mu1['m00'], mu1['m01'] / mu1['m00']]
    # mc2 = [mu2['m10'] / mu2['m00'], mu2['m01'] / mu2['m00']]
    # Approximate contour
    approx1, approx2 = approx_contours(image1, contours1[0], 0.001), approx_contours(image2, contours2[0], 0.001)

    # split contour
    mmp1, segments_si1, = split_contours(contours1[0], approx1)
    mmp2, segments_si2 = split_contours(contours2[0], approx2)
    # segments_sik1, segments_sik2 = split_color(image1, segments_si1), split_color(image2, segments_si2)
    
    # search match segments
    match_segments1, match_segments2 = search_match_segments(image1, image2, segments_si1, segments_si2)
    
    logging.info(f"Find image {i} and image {j} transformation_matrix, the num of match_segments is {len(match_segments1)}")
    # prob_matrix, prb_score = [], []
    
    for k in range(len(match_segments1)):
        matrix = calculate_transform_matrix(match_segments1[k], match_segments2[k])

        if fusion_image(image1, image2, matrix, bg_color)[1] < 0.05:
            score = calculate_prob(image1, image2, segments_si1, segments_si2,
                                    matrix)
            if score > 50:
                # prob_matrix.append(matrix1)
                # prb_score.append(score1)
                points1, points2 = contours1[0].squeeze(1), contours2[0].squeeze(1)[::-1]
                lcs1, lcs2 = longest_common_subsequence(points1, points2, matrix, 2)
                cloud1, cloud2 = np.array([points1[i] for i in lcs1]), np.array([points2[i] for i in lcs2])
                cloud2_transformed = np.dot(np.column_stack((cloud2, np.ones(len(cloud2)))), matrix.T)[:, :2]
                final_transform = icp(cloud1, cloud2_transformed)
                final_transform = np.matmul(final_transform, matrix)
                final_score = int(calculate_prob(image1, image2, segments_si1, segments_si2, final_transform))
                # lcs1, lcs2 = longest_common_continuous_subsequence_circular(points1, points2, final_transform, 4)
                # cloud2 = np.array([points2[i] for i in lcs2])
                points = cloud2[::-1]
                # offset_maxtrix = fusion_image(image1, image2, matrix, bg_color)[2]
                # line = np.dot(np.column_stack((line, np.ones(len(line)))), offset_maxtrix.T)[:, :2]
                logging.info(f"lock the write mutex")
                with write_lock:
                    with open(config.alignments_file, "a+", encoding='utf-8') as f:
                        f.write(f"{i-1} {j-1} {final_score} ")
                        # f.write("%f %f %f %f %f %f 0.000000 0.000000 1.000000 line" % (
                        #     final_transform[0, 0], final_transform[0, 1], final_transform[0, 2], final_transform[1, 0], final_transform[1, 1],
                        #     final_transform[1, 2]))
                        f.write("%f %f %f %f %f %f 0.000000 0.000000 1.000000 line" % (
                            final_transform[0, 0], final_transform[1, 0], final_transform[1, 2], final_transform[0, 1], final_transform[1, 1],
                            final_transform[0, 2]))
                        for point in points:
                            f.write(f" {point[0]} {point[1]}")
                        f.write(f"\n")
        

    t2 = time.time()
    logging.info(f"Processing image {i} and image {j} complete, cost {t2-t1} seconds")


def search_pairwise_match_candidates(dataset_path, bg_color, num_fragment, processes):
    logging.info("Starting search_pairwise_match_candidates")
    args_list = [(i, j, dataset_path, bg_color) for i in range(1, num_fragment) for j in range(i + 1, num_fragment + 1)]
    
    with Pool(processes=processes) as pool:
        pool.map(process_image_pair, args_list)
    
    logging.info("Complete")
    

if __name__ == '__main__':
    num_fragment = 0
    setup_logging()

    t1 = time.time()
    logging.info("=====================================================================================================")
    logging.info("Starting main program")
    logging.info(f"Processing the dataset {config.dataset} with {config.num_fragment} fragments and {config.num_processes} processes")
    # clear the txt file 
    with open(config.alignments_file , "w+", encoding='utf-8') as f:
        for i in range(config.num_fragment):
            f.write(f"Node {i}\n")   

    search_pairwise_match_candidates(config.dataset_path, config.bg_color, config.num_fragment, config.num_processes)
    logging.info(f"Complete main program, cost {(time.time()-t1)//60} minutes {(time.time()-t1)%60} seconds")
    logging.info("=====================================================================================================\n\n")
