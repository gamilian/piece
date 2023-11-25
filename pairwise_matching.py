import cv2
import os
from coutours_process import *
from image_process import *


def search_pairwise_match_candidates(dataset, bg_color, num_fragment):
    print(f"dataset is {dataset}")
    for i in range(1, num_fragment - 1):
        # load image
        path1 = f"{dataset}/fragment_{str(i).rjust(4, '0')}.png"
        for j in range(i + 1, num_fragment):
            path2 = f"{dataset}/fragment_{str(j).rjust(4, '0')}.png"
            image1 = cv2.imread(path1)
            image2 = cv2.imread(path2)
            print(f"fusion image {i} and image {j}")
            # Calculate contour
            contours1, contours2 = get_coutours(image1, bg_color), get_coutours(image2, bg_color)
    
            # Approximate contour
            approx1, approx2 = approx_contours(image1, contours1[0]), approx_contours(image2, contours2[0])
    
            # split contour
            mmp1, segments_si1, = split_contours(contours1[0], approx1)
            mmp2, segments_si2 = split_contours(contours2[0], approx2)
            segments_sik1, segments_sik2 = split_color(image1, segments_si1), split_color(image2, segments_si2)
            
            # search match segments
            match_segments1, match_segments2 = search_match_segments(image1, image2, segments_si1, segments_si2)
            
            print(f"the num of match_segments is {len(match_segments1)}")
            prob_matrix, prb_score = [], []
            with open(f'{dataset}.txt', "a+", encoding='utf-8') as f:
                for k in range(len(match_segments1)):
                    matrix1, matrix2 = calculate_transform_matrix(match_segments1[k], match_segments2[k])

                    if fusion_image(image1, image2, matrix1, bg_color)[1] < 0.05:
                        score1 = calculate_prob(image1, image2, segments_si1, segments_si2,
                                                matrix1)
                        if score1 > 200:
                            prob_matrix.append(matrix1)
                            prb_score.append(score1)
                            f.write(f"{i} {j} {score1} ")
                            f.write("[[%f, %f, %f], [%f, %f, %f], [0, 0, 1]] line \n" % (
                                matrix1[0, 0], matrix1[0, 1], matrix1[0, 2], matrix1[1, 0], matrix1[1, 1],
                                matrix1[1, 2]))
                    if fusion_image(image1, image2, matrix2, bg_color)[1] < 0.05:
                        score2 = calculate_prob(image1, image2, segments_si1, segments_si2,
                                                matrix2)
                        if score2 > 200:
                            prob_matrix.append(matrix2)
                            prb_score.append(score2)
                            f.write(f"{i} {j} {score2} ")
                            f.write("[[%f, %f, %f], [%f, %f, %f], [0, 0, 1]] line \n" % (
                                matrix2[0, 0], matrix2[0, 1], matrix2[0, 2], matrix2[1, 0], matrix2[1, 1],
                                matrix2[1, 2]))

if __name__ == '__main__':
    # background color
    # bg_color = [232, 8, 248]
    bg_color = [8, 248, 8]
    dataset = "BGU_ex"
    # dataset = "MIT_ex"

    # Find how many fragments there are
    num_fragment = 0
    for filename in os.listdir(dataset):
        if filename.startswith("fragment") and filename.endswith(".png"):
            num_fragment += 1

    search_pairwise_match_candidates(dataset, bg_color, num_fragment)
                  
