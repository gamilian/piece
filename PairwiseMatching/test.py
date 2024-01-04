import cv2
import numpy as np
from coutours_process import get_coutours, approx_contours, split_contours, split_color, search_match_segments
from image_process import fusion_image, calculate_transform_matrix, calculate_prob



if __name__ == '__main__':
    # 背景颜色
    # bg_color = [232, 8, 248]
    bg_color = [8, 248, 8]
    dataset = "BGU_ex"
    # 读取图片
    path1 = f"{dataset}/fragment_{str(1).rjust(4, '0')}.png"
    path2 = f"{dataset}/fragment_{str(2).rjust(4, '0')}.png"
    image1 = cv2.imread(path1)
    image2 = cv2.imread(path2)
    print(f"fusion image {1} and image {2} \n")
    # 计算轮廓
    contours1, contours2 = get_coutours(image1, bg_color), get_coutours(image2, bg_color)

    #  近似轮廓
    approx1, approx2 = approx_contours(image1, contours1[0]), approx_contours(image2, contours2[0])

    # 分割集群
    mmp1, segments_si1, = split_contours(contours1[0], approx1)
    mmp2, segments_si2 = split_contours(contours2[0], approx2)
    segments_sik1, segments_sik2 = split_color(image1, segments_si1), split_color(image2, segments_si2)
    match_segments1, match_segments2 = search_match_segments(image1, image2, segments_si1, segments_si2)
    
    print(f"the num of match_segments is {len(match_segments1)}")
    prob_matrix, prb_score = [], []
    with open('mit.txt', "a+", encoding='utf-8') as f:
        for k in range(len(match_segments1)):
            matrix1, matrix2 = calculate_transform_matrix(match_segments1[k], match_segments2[k])
        
            if fusion_image(image1, image2, matrix1, bg_color)[1] < 0.09:
                score1 = calculate_prob(image1, image2, segments_si1, segments_si2,
                                        matrix1)
                if score1 > 200:
                    prob_matrix.append(matrix1)
                    prb_score.append(score1)
                    f.write(f"{1} {2} {score1} {match_segments1[k][0]} , {match_segments1[k][-1]}, {match_segments2[k][0]} , {match_segments2[k][-1]} ")
                    f.write("[[%f, %f, %f], [%f, %f, %f], [0, 0, 1]] line \n" % (
                        matrix1[0, 0], matrix1[0, 1], matrix1[0, 2], matrix1[1, 0], matrix1[1, 1],
                        matrix1[1, 2]))
            
            if fusion_image(image1, image2, matrix2, bg_color)[1] < 0.09:
                score2 = calculate_prob(image1, image2, segments_si1, segments_si2,
                                        matrix2)
                if score2 > 200:
                    prob_matrix.append(matrix2)
                    prb_score.append(score2)
                    f.write(f"{1} {2} {score2} ")
                    f.write("[[%f, %f, %f], [%f, %f, %f], [0, 0, 1]] line \n" % (
                        matrix2[0, 0], matrix2[0, 1], matrix2[0, 2], matrix2[1, 0], matrix2[1, 1],
                        matrix2[1, 2]))
            

   
    # # 将image2拼接到image1上
    # item = fusion_image(image1, image2, transformation_matrix, bg_color)
    # score = calculate_prob(image1, image2, segments_si1, segments_si2, transformation_matrix)
    # print(item[1], score)
    # # 显示图片
    # cv2.imshow('Result', item[0])
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
