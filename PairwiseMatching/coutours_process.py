import cv2
import numpy as np
import sys
sys.path.append("/data/csl/code/piece/PairwiseMatching")

from util import calculate_euclidean_distance, calculate_color_similarity, calculate_average_color

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


# 使用dp算法近似轮廓
def approx_contours(image, contour, beta=0.0008):
    '''
    Approximate contours using the dp algorithm
    '''
    # Find the farthest vertex v1 and v2
    max_distance = 0
    v1, v2 = 0, 0
    for i in range(len(contour)):
        for j in range(i + 1, len(contour)):
            # Calculate the Euclidean distance between two points
            distance = calculate_euclidean_distance(contour[i][0], contour[j][0])
            if distance > max_distance:
                max_distance = distance
                v1, v2 = i, j
    # print(f"v1:{contour[v1][0]}\nv2:{contour[v2][0]}\n")

    if v1 < v2:
        # Extract the contour of the v1v2 interval
        v1v2_contour = contour[v1:v2 + 1]  
        # Extract the contour of the v2v1 interval
        v2v1_contour = np.concatenate((contour[v2 + 1:], contour[:v1]))
    else:
        # Extract the contour of the v1v2 interval
        v1v2_contour = np.concatenate(contour[v1:], contour[:v2 + 1])
        # Extract the contour of the v2v1 interval
        v2v1_contour = contour[v2 + 1:v1]

    epsilon12 = beta * cv2.arcLength(v1v2_contour, True)
    epsilon21 = beta * cv2.arcLength(v2v1_contour, True)
    # Execute the DP algorithm on v1 to v2
    dp_v1v2 = cv2.approxPolyDP(v1v2_contour, epsilon12, closed=True)  
    # Execute the DP algorithm on v2 to v1
    dp_v2v1 = cv2.approxPolyDP(v2v1_contour, epsilon21, closed=True)
    # Combine the two contours  
    approx = np.concatenate((dp_v1v2, dp_v2v1))
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


# 分割轮廓曲线为多个曲线段
def split_contours(contour, approx):
    mmp = {}
    segment_si = []
    num_seg = len(approx)
    contour = np.squeeze(contour, axis=1)
    approx = np.squeeze(approx, axis=1)
     
    for i, point in enumerate(contour):
        mmp[tuple(point)] = i

    st, ed, i = 0, 0, 0
    while i < num_seg:
        st, ed = mmp[tuple(approx[i % num_seg])], mmp[tuple(approx[(i + 1) % num_seg])]
        i += 1
        if st > ed:
            break
        if len(contour[st:ed + 1]):
            segment_si.append(contour[st:ed + 1])
    if st > ed:
        l1 = list(contour[st:])
        l2 = list(contour[:ed + 1])
        l1.extend(l2)
        if len(l1):
            segment_si.append(np.array(l1))
    while i < num_seg:
        st, ed = mmp[tuple(approx[i % num_seg])], mmp[tuple(approx[(i + 1) % num_seg])]
        i += 1
        if len(contour[st:ed + 1]):
            segment_si.append(contour[st:ed + 1])
    return mmp, segment_si

# 根据颜色对集群Si进一步划分
def split_color(image, segments_si):
    # 定义颜色相似度的阈值
    color_similarity_threshold = 0.8

    # 存储划分后的段Sik列表
    segments_sik = []

    # 遍历每个片段Si
    for i, segment_si in enumerate(segments_si):
        # 初始化当前片段si的第一个段sik
        current_segment_sik = [segment_si[0]]  # 以片段si的第一个点作为当前段sik的起始点
        current_color = image[segment_si[0][1]][segment_si[0][0]]  # 当前段sik的颜色信息

        # 遍历片段Si的每个点（从第二个点开始）
        for j in range(1, len(segment_si)):
            point = segment_si[j]  # 当前点
            point_color = image[point[1]][point[0]]  # 当前点的颜色信息
            # 计算当前点与当前段Sik的颜色相似度
            color_similarity = calculate_color_similarity(point_color, current_color)

            # 如果颜色相似度超过阈值，则将当前点添加到当前段Sik中
            if color_similarity > color_similarity_threshold:
                current_segment_sik.append(point)
            else:
                # 如果颜色相似度不足，则将当前段sik添加到划分后的段sik列表中，并创建新的当前段sik
                segments_sik.append(current_segment_sik)
                current_segment_sik = [point]
                current_color = point_color

        # 将最后一个当前段sik添加到划分后的段sik列表中
        segments_sik.append([current_segment_sik])
    # 输出划分后的段sik列表
    # for i, segment_sik in enumerate(segments_sik):
    #     print(f"段Sik {i + 1}: {segment_sik}")

    # return segments_sik


def search_match_segments(img_1, img_2, segments_si1, segments_si2):
    point_cloud_1 = []
    point_cloud_2 = []
    length_threshold = 3
    color_thresshold = 0.98
    # 遍历每个片段Si
    for segment1_si in segments_si1:
        # 初始化当前片段si的第一个段sik
        len1 = len(segment1_si)
        for segment2_si in segments_si2:
            len2 = len(segment2_si)
            avg_color1, avg_color2 = calculate_average_color(img_1, segment1_si), calculate_average_color(img_2,
                                                                                                      segment2_si)
            color_similarity = calculate_color_similarity(avg_color1, avg_color2)
            if abs(len1 - len2) < length_threshold and color_similarity > color_thresshold:
                point_cloud_1.append(segment1_si)
                point_cloud_2.append(segment2_si)
    return point_cloud_1, point_cloud_2