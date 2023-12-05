import cv2
import numpy as np
import sys
sys.path.append("/work/csl/code/piece/PairwiseMatching")
from util import calculate_color_similarity, calculate_average_color

# calculate the rigid transformation matrix
def calculate_transform_matrix(cloud1, cloud2):
    pt1, pt2 = cloud1[0], cloud1[-1]
    pt3, pt4 = cloud2[0], cloud2[-1]
    # Calculate the center of mass
    center1 = ((pt1[0] + pt2[0]) / 2, (pt1[1] + pt2[1]) / 2)
    center2 = ((pt3[0] + pt4[0]) / 2, (pt3[1] + pt4[1]) / 2)

    # Calculate the angle
    angle1 = np.arctan2(pt2[1] - pt1[1], pt2[0] - pt1[0])
    angle2 = np.arctan2(pt3[1] - pt4[1], pt3[0] - pt4[0])
    # Calculate rotation angle
    rotation_angle = angle1 - angle2
    rotation_matrix = np.array([[np.cos(rotation_angle), -np.sin(rotation_angle)],
                                [np.sin(rotation_angle), np.cos(rotation_angle)]])
    
    # Apply rotation matrix and calculate translation vector
    rotated_center = rotation_matrix@(np.array([center2[0], center2[1]]))
    translation_vector = np.array([center1[0] - rotated_center[0], center1[1] - rotated_center[1]])
    # Construct rigid transformation matrix
    rigid_transform_matrix = np.eye(3)
    # Place the rotation matrix and translation vector into a rigid transformation matrix
    rigid_transform_matrix[:2, :2] = rotation_matrix
    rigid_transform_matrix[:2, 2] = translation_vector
    return rigid_transform_matrix



# Splice dst to src
def fusion_image(src, dst, transform, bg_color=[0, 0, 0]):
    black_bg = [0, 0, 0]
    if bg_color != black_bg:
        # Set pixels in src and dst that have the same color as the background to black
        src[np.where((src == bg_color).all(axis=2))] = [0, 0, 0]
        dst[np.where((dst == bg_color).all(axis=2))] = [0, 0, 0]

    # Get the position of the non-black (actual image) pixels in the dst image
    color_idx = np.where((dst != black_bg).any(axis=2))

    # Increase dimensions to facilitate matrix multiplication
    one = np.ones(len(color_idx[0]))
    color_idx = list(color_idx)
    color_idx = [color_idx[1], color_idx[0]]
    color_idx.append(one)
    color_idx = np.array(color_idx)

    # Apply rigid transformation to color index
    transformed_points = np.matmul(transform, color_idx)
    try:
        # Compute the bounding box of the transformed image
        dst_min_col = np.floor(np.min(transformed_points[0])).astype(int)
        dst_min_row = np.floor(np.min(transformed_points[1])).astype(int)
        dst_max_col = np.ceil(np.max(transformed_points[0])).astype(int)
        dst_max_row = np.ceil(np.max(transformed_points[1])).astype(int)
    except ValueError:
        return []
    # print(dst_min_row, dst_min_col, dst_max_row, dst_max_col)
    # Compute global bounding box
    src_color_indices = np.where((src != black_bg).any(axis=2))
    try:
        src_min_row = np.floor(np.min(src_color_indices[0])).astype(int)
        src_min_col = np.floor(np.min(src_color_indices[1])).astype(int)
        src_max_row = np.ceil(np.max(src_color_indices[0])).astype(int)
        src_max_col = np.ceil(np.max(src_color_indices[1])).astype(int)
    except ValueError:
        return []

    # Calculate the minimum bounding box
    min_row = min(dst_min_row, src_min_row)
    max_row = max(dst_max_row, src_max_row)
    min_col = min(dst_min_col, src_min_col)
    max_col = max(dst_max_col, src_max_col)
    offset_row = -min_row
    offset_col = -min_col

    # the offset rigid transformation, also the transformation for the src image
    offset_transform = np.float32([[1, 0, offset_col], [0, 1, offset_row]])

    # the rigid transformation for the dst image
    dst_transform = np.matmul(np.matrix([[1, 0, offset_col], [0, 1, offset_row], [0, 0, 1]]), transform)
    # Convert the coordinate system to coordinates in open cv
    dst_transform = np.float32([[dst_transform[0, 0], dst_transform[0, 1], dst_transform[0, 2]],
                                [dst_transform[1, 0], dst_transform[1, 1], dst_transform[1, 2]]])

    # Apply rigid transformation
    src_transformed = cv2.warpAffine(src, offset_transform, (max_col - min_col, max_row - min_row))
    dst_transformed = cv2.warpAffine(dst, dst_transform, (max_col - min_col, max_row - min_row))

    # Overlap detection
    a = np.all(src_transformed == black_bg, axis=2)
    b = np.all(dst_transformed != black_bg, axis=2)
    c = np.logical_and(a, b)
    c = c.reshape((c.shape[0], c.shape[1]))
    non_overlap_indices = np.where(c)
    if len(np.where(b)[0]) == 0:
        assert False and "no valid pixels in transformed dst image, please check the transform process"
    else:
        overlap_ratio = 1 - len(non_overlap_indices[0]) / len(np.where(b)[0])

    # Splicing, replace the pixel corresponding to src with dst
    bg_indices = np.where(a)
    src_transformed[bg_indices] = dst_transformed[bg_indices]

    return [src_transformed, overlap_ratio, offset_transform]


# 计算刚性变换的成对兼容度
def calculate_prob(image1, image2, segments_sik1, segments_sik2, transform, threshold=3.6, alp=5):

    w1, w2, w3 = 0, 0, 0
    for segment_sik1 in segments_sik1:
        if len(segment_sik1) < 3:
            continue
        for segment_sik2 in segments_sik2:
            if len(segment_sik2) < 3:
                continue
            center_point1, center_point2 = np.mean(segment_sik1, axis=0), np.mean(segment_sik2, axis=0)
            center_point2 = np.append(center_point2, 1)
            # 对质心进行刚性变换
            center_point2_translate = np.matmul(transform, center_point2)
            center_point2 = center_point2_translate[:2]
            if np.linalg.norm(center_point1 - center_point2) < threshold:
                l1, l2 = len(segment_sik1), len(segment_sik2)
                w1 += (l1 + l2) / 2
                c1, c2 = calculate_average_color(image1, segment_sik1), calculate_average_color(image2, segment_sik2)
                # w2 += (l1 + l2) / 2 * (1 - compute_color_similarity(c1, c2))
                similarity = calculate_color_similarity(c1, c2)
                # if similarity > 0.98:
                #     w2 += similarity
                w2 += (l1 + l2) / 2 * similarity
                w3 += 1

    # if w2 < 0.3:
    #     return 0
    # w = (w1 + alp * w3) / w2
    # w2 /= w3
    w = w2 + alp * w3
    # print(f"w1 = {w1}， w2 = {w2}， w3 = {w3}， w = {w}")
    return w
