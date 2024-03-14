'''
Convert pairwise alignment transformation to one stitched image
'''

import cv2
import numpy as np

'''
transform dst to src
'''
def FusionImage2(src, dst, transform, bg_color=[0,0,0]):
    black_bg = [0,0,0]
    if bg_color!=black_bg:
        src[np.where((src == bg_color).all(axis=2))] = [0,0,0]
        dst[np.where((dst == bg_color).all(axis=2))] = [0, 0, 0]

    color_indices = np.where((dst != black_bg).any(axis=2))
    color_pt_num = len(color_indices[0])
    one = np.ones(color_pt_num)

    color_indices = list(color_indices)
    color_indices.append(one)
    color_indices = np.array(color_indices)

    transformed_lin_pts = np.matmul(transform, color_indices)
    # bounding box after transform
    try:
        dst_min_row = np.floor(np.min(transformed_lin_pts[0])).astype(int)
        dst_min_col = np.floor(np.min(transformed_lin_pts[1])).astype(int)
        dst_max_row = np.ceil(np.max(transformed_lin_pts[0])).astype(int)
        dst_max_col = np.ceil(np.max(transformed_lin_pts[1])).astype(int)
    except ValueError:
        return []       # the src or dst image has the same color with background. e.g totally black.

    # global bounding box
    src_color_indices = np.where((src != black_bg).any(axis=2))
    try:
        src_min_row = np.floor(np.min(src_color_indices[0])).astype(int)
        src_min_col = np.floor(np.min(src_color_indices[1])).astype(int)
        src_max_row = np.ceil(np.max(src_color_indices[0])).astype(int)
        src_max_col = np.ceil(np.max(src_color_indices[1])).astype(int)
    except ValueError:
        return []       # the src or dst image has the same color with background. e.g totally black.

    min_row = min(dst_min_row, src_min_row)
    max_row = max(dst_max_row, src_max_row)
    min_col = min(dst_min_col, src_min_col)
    max_col = max(dst_max_col, src_max_col)

    offset_row = -min_row
    offset_col = -min_col

    offset_transform = np.float32([[1,0,offset_col],[0,1,offset_row]])
    dst_transform = np.matmul(np.matrix([[1,0,offset_row],[0,1,offset_col],[0,0,1]]), transform)
    # convert row, col to opencv x,y
    dst_transform = np.float32([[dst_transform[0,0], dst_transform[1,0], dst_transform[1,2]], [dst_transform[0,1], dst_transform[1,1], dst_transform[0,2]]])

    src_transformed = cv2.warpAffine(src, offset_transform, (max_col-min_col, max_row-min_row))
    dst_transformed = cv2.warpAffine(dst, dst_transform, (max_col-min_col, max_row-min_row))

    # overlap detection
    a = np.all(src_transformed == black_bg, axis=2)
    b = np.all(dst_transformed != black_bg, axis=2)
    c = np.logical_and(a, b)
    c = c.reshape((c.shape[0], c.shape[1]))
    non_overlap_indices = np.where(c)
    if len(np.where(b)[0]) == 0:
        assert False and "no valid pixels in transformed dst image, please check the transform process"
    else:
        overlap_ratio = 1 - len(non_overlap_indices[0]) / len(np.where(b)[0])

    # fusion
    bg_indices = np.where(a)
    src_transformed[bg_indices] = dst_transformed[bg_indices]

    offset_transform_matrix = np.float32([[1, 0, offset_row], [0, 1, offset_col], [0,0,1]])
    return [src_transformed, overlap_ratio, offset_transform_matrix]

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


def FusionImage2(src, dst, transform, bg_color=[0,0,0]):
    black_bg = [0,0,0]
    if bg_color!=black_bg:
        src[np.where((src == bg_color).all(axis=2))] = [0,0,0]
        dst[np.where((dst == bg_color).all(axis=2))] = [0, 0, 0]

    color_indices = np.where((dst != black_bg).any(axis=2))
    color_pt_num = len(color_indices[0])
    one = np.ones(color_pt_num)

    color_indices = list(color_indices)
    color_indices.append(one)
    color_indices = np.array(color_indices)

    transformed_lin_pts = np.matmul(transform, color_indices)
    # bounding box after transform
    try:
        dst_min_row = np.floor(np.min(transformed_lin_pts[0])).astype(int)
        dst_min_col = np.floor(np.min(transformed_lin_pts[1])).astype(int)
        dst_max_row = np.ceil(np.max(transformed_lin_pts[0])).astype(int)
        dst_max_col = np.ceil(np.max(transformed_lin_pts[1])).astype(int)
    except ValueError:
        return []       # the src or dst image has the same color with background. e.g totally black.

    # global bounding box
    src_color_indices = np.where((src != black_bg).any(axis=2))
    try:
        src_min_row = np.floor(np.min(src_color_indices[0])).astype(int)
        src_min_col = np.floor(np.min(src_color_indices[1])).astype(int)
        src_max_row = np.ceil(np.max(src_color_indices[0])).astype(int)
        src_max_col = np.ceil(np.max(src_color_indices[1])).astype(int)
    except ValueError:
        return []       # the src or dst image has the same color with background. e.g totally black.

    min_row = min(dst_min_row, src_min_row)
    max_row = max(dst_max_row, src_max_row)
    min_col = min(dst_min_col, src_min_col)
    max_col = max(dst_max_col, src_max_col)

    offset_row = -min_row
    offset_col = -min_col

    offset_transform = np.float32([[1,0,offset_col],[0,1,offset_row]])
    dst_transform = np.matmul(np.matrix([[1,0,offset_row],[0,1,offset_col],[0,0,1]]), transform)
    # convert row, col to opencv x,y
    dst_transform = np.float32([[dst_transform[0,0], dst_transform[1,0], dst_transform[1,2]], [dst_transform[0,1], dst_transform[1,1], dst_transform[0,2]]])

    src_transformed = cv2.warpAffine(src, offset_transform, (max_col-min_col, max_row-min_row))
    dst_transformed = cv2.warpAffine(dst, dst_transform, (max_col-min_col, max_row-min_row))

    # overlap detection
    a = np.all(src_transformed == black_bg, axis=2)
    b = np.all(dst_transformed != black_bg, axis=2)
    d = np.where(np.all(src_transformed != black_bg, axis=2))
    f = np.where(b)
    src_roi_min_row = np.floor(np.min(d[0]))
    src_roi_min_col = np.floor(np.min(d[1]))
    src_roi_max_row = np.ceil(np.max(d[0]))
    src_roi_max_col = np.ceil(np.max(d[1]))
    
    dst_roi_min_row = np.floor(np.min(f[0]))
    dst_roi_min_col = np.floor(np.min(f[1]))
    dst_roi_max_row = np.ceil(np.max(f[0]))
    dst_roi_max_col = np.ceil(np.max(f[1]))

    min_roi_row = max(src_roi_min_row, dst_roi_min_row)
    max_roi_row = min(src_roi_max_row, dst_roi_max_row)
    min_roi_col = max(src_roi_min_col, dst_roi_min_col)
    max_roi_col = min(src_roi_max_col, dst_roi_max_col)

    cols, rows = max_col-min_col, max_row-min_row
    new_min_row_ratio = min_roi_row/rows
    new_min_col_ratio = min_roi_col/cols
    new_max_row_ratio = max_roi_row/rows
    new_max_col_ratio = max_roi_col/cols
    roi = (new_min_col_ratio, new_min_row_ratio, new_max_col_ratio, new_max_row_ratio)


    c = np.logical_and(a, b)
    c = c.reshape((c.shape[0], c.shape[1]))
    non_overlap_indices = np.where(c)
    if len(np.where(b)[0]) == 0:
        assert False and "no valid pixels in transformed dst image, please check the transform process"
    else:
        overlap_ratio = 1 - len(non_overlap_indices[0]) / len(np.where(b)[0])

    # fusion
    bg_indices = np.where(a)
    src_transformed[bg_indices] = dst_transformed[bg_indices]

    offset_transform_matrix = np.float32([[1, 0, offset_row], [0, 1, offset_col], [0,0,1]])
    return [src_transformed, overlap_ratio, offset_transform_matrix, roi]


if __name__=="__main__":
    src = cv2.imread("/work/csl/code/piece/dataset/MIT_ex/fragment_0001.png")
    dst = cv2.imread("/work/csl/code/piece/dataset/MIT_ex/fragment_0002.png")
    transform = np.float32([[-0.194909, 0.980821, 184.420000], [-0.980821, -0.194909, 334.444000], [0, 0, 1]])
    item = FusionImage2(src, dst, transform)
    cv2.imshow('Result', item[0])
    cv2.waitKey(0)
    cv2.destroyAllWindows()
