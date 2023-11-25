import numpy as np
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import cv2
from coutours_process import get_coutours, approx_contours, split_contours
from util import transform_cloud, tranform_ponit, longest_common_subsequence
from image_process import calculate_prob,fusion_image


def nearest_neighbor(src, dst):
    """
    Find the nearest (Euclidean) neighbor in dst for each point in src.
    """
    indices = []
    for p in src:
        distances = np.sum((dst - p) ** 2, axis=1)
        index = np.argmin(distances)
        indices.append(index)
    return np.array(indices)


def best_fit_transform(src, dst):
    """
    Calculates the least-squares best-fit transform between corresponding 2D points.
    """
    src_mean = np.mean(src, axis=0)
    dst_mean = np.mean(dst, axis=0)

    src_centered = src - src_mean
    dst_centered = dst - dst_mean

    H = src_centered.T @ dst_centered
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # Special reflection case
    if np.linalg.det(R) < 0:
        Vt[1, :] *= -1
        R = Vt.T @ U.T

    t = dst_mean.T - R @ src_mean.T

    return R, t


def icp(A, B, max_iterations=20, tolerance=0.001):
    """
    The Iterative Closest Point method.
    """
    src = np.copy(B)
    dst = np.copy(A)

    prev_error = 0

    for i in range(max_iterations):
        # Find the nearest neighbors between the current source and destination points
        indices = nearest_neighbor(src, dst)

        # Compute the best-fit transform
        R, t = best_fit_transform(src, dst[indices])

        # Update the current source
        src = (R @ src.T).T + t

        # Check error
        mean_error = np.mean(np.sum((src - dst[indices]) ** 2, axis=1))
        if np.abs(prev_error - mean_error) < tolerance:
            break
        prev_error = mean_error

    R, t = best_fit_transform(A, src)
    
    final_transform = np.eye(3)
    # Place the rotation matrix and translation vector into a rigid transformation matrix
    final_transform[:2, :2] = R
    final_transform[:2, 2] = t

    return final_transform

if __name__ == "__main__":
    # test ICP
    bg_color = [232, 8, 248]
    path1 = f"MIT_ex/fragment_0001.png"
    path2 = f"MIT_ex/fragment_0002.png"
    image1 = cv2.imread(path1)
    image2 = cv2.imread(path2)
    # 计算轮廓
    contours1, contours2 = get_coutours(image1, bg_color), get_coutours(image2, bg_color)

    #  近似轮廓
    approx1, approx2 = approx_contours(image1, contours1[0]), approx_contours(image2, contours2[0])

    # 分割集群
    mmp1, segments_si1, = split_contours(contours1[0], approx1)
    mmp2, segments_si2 = split_contours(contours2[0], approx2)
    cloud1, cloud2 = contours1[0].squeeze(1), contours2[0].squeeze(1)
    # 考虑正反
    cloud2 = cloud2[::-1]

    # 初始变换矩阵
    initial_transform = np.array([[-0.196116, -0.980581, 333.217268], [0.980581, -0.196116, 185.508748], [0, 0, 1]])

    lcs1, lcs2 = longest_common_subsequence(image1, image2, cloud1, cloud2, initial_transform, 2)

    print(lcs1)
    print(lcs2)

    cloud1, cloud2, cloud3 = (np.array([cloud1[i] for i in lcs1]), np.array([cloud2[i] for i in lcs2]),
                            np.array([tranform_ponit(cloud2[i], initial_transform) for i in lcs2]))

    print("clou1:")
    print(cloud1)
    print("--------")
    print("clou2:")
    print(cloud2)
    print("--------")
    print("clou3")
    print(cloud3)
    print("--------")
    print(len(lcs1), len(lcs2))

    # 应用初始变换
    cloud2_transformed = np.dot(np.column_stack((cloud2, np.ones(len(cloud2)))), initial_transform.T)[:, :2]


    # 执行ICP算法
    final_transform = icp(cloud1, cloud2_transformed)

    final_transform = np.matmul(final_transform, initial_transform)
    cloud2_aligned = transform_cloud(cloud2, final_transform)

    print(f"final_transform:\n{final_transform}")

    print("calculate_probcalculate_prob:\n{calculate_probcalculate_prob(image1, image2, segments_si1, segments_si2, final_transform)}")
    # item = fusion_image(image1, image2, final_transform, bg_color)
    # cv2.namedWindow("fusion_image")

    # cv2.imshow("fusion_image", item[0])
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # print(f'init transform:\n{initial_transform}')

    # print(f'Final transform:\n{final_transform}')
    # # 可视化结果
    # plt.figure(figsize=(10, 5))

    # plt.subplot(1, 2, 1)
    # plt.scatter(cloud1[:, 0], cloud1[:, 1], color='blue', label='Cloud 1')
    # plt.scatter(cloud2[:, 0], cloud2[:, 1], color='red', label='Cloud 2')
    # plt.title('Original Clouds')
    # plt.legend()

    # plt.subplot(1, 2, 2)
    # plt.scatter(cloud1[:, 0], cloud1[:, 1], color='blue', label='Cloud 1')
    # plt.scatter(cloud2_aligned[:, 0], cloud2_aligned[:, 1], color='green', label='Aligned Cloud 2')
    # plt.title('Aligned Clouds')
    # plt.legend()

    # plt.show()
