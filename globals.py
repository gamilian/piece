import cv2
import numpy as np
from image_process import fusion_image
from util import Alignment2d


image1 = f"MIT_ex/fragment_0001.png"
relative_transform_filename = "filter_file2.txt"
bg_color = [232, 8, 248]
align = Alignment2d(relative_transform_filename)
# image1为待拼接的初始图片
i = 1
image1 = cv2.imread(f"MIT_ex/fragment_000{i}.png")
# offset_matrix为拼接时候对初始图片累计的偏移矩阵
offset_matrix = np.identity(3, dtype=np.float32)
# visited表示还未遍历过的顶点
visited = [1]
unvisit = [2, 3, 4, 5, 6, 7, 8, 9]
vec_trans = {1: np.identity(3, dtype=np.float32)}

for _ in range(8):
    # 1.找到与image1相邻的碎片image2
    j, k = 0, 0
    flag = False
    for v in visited:
        if v not in align.frame_to_data:
            continue
        for t in align.frame_to_data[v]:
            if align.data[t].frame2 in unvisit:
                i, j = v, t
                flag = True
                break
        if flag:
            break

    image2 = cv2.imread(f"MIT_ex/fragment_000{align.data[j].frame2}.png")
    visited.append(align.data[j].frame2)
    unvisit.remove(align.data[j].frame2)
    trans = offset_matrix

    vec_trans[align.data[j].frame2] = np.matmul(vec_trans[i], align.data[j].transform)
    trans = np.matmul(offset_matrix, vec_trans[align.data[j].frame2])

    # 2.拼接相邻的矩阵
    item = fusion_image(image1, image2, trans, bg_color)
    # 修改image为拼接后的图片
    image1 = item[0]
    i = align.data[j].frame2
    # 计算当前累计的offset_matrix
    offset_matrix = np.matmul(offset_matrix, item[2])

cv2.imshow('Result', item[0])
cv2.waitKey(0)
cv2.destroyAllWindows()
