# import glob
# import numpy as np
# import os
# from JigsawNet.JigsawCNN.Utils import GtPose

# raw_dataset_path = glob.glob(os.path.join("/work/csl/code/piece/dataset/raw_dataset", "*"))


# for path in raw_dataset_path:
#     gt_pose = GtPose(os.path.join(path, "groundTruth.txt"))
#     gt_path = os.path.join(path, "ground_truth.txt")
#     with open(gt_path, "w") as f:
#         for i in range(len(gt_pose.data)):
#             f.write(f"{i}\n")
#             matrix = gt_pose.data[i]
#             for j in range(3):
#                 for k in range(3):
#                     f.write(f"{matrix[j][k]} ")
#             f.write("\n")
