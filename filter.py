import numpy as np
import operator
path = "JigsawNet/Measure/6_ex/filtered_alignments.txt"

class Transform2d:
    def __init__(self, v1=-1, v2=-1, score=-1, transform=np.identity(3)):
        self.frame1 = v1
        self.frame2 = v2
        self.score = score
        self.transform = transform

data = []
relative_transform_filename = "JigsawNet/Measure/6_ex/alignments.txt"
with open(relative_transform_filename) as f:
    all_line = [line.rstrip() for line in f]
    node_num = 0
    for line in all_line:
        if line[0:4] == "Node":
            node_num += 1
        else:
            data_str_list = line.split()
            v1,v2,score, m1,m2,m3,m4,m5,m6,m7,m8,m9 = [t(s) for t,s in zip((int,int, float, float,float,float,float,float,float,float,float,float), data_str_list[0:12])]
            transform = np.array([[m1,m2,m3], [m4,m5,m6], [m7,m8,m9]])
            data.append(Transform2d(v1, v2, score, transform))



# data = []
# with open(path, 'r+') as f:
#     all_line = [line.rstrip() for line in f]
#     for i in range(0, len(all_line), 4):
#         line = all_line[i].rsplit("\t")
#         frame1, frame2, score = int(line[0]), int(line[1]), float(line[2])
#         group = all_line[i + 1:i + 4]
#         transform = []
#         for l in group:
#             vector = [float(num) for num in l.rsplit()]
#             transform.append(vector)
#         transform = np.matrix(transform)
#         data.append(Transform2d(frame1, frame2, score, transform))

max_score_dict = {}

for item in data:
    key = (item.frame1, item.frame2)
    score = item.score
    # 如果这对 frame1 和 frame2 不存在于字典中，或者当前 score 更高，则更新字典
    if key not in max_score_dict or item.score > max_score_dict[key].score:
        max_score_dict[key] = item



# filtered_data 现在包含了每对 frame1 和 frame2 中 score 最高的实例
filtered_data = list(max_score_dict.values())

with open(relative_transform_filename, 'w+') as f:
    for item in filtered_data:
        frame1, frame2, score, trans = item.frame1, item.frame2, item.score, item.transform
        f.write(f"{frame1} {frame2} {score} ")
        f.write(f"{trans[0, 0]} {trans[0, 1]} {trans[0, 2]} {trans[1, 0]} {trans[1, 1]} {trans[1, 2]} 0 0 1\n")


# with open(path, 'w+') as f:
#     for item in filtered_data:
#         frame1, frame2, score, trans = item.frame1, item.frame2, item.score, item.transform
#         f.write(f"{frame1}\t{frame2}\t{score}\t0\n")
#         f.write(f"{trans[0, 0]} {trans[0, 1]} {trans[0, 2]}\n{trans[1, 0]} {trans[1, 1]} {trans[1, 2]}\n0 0 1\n")






