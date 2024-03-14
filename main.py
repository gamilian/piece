# import glob
# import numpy as np
# import os
# from JigsawNet.JigsawCNN.Utils import GtPose

# raw_dataset_path = glob.glob(os.path.join("/data/csl/dataset/jigsaw_dataset/raw_dataset", "*"))


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

class SZPDataset(Dataset):
    def __init__(self, alignments, fragments_dir, bg_color, transform):
        self.alignments = alignments
        self.fragments_dir = fragments_dir
        self.bg_color = bg_color
        self.transform = transform
        self.image_cache = {}
        self.data = self.load_data()
        

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img, v1, v2, score, trans = self.data[idx]
        return img, v1, v2, score, trans

    def load_data(self):
        data = []
        for alignment in self.alignments.data:
            v1, v2 = alignment.frame1, alignment.frame2
            score, trans = alignment.score, alignment.transform

            # 从缓存中读取或加载图像
            if v1 not in  self.image_cache:
                self.image_cache[v1] = cv2.imread(os.path.join(self.fragments_dir, f"fragment_{v1 + 1:04}.png"))
            if v2 not in  self.image_cache:
                self.image_cache[v2] = cv2.imread(os.path.join(self.fragments_dir, f"fragment_{v2 + 1:04}.png"))

            image1, image2 =  self.image_cache[v1],  self.image_cache[v2]
            item = FusionImage2(image1, image2, trans, self.bg_color)
            if len(item) <= 0 or item[1] > 0.05:  # overlap_ratio
                continue

            image_rgb = cv2.cvtColor(item[0], cv2.COLOR_BGR2RGB)  # image_cv2
            pil_image = Image.fromarray(image_rgb)

            img = self.transform(pil_image)
            data.append((img, v1, v2, score, trans))
        return data

def filter_pairwise(alignments, fragments_dir, net, bg_color, device, batch_size=32):
    net.eval()
    output_lines = ["Node {}\n".format(i) for i in range(alignments.node_num)]
  
    # 收集所有图像
    print("loading images ...")
    eval_data = SZPDataset(alignments, fragments_dir, bg_color, transform)
    dataload = DataLoader(eval_data, batch_size=batch_size, shuffle=False, num_workers=64, drop_last=False)
    print("evaluating ...")
    with torch.no_grad():
        for batch_tensor in dataload:
            print(batch_tensor.shape)
            batch_images = batch_tensor[:,0].to(device)
            output_batch = net(batch_images)
            probabilities = torch.nn.functional.softmax(output_batch, dim=1)
            correct_probabilities = probabilities[:, 1]
            # 处理批量推理结果
            for idx, correct_probability in enumerate(correct_probabilities):
                if correct_probability > 0.5:
                    v1, v2, score, trans = batch_tensor[idx][1:].tolist()
                    line = f"{v1} {v2} {score} "
                    line += "%f %f %f %f %f %f 0.000000 0.000000 1.000000\n" % (
                        trans[0, 0], trans[0, 1], trans[0, 2], trans[1, 0], trans[1, 1], trans[1, 2])
                    output_lines.append(line)

    # 将结果一次性写入文件
    with open(os.path.join(fragments_dir, "alignments.txt"), 'w+') as f:
        f.writelines(output_lines)
