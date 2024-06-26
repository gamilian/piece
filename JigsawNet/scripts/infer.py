import os, sys
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch
import argparse

import time
import cv2
from tqdm import tqdm
import torch.nn.functional as F
import numpy as np
from torchvision import transforms
from PIL import Image
from glob import glob
from torch.utils.data import Dataset, DataLoader


sys.path.append("/data/csl/code/piece/JigsawNet")
from JigsawCNN import Utils
from JigsawCNN.PairwiseAlignment2Image import FusionImage2
from JiT import logger, dist_util 
from JiT.JigsawViT import JigsawViT
from JiT.script_util import add_dict_to_argparser

transform = transforms.Compose([    
    transforms.Resize((224, 224)),  
    transforms.ToTensor(), # 转化为张量并归一化至[0-1]
    ## 图像标准化处理
    transforms.Normalize([0.485, 0.456, 0.406], 
                            [0.229, 0.224, 0.225]) 
]) 


# class SZPDataset(Dataset):
#     def __init__(self, alignments, fragments_dir, transform=None):
#         self.alignments = alignments
#         self.fragments_dir = fragments_dir
#         self.transform = transform

#     def __len__(self):
#         return len(self.alignments.data)

#     def __getitem__(self, idx):
#         alignment = self.alignments.data[idx]
#         v1, v2 = alignment.frame1, alignment.frame2
#         trans = alignment.transform

#         image1_path = os.path.join(self.fragments_dir, f"fragment_{v1 + 1:04}.png")
#         image2_path = os.path.join(self.fragments_dir, f"fragment_{v2 + 1:04}.png")

#         image1 = cv2.imread(image1_path)
#         image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
#         image2 = cv2.imread(image2_path)
#         image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)

#         if self.transform:
#             image1 = self.transform(Image.fromarray(image1))
#             image2 = self.transform(Image.fromarray(image2))

#         return image1, image2, trans, v1, v2






def ValidatePathNet(alignments, gt_pose, fragments_dir, net, bg_color, device):
    print("ValidatePathNet start!")
    net.eval()
    r_err_threshold = 4
    t_err_threshold = 50
    tp, tn, fp, fn = 0, 0, 0, 0
    with open(os.path.join(fragments_dir, "filtered_alignments.txt"), 'w+') as f:
        for alignment in alignments.data:
            v1 = alignment.frame1
            v2 = alignment.frame2
            trans = alignment.transform

            # gt judgement
            gt = 0
            pose1 = gt_pose.data[v1]
            pose2 = gt_pose.data[v2]
            gt_trans = np.matmul(np.linalg.inv(pose1), pose2)
            theta_err = np.abs(np.arccos(trans[0, 0]) * 180 / 3.1415926 - np.arccos(gt_trans[0, 0]) * 180 / 3.1415926)
            translation_err = np.sqrt((trans[0, 2] - gt_trans[0, 2]) ** 2 + (trans[1, 2] - gt_trans[1, 2]) ** 2)
            if theta_err < r_err_threshold and translation_err < t_err_threshold:
                gt = 1

            # neural network judgement
            image1 = cv2.imread(os.path.join(fragments_dir, "fragment_{0:04}.png".format(v1 + 1)))
            image2 = cv2.imread(os.path.join(fragments_dir, "fragment_{0:04}.png".format(v2 + 1)))
            item = FusionImage2(image1, image2, trans, bg_color)
            if len(item) <= 0:
                continue
            image_cv2, overlap_ratio, transform_offset = item[0], item[1], item[2]
            # OpenCV 读取的图像默认是 BGR 格式，将其转换为 RGB 格式
            image_rgb = cv2.cvtColor(image_cv2, cv2.COLOR_BGR2RGB)

            # 将 OpenCV 图像转换为 PIL Image
            pil_image = Image.fromarray(image_rgb)
            img = transform(pil_image)
            with torch.no_grad():
                output = net(img.unsqueeze(0).to(device))
            output = F.softmax(output, dim=1).squeeze(0)
            correct_probability = output.tolist()[1]
            if correct_probability > 0.5 and overlap_ratio < 0.05:
                final_class = 1
            else:
                final_class = 0
        
            if final_class == gt and final_class == 1:
                tp += 1
                f.write(f"{v1}\t{v2}\t{correct_probability}\t0\n")
                f.write(f"{trans[0, 0]} {trans[0, 1]} {trans[0, 2]}\n{trans[1, 0]} {trans[1, 1]} {trans[1, 2]}\n0 0 1\n" )
            if final_class == gt and final_class == 0:
                tn += 1
            if final_class != gt and final_class == 1:
                fp += 1
                f.write(f"{v1}\t{v2}\t{correct_probability}\t1\n")
                f.write(f"{trans[0, 0]} {trans[0, 1]} {trans[0, 2]}\n{trans[1, 0]} {trans[1, 1]} {trans[1, 2]}\n0 0 1\n" )
            if final_class != gt and final_class == 0:
                fn += 1 
                f.write(f"{v1}\t{v2}\t{correct_probability}\t2\n")
                f.write(f"{trans[0, 0]} {trans[0, 1]} {trans[0, 2]}\n{trans[1, 0]} {trans[1, 1]} {trans[1, 2]}\n0 0 1\n" )
    
    print(f'tp: {tp}, tn: {tn}, fp: {fp}, fn: {fn}')
    print(f'accuracy: {(tp + tn) / (tp + tn + fp + fn)}')
    print(f'precision: {tp / (tp + fp)}')
    print(f'recall: {tp / (tp + fn)}')

def meassure_pairwise(alignments, fragments_dir, net, bg_color, device):
    print("meassure_pairwise--------------")
    net.eval()
    with open(os.path.join(fragments_dir, "filtered_alignments.txt"), 'w+') as f:
        for alignment in alignments.data:
            v1 = alignment.frame1
            v2 = alignment.frame2
            trans = alignment.transform


            # neural network judgement
            image1 = cv2.imread(os.path.join(fragments_dir, "fragment_{0:04}.png".format(v1 + 1)))
            image2 = cv2.imread(os.path.join(fragments_dir, "fragment_{0:04}.png".format(v2 + 1)))
            item = FusionImage2(image1, image2, trans, bg_color)
            
            if len(item) <= 0:
                continue
            
            image_cv2, overlap_ratio, transform_offset = item[0], item[1], item[2]
            # black_bg = [0,0,0]
            # image_cv2[np.where((image_cv2 == bg_color).all(axis=2))] = [0,0,0]
            # image_cv2[np.where((image_cv2 != black_bg).all(axis=2))] = [255,255,255]
            # OpenCV 读取的图像默认是 BGR 格式，将其转换为 RGB 格式
            image_rgb = cv2.cvtColor(image_cv2, cv2.COLOR_BGR2RGB)

            # 将 OpenCV 图像转换为 PIL Image
            pil_image = Image.fromarray(image_rgb)
            img = transform(pil_image)
            with torch.no_grad():
                output = net(img.unsqueeze(0).to(device))
            output = F.softmax(output, dim=1).squeeze(0)
            correct_probability = output.tolist()[1]
        
            if correct_probability > 0.6:
                # cv2.imwrite(os.path.join(fragments_dir, "test" , f"fusion_{v1 + 1}_{v2 + 1}_{trans[0][0]}{trans[0][1]}_{correct_probability}.png"), item[0])
                f.write(f"{v1}\t{v2}\t{correct_probability}\t0\n")
                f.write(f"{trans[0, 0]} {trans[0, 1]} {trans[0, 2]}\n{trans[1, 0]} {trans[1, 1]} {trans[1, 2]}\n0 0 1\n" )
    print("meanssure_pairwise complete!")


# def filter_pairwise(alignments, fragments_dir, net, bg_color, device, batch_size=128):
#     net.eval()
#     image_cache = {}
#     output_lines = ["Node {}\n".format(i) for i in range(alignments.node_num)]
#     all_images = []
#     all_info = []
#     print("loading images ...")
#     # 收集所有图像
#     t1 = time.time()
#     for alignment in alignments.data:
#         v1, v2 = alignment.frame1, alignment.frame2
#         score, trans = alignment.score, alignment.transform

#         # 从缓存中读取或加载图像
#         if v1 not in image_cache:
#             image_cache[v1] = cv2.imread(os.path.join(fragments_dir, f"fragment_{v1 + 1:04}.png"))
#         if v2 not in image_cache:
#             image_cache[v2] = cv2.imread(os.path.join(fragments_dir, f"fragment_{v2 + 1:04}.png"))
        
#         image1, image2 = image_cache[v1], image_cache[v2]
#         # image1 = cv2.imread(os.path.join(fragments_dir, f"fragment_{v1 + 1:04}.png"))
#         # image2 = cv2.imread(os.path.join(fragments_dir, f"fragment_{v2 + 1:04}.png"))
#         item = FusionImage2(image1, image2, trans, bg_color)
#         if len(item) <= 0 or item[1] > 0.05:  # overlap_ratio
#             continue

#         image_rgb = cv2.cvtColor(item[0], cv2.COLOR_BGR2RGB)  # image_cv2
#         pil_image = Image.fromarray(image_rgb)
#         img = transform(pil_image)  # 应用预处理转换
#         all_images.append(img)
#         all_info.append((v1, v2, score, trans))
#     print(f"load images complete, cost {(time.time() - t1) / 60} minutes")
#     print("evaluating ...")
#     # 分批处理图像
#     num_batches = len(all_images) // batch_size + (1 if len(all_images) % batch_size else 0)
#     for batch_idx in tqdm(range(num_batches)):
#         start_idx = batch_idx * batch_size
#         end_idx = min((batch_idx + 1) * batch_size, len(all_images))
#         batch_images = all_images[start_idx:end_idx]
#         batch_info = all_info[start_idx:end_idx]

#         # 构建批量数据并进行批量推理
#         batch_tensor = torch.stack(batch_images).to(device)
#         # print(net.device)
#         # print(batch_tensor.device)
#         with torch.no_grad():
#             output_batch = net(batch_tensor)

#         # 处理批量推理结果
#         for idx, output in enumerate(output_batch):
#             correct_probability = torch.nn.functional.softmax(output, dim=0)[1].item()
#             if correct_probability > 0.66:
#                 v1, v2, score, trans = batch_info[idx]
#                 line = f"{v1} {v2} {score} "
#                 line += "%f %f %f %f %f %f 0.000000 0.000000 1.000000\n" % (
#                     trans[0, 0], trans[0, 1], trans[0, 2], trans[1, 0], trans[1, 1], trans[1, 2])
#                 output_lines.append(line)
#     print("writing ...")
#     # 将结果一次性写入文件
#     with open(os.path.join(fragments_dir, "vit_filter_alignments.txt"), 'w+') as f:
#         f.writelines(output_lines)


def filter_pairwise(alignments, fragments_dir, net, bg_color, device, batch_size=128):
    net.eval()
    with open(os.path.join(fragments_dir, "vit_filter_alignments.txt"), 'w+') as f:
        for i in range(alignments.node_num):
            f.write(f"Node {i}\n")   
    
    for alignment in alignments.data:
        v1 = alignment.frame1
        v2 = alignment.frame2
        score = alignment.score
        trans = alignment.transform
        # neural network judgement
        image1 = cv2.imread(os.path.join(fragments_dir, "fragment_{0:04}.png".format(v1 + 1)))
        image2 = cv2.imread(os.path.join(fragments_dir, "fragment_{0:04}.png".format(v2 + 1)))
        item = FusionImage2(image1, image2, trans, bg_color)
             
        if len(item) <= 0:
            continue
        image_cv2, overlap_ratio, transform_offset = item[0], item[1], item[2]
        if overlap_ratio > 0.05:
            continue

        # OpenCV 读取的图像默认是 BGR 格式，将其转换为 RGB 格式
        image_rgb = cv2.cvtColor(image_cv2, cv2.COLOR_BGR2RGB)

        # 将 OpenCV 图像转换为 PIL Image
        pil_image = Image.fromarray(image_rgb)
        img = transform(pil_image)
        
        with torch.no_grad():
            output = net(img.unsqueeze(0).to(device))
        output = F.softmax(output, dim=1).squeeze(0)
        correct_probability = output.tolist()[1]
        # if correct_probability > 0.2:
        cv2.imwrite(os.path.join(fragments_dir, "test" , f"fusion_{v1 + 1}_{v2 + 1}_{trans[0][0]}{trans[0][1]}_{correct_probability}.png"), item[0])
         
        with open(os.path.join(fragments_dir, "vit_filter_alignments.txt"), 'a+') as f:
            if correct_probability > 0.5:
                f.write(f"{v1} {v2} {score} ")
                f.write("%f %f %f %f %f %f 0.000000 0.000000 1.000000" % (
                    trans[0, 0], trans[0, 1], trans[0, 2], trans[1, 0], trans[1, 1],
                    trans[1, 2]))
                f.write("\n")

def main():
    args = create_argparser().parse_args()
    dist_util.setup_dist()
    device = dist_util.dev()
    logger.configure(dir=f'logs/{args.exp_name}')
    logger.log(f"using device {device} ...")
    logger.log("create and load model ...")
    model = JigsawViT(
        pretrained_cfg_file=args.pretrained_cfg_file,
        num_labels=2
    ).to(device)
    logger.log("load resume_checkpoint ...")

    resume_checkpoint = os.path.join(args.resume_checkpoint_dir, "pit_s-distilled_resume.pth")
    model.load_state_dict(torch.load(resume_checkpoint))
    fragments_dirs = glob(os.path.join(args.measure_data_root, "*image"))
    for i in range(len(fragments_dirs)):
        logger.log(f"dataset {i+1}/{len(fragments_dirs)}:  {fragments_dirs[i]}")
        # if not os.path.exists(os.path.join(fragments_dirs[i], "alignments.txt")):
        #     continue
        bg_color_file = os.path.join(fragments_dirs[i], "bg_color.txt")
        with open(bg_color_file) as f:
            for line in f:
                line = line.split()
                if line:
                    bg_color = [int(i) for i in line]
                    bg_color = bg_color[::-1]


        if args.mode == "filter":
            alignment = os.path.join(fragments_dirs[i], "alignments_tmp.txt")
            alignments = Utils.Alignment2d(alignment)
            filter_pairwise(alignments, fragments_dirs[i], model, bg_color, device)
        elif args.mode == "infer":
            print("-----infer-----------")
            filtered_alignment = os.path.join(fragments_dirs[i], "alignments.txt")
            gt_pose_path = os.path.join(fragments_dirs[i], "ground_truth.txt")
            
            alignments = Utils.Alignment2d(filtered_alignment)
            if (os.path.exists(gt_pose_path)):
                gt_pose = Utils.GtPose(gt_pose_path)
                ValidatePathNet(alignments, gt_pose, fragments_dirs[i], model, bg_color, device)
            else:
                meassure_pairwise(alignments, fragments_dirs[i], model, bg_color, device)
        logger.log("----------------")
 
def create_argparser():
    defaults = dict(
        measure_data_root = "../Measure",
        batch_size = 128,
        pretrained_cfg_file='/data/csl/code/piece/models/pit_s-distilled_224/model.safetensors',
        resume_checkpoint_dir = "/data/csl/code/piece/checkpoints/JigsawVIT_checkpoint/",
        exp_name = "tmp",
        mode = "filter"
    )
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser

if __name__ == "__main__":
    main()