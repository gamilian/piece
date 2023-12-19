import torch
import argparse
import os, sys
import cv2
import torch.nn.functional as F
import numpy as np
from torchvision import transforms
from PIL import Image
from glob import glob

sys.path.append("/work/csl/code/piece/JigsawNet")
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

def ValidatePathNet(alignments, gt_pose, fragments_dir, net, bg_color, device):
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
        
            # err_trans = np.matmul(trans, np.linalg.inv(gt_trans))
            # if np.abs(err_trans[0, 0] - 1) < 1e-3:
            #     err_trans[0, 0] = 1
            # if np.abs(err_trans[0, 0] + 1) < 1e-3:
            #     err_trans[0, 0] = -1
            # theta_err = np.arccos(err_trans[0, 0]) * 180 / 3.1415926
            # translation_err = np.sqrt(err_trans[0, 2] ** 2 + err_trans[1, 2] ** 2)
            # if theta_err < r_err_threshold and translation_err < t_err_threshold:

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
            net.eval()
            with torch.no_grad():
                output = net(img.unsqueeze(0).to(device))
            output = F.softmax(output, dim=1).squeeze(0)
            correct_probability = output.tolist()[1]
            if correct_probability > 0.6:
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
                f.write(f"{v1}\t{v2}\t{correct_probability} {theta_err} {translation_err}\t1\n")
                f.write(f"{trans[0, 0]} {trans[0, 1]} {trans[0, 2]}\n{trans[1, 0]} {trans[1, 1]} {trans[1, 2]}\n0 0 1\n" )
            if final_class != gt and final_class == 0:
                fn += 1 
                f.write(f"{v1}\t{v2}\t{correct_probability} {theta_err} {translation_err}\t2\n")
                f.write(f"{trans[0, 0]} {trans[0, 1]} {trans[0, 2]}\n{trans[1, 0]} {trans[1, 1]} {trans[1, 2]}\n0 0 1\n" )
    
    print(f'tp: {tp}, tn: {tn}, fp: {fp}, fn: {fn}')
    print(f'accuracy: {(tp + tn) / (tp + tn + fp + fn)}')
    print(f'precision: {tp / (tp + fp)}')
    print(f'recall: {tp / (tp + fn)}')

def meassure_pairwise(alignments, fragments_dir, net, bg_color, device):
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
            # OpenCV 读取的图像默认是 BGR 格式，将其转换为 RGB 格式
            image_rgb = cv2.cvtColor(image_cv2, cv2.COLOR_BGR2RGB)

            # 将 OpenCV 图像转换为 PIL Image
            pil_image = Image.fromarray(image_rgb)
            img = transform(pil_image)
            net.eval()
            with torch.no_grad():
                output = net(img.unsqueeze(0).to(device))
            output = F.softmax(output, dim=1).squeeze(0)
            probs = output.tolist()[1]
        
            if probs > 0.5:
                # cv2.imwrite(os.path.join(fragments_dir, "test" , f"fusion_{v1 + 1}_{v2 + 1}_{trans[0][0]}{trans[0][1]}_{correct_probability}.png"), item[0])
                f.write(f"{v1}\t{v2}\t{probs}\t0\n")
                f.write(f"{trans[0, 0]} {trans[0, 1]} {trans[0, 2]}\n{trans[1, 0]} {trans[1, 1]} {trans[1, 2]}\n0 0 1\n" )
    print("meanssure_pairwise complete!")

def main():
    args = create_argparser().parse_args()
    dist_util.setup_dist()
    device = dist_util.dev()
    logger.configure(dir=f'logs/{args.exp_name}')
    logger.log(f"using device {device} ...")
    logger.log("create and load model ...")
    model = JigsawViT(
        pretrained_cfg_file=args.pretrained_cfg_file,
        num_labels=2,
    ).to(device)
    resume_checkpoint = os.path.join(args.resume_checkpoint_dir, "pit_s_distilled_epoch3.pth")
    model.load_state_dict(torch.load(resume_checkpoint))

    fragments_dirs = glob(os.path.join(args.measure_data_root, "*_ex"))
    for i in range(len(fragments_dirs)):
        print(f"dataset {i+1}/{len(fragments_dirs)}:  {fragments_dirs[i]}")
        if not os.path.exists(os.path.join(fragments_dirs[i], "alignments.txt")):
            continue
        bg_color_file = os.path.join(fragments_dirs[i], "bg_color.txt")
        with open(bg_color_file) as f:
            for line in f:
                line = line.split()
                if line:
                    bg_color = [int(i) for i in line]
                    bg_color = bg_color[::-1]
    
        relative_alignment = os.path.join(fragments_dirs[i], "alignments.txt")
        gt_pose_path = os.path.join(fragments_dirs[i], "groundTruth.txt")
        
        alignments = Utils.Alignment2d(relative_alignment)
        if (os.path.exists(gt_pose_path)):
            gt_pose = Utils.GtPose(gt_pose_path)
            ValidatePathNet(alignments, gt_pose, fragments_dirs[i], model, bg_color, device)
        else:
            meassure_pairwise(alignments, fragments_dirs[i], model, bg_color, device)
        print("----------------")

def create_argparser():
    defaults = dict(
        measure_data_root = "../Measure",
        batch_size = 256,
        pretrained_cfg_file = '/work/csl/code/piece/models/pit_s-distilled_224/model.safetensors',
        resume_checkpoint_dir="/work/csl/code/piece/checkpoints/JigsawVIT_checkpoint/",
        exp_name = "tmp"
    )
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser

if __name__ == "__main__":
    main()