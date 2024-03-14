import torch
import torch.nn as nn
import argparse
import os
from torch.utils.tensorboard import SummaryWriter  
from tqdm import tqdm

import sys
sys.path.append("/data/csl/code/piece/JiT")
from script_util import add_dict_to_argparser
from dataload import get_test_dataloader
from JiT import JigsawViT
from train_util import train_loop, set_seed
import dist_util, logger


def predict(test_dataloader, model, device):
    tp, tn, fp, fn = 0, 0, 0, 0
    model.eval()
    with torch.no_grad():
        for test_image, test_label, test_roi in tqdm(test_dataloader):
            test_image = test_image.to(device)
            test_label = test_label.to(device)
            test_roi = test_roi.to(device)
            output = model(test_image, test_roi)
            pred = output.argmax(dim=1)
            
            for i in range(len(pred)):
                if pred[i] == 1:
                    if test_label[i] == 1:
                        tp += 1
                    else:
                        fp += 1
                else:
                    if test_label[i] == 1:
                        fn += 1
                    else:
                        tn += 1

    print(f'tp: {tp}, tn: {tn}, fp: {fp}, fn: {fn}')
    print(f'accuracy: {(tp + tn) / (tp + tn + fp + fn)}')
    print(f'precision: {tp / (tp + fp)}')
    print(f'recall: {tp / (tp + fn)}')


def main():
    args = create_argparser().parse_args()
    dist_util.setup_dist()
    device = dist_util.dev()
    logger.configure(dir=f'logs/{args.exp_name}')
    logger.log(f"using device {device} ...")
    logger.log("create and load model ...")
    model = JigsawViT(pretrained=False).to(device)
    resume_checkpoint = os.path.join(args.resume_checkpoint_dir, "jit_epoch4.pth")
    model.load_state_dict(torch.load(resume_checkpoint))

    logger.log("creating data loader...")
    test_dataloader = get_test_dataloader(args.test_dataset_path, args.batch_size)
    logger.log("test...")
    predict(test_dataloader=test_dataloader, model=model, device=device)

    logger.log("done.")


def create_argparser():
    defaults = dict(
        test_dataset_path = "/data/csl/dataset/jigsaw_dataset/szp_test_roi",
        batch_size = 32,
        # pretrained_cfg_file='/work/csl/code/piece/models/crossvit_base_240/model.safetensors',
        resume_checkpoint_dir="/data/csl/code/piece/checkpoints/JiT_checkpoint",
        exp_name = "tmp"
    )
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
    