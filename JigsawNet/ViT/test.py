import torch
import torch.nn as nn
import argparse

import dist_util
from tqdm import tqdm
import my_logger as logger
from torch.utils.tensorboard import SummaryWriter  
from JigsawViT import JigsawViT
from dataload import get_test_dataloader
from script_util import add_dict_to_argparser


def predict(test_dataloader, model, device):
    tp, tn, fp, fn = 0, 0, 0, 0
    model.eval()
    with torch.no_grad():
        for test_image, test_label in tqdm(test_dataloader):
            test_image = test_image.to(device)
            test_label = test_label.to(device)
            output = model(test_image)
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


def main():
    args = create_argparser().parse_args()
    dist_util.setup_dist()
    device = dist_util.dev()
    logger.configure(dir=f'logs/{args.exp_name}')
    logger.log(f"using device {device} ...")
    logger.log("create and load model ...")
    model = JigsawViT(
        pretrained_model=args.pretrained_model,
        num_labels=2,
    ).to(device)
    resume_checkpoint = os.path.join(args.resume_checkpoint_dir, "model_epoch5.pth")
    model.load_state_dict(torch.load(resume_checkpoint))

    logger.log("creating data loader...")
    test_dataloader = get_test_dataloader(args.test_dataset_path, args.batch_size)
    logger.log("test...")
    predict(test_dataloader=test_dataloader, model=model, device=device)

    logger.log("done.")


def create_argparser():
    defaults = dict(
        test_dataset_path = "/work/csl/code/piece/dataset/test_dataset",
        batch_size = 64,
        pretrained_model = '/work/csl/code/piece/models/vit-base-patch16-224-in21k',
        resume_checkpoint_dir="/work/csl/code/piece/checkpoints/JigsawVIT_checkpoint",
        exp_name = "tmp"
    )
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
    