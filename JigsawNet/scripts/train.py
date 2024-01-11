from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.nn as nn
import torch
import argparse
import shutil
import sys
import os

sys.path.append("/work/csl/code/piece/JigsawNet")
from JiT.script_util import add_dict_to_argparser
from JiT.dataload import get_train_dataloader
from JiT.JigsawViT import JigsawViT
from JiT.train_util import train_loop, set_seed
from JiT import dist_util, logger, dataload



def main():
    print("set_seed")
    set_seed(1)
    args = create_argparser().parse_args()
    dist_util.setup_dist()
    device = dist_util.dev()
    logger.configure(dir=f'logs/{args.exp_name}')
    logger.log(f"using device {device} ...")

    if os.path.exists(args.tensorboard_dir):
        # 删除目录及其内容
        try:
            shutil.rmtree(args.tensorboard_dir)
            logger.log(
                f"tensorboard_dir '{args.tensorboard_dir}' has been deleted.")
        except OSError as e:
            logger.log(f"delelte tensorboard_dir error: {e}")
    else:
        logger.log(f"dir '{args.tensorboard_dir}' not exists.")
    os.makedirs(args.tensorboard_dir)
    writer = SummaryWriter(args.tensorboard_dir)

    logger.log("creating tensorboard_dir and model ...")

    model = JigsawViT(
        pretrained_cfg_file=args.pretrained_cfg_file,
        num_labels=2
    ).to(device)
    
    # resume_checkpoint = os.path.join(
    #     args.resume_checkpoint_dir, "cross_vit_epoch4.pth")
    # logger.log(f"load checkpoint: {resume_checkpoint}...")
    # model.load_state_dict(torch.load(resume_checkpoint))
    logger.log("creating data loader...")
    train_dataloader, valid_dataloader = get_train_dataloader(
        args.train_dataset_path, args.batch_size)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    lr_scheduler = CosineAnnealingLR(optimizer, T_max=50)
    logger.log("training...")
    train_loop(train_dataloader=train_dataloader, valid_dataloader=valid_dataloader, model=model, device=device, epochs=args.epochs, batch_size=args.batch_size,
               criterion=criterion, optimizer=optimizer, lr_scheduler=lr_scheduler, writer=writer, resume_checkpoint_dir=args.resume_checkpoint_dir)

    logger.log("done.")


def create_argparser():
    defaults = dict(
        train_dataset_path="/work/csl/code/piece/dataset/szp_training_dataset",
        lr=1e-5,
        epochs=20,
        batch_size=64,
        pretrained_cfg_file='/work/csl/code/piece/models/crossvit_base_240/model.safetensors',
        resume_checkpoint_dir="/work/csl/code/piece/checkpoints/JigsawVIT_checkpoint/",
        tensorboard_dir="logs/tensorboard",
        exp_name="tmp"
    )
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
