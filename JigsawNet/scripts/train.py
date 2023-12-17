import os, sys
import shutil
import argparse
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter


sys.path.append("/work/csl/code/piece/JigsawNet")
from JiT import dist_util, logger, dataload
from JiT.train_util import train_loop, set_seed
from JiT.JigsawViT import JigsawViT
from JiT.dataload import get_train_dataloader
from JiT.script_util import add_dict_to_argparser

def main():
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
            logger.log(f"tensorboard_dir '{args.tensorboard_dir}' has been deleted.")
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
    # resume_checkpoint = os.path.join(args.resume_checkpoint_dir, "pit_s_distilled_epoch2.pth")
    # model.load_state_dict(torch.load(resume_checkpoint))
    logger.log("creating data loader...")
    train_dataloader = get_train_dataloader(args.train_dataset_path, args.batch_size)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=50)
    logger.log("training...")
    # train_loop(train_dataloader=train_dataloader, model=model, device=device, epochs=args.epochs, batch_size=args.batch_size,
    #        criterion=criterion, optimizer=optimizer, scheduler=scheduler, writer=writer, resume_checkpoint_dir=args.resume_checkpoint_dir)

    # logger.log("done.")


def create_argparser():
    defaults = dict(
        train_dataset_path = "/work/csl/code/piece/dataset/12_17",
        lr=1e-4,
        epochs = 10,
        batch_size = 1,
        pretrained_cfg_file = '/work/csl/code/piece/models/pit_s-distilled_224/model.safetensors',
        resume_checkpoint_dir="/work/csl/code/piece/checkpoints/JigsawVIT_checkpoint/",
        tensorboard_dir = "logs/tensorboard",
        exp_name = "tmp"
    )
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()