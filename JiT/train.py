import random
import numpy as np
import os
import time
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.nn as nn
import torch
import argparse
import shutil
import sys
import logger as logger
import dist_util
from script_util import add_dict_to_argparser, update_arg_parser
from transformer import TransformerModel
from jit_dataset import get_train_dataloader
import torch.nn.functional as F

class TranslationRotationLoss(nn.Module):
    def __init__(self, lambda_translation=0.2, lambda_rotation=0.8):
        """
        初始化损失函数。
        :param lambda_translation: 平移损失的权重
        :param lambda_rotation: 旋转损失的权重
        """
        super(TranslationRotationLoss, self).__init__()
        self.lambda_translation = lambda_translation
        self.lambda_rotation = lambda_rotation
        self.mse_loss = nn.MSELoss()  # 平移的均方误差损失

    def forward(self, predictions, labels):
        # 假设labels和predictions的形状都是[n, 3]
        # 其中第一列是角度，后两列是偏移

        # 提取角度和偏移
        angle_pred, offsets_pred = predictions[:, 0], predictions[:, 1:]
        angle_true, offsets_true = labels[:, 0], labels[:, 1:]

        # 计算角度的余弦相似度损失
        angle_loss = 1 - F.cosine_similarity(angle_pred.unsqueeze(1), angle_true.unsqueeze(1))
        angle_loss = angle_loss.mean()  
        # 计算偏移的均方误差损失
        offsets_loss = self.mse_loss(offsets_pred, offsets_true)

        # 组合两种损失
        total_loss = self.lambda_rotation*angle_loss + self.lambda_translation*offsets_loss

        return total_loss

def predict(model, input_tensor, cond):
    output, _ = model(input_tensor, **cond)
    return output



def set_seed(seed=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def valiad_loop(valid_dataloader, model, device):
    model.eval()
    total_acc_valid = 0
    with torch.no_grad():
        for input_tensor, valid_label, cond in tqdm(valid_dataloader):
            input_tensor = input_tensor.to(device)
            valid_label = valid_label.to(device)
            output, _ = model(input_tensor, **cond)
            theta_err = (output[:, 0] * 180 / 3.1415926 - valid_label[:, 0] * 180 / 3.1415926).abs()
            translation_err = torch.sqrt((output[:, 1]*1000 - valid_label[:, 1]*1000) ** 2 + (output[:, 2]*1000 - valid_label[:, 2]*1000) ** 2)
            acc = ((theta_err <= 4) * (translation_err <= 30)).sum().item()
            total_acc_valid += acc
    return total_acc_valid / len(valid_dataloader) / valid_dataloader.batch_size

def train_loop(train_dataloader, valid_dataloader, model, device, epochs, batch_size, criterion, optimizer, scheduler, writer, resume_checkpoint_dir):
    global_step = 0
    # Fine tuning loop
    # scaler = GradScaler()
    for epoch in range(epochs):
        model.train()
        logger.log(f"Epoch{epoch + 1}:")
        total_acc_train = 0
        total_loss_train = 0.0
     
        for input_tensor, train_label, cond in tqdm(train_dataloader):
            input_tensor = input_tensor.to(device)
            train_label = train_label.to(device)
            # with autocast():
            output, _ = model(input_tensor, **cond)
            loss = criterion(output, train_label)
            total_loss_train += loss.item()
            theta_err = (output[:, 0] * 180 / 3.1415926 - train_label[:, 0] * 180 / 3.1415926).abs()
            translation_err = torch.sqrt((output[:, 1]*1000 - train_label[:, 1]*1000) ** 2 + (output[:, 2]*1000 - train_label[:, 2]*1000) ** 2)
            writer.add_scalar('theta_err', (theta_err/batch_size).mean().item(), global_step)
            writer.add_scalar('translation_err', (translation_err/batch_size).mean().item(), global_step)
            acc = ((theta_err <= 4) * (translation_err <= 30)).sum().item()
            total_acc_train += acc
            # scaler.scale(loss).backward()
            # # 使用 scaler 进行梯度更新
            # scaler.step(optimizer)
            # scaler.update()

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            writer.add_scalar('Loss', loss.item(), global_step)
            writer.add_scalar('acc', acc / batch_size, global_step)
       
            # if (global_step + 1) % 10000 == 0:
            #     save_path = os.path.join(resume_checkpoint_dir, f'cross_vit_step{global_step}.pth')
            #     torch.save(model.state_dict(), save_path)
            global_step += 1

        valid_acc = valiad_loop(valid_dataloader, model, device)
        scheduler.step()
        logger.log(f'Learning Rate: {optimizer.param_groups[0]["lr"]} | Loss: {total_loss_train / len(train_dataloader): .6f} | Accuracy: {total_acc_train / len(train_dataloader) / batch_size: .5f} | Valid Accuracy: {valid_acc: .5f}')
        if (epoch + 1) % 2 == 0:
            save_path = os.path.join(resume_checkpoint_dir, f'jit_epoch{epoch+1}.pth')
            torch.save(model.state_dict(), save_path)
        writer.add_scalar('Train_Acc', total_acc_train / len(train_dataloader) / batch_size, epoch)
        writer.add_scalar('Valid_Acc', valid_acc, epoch)
    return model


def main():
    print("set_seed")
    set_seed(1)
    args = create_argparser().parse_args()
    update_arg_parser(args)
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
    model = TransformerModel(
        in_channels=args.input_channels,
        model_channels=512,
        out_channels=args.out_channels,
        use_checkpoint=args.use_checkpoint
    ).to(device)
    
    # resume_checkpoint = os.path.join(
    #     args.resume_checkpoint_dir, "jit_epoch.pth")
    # logger.log(f"load checkpoint: {resume_checkpoint}...")
    # model.load_state_dict(torch.load(resume_checkpoint))
    # logger.log("creating data loader...")
    train_dataloader, valid_dataloader = get_train_dataloader(
        args.train_dataset_path, args.batch_size)
    criterion = nn.SmoothL1Loss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=50)
    logger.log("training...")
    train_loop(train_dataloader=train_dataloader, valid_dataloader=valid_dataloader, model=model, device=device, epochs=args.epochs, batch_size=args.batch_size,
               criterion=criterion, optimizer=optimizer, scheduler=scheduler, writer=writer, resume_checkpoint_dir=args.resume_checkpoint_dir)

    logger.log("done.")


def create_argparser():
    defaults = dict(
        train_dataset_path="/work/csl/code/piece/dataset/jit_dataset/train.csv",
        lr=1e-4,
        epochs=25,
        batch_size=8,
        use_checkpoint=False,
        resume_checkpoint_dir="/work/csl/code/piece/checkpoints/JiT_checkpoint/",
        tensorboard_dir="logs/tensorboard",
        exp_name="tmp"
    )
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()

    