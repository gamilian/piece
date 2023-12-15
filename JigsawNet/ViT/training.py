import random
import numpy as np
import torch
import shutil
import os
import argparse
import torch.nn as nn
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import autocast, GradScaler
import dist_util
import my_logger as logger
from JigsawViT import JigsawViT
from dataload import get_train_dataloader
from script_util import add_dict_to_argparser

def set_seed(seed=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)



def train(train_dataloader,model, device, epochs, batch_size, criterion, optimizer, scheduler, writer, resume_checkpoint_dir):
    global_step = 0
    # Fine tuning loop
    # scaler = GradScaler()
    for epoch in range(epochs):
        total_acc_train = 0
        total_loss_train = 0.0
        
        for train_image, train_label in tqdm(train_dataloader):
            train_image = train_image.to(device)
            train_label = train_label.to(device)
            # with autocast():
            output = model(train_image)
            loss = criterion(output, train_label)
            acc = (output.argmax(dim=1) == train_label).sum().item()
            total_acc_train += acc
            total_loss_train += loss.item()
            
            # scaler.scale(loss).backward()

            # # 使用 scaler 进行梯度更新
            # scaler.step(optimizer)
            # scaler.update()

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            writer.add_scalar('Loss', loss.item() / batch_size, global_step)
            
            global_step += 1

        scheduler.step()
        print(f'Epochs{epoch + 1}: Learning Rate: {optimizer.param_groups[0]["lr"]} | Loss: {total_loss_train / len(train_dataloader) / batch_size: .6f} | Accuracy: {total_acc_train / len(train_dataloader) / batch_size: .5f}')
        writer.add_scalar('Accuracy', total_acc_train / len(train_dataloader) / batch_size, epoch)
        if (epoch + 1) % 2 == 0:
            save_path = os.path.join(resume_checkpoint_dir, f'model_epoch{epoch+1}.pth')
            torch.save(model.state_dict(), save_path)
    return model



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
            print(f"tensorboard_dir '{args.tensorboard_dir}' has been deleted.")
        except OSError as e:
            print(f"delelte tensorboard_dir error: {e}")
    else:
        print(f"dir '{args.tensorboard_dir}' not exists.")
    os.makedirs(args.tensorboard_dir)
    writer = SummaryWriter(args.tensorboard_dir)

    logger.log("creating tensorboard_dir and model ...")
    model = JigsawViT(
        pretrained_model=args.pretrained_model,
        num_labels=2
    ).to(device)
    # resume_checkpoint = os.path.join(args.resume_checkpoint_dir, "model_epoch5.pth")
    # model.load_state_dict(torch.load(resume_checkpoint))
    logger.log("creating data loader...")
    train_dataloader = get_train_dataloader(args.train_dataset_path, args.batch_size)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=50)
    logger.log("training...")
    train(train_dataloader=train_dataloader, model=model, device=device, epochs=args.epochs, batch_size=args.batch_size,
           criterion=criterion, optimizer=optimizer, scheduler=scheduler, writer=writer, resume_checkpoint_dir=args.resume_checkpoint_dir)

    logger.log("done.")


def create_argparser():
    defaults = dict(
        train_dataset_path = "/work/csl/code/piece/dataset/12_15",
        lr=1e-4,
        epochs = 10,
        batch_size = 16,
        pretrained_model = '/work/csl/code/piece/models/vit-base-patch16-224-in21k',
        resume_checkpoint_dir="/work/csl/code/piece/checkpoints/JigsawVIT_checkpoint/",
        tensorboard_dir = "logs/tensorboard",
        exp_name = "tmp"
    )
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
    