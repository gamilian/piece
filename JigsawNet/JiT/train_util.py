import random
import numpy as np
import os
import time
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
from JiT import logger

def set_seed(seed=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def validate_loop(valid_dataloader, model, device):
    tp, tn, fp, fn = 0, 0, 0, 0
    model.eval()
    with torch.no_grad():
        for valid_image, valid_label in tqdm(valid_dataloader):
            valid_image = valid_image.to(device)
            valid_label = valid_label.to(device)
            output = model(valid_image)
            pred = output.argmax(dim=1)
            
            for i in range(len(pred)):
                if pred[i] == 1:
                    if valid_label[i] == 1:
                        tp += 1
                    else:
                        fp += 1
                else:
                    if valid_label[i] == 1:
                        fn += 1
                    else:
                        tn += 1

    logger.log(f'tp: {tp}, tn: {tn}, fp: {fp}, fn: {fn}')
    logger.log(f'accuracy: {(tp + tn) / (tp + tn + fp + fn)}')
    logger.log(f'precision: {tp / (tp + fp)}')
    logger.log(f'recall: {tp / (tp + fn)}')
    
def train_loop(train_dataloader, valid_dataloader, model, device, epochs, batch_size, criterion, optimizer, lr_scheduler, writer, resume_checkpoint_dir):
    global_step = 0
    # Fine tuning loop
    # scaler = GradScaler()
    for epoch in range(epochs):
        model.train()
        logger.log(f"Epoch{epoch + 1}:")
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
            writer.add_scalar('Acc', acc / batch_size, global_step)
       
            # if (global_step + 1) % 10000 == 0:
            #     save_path = os.path.join(resume_checkpoint_dir, f'cross_vit_step{global_step}.pth')
            #     torch.save(model.state_dict(), save_path)
            global_step += 1

        lr_scheduler.step()
        logger.log(f'Learning Rate: {optimizer.param_groups[0]["lr"]} | Loss: {total_loss_train / len(train_dataloader) / batch_size: .6f} | Accuracy: {total_acc_train / len(train_dataloader) / batch_size: .5f}')
        writer.add_scalar('Accuracy', total_acc_train / len(train_dataloader) / batch_size, epoch)
        
        if (epoch + 1) % 1 == 0:
            save_path = os.path.join(resume_checkpoint_dir, f'cross_vit_epoch{epoch+1}.pth')
            torch.save(model.state_dict(), save_path)

        logger.log(f"save model to {save_path}")
        logger.log(f"validating...")
        validate_loop(valid_dataloader, model, device)
    return model




    