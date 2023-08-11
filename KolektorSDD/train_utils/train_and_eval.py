import torch
from torch import nn
import train_utils.distributed_utils as utils
import torch.nn.functional as F
import time
import numpy as np
import cv2 as cv


def loss_calc(inputs, target, device, weight=None):
    '''需要按照不同的条件下，调整训练时损失，使其在1.0左右'''

    # 由标签信息引入的边界权重
    def label_boundary_weight(target, device, use_boundary=True):
        with torch.no_grad():
            '''边界权重涉及，越靠近边界，权重信息越大'''
            target = (target != 0).long()

            target = np.array(target.cpu().numpy(), dtype=np.float)
            ksize, sigma, c0 = 13, 5, 0.05
            weight = []
            for i in range(len(target)):
                image_i = target[i]
                image_i = 0.5 * np.abs(cv.Sobel(image_i, cv.CV_16S, 1, 0)) + 0.5 * np.abs(
                    cv.Sobel(image_i, cv.CV_16S, 0, 1))
                image_i = cv.GaussianBlur(image_i, ksize=(ksize, ksize), sigmaX=sigma, )
                weight_i = image_i + c0
                weight.append(weight_i)
            weight = np.array(weight, dtype=np.float)
            weight = torch.tensor(weight).to(device)
            if use_boundary:
                return weight
            else:
                weight = torch.ones(weight.shape, dtype=weight.dtype).to(device)
                return weight

    # 由预测信息引入的边界权重
    def pred_boundary_weight(inputs, device, use_boundary=True):
        with torch.no_grad():
            '''边界权重涉及，越靠近边界，权重信息越大'''
            inputs = inputs.argmax(1)
            inputs = (inputs != 0).long()
            inputs = np.array(inputs.cpu().numpy(), dtype=np.float)
            ksize, sigma, c0 = 13, 5, 0.05
            weight = []
            for i in range(len(inputs)):
                image_i = inputs[i]
                image_i = 0.5 * np.abs(cv.Sobel(image_i, cv.CV_16S, 1, 0)) + 0.5 * np.abs(
                    cv.Sobel(image_i, cv.CV_16S, 0, 1))
                image_i = cv.GaussianBlur(image_i, ksize=(ksize, ksize), sigmaX=sigma, )
                weight_i = image_i + c0
                weight.append(weight_i)
            weight = np.array(weight, dtype=np.float)
            weight = torch.tensor(weight).to(device)
            if use_boundary:
                return weight
            else:
                weight = torch.ones(weight.shape, dtype=weight.dtype).to(device)
                return weight

    label_boundary_target = label_boundary_weight(target, device)
    pred_boundary_target = pred_boundary_weight(inputs, device)
    # import matplotlib.pyplot as plt
    # plt.figure()
    # for i in range(len(inputs)):
    #     plt.subplot(4, 4, 4 * i + 1)
    #     plt.imshow(np.array(inputs[i].argmax(0).cpu().detach().numpy(), dtype=np.float32))
    #     plt.axis('off')
    #     plt.colorbar()
    #     plt.subplot(4, 4, 4 * i + 2)
    #     plt.imshow(np.array(target[i].cpu().detach().numpy(), dtype=np.float32))
    #     plt.axis('off')
    #     plt.colorbar()
    #     plt.subplot(4, 4, 4 * i + 3)
    #     plt.imshow(label_boundary_target[i].cpu().detach().numpy())
    #     plt.axis('off')
    #     plt.colorbar()
    #     plt.subplot(4, 4, 4 * i + 4)
    #     plt.imshow(pred_boundary_target[i].cpu().detach().numpy())
    #     plt.axis('off')
    #     plt.colorbar()
    # plt.show()
    # plt.close()
    # exit(1)
    '''交叉熵损失'''
    # logpt = nn.CrossEntropyLoss(ignore_index=255, reduction='none', weight=torch.tensor(weight).to(device), )(inputs,
    #                                                                                                           target)  # 忽略target中值为255的像素，255的像素是目标边缘或者padding填充
    boundary_logpt = nn.CrossEntropyLoss(ignore_index=255, reduction='none', )(inputs,
                                                                               target)  # 忽略target中值为255的像素，255的像素是目标边缘或者padding填充
    boundary_logpt = 1.0 * boundary_logpt * (label_boundary_target + pred_boundary_target)
    weight_loss = F.cross_entropy(inputs, target, weight=torch.tensor(weight).to(device), ignore_index=255)
    loss = boundary_logpt.mean() + weight_loss
    # loss = weight_loss

    # '''Focus损失'''
    # n, c, h, w = inputs.size()
    # temp_inputs = inputs.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    # temp_target = target.view(-1)
    # weight_target = weight_target.view(-1)
    # # logpt = -nn.CrossEntropyLoss(ignore_index=255, reduction='none', weight=torch.tensor(weight).to(device), )(
    # #     temp_inputs, temp_target)
    # logpt = -nn.CrossEntropyLoss(ignore_index=255, reduction='none', )(temp_inputs, temp_target)
    # pt = torch.exp(logpt)
    # logpt = -((1 - pt) ** 2) * logpt
    # loss = logpt * weight_target
    # loss = loss.mean() * 1e1
    # losses['out'] = losses['out'] + ratio[1] * loss
    # #
    # '''Dice损失'''
    # beta, smooth = 1, 1e-5
    # n, c, h, w = inputs.size()
    # # 需要把target(b，h，w），经过onehot编码，增加channel维度，保持标签与预测结果的size一致
    # target_onehot = F.one_hot(target, num_classes).permute(0, 3, 1, 2).float()
    # nt, ct, ht, wt = target_onehot.size()
    # temp_inputs = torch.softmax(inputs.transpose(1, 2).transpose(2, 3).contiguous().view(n, -1, c), -1)
    # temp_target = target_onehot.contiguous().view(n, -1, ct)
    # tp = torch.sum(temp_target * temp_inputs, axis=[0, 1])
    # fp = torch.sum(temp_inputs, axis=[0, 1]) - tp
    # fn = torch.sum(temp_target, axis=[0, 1]) - tp
    # score = (2 * tp + smooth) / (2 * tp + fn + fp + smooth)
    # dice_loss = 1 - torch.mean(score)
    # losses['out'] = losses['out'] + ratio[2] * dice_loss
    return loss


def criterion(inputs, target, device, weight=None, loss_method='MyMethod', epoch_ratio=0):
    if (loss_method == 'original') or (epoch_ratio < 0.5):
        losses = []
        for i in range(len(inputs)):
            losses.append(F.cross_entropy(inputs[i], target, weight=torch.tensor(weight).to(device), ignore_index=255))
        return sum(losses)
    else:
        losses = []
        for i in range(len(inputs)):
            losses.append(loss_calc(inputs[i], target, device, weight))
        return sum(losses)


def evaluate(model, data_loader, device, num_classes):
    model.eval()
    confmat = utils.ConfusionMatrix(num_classes)
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    with torch.no_grad():
        for image, target in metric_logger.log_every(data_loader, 100, header):
            image, target = image.to(device), target.to(device)
            output = model(image)
            # output = output['out']
            confmat.update(target.flatten(), output.argmax(1).flatten())
        confmat.reduce_from_all_processes()

    return confmat


def train_one_epoch(model, optimizer, data_loader, device, epoch, lr_scheduler, print_freq=10, scaler=None,
                    weight=None, epoch_ratio=0):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    for image, target in metric_logger.log_every(data_loader, print_freq, header):
        image, target = image.to(device), target.to(device)
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            output = model(image)
            loss = criterion(output, target, device, weight, epoch_ratio=epoch_ratio)
        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        lr_scheduler.step()

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(loss=loss.item(), lr=lr)

    return metric_logger.meters["loss"].global_avg, lr


def create_lr_scheduler(optimizer,
                        num_step: int,
                        epochs: int,
                        warmup=True,
                        warmup_epochs=1,
                        warmup_factor=1e-3):
    assert num_step > 0 and epochs > 0
    if warmup is False:
        warmup_epochs = 0

    def f(x):
        """
        根据step数返回一个学习率倍率因子，
        注意在训练开始之前，pytorch会提前调用一次lr_scheduler.step()方法
        """
        if warmup is True and x <= (warmup_epochs * num_step):
            alpha = float(x) / (warmup_epochs * num_step)
            # warmup过程中lr倍率因子从warmup_factor -> 1
            return warmup_factor * (1 - alpha) + alpha
        else:
            # warmup后lr倍率因子从1 -> 0
            # 参考deeplab_v2: Learning rate policy
            return (1 - (x - warmup_epochs * num_step) / ((epochs - warmup_epochs) * num_step)) ** 0.9

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=f)
