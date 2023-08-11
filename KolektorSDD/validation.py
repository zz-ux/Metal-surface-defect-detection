import os
import time
import datetime
import torch
import torchvision
from src import u2net_full, u2net_lite
from train_utils import train_one_epoch, evaluate, create_lr_scheduler
from KolektorSDD_dataset import KoSDDDataset, category_weights, KoSDDDatasetSelectMethod, KoSDDDatasetAddDefectMethod
from torch.utils.data import DataLoader
import numpy as np


def create_model(aux, num_classes, pretrain=True):
    model = u2net_lite(out_ch=num_classes)
    weights_dict = torch.load(
        '/home/txy/zhangzhao/steel_segmentation/Second_Article/ShowImage/AblationExperiment/KolektorSDD/KolektorSDD_new_weight_transfer_dual_boundary.pth',
        map_location='cpu')['model']
    model.load_state_dict(weights_dict)
    return model


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print('显卡使用：', device)
    batch_size = args.batch_size
    num_classes = args.num_classes + 1

    # 数据集
    val_dataset = KoSDDDataset(train_val_state='val')
    num_workers = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=2,
                                             num_workers=num_workers,
                                             pin_memory=True)
    model = create_model(aux=args.aux, num_classes=num_classes)
    model.to(device)
    # 训练后，在验证集上测试
    val_confmat = evaluate(model, val_loader, device=device, num_classes=num_classes)
    val_data_info = str(val_confmat)
    print('验证集结果')
    print(val_data_info)


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="pytorch fcn training")
    parser.add_argument("--num-classes", default=1, type=int)
    parser.add_argument("--aux", default=False, type=bool, help="auxilier loss")
    parser.add_argument("--device", default="cuda:1", help="training device")
    parser.add_argument("-b", "--batch-size", default=4, type=int)
    parser.add_argument("--epochs", default=250, type=int, metavar="N",
                        help="number of total epochs to train")

    parser.add_argument('--lr', default=0.0001, type=float, help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=5e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument('--print-freq', default=800, type=int, help='print frequency')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    # Mixed precision training parameters
    parser.add_argument("--amp", default=False, type=bool,
                        help="Use torch.cuda.amp for mixed precision training")

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    torch.random.manual_seed(0)
    np.random.seed(0)
    args = parse_args()
    main(args)
