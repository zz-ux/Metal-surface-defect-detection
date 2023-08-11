import os
import time
import datetime
import torch
import torchvision
from src import u2net_full, u2net_lite
from train_utils import train_one_epoch, evaluate, create_lr_scheduler
from KolektorSDD_dataset import KoSDDDataset, category_weights, KoSDDDatasetSelectMethod, KoSDDDatasetAddDefectMethod
from torch.utils.data import DataLoader


def create_model(aux, num_classes, pretrain=True):
    model = u2net_lite(out_ch=num_classes)
    return model


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print('显卡使用：', device)
    batch_size = args.batch_size

    # 改变U2Net结构，让其输出多个通道
    num_classes = args.num_classes + 1

    # 用来保存训练以及验证过程中信息
    save_name = 'u2net_lite'
    results_file = save_name + ".txt"
    # 数据集
    train_dataset = KoSDDDatasetAddDefectMethod(train_val_state='train')
    val_dataset = KoSDDDataset(train_val_state='val')
    train_save_dataset = KoSDDDataset(train_val_state='train')
    num_workers = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              num_workers=num_workers,
                              shuffle=True,
                              pin_memory=True, drop_last=True)
    train_save_loader = DataLoader(train_save_dataset,
                                   batch_size=batch_size,
                                   num_workers=num_workers,
                                   pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset,
                            batch_size=2,
                            num_workers=num_workers,
                            pin_memory=True, drop_last=True)
    model = create_model(aux=args.aux, num_classes=num_classes)
    model.to(device)

    params_to_optimize = [
        {"params": [p for p in model.parameters() if p.requires_grad]},
    ]

    if args.aux:
        params = [p for p in model.aux_classifier.parameters() if p.requires_grad]
        params_to_optimize.append({"params": params, "lr": args.lr * 10})

    optimizer = torch.optim.SGD(params_to_optimize, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    scaler = torch.cuda.amp.GradScaler() if args.amp else None

    # 创建学习率更新策略，这里是每个step更新一次(不是每个epoch)
    lr_scheduler = create_lr_scheduler(optimizer, len(train_loader), args.epochs, warmup=True)

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        args.start_epoch = checkpoint['epoch'] + 1
        if args.amp:
            scaler.load_state_dict(checkpoint["scaler"])

    start_time = time.time()
    Iou0 = 0
    for epoch in range(args.start_epoch, args.epochs):
        mean_loss, lr = train_one_epoch(model, optimizer, train_loader, device, epoch,
                                        lr_scheduler=lr_scheduler, print_freq=args.print_freq, scaler=scaler,
                                        weight=category_weights, epoch_ratio=float(epoch) / float(args.epochs))
        train_confmat = evaluate(model, train_save_loader, device=device, num_classes=num_classes)
        train_data_info = str(train_confmat)
        print(train_data_info)
        # write into txt
        with open(results_file, "a") as f:
            # 记录每个epoch对应的train_loss、lr以及训练集各指标
            train_info = f"[epoch: {epoch}]\n" \
                         f"train_loss: {mean_loss:.4f}\n" \
                         f"lr: {lr:.6f}\n"
            f.write(train_info + train_data_info + "\n\n")
        # train_acc_global, train_acc, train_iu = train_confmat.compute()
        train_acc_global, train_acc, train_recall, train_iu = train_confmat.compute()
        if train_iu.mean().item() > Iou0:
            Iou0 = train_iu.mean().item()
            save_file = {"model": model.state_dict(),
                         "optimizer": optimizer.state_dict(),
                         "lr_scheduler": lr_scheduler.state_dict(),
                         "epoch": epoch,
                         "args": args}
            if args.amp:
                save_file["scaler"] = scaler.state_dict()
            torch.save(save_file, "save_weights/" + save_name + '.pth')
    '''加载最好时刻数据'''
    model.cpu().load_state_dict(torch.load("save_weights/" + save_name + '.pth', map_location='cpu')['model'])
    model.to(device)
    # 训练后，在验证集上测试
    val_confmat = evaluate(model, val_loader, device=device, num_classes=num_classes)
    val_data_info = str(val_confmat)
    print('验证集结果')
    print(val_data_info)
    with open(results_file, "a") as f:
        # 记录每个epoch对应的train_loss、lr以及训练集各指标
        val_info = "验证集效果\n"
        f.write(val_info + val_data_info + "\n\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("training time {}".format(total_time_str))


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="pytorch fcn training")
    parser.add_argument("--num-classes", default=1, type=int)
    parser.add_argument("--aux", default=False, type=bool, help="auxilier loss")
    parser.add_argument("--device", default="cuda:1", help="training device")
    parser.add_argument("-b", "--batch-size", default=4, type=int)
    parser.add_argument("--epochs", default=250, type=int, metavar="N",
                        help="number of total epochs to train")

    parser.add_argument('--lr', default=0.001, type=float, help='initial learning rate')
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
    args = parse_args()
    print('KolektorSDD u2net_lite')

    if not os.path.exists("./save_weights"):
        os.mkdir("./save_weights")

    main(args)
