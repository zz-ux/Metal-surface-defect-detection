import os
import random
import numpy as np
import torch
import torch.utils.data as Data
from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode

'''针对KolektorSDD表面缺陷分割数据集，原图像500*500（高*宽），插值到256*256，有1+1（背景）种类别'''
ROOT_PATH = '/home/txy/zhangzhao/dataset/KolektorSDD/'    # 运行时要改变数据集位置信息
category_list = [
    [0, 0, 0],  # 标记0类（背景） 占比99.93%------黑色
    [255, 255, 255],  # 标记1类    占比0.07%------白色
]
# category_weights = [0.50776377, 32.70085929]  # 中值频率平衡


category_weights = [0.5813477, 3.57322767]  # 论文二的方法


def TrainCategory(Input_name):
    # 考虑到类别严重不均衡，有的类别仅有200张图像，有的类别则有几千张，因此进行训练分类，在取batch时，更多的向小数据的倾斜
    category_0, category_1 = [], []
    with open(ROOT_PATH + 'file_txts/' + 'category_0.txt', 'r') as File:
        for line in File.readlines():
            line = line.replace('\n', '')
            category_0.append(line)
    with open(ROOT_PATH + 'file_txts/' + 'category_1.txt', 'r') as File:
        for line in File.readlines():
            line = line.replace('\n', '')
            category_1.append(line)
    return_category_0, return_category_1 = [], []
    for i in range(len(Input_name)):
        if Input_name[i] in category_0:
            return_category_0.append(Input_name[i])
            continue  # 在0类别中的一定不在其他类别中
        if Input_name[i] in category_1:  # 不加continue，因为有可能有的图有多种缺陷
            return_category_1.append(Input_name[i])
    return [return_category_0, return_category_1]


class KoSDDDataset(Data.Dataset):
    def __init__(self, train_val_state='train', train_ratio=0.8):
        super(KoSDDDataset, self).__init__()
        images_names = os.listdir(ROOT_PATH + 'Images')
        for i in range(len(images_names)):
            images_names[i] = images_names[i][:-4]
        random.seed(0)  # 随机打乱，增加随机性
        random.shuffle(images_names)
        self.train_part = images_names[:int(train_ratio * len(images_names))]  # 交叉验证时更改这一部分
        self.val_part = images_names[int(train_ratio * len(images_names)):]  # 交叉验证时更改这一部分
        if train_val_state == 'train':
            self.train_state = True
        else:
            self.train_state = False

    def __len__(self):
        if self.train_state:
            return len(self.train_part)
        else:
            return len(self.val_part)

    def __getitem__(self, item):
        if self.train_state:
            name = self.train_part[item]
        else:
            name = self.val_part[item]
        image_name = ROOT_PATH + 'Images/' + name + '.jpg'
        label_name = ROOT_PATH + 'Labels/' + name + '.bmp'
        image = Image.open(image_name).convert('RGB')
        label = Image.open(label_name).convert('RGB')
        label = self.Label_RGB_TO_L(label)
        image, label = self.Image_Transform(image, label)
        return image, label

    def Label_RGB_TO_L(self, label_RGB):
        # 将RGB格式的图像变成0,1,2,3,4这种形式，方便训练学习
        label_RGB = label_RGB.__array__()
        label_L = np.zeros(label_RGB.shape[:-1])
        for category_i in range(len(category_list)):
            check_label = (label_RGB == np.array(category_list[category_i]))
            check_label = check_label[:, :, 0] * check_label[:, :, 1] * check_label[:, :, 2]
            label_L[check_label] = category_i
        label_L = Image.fromarray(label_L)  # 变成PIL格式与image统一
        return label_L

    def Image_Transform(self, image_RGB, label_L):
        if self.train_state:  # 加上随机翻转
            image_transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                                  transforms.RandomVerticalFlip(),
                                                  transforms.Resize(size=(256, 256)),
                                                  transforms.ToTensor(),
                                                  transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                                  ])
            label_transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                                  transforms.RandomVerticalFlip(),
                                                  transforms.Resize(size=(256, 256),
                                                                    interpolation=InterpolationMode.NEAREST),
                                                  transforms.ToTensor()
                                                  ])
            seed = np.random.randint(0, 125416584)
            torch.random.manual_seed(seed)
            image = image_transform(image_RGB)
            torch.random.manual_seed(seed)
            label = label_transform(label_L).long()
            return image, label[0]
        else:
            image_transform = transforms.Compose([transforms.Resize(size=(256, 256)),
                                                  transforms.ToTensor(),
                                                  transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                                  ])
            label_transform = transforms.Compose([transforms.Resize(size=(256, 256),
                                                                    interpolation=InterpolationMode.NEAREST),
                                                  transforms.ToTensor()])
            image = image_transform(image_RGB)
            label = label_transform(label_L).long()
            return image, label[0]

    def Show_Image(self, image_Norm, label_L):
        # 将经过均值化的图像及标签（L格式）恢复成原始形状（单一格式）
        # 对图像操作
        image_Norm = image_Norm.cpu().numpy().transpose((1, 2, 0))
        image_Norm = image_Norm * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        image_Norm = image_Norm.clip(0, 1)
        # 对标签操作
        high, width = label_L.shape
        label_L = label_L.cpu().numpy().reshape(high * width, -1)
        image = np.zeros((high * width, 3), dtype=np.uint32)
        for ii in range(len(category_list)):
            index = np.where(label_L == ii)
            image[index, :] = category_list[ii]
        return image_Norm, image.reshape((high, width, 3))


class KoSDDDatasetSelectMethod(Data.Dataset):
    def __init__(self, train_val_state='train', train_ratio=0.8):
        super(KoSDDDatasetSelectMethod, self).__init__()
        images_names = os.listdir(ROOT_PATH + 'Images')
        for i in range(len(images_names)):
            images_names[i] = images_names[i][:-4]
        random.seed(0)  # 随机打乱，增加随机性
        random.shuffle(images_names)
        self.train_part = images_names[:int(train_ratio * len(images_names))]  # 交叉验证时更改这一部分
        self.val_part = images_names[int(train_ratio * len(images_names)):]  # 交叉验证时更改这一部分
        self.train_0, self.train_1 = TrainCategory(self.train_part)
        if train_val_state == 'train':
            self.train_state = True
        else:
            self.train_state = False

    def __len__(self):
        if self.train_state:
            return len(self.train_part)
        else:
            return len(self.val_part)

    def __getitem__(self, item):
        name = self.get_name(item)
        image_name = ROOT_PATH + 'Images/' + name + '.jpg'
        label_name = ROOT_PATH + 'Labels/' + name + '.bmp'
        image = Image.open(image_name).convert('RGB')
        label = Image.open(label_name).convert('RGB')
        label = self.Label_RGB_TO_L(label)
        image, label = self.Image_Transform(image, label)
        return image, label

    def get_name(self, item):
        if self.train_state:
            seed = np.random.rand(1)[0]
            if seed <= 0.5:  # 对第0类图像进行提取
                item = np.random.randint(len(self.train_0))
                return self.train_0[item]
            else:
                item = np.random.randint(len(self.train_1))
                return self.train_1[item]
        else:
            return self.val_part[item]

    def Label_RGB_TO_L(self, label_RGB):
        # 将RGB格式的图像变成0,1,2,3,4这种形式，方便训练学习
        label_RGB = label_RGB.__array__()
        label_L = np.zeros(label_RGB.shape[:-1])
        for category_i in range(len(category_list)):
            check_label = (label_RGB == np.array(category_list[category_i]))
            check_label = check_label[:, :, 0] * check_label[:, :, 1] * check_label[:, :, 2]
            label_L[check_label] = category_i
        label_L = Image.fromarray(label_L)  # 变成PIL格式与image统一
        return label_L

    def Image_Transform(self, image_RGB, label_L):
        if self.train_state:  # 加上随机翻转
            image_transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                                  transforms.RandomVerticalFlip(),
                                                  transforms.Resize(size=(256, 256)),
                                                  transforms.ToTensor(),
                                                  transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                                  ])
            label_transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                                  transforms.RandomVerticalFlip(),
                                                  transforms.Resize(size=(256, 256),
                                                                    interpolation=InterpolationMode.NEAREST),
                                                  transforms.ToTensor()
                                                  ])
            seed = np.random.randint(0, 125416584)
            torch.random.manual_seed(seed)
            image = image_transform(image_RGB)
            torch.random.manual_seed(seed)
            label = label_transform(label_L).long()
            return image, label[0]
        else:
            image_transform = transforms.Compose([transforms.Resize(size=(256, 256)),
                                                  transforms.ToTensor(),
                                                  transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                                  ])
            label_transform = transforms.Compose([transforms.Resize(size=(256, 256),
                                                                    interpolation=InterpolationMode.NEAREST),
                                                  transforms.ToTensor()])
            image = image_transform(image_RGB)
            label = label_transform(label_L).long()
            return image, label[0]

    def Show_Image(self, image_Norm, label_L):
        # 将经过均值化的图像及标签（L格式）恢复成原始形状（单一格式）
        # 对图像操作
        image_Norm = image_Norm.cpu().numpy().transpose((1, 2, 0))
        image_Norm = image_Norm * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        image_Norm = image_Norm.clip(0, 1)
        # 对标签操作
        high, width = label_L.shape
        label_L = label_L.cpu().numpy().reshape(high * width, -1)
        image = np.zeros((high * width, 3), dtype=np.uint32)
        for ii in range(len(category_list)):
            index = np.where(label_L == ii)
            image[index, :] = category_list[ii]
        return image_Norm, image.reshape((high, width, 3))


class KoSDDDatasetAddDefectMethod(Data.Dataset):
    def __init__(self, train_val_state='train', train_ratio=0.8):
        super(KoSDDDatasetAddDefectMethod, self).__init__()
        images_names = os.listdir(ROOT_PATH + 'Images')
        for i in range(len(images_names)):
            images_names[i] = images_names[i][:-4]
        random.seed(0)  # 随机打乱，增加随机性
        random.shuffle(images_names)
        self.train_part = images_names[:int(train_ratio * len(images_names))]  # 交叉验证时更改这一部分
        self.val_part = images_names[int(train_ratio * len(images_names)):]  # 交叉验证时更改这一部分
        self.train_0, self.train_1 = TrainCategory(self.train_part)
        if train_val_state == 'train':
            self.train_state = True
        else:
            self.train_state = False

    def __len__(self):
        if self.train_state:
            return len(self.train_part)
        else:
            return len(self.val_part)

    def __getitem__(self, item):
        if self.train_state:
            name = self.train_part[item]
        else:
            name = self.val_part[item]
        image_name = ROOT_PATH + 'Images/' + name + '.jpg'
        label_name = ROOT_PATH + 'Labels/' + name + '.bmp'
        image = Image.open(image_name).convert('RGB')
        label = Image.open(label_name).convert('RGB')
        label = self.Label_RGB_TO_L(label)
        image, label = self.AddDefect(image, label)
        image, label = self.Image_Transform(image, label)
        return image, label

    def AddDefect(self, image_RGB, label_L, p=0.5):
        '''对无缺陷的样品，把别的有缺陷样本的缺陷加过来'''
        if np.sum(label_L.__array__()) > 0:
            '''本身有缺陷就不用添加了'''
            return image_RGB, label_L
        else:
            seed = np.random.rand(1)[0]
            if seed < p:
                '''有一定的概率(p)不参与计算'''
                return image_RGB, label_L
            else:
                '''添加有缺陷样本的缺陷'''
                '''首先获取缺陷样本图像image与label'''
                item = np.random.randint(len(self.train_1))
                image_name = ROOT_PATH + 'Images/' + self.train_1[item] + '.jpg'
                label_name = ROOT_PATH + 'Labels/' + self.train_1[item] + '.bmp'
                image = Image.open(image_name).convert('RGB')
                label = Image.open(label_name).convert('RGB')
                label = self.Label_RGB_TO_L(label)
                '''给随机缺陷样本翻转等'''
                image_transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                                      transforms.RandomVerticalFlip()])
                label_transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                                      transforms.RandomVerticalFlip()])
                seed = np.random.randint(0, 125416584)
                torch.random.manual_seed(seed)
                image = image_transform(image)
                torch.random.manual_seed(seed)
                label = label_transform(label)
                '''添加缺陷样本的缺陷区域'''
                defect_area = np.array(
                    np.reshape(label.__array__(), newshape=(label.size[1], label.size[0], 1)) * image.__array__(),
                    dtype=np.uint8)  # 获取缺陷区域
                free_area = np.array(np.reshape(1.0 - label.__array__(),
                                                newshape=(label.size[1], label.size[0], 1)) * image_RGB.__array__(),
                                     dtype=np.uint8)  # 保存无缺陷图像的其他区域
                FinalImage = Image.fromarray(defect_area + free_area)
                return FinalImage, label
                # # # 绘图显示
                # import matplotlib.pyplot as plt
                # plt.figure()
                # plt.subplot(2, 3, 1)
                # plt.imshow(image.__array__())
                # plt.subplot(2, 3, 2)
                # plt.imshow(label.__array__())
                # plt.subplot(2, 3, 3)
                # plt.imshow(defect_area)
                # plt.subplot(2, 3, 4)
                # plt.imshow(free_area)
                # plt.subplot(2, 3, 5)
                # plt.imshow(image_RGB)
                # plt.subplot(2, 3, 6)
                # plt.imshow(FinalImage)
                # plt.show()
                # plt.close()
                #
                # exit(1)

    def Label_RGB_TO_L(self, label_RGB):
        # 将RGB格式的图像变成0,1,2,3,4这种形式，方便训练学习
        label_RGB = label_RGB.__array__()
        label_L = np.zeros(label_RGB.shape[:-1])
        for category_i in range(len(category_list)):
            check_label = (label_RGB == np.array(category_list[category_i]))
            check_label = check_label[:, :, 0] * check_label[:, :, 1] * check_label[:, :, 2]
            label_L[check_label] = category_i
        label_L = Image.fromarray(label_L)  # 变成PIL格式与image统一
        return label_L

    def Image_Transform(self, image_RGB, label_L):
        if self.train_state:  # 加上随机翻转
            image_transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                                  transforms.RandomVerticalFlip(),
                                                  transforms.Resize(size=(256, 256)),
                                                  transforms.ToTensor(),
                                                  transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                                  ])
            label_transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                                  transforms.RandomVerticalFlip(),
                                                  transforms.Resize(size=(256, 256),
                                                                    interpolation=InterpolationMode.NEAREST),
                                                  transforms.ToTensor()
                                                  ])
            seed = np.random.randint(0, 125416584)
            torch.random.manual_seed(seed)
            image = image_transform(image_RGB)
            torch.random.manual_seed(seed)
            label = label_transform(label_L).long()
            return image, label[0]
        else:
            image_transform = transforms.Compose([transforms.Resize(size=(256, 256)),
                                                  transforms.ToTensor(),
                                                  transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                                  ])
            label_transform = transforms.Compose([transforms.Resize(size=(256, 256),
                                                                    interpolation=InterpolationMode.NEAREST),
                                                  transforms.ToTensor()])
            image = image_transform(image_RGB)
            label = label_transform(label_L).long()
            return image, label[0]

    def Show_Image(self, image_Norm, label_L):
        # 将经过均值化的图像及标签（L格式）恢复成原始形状（单一格式）
        # 对图像操作
        image_Norm = image_Norm.cpu().numpy().transpose((1, 2, 0))
        image_Norm = image_Norm * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        image_Norm = image_Norm.clip(0, 1)
        # 对标签操作
        high, width = label_L.shape
        label_L = label_L.cpu().numpy().reshape(high * width, -1)
        image = np.zeros((high * width, 3), dtype=np.uint32)
        for ii in range(len(category_list)):
            index = np.where(label_L == ii)
            image[index, :] = category_list[ii]
        return image_Norm, image.reshape((high, width, 3))


if __name__ == "__main__":
    dataset = KoSDDDatasetAddDefectMethod(train_val_state='train')

    for i in range(4):
        image, label = dataset[i]
        image, label = dataset.Show_Image(image, label)
        import matplotlib.pyplot as plt

        plt.figure()
        plt.subplot(1, 2, 1)
        plt.imshow(image)
        plt.subplot(1, 2, 2)
        plt.imshow(label)
        plt.show()
        plt.close()
