import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

'''融合各尺度信息（stage4）'''
BN_MOMENTUM = 0.1


class SpaceAttention(nn.Module):
    def __init__(self, in_channel):
        super(SpaceAttention, self).__init__()
        self.space_att1 = nn.Sequential(nn.Conv2d(in_channels=in_channel, out_channels=1, kernel_size=3, padding=1),
                                        nn.BatchNorm2d(1),
                                        nn.Sigmoid())
        self.space_att2 = nn.Conv2d(in_channels=in_channel, out_channels=in_channel, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.space_att2(x) * self.space_att1(x) + x
        return x


class HighLowChannelAttention(nn.Module):
    def __init__(self, high_channel, low_channel):
        # high/low取决于分辨率
        super(HighLowChannelAttention, self).__init__()
        self.channel_att1 = nn.Sequential(
            nn.Conv2d(in_channels=low_channel, out_channels=high_channel, kernel_size=1),
            nn.BatchNorm2d(high_channel),
            nn.Sigmoid())
        self.channel_att2 = nn.Sequential(
            nn.Conv2d(in_channels=high_channel, out_channels=high_channel, kernel_size=3, padding=1),
        )

    def forward(self, features):
        high_features, low_features = features
        high_residual = high_features
        Avgpool = nn.AvgPool2d(kernel_size=(low_features.size(2), low_features.size(3)),
                               stride=(low_features.size(2), low_features.size(3)))
        # Maxpool = nn.MaxPool2d(kernel_size=(low_features.size(2), low_features.size(3)),
        #                        stride=(low_features.size(2), low_features.size(3)))
        # low_features = torch.cat((Avgpool(low_features), Maxpool(low_features)), dim=1)
        low_features = Avgpool(low_features)
        low_att = self.channel_att1(low_features)
        high_features = low_att * self.channel_att2(high_features)
        result = high_residual + high_features
        return result


class MultiScaleDualAttention(nn.Module):
    def __init__(self, channel_lists):
        super(MultiScaleDualAttention, self).__init__()
        self.SpaceAttentionList = []
        '''空间注意力'''
        for i in range(6):
            self.SpaceAttentionList.append(SpaceAttention(channel_lists[i]))
        '''通道注意力'''
        self.MultiScaleChannelAttentionList = []
        for i in range(5):
            self.MultiScaleChannelAttentionList.append(
                HighLowChannelAttention(high_channel=channel_lists[i], low_channel=channel_lists[i + 1]))
        self.SpaceAttentionList = nn.ModuleList(self.SpaceAttentionList)
        self.MultiScaleChannelAttentionList = nn.ModuleList(self.MultiScaleChannelAttentionList)

    def forward(self, x):
        assert len(x) == 6
        '''先经过一个多尺度通道注意力'''
        for i, m in enumerate(self.MultiScaleChannelAttentionList):
            x[i] = m([x[i], x[i + 1]])
        '''再经过一个空间注意力'''
        for i, m in enumerate(self.SpaceAttentionList):
            x[i] = m(x[i])
        return x


# class Residual_branch(nn.Module):
#     def __init__(self):
#         super(Residual_branch, self).__init__()
#         self.branch1 = nn.Conv2d(3, 3, kernel_size=(3, 3), padding=1)
#         self.branch2 = nn.Conv2d(3, 3, kernel_size=(3, 3), padding=2, dilation=2)
#         self.branch3 = nn.Conv2d(3, 3, kernel_size=(3, 3), padding=3, dilation=3)
#         self.branch4 = nn.Conv2d(3, 3, kernel_size=(3, 3), padding=4, dilation=4)
#         self.bn = nn.BatchNorm2d(3)
#         self.relu = nn.ReLU()
#         self.conv = nn.Conv2d(3, 3, kernel_size=(3, 3), padding=1)
#
#     def forward(self, x):
#         x_residual = x
#         x = self.conv(self.relu(self.bn(self.branch1(x) + self.branch2(x) + self.branch3(x) + self.branch4(x))))
#         return x + x_residual

class Residual_branch(nn.Module):
    def __init__(self):
        super(Residual_branch, self).__init__()
        self.branch = nn.Sequential(nn.Conv2d(3, 3, kernel_size=(3, 3), padding=1),
                                    nn.BatchNorm2d(3),
                                    nn.ReLU(),
                                    nn.Conv2d(3, 3, kernel_size=(3, 3), padding=1))

    def forward(self, x):
        return x + self.branch(x)


class FeatureResidualOriginalFusing(nn.Module):
    def __init__(self, channel_lists, out_ch=2, N=1):
        assert len(channel_lists) == 6
        super(FeatureResidualOriginalFusing, self).__init__()
        self.DualAttentionBlock = []
        self.Residual_branchs = []
        for i in range(N):
            self.DualAttentionBlock.append(MultiScaleDualAttention(channel_lists))
        self.side_list = []
        for i in range(6):
            self.side_list.append(nn.Conv2d(channel_lists[i] + 3, out_ch, kernel_size=3, padding=1))
            self.Residual_branchs.append(Residual_branch())  # 6个支路
        self.Residual_branchs.append(Residual_branch())  # 1个主路
        '''最终输出支路'''
        self.DualAttentionBlock = nn.ModuleList(self.DualAttentionBlock)
        self.side_list = nn.ModuleList(self.side_list)
        self.Residual_branchs = nn.ModuleList(self.Residual_branchs)
        self.out_branch = nn.Conv2d(6 * out_ch + 3, out_ch, kernel_size=3, padding=1)

    def forward(self, x, original):
        assert len(x) == 6
        _, _, h, w = x[0].shape
        '''先经过双注意力特征融合模块'''
        for i, m in enumerate(self.DualAttentionBlock):
            x = m(x)
        '''产生多监督分支，并插值到标准尺度'''
        for i, m in enumerate(self.side_list):
            original_i = F.interpolate(original, size=[x[i].shape[2], x[i].shape[3]], mode='bilinear',
                                       align_corners=False)
            x[i] = F.interpolate(m(torch.cat((x[i], self.Residual_branchs[i](original_i)), dim=1)), size=[h, w],
                                 mode='bilinear', align_corners=False)
        output = self.out_branch(torch.cat((*x, self.Residual_branchs[6](original)), dim=1))
        if self.training:
            return x + [output]  # do not use torch.sigmoid for amp safe
        else:
            return output


# class FeatureResidualOriginalFusing(nn.Module):
#     def __init__(self, channel_lists, out_ch=2):
#         super(FeatureResidualOriginalFusing, self).__init__()
#         assert len(channel_lists) == 6
#         '''空间注意力与输出拼接模块'''
#         self.SpaceAttentionList1 = []
#         self.side_list = []
#         for i in range(6):
#             self.SpaceAttentionList1.append(SpaceAttention(channel_lists[i]))
#             self.side_list.append(nn.Conv2d(channel_lists[i] + 3, out_ch, kernel_size=3, padding=1))
#         '''多尺度通道注意力'''
#         self.MutiScaleChannelAttentionList = []
#         for i in range(5):
#             self.MutiScaleChannelAttentionList.append(
#                 HighLowChannelAttention(high_channel=channel_lists[i], low_channel=channel_lists[i + 1]))
#         '''最终输出支路'''
#         self.SpaceAttentionList1 = nn.ModuleList(self.SpaceAttentionList1)
#         self.MutiScaleChannelAttentionList = nn.ModuleList(self.MutiScaleChannelAttentionList)
#         self.side_list = nn.ModuleList(self.side_list)
#         self.out_branch = nn.Conv2d(6 * out_ch + 3, out_ch, kernel_size=3, padding=1)
#
#     def forward(self, x, original):
#         _, _, h, w = x[0].shape
#         # x_temp = x
#         assert len(x) == 6
#         '''先经过一个空间注意力'''
#         for i, m in enumerate(self.SpaceAttentionList1):
#             x[i] = m(x[i])
#         '''再经过一个多尺度通道注意力'''
#         for i, m in enumerate(self.MutiScaleChannelAttentionList):
#             x[i] = m([x[i], x[i + 1]])
#         '''产生多监督分支，并插值到标准尺度'''
#         for i, m in enumerate(self.side_list):
#             original_i = F.interpolate(original, size=[x[i].shape[2], x[i].shape[3]], mode='bilinear',
#                                        align_corners=False)
#             x[i] = F.interpolate(m(torch.cat((x[i], original_i), dim=1)), size=[h, w],
#                                  mode='bilinear', align_corners=False)
#         output = self.out_branch(torch.cat((*x, original), dim=1))
#         if self.training:
#             return x + [output]  # do not use torch.sigmoid for amp safe
#         else:
#             return output


if __name__ == "__main__":
    pass
    # block = FeatureResidualOriginalProcessPredictProcessFusing()
    # print(block(torch.rand([2, 3, 256, 256])).shape)
