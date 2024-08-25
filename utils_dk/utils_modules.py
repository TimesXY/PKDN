import torch
import cv2 as cv
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from .utils_sr import DropPath
from skimage import feature, exposure


# 边缘特征计算模块
def edge_computing(batch_images):
    # 设置中间变量
    list_images = []

    for i in range(batch_images.shape[0]):
        # 获取图像
        gary_image = batch_images[i, :, :, :]

        # 图像维度转换
        gary_image = gary_image.permute(1, 2, 0).cpu().numpy()

        # 转换为灰度图
        image_gray = cv.cvtColor(gary_image, cv.COLOR_RGB2GRAY)

        # HOG算法实现特征提取
        _, hog_image = feature.hog(image_gray, orientations=9, pixels_per_cell=(16, 16),
                                   cells_per_block=(4, 2), visualize=True)

        # 重新调整直方图比例，以显示结果
        hog_image = exposure.rescale_intensity(hog_image, in_range=(0, 10))

        # 在新的维度拼接图像
        list_images.append(hog_image)

    # 数据格式转换
    list_ims = np.array(list_images)
    list_ims = torch.FloatTensor(list_ims)

    return list_ims


# 色彩信息计算模块
def color_computing(batch_images):
    # 设置中间变量
    hist_images = []

    for i in range(batch_images.shape[0]):
        # 获取图像
        gary_image = batch_images[i, :, :, :]

        # 图像维度转换
        gary_image = gary_image.permute(1, 2, 0).cpu().numpy()

        # 图像格式转换
        image_hls = cv.cvtColor(gary_image, cv.COLOR_RGB2HLS)

        # 图像维度转换
        per_image = image_hls.transpose(2, 0, 1)

        hist_images.append(per_image[:2, :, :])

    # 数据格式转换
    hist_images = np.array(hist_images)
    hist_images = torch.Tensor(hist_images)
    return hist_images


# 领域知识引导模块
class DKGMBlock(nn.Module):

    def __init__(self):
        super(DKGMBlock, self).__init__()

        self.conv = nn.Conv2d(3, 3, kernel_size=(3, 3), stride=(1, 1), padding=1)

    def forward(self, x):
        # 获取色彩信息曲线
        hist_images = color_computing(x).to("cuda")

        # 获取图像的边缘特征信息
        embedding_edges = edge_computing(x).to("cuda")

        # 扩展数据维度，用于合并
        embedding_edges = torch.unsqueeze(embedding_edges, dim=1)

        # 信息编码
        embedding = torch.cat((hist_images[:, :2, :, :], embedding_edges), dim=1)
        embedding = self.conv(embedding)

        embedding = x + embedding

        return embedding


# 门控通道注意模块 GCAM
class GCAMBlock(nn.Module):

    def __init__(self, in_channels, mid_channels, out_channels):
        super(GCAMBlock, self).__init__()

        # 超参数赋值
        self.in_channels = in_channels
        self.out_channels = out_channels

        # 模块的第一个卷积层
        self.conv_1 = nn.Conv2d(in_channels, mid_channels, kernel_size=(3, 3), padding=1)
        self.bn_1 = nn.BatchNorm2d(mid_channels)
        self.relu_1 = nn.ReLU()

        # 正则化层
        self.drop = DropPath(0.2)

        # 模块的第二个卷积层
        self.conv_2 = nn.Conv2d(mid_channels, mid_channels, kernel_size=(3, 3), padding=1)
        self.bn_2 = nn.BatchNorm2d(mid_channels)
        self.relu_2 = nn.ReLU()

        # 模块的注意力机制
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.alpha = nn.Parameter(torch.rand(mid_channels, 1, 1))
        self.beta = nn.Parameter(torch.rand(mid_channels, 1, 1))
        self.softmax = nn.Softmax(dim=0)
        self.tan_sig = nn.Tanh()

        # 模块的第三个卷积层
        self.conv_3 = nn.Conv2d(mid_channels, out_channels, kernel_size=(3, 3), padding=1)
        self.bn_3 = nn.BatchNorm2d(out_channels)
        self.relu_3 = nn.ReLU()

    def forward(self, x):
        # 残差连接
        short_x = x

        # 卷积层 1
        x = self.conv_1(x)
        x = self.bn_1(x)
        x = self.relu_1(x)

        # 正则化层
        x = self.drop(x)

        # 卷积层 2
        x = self.conv_2(x)
        x = self.bn_2(x)
        x = self.relu_2(x)

        # GCAM 门控通道注意
        x_c = self.gap(x)
        x_c = self.softmax(x_c)
        x_c = self.alpha * x_c + self.beta
        x_c = self.tan_sig(x_c)
        x_z = x_c * x

        # 获取新的特征图
        x = x_z + x

        # 卷积层 3
        x = self.conv_3(x)
        x = self.bn_3(x)
        x = self.relu_3(x)

        # 残差连接
        if self.in_channels == self.out_channels:
            x = x + short_x

        return x, x_c


# 病变区域注意模块
class LRAMBlock(nn.Module):

    def __init__(self, in_channels, mid_channels, out_channels):
        super(LRAMBlock, self).__init__()

        # 超参数赋值
        self.in_channels = in_channels
        self.mid_channels = mid_channels
        self.out_channels = out_channels

        # 模块的第一个卷积层
        self.conv_1 = nn.Conv2d(in_channels, mid_channels, kernel_size=(3, 3), padding=1)
        self.bn_1 = nn.BatchNorm2d(mid_channels)
        self.relu_1 = nn.ReLU()

        # 模块的第二个卷积层
        self.conv_2 = nn.Conv2d(mid_channels, mid_channels, kernel_size=(3, 3), padding=1)
        self.bn_2 = nn.BatchNorm2d(mid_channels)
        self.relu_2 = nn.ReLU()

        # 模块的注意力机制
        self.conv_21 = nn.Conv2d(mid_channels // 3, mid_channels // 3, kernel_size=(3, 3), stride=(3, 3),
                                 padding=1, dilation=(1, 1))
        self.conv_22 = nn.Conv2d(mid_channels // 3, mid_channels // 3, kernel_size=(3, 3), stride=(5, 5),
                                 padding=2, dilation=(2, 2))
        self.conv_23 = nn.Conv2d(mid_channels - 2 * (mid_channels // 3), mid_channels - 2 * (mid_channels // 3),
                                 kernel_size=(3, 3), stride=(9, 9), padding=4, dilation=(4, 4))

        # 线性映射过程
        self.conv_24 = nn.Conv2d(mid_channels, 1, kernel_size=(3, 3), stride=(1, 1), padding=1)

        # 模块的第三个卷积层
        self.conv_3 = nn.Conv2d(mid_channels, out_channels, kernel_size=(3, 3), padding=1)
        self.bn_3 = nn.BatchNorm2d(out_channels)
        self.relu_3 = nn.ReLU()

    def forward(self, x):
        # 残差连接
        short_x = x

        # 卷积层 1
        x = self.conv_1(x)
        x = self.bn_1(x)
        x = self.relu_1(x)

        # 卷积层 2
        x = self.conv_2(x)
        x = self.bn_2(x)
        x = self.relu_2(x)

        # 病变区域注意
        x_1 = x[:, : self.mid_channels // 3, :, :]
        x_2 = x[:, self.mid_channels // 3: 2 * (self.mid_channels // 3), :, :]
        x_3 = x[:, 2 * (self.mid_channels // 3): self.mid_channels, :, :]

        x_1 = self.conv_21(x_1)
        x_2 = self.conv_22(x_2)
        x_3 = self.conv_23(x_3)

        # 上采样过程
        x_1 = F.interpolate(x_1, size=x.shape[2:])
        x_2 = F.interpolate(x_2, size=x.shape[2:])
        x_3 = F.interpolate(x_3, size=x.shape[2:])

        # 特征图的合并
        x_c = torch.cat((x_1, x_2, x_3), dim=1)

        # 线性映射过程
        x_c = self.conv_24(x_c)

        # 获取新的特征图
        x = x_c * x

        # 卷积层 3
        x = self.conv_3(x)
        x = self.bn_3(x)
        x = self.relu_3(x)

        # 残差连接
        if self.in_channels == self.out_channels:
            x = x + short_x

        return x, x_c


# 定义特征蒸馏损失函数
class HintLoss(nn.Module):
    def __init__(self):
        super(HintLoss, self).__init__()
        self.criterion = nn.MSELoss()

    def forward(self, feature_s, feature_t):
        loss = self.criterion(feature_s, feature_t)
        return loss


# 掩码拼接
def cat_mask(t, mask1, mask2):
    t1 = (t * mask1).sum(dim=1, keepdims=True)
    t2 = (t * mask2).sum(1, keepdims=True)
    rt = torch.cat([t1, t2], dim=1)
    return rt


# 解耦蒸馏损失
def dkd_loss(logit_student, logit_teacher, target, alpha, beta, temperature):
    #  获取掩码
    target_reshape = target.reshape(-1)
    gt_mask = torch.zeros_like(logit_student).scatter_(1, target_reshape.unsqueeze(1), 1).bool()
    other_mask = torch.ones_like(logit_student).scatter_(1, target_reshape.unsqueeze(1), 0).bool()

    # 采用掩码进行数据选择
    predict_student_1 = F.softmax(logit_student / temperature, dim=1)
    predict_teacher_1 = F.softmax(logit_teacher / temperature, dim=1)
    predict_student = cat_mask(predict_student_1, gt_mask, other_mask)
    predict_teacher = cat_mask(predict_teacher_1, gt_mask, other_mask)

    # 计算目标类别损失
    t_loss = (F.kl_div(torch.log(predict_student), predict_teacher, reduction='sum')
              * (temperature ** 2) / target.shape[0])

    # 采用掩码进行数据选择
    # predict_teacher_2 = F.softmax(logit_teacher / temperature - 1000.0 * gt_mask, dim=1)
    # log_predict_student_2 = F.log_softmax(logit_student / temperature - 1000.0 * gt_mask, dim=1)
    predict_teacher_2 = predict_teacher_1[other_mask]
    log_predict_student_2 = predict_student_1[other_mask]

    # 计算非目标类别损失
    # n_loss = (F.kl_div(log_predict_student_2, predict_teacher_2, reduction='sum')
    #           * (temperature ** 2) / target.shape[0])
    n_loss = F.mse_loss(predict_teacher_2, log_predict_student_2)

    return alpha * t_loss + beta * n_loss


# 权重初始化
def init_weight(model):
    if type(model) in [nn.Linear]:
        nn.init.xavier_normal_(model.weight)
    if type(model) in [nn.Conv2d]:
        nn.init.kaiming_normal_(model.weight)
    if type(model) in [nn.BatchNorm2d]:
        nn.init.ones_(model.weight)
