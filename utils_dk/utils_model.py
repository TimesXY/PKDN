import torch.nn as nn
from .utils_modules import GCAMBlock, LRAMBlock, DKGMBlock


class DKNet(nn.Module):
    def __init__(self, num_class):
        super(DKNet, self).__init__()

        # 领域知识引导阶段
        self.dk_1 = DKGMBlock()
        self.dk_2 = DKGMBlock()

        # 初始阶段 1
        self.conv_11 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.bn_11 = nn.BatchNorm2d(64)
        self.act_11 = nn.ReLU()

        self.conv_21 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.bn_21 = nn.BatchNorm2d(64)
        self.act_21 = nn.ReLU()

        # 病变区域注意阶段 1
        self.lr_11 = LRAMBlock(in_channels=64, mid_channels=32, out_channels=64)
        self.lr_21 = LRAMBlock(in_channels=64, mid_channels=32, out_channels=64)

        # 下采样阶段 1  # (112)
        self.down_s11 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.down_s21 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        # 病变区域注意阶段 2
        self.lr_12 = LRAMBlock(in_channels=64, mid_channels=64, out_channels=128)
        self.lr_22 = LRAMBlock(in_channels=64, mid_channels=64, out_channels=128)

        # 下采样阶段 2  # (56)
        self.down_s12 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.down_s22 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        # GCAM 阶段 3
        self.gcam_13 = GCAMBlock(in_channels=128, mid_channels=64, out_channels=128)
        self.gcam_23 = GCAMBlock(in_channels=128, mid_channels=64, out_channels=128)

        # 下采样阶段 3  # (28)
        self.down_s13 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.down_s23 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        # GCAM 阶段 4
        self.gcam_14 = GCAMBlock(in_channels=128, mid_channels=64, out_channels=128)
        self.gcam_24 = GCAMBlock(in_channels=128, mid_channels=64, out_channels=128)

        # 下采样阶段 4  # (14)
        self.down_s14 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.down_s24 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        # GCAM 阶段 5
        self.gcam_15 = GCAMBlock(in_channels=128, mid_channels=128, out_channels=256)
        self.gcam_25 = GCAMBlock(in_channels=128, mid_channels=128, out_channels=256)

        # 下采样阶段 5  # (7)
        self.down_s15 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.down_s25 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        # GCAM 阶段 6
        self.gcam_16 = GCAMBlock(in_channels=256, mid_channels=128, out_channels=256)
        self.gcam_26 = GCAMBlock(in_channels=256, mid_channels=128, out_channels=256)

        # 全局平均池化层
        self.gap_1 = nn.AdaptiveAvgPool2d((1, 1))
        self.gap_2 = nn.AdaptiveAvgPool2d((1, 1))

        self.flatten = nn.Flatten()

        self.linear_17 = nn.Linear(256, num_class)
        self.linear_27 = nn.Linear(256, num_class)

    def forward(self, x):

        # 领域知识引导阶段
        x_1 = self.dk_1(x)
        x_2 = self.dk_2(x)

        # 初始卷积阶段
        x_1 = self.conv_11(x_1)
        x_1 = self.bn_11(x_1)
        x_1 = self.act_11(x_1)

        x_2 = self.conv_21(x_2)
        x_2 = self.bn_21(x_2)
        x_2 = self.act_21(x_2)

        # 病变区域注意阶段 1
        lx_11, lx_at_11 = self.lr_11(x_1)
        lx_21, lx_at_21 = self.lr_21(x_2)

        # 下采样阶段 1
        dx_11 = self.down_s11(lx_11)
        dx_21 = self.down_s21(lx_21)

        # 病变区域注意阶段 2
        lx_12, lx_at_12 = self.lr_12(dx_11)
        lx_22, lx_at_22 = self.lr_22(dx_21 + dx_11)

        # 下采样阶段 2
        dx_12 = self.down_s12(lx_12)
        dx_22 = self.down_s22(lx_22)

        # GCAM 阶段 3
        gx_13, gx_at_13 = self.gcam_13(dx_12)
        gx_23, gx_at_23 = self.gcam_23(dx_22)

        # 下采样阶段 3
        dx_13 = self.down_s13(gx_13)
        dx_23 = self.down_s23(gx_23)

        # GCAM 阶段 4
        gx_14, gx_at_14 = self.gcam_14(dx_13)
        gx_24, gx_at_24 = self.gcam_24(dx_23 + dx_13)

        # 下采样阶段 4
        dx_14 = self.down_s14(gx_14)
        dx_24 = self.down_s24(gx_24)

        # GCAM 阶段 5
        gx_15, gx_at_15 = self.gcam_15(dx_14)
        gx_25, gx_at_25 = self.gcam_25(dx_24)

        # 下采样阶段 5
        dx_15 = self.down_s15(gx_15)
        dx_25 = self.down_s25(gx_25)

        # GCAM 阶段 6
        gx_16, gx_at_16 = self.gcam_16(dx_15)
        gx_26, gx_at_26 = self.gcam_26(dx_25 + dx_15)

        # 最后的 GAP
        y_1 = self.gap_1(gx_16)
        y_1 = self.flatten(y_1)
        y_1 = self.linear_17(y_1)

        y_2 = self.gap_2(gx_26)
        y_2 = self.flatten(y_2)
        y_2 = self.linear_27(y_2)

        # 保存结果
        model = {"Y1": y_1, "Y2": y_2, "FA_1": lx_at_11, "FA_2": lx_at_21,
                 "GA_1": gx_at_13, "GA_2": gx_at_23, "GA_3": gx_at_15, "GA_4": gx_at_25}

        return model
