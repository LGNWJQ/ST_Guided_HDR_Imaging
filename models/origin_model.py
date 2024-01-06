import torch
import torch.nn as nn


class HDR_Network(nn.Module):
    def __init__(self, in_channels=7, basic_channels=64, depths=[2, 2, 2]):
        super().__init__()
        self.encoder = Feature_Extraction_Net(in_channels=in_channels, basic_channels=basic_channels)
        self.decoder = HDR_Reconstruction_Net(basic_channels=basic_channels, depths=depths)

    def forward(self, X1, X2):
        encode_result, global_skip = self.encoder(X1, X2)
        hdr_image = self.decoder(encode_result, global_skip)
        return hdr_image


# 特征提取网络
class Feature_Extraction_Net(nn.Module):
    def __init__(self, in_channels=7, basic_channels=64):
        super().__init__()
        self.encoder1 = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=basic_channels,
                kernel_size=3,
                stride=1, padding=1
            ),
            ResnetBlock(in_channels=basic_channels)
        )
        self.encoder2 = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=basic_channels,
                kernel_size=3,
                stride=1, padding=1
            ),
            ResnetBlock(in_channels=basic_channels)
        )
        self.SelfAttention1 = CrossLinearAttention(dim=basic_channels)
        self.SelfAttention2 = CrossLinearAttention(dim=basic_channels)
        self.CrossAttention = CrossLinearAttention(dim=basic_channels)

    def forward(self, X1, X2):
        feature1 = self.encoder1(X1)
        feature2 = self.encoder2(X2)
        SA1 = self.SelfAttention1(feature1)
        SA2 = self.SelfAttention2(feature2)
        CA12 = self.CrossAttention(x=feature2, y=feature1)
        encode_result = torch.cat([SA1, CA12, SA2], dim=1)
        global_skip = feature2

        return encode_result, global_skip


# HDR 重建网络
class HDR_Reconstruction_Net(nn.Module):
    def __init__(self, basic_channels=64, depths=[2, 2, 2]):
        super().__init__()
        self.in_conv = nn.Conv2d(
            in_channels=basic_channels * 3,
            out_channels=basic_channels,
            kernel_size=3,
            stride=1, padding=1
        )
        self.decoder1 = SRAM(in_channels=basic_channels, num_layer=depths[0])
        self.decoder2 = SRAM(in_channels=basic_channels, num_layer=depths[1])
        self.decoder3 = SRAM(in_channels=basic_channels, num_layer=depths[2])
        self.out_conv1 = Block(
            in_channels=basic_channels * 3,
            out_channels=basic_channels
        )
        self.out_conv2 = Block(
            in_channels=basic_channels,
            out_channels=3
        )

    def forward(self, encode_result, global_skip):
        # 3 x basic_channels -> basic_channels
        F0 = self.in_conv(encode_result)

        F1 = self.decoder1(F0)
        F2 = self.decoder2(F1)
        F3 = self.decoder3(F2)

        Fc = torch.cat([F1, F2, F3], dim=1)

        F4 = self.out_conv1(Fc) + global_skip
        F5 = self.out_conv2(F4)

        return torch.tanh(F5) * 0.5 + 0.5


# Stacked residual attention modules
class SRAM(nn.Module):
    def __init__(self, in_channels=64, num_layer=2):
        super().__init__()
        layer_list = []
        for i in range(num_layer):
            res_block = ResnetBlock(in_channels)
            layer_list.append(res_block)
        layer_list.append(CrossLinearAttention(dim=in_channels))
        self.layers = nn.Sequential(*layer_list)

    def forward(self, x):
        return self.layers(x)


# ResnetBlock的子模块
class Block(nn.Module):
    def __init__(self, in_channels, out_channels, use_norm=True, use_act=True):
        super().__init__()
        self.norm = nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6,
                                 affine=True) if use_norm else nn.Identity()
        self.act = nn.SiLU() if use_act else nn.Identity()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1, padding=1
        )

    def forward(self, x):
        x = self.norm(x)
        x = self.act(x)
        x = self.conv(x)
        return x


# 残差网络模块
class ResnetBlock(nn.Module):
    def __init__(self, in_channels, out_channels=None, conv_shortcut=False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels if out_channels is None else out_channels
        self.use_conv_shortcut = conv_shortcut

        self.block1 = Block(in_channels=self.in_channels, out_channels=self.out_channels)
        self.block2 = Block(in_channels=self.out_channels, out_channels=self.out_channels)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    stride=1, padding=1
                )
            else:
                self.nin_shortcut = nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=1,
                    stride=1, padding=0
                )

    def forward(self, x):
        h = x
        h = self.block1(h)
        h = self.block2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x + h


# Attention
class CrossLinearAttention(nn.Module):
    """
    https://github.com/lucidrains/linear-attention-transformer
    """

    def __init__(self, dim, heads=4, num_head_channels=32):
        super().__init__()
        # 缩放系数
        self.scale = num_head_channels ** -0.5
        # 头数
        self.heads = heads
        # 每一个头的通道数量
        self.num_head_channels = num_head_channels
        # qkv维度
        hidden_dim = num_head_channels * heads

        self.to_q = nn.Conv2d(
            in_channels=dim,
            out_channels=hidden_dim,
            kernel_size=1,
            stride=1, padding=0,
            bias=False
        )
        self.to_k = nn.Conv2d(
            in_channels=dim,
            out_channels=hidden_dim,
            kernel_size=1,
            stride=1, padding=0,
            bias=False
        )
        self.to_v = nn.Conv2d(
            in_channels=dim,
            out_channels=hidden_dim,
            kernel_size=1,
            stride=1, padding=0,
            bias=False
        )

        self.to_out = nn.Sequential(
            nn.Conv2d(
                in_channels=hidden_dim,
                out_channels=dim,
                kernel_size=1,
                stride=1, padding=0
            ),
            nn.GroupNorm(num_groups=1, num_channels=dim)
        )

    def forward(self, x, y=None):
        b, c, hight, width = x.shape

        y = x if y is None else y

        # [b, c, h, w] -> [b, head x head_dim, h, w] -> [b x head, head_dim, h x w]
        q = self.to_q(x).view(b * self.heads, self.num_head_channels, hight * width)
        k = self.to_k(y).view(b * self.heads, self.num_head_channels, hight * width)
        v = self.to_v(y).view(b * self.heads, self.num_head_channels, hight * width)

        # 缩放
        q = q.softmax(dim=-2) * self.scale
        k = k.softmax(dim=-1)

        # [b x head, head_dim_k, h x w] x [b x head, h x w, head_dim_v]
        # = [b x head, head_dim_k, head_dim_v]
        context = torch.bmm(k, v.permute(0, 2, 1))
        # [b x head, head_dim_v, head_dim_k] x [b x head, head_dim, h x w]
        # = [b x head, head_dim_v, h x w]
        output = torch.bmm(context.permute(0, 2, 1), q)
        output = output.view(b, self.heads * self.num_head_channels, hight, width)

        return self.to_out(output) + x