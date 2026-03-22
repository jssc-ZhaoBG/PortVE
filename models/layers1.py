import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, bias=True, norm=False, relu=True, transpose=False):
        super(BasicConv, self).__init__()
        if bias and norm:
            bias = False

        padding = kernel_size // 2
        layers = list()
        if transpose:
            padding = kernel_size // 2 -1
            layers.append(nn.ConvTranspose2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        else:
            layers.append(
                nn.Conv2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        if norm:
            layers.append(nn.BatchNorm2d(out_channel))
        if relu:
            layers.append(nn.GELU())
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)


class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, filter=False):
        super(ResBlock, self).__init__()
        self.main = nn.Sequential(
            BasicConv(in_channel, out_channel, kernel_size=3, stride=1, relu=True),
            DeepPoolLayer(in_channel, out_channel) if filter else nn.Identity(),
            BasicConv(out_channel, out_channel, kernel_size=3, stride=1, relu=False)
        )

    def forward(self, x):
        return self.main(x) + x


class DeepPoolLayer(nn.Module):
    def __init__(self, k, k_out):
        super(DeepPoolLayer, self).__init__()
        self.pools_sizes = [8,4,2]
        dilation = [7,9,11]
        pools, convs, dynas = [],[],[]
        for j, i in enumerate(self.pools_sizes):
            pools.append(PoolConvDown(k, i))
            convs.append(nn.Conv2d(k, k, 3, 1, 1, bias=False))
            # dynas.append(MultiShapeKernel(dim=k, kernel_size=3, dilation=dilation[j]))
            dynas.append(FDMBlock(dim=k))
        self.pools = nn.ModuleList(pools)
        self.convs = nn.ModuleList(convs)
        self.dynas = nn.ModuleList(dynas)
        self.relu = nn.GELU()
        self.conv_sum = nn.Conv2d(k, k_out, 3, 1, 1, bias=False)

    def forward(self, x):
        x_size = x.size()
        resl = x
        for i in range(len(self.pools_sizes)):
            if i == 0:
                y = self.dynas[i](self.convs[i](self.pools[i](x)))
            else:
                y = self.dynas[i](self.convs[i](self.pools[i](x)+y_up))
            resl = torch.add(resl, F.interpolate(y, x_size[2:], mode='bilinear', align_corners=True))
            if i != len(self.pools_sizes)-1:
                y_up = F.interpolate(y, scale_factor=2, mode='bilinear', align_corners=True)
        resl = self.relu(resl)
        resl = self.conv_sum(resl)

        return resl



class ChannelAttention(nn.Module):

    def __init__(self, channels, ratio=16):
        super().__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels//ratio, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels//ratio, channels, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):

        w = self.avg_pool(x)
        w = self.fc(w)

        return x * w

class LightMSM(nn.Module):

    def __init__(self, dim):

        super().__init__()

        self.conv1 = nn.Conv2d(dim, dim, 1)

        self.dw3 = nn.Conv2d(
            dim, dim,
            kernel_size=3,
            padding=1,
            groups=dim
        )

        self.dw5 = nn.Conv2d(
            dim, dim,
            kernel_size=5,
            padding=2,
            groups=dim
        )

        self.dw7 = nn.Conv2d(
            dim, dim,
            kernel_size=7,
            padding=3,
            groups=dim
        )

        self.fuse = nn.Conv2d(dim*3, dim, 1)

        self.ca = ChannelAttention(dim)

        self.act = nn.GELU()

    def forward(self, x):

        identity = x

        x = self.conv1(x)

        f1 = self.dw3(x)
        f2 = self.dw5(x)
        f3 = self.dw7(x)

        f = torch.cat([f1, f2, f3], dim=1)

        f = self.fuse(f)

        f = self.ca(f)

        f = self.act(f)

        return identity + f


class FDMBlock(nn.Module):

    def __init__(self, dim):
        super().__init__()

        # multi-scale depthwise conv
        self.dw3 = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)
        self.dw5 = nn.Conv2d(dim, dim, 5, 1, 2, groups=dim)
        self.dw7 = nn.Conv2d(dim, dim, 7, 1, 3, groups=dim)

        # multi-scale aggregation
        self.reduce = nn.Conv2d(dim*3, dim, 1)

        # directional modeling
        self.dir_h = nn.Conv2d(dim, dim, (1,5), padding=(0,2), groups=dim)
        self.dir_v = nn.Conv2d(dim, dim, (5,1), padding=(2,0), groups=dim)

        # frequency decomposition
        self.pool = nn.AvgPool2d(3,1,1)

        # frequency modulation
        self.freq_attn = nn.Sequential(
            nn.Conv2d(dim, dim//4, 1),
            nn.GELU(),
            nn.Conv2d(dim//4, dim, 1),
            nn.Sigmoid()
        )

        self.fuse = nn.Conv2d(dim, dim, 1)

    def forward(self,x):

        identity = x

        # multi-scale
        f3 = self.dw3(x)
        f5 = self.dw5(x)
        f7 = self.dw7(x)

        ms = torch.cat([f3,f5,f7], dim=1)
        ms = self.reduce(ms)

        # directional
        dir_feat = self.dir_h(ms) + self.dir_v(ms)

        # frequency decomposition
        low = self.pool(dir_feat)
        high = dir_feat - low

        # modulation
        att = self.freq_attn(high)

        out = dir_feat * att

        out = self.fuse(out)

        return out + identity


class PoolConvDown(nn.Module):

    def __init__(self, channels, scale):
        """
        Pool-Conv Downsampling Module (ConvIR compatible)

        Args:
            channels: 输入输出通道数 (保持不变)
            scale: 下采样倍数 (2 / 4 / 8)
        """

        super(PoolConvDown, self).__init__()

        # 分支1：MaxPool 保留强响应
        self.pool = nn.MaxPool2d(kernel_size=scale, stride=scale)

        # 分支2：Depthwise stride conv 提取特征
        self.conv = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=scale,
                stride=scale,
                groups=channels,   # depthwise
                bias=False
            ),
            nn.Conv2d(
                channels,
                channels,
                kernel_size=1,
                bias=False
            ),
            nn.GELU()
        )

    def forward(self, x):

        pool_feat = self.pool(x)
        conv_feat = self.conv(x)

        # 融合两个分支
        out = pool_feat + conv_feat

        return out