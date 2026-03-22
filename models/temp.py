import torch
import torch.nn as nn


class PoolConvDown(nn.Module):
    def __init__(self, in_channels, out_channels):
        """
        Pool-Conv-Down 模块实现
        :param in_channels: 输入通道数 Ci
        :param out_channels: 输出通道数 Co
        """
        super(PoolConvDown, self).__init__()

        # 确保输出通道大于输入通道，否则 Co - Ci 无意义
        assert out_channels > in_channels, "out_channels must be greater than in_channels"

        # 分支 1: Max-pooling (2x2, stride=2)
        # 输出尺寸: Ci x (H/2) x (W/2)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        # 分支 2: Conv (kernel=2x2, stride=2)
        # 输出通道数为 Co - Ci，卷积核大小为 2，步长为 2
        # 输出尺寸: (Co - Ci) x (H/2) x (W/2)
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels - in_channels,
            kernel_size=2,
            stride=2
        )

    def forward(self, x):
        # x 形状: (Batch, Ci, H, W)

        out_pool = self.maxpool(x)  # 得到 Ci 通道
        out_conv = self.conv(x)  # 得到 Co - Ci 通道

        # 在通道维度 (dim=1) 进行拼接 (Concat)
        # 最终输出通道: Ci + (Co - Ci) = Co
        output = torch.cat([out_pool, out_conv], dim=1)

        return output


# --- 测试代码 ---
if __name__ == "__main__":
    # 假设输入通道为 16，目标输出通道为 64，输入尺寸为 224x224
    ci, co = 16, 64
    model = PoolConvDown(in_channels=ci, out_channels=co)

    input_tensor = torch.randn(1, ci, 224, 224)
    output_tensor = model(input_tensor)

    print(f"输入形状: {input_tensor.shape}")
    print(f"输出形状: {output_tensor.shape}")  # 应该是 [1, 64, 112, 112]