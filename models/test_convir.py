import torch
import torch.nn as nn
from thop import profile, clever_format

# === 导入 ConvIR 模型（修改为文件一的结构） ===
# 注意：文件一的 ConvIR 类需要 version 参数
from ConvIR1 import build_net  # 导入文件一的构建函数

def test_convIR_forward_backward():
    print("=== ConvIR Unit Test (文件一版本) ===")

    # 1️⃣ 初始化模型 - 使用版本参数而非 num_res
    # 文件一支持 'small'(4), 'base'(8), 'large'(16) 三种版本
    version = 'small'  # 可改为 'base' 或 'large'
    model = build_net(version)  # 使用文件一的构建函数
    model.eval()
    model = model.cuda() if torch.cuda.is_available() else model
    device = next(model.parameters()).device
    print(f"Model on device: {device}")
    print(f"Model version: {version}")

    # 2️⃣ 随机输入 - 注意：将输入尺寸从192x192调整为256x256
    # 原始模型对输入尺寸有要求，256x256是最小尺寸之一
    input_size = 192
    x = torch.randn(1, 3, input_size, input_size).to(device)
    print(f"Input shape: {tuple(x.shape)}")

    # 3️⃣ 前向传播
    with torch.no_grad():
        outputs = model(x)

    # 4️⃣ 打印每个输出的形状
    for i, out in enumerate(outputs):
        print(f"Output[{i}] shape: {tuple(out.shape)}")

    # 5️⃣ 测试梯度是否能传播
    x.requires_grad = True
    y = model(x)[-1].mean()  # 只取最后一个输出
    y.backward()
    print(f"Gradients on input: {x.grad.abs().mean().item():.6f}")

    # 检查模型参数是否有梯度
    has_grad = any(p.grad is not None for p in model.parameters())
    print(f"Model parameters have gradients: {has_grad}")

    # 6️⃣ 打印参数总量
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {total_params / 1e6:.3f} M")

    # 7️⃣ 计算 FLOPs - 使用相同的输入尺寸
    dummy_input = torch.randn(1, 3, input_size, input_size).to(device)
    flops, params = profile(model, inputs=(dummy_input,), verbose=False)
    flops, params = clever_format([flops, params], "%.3f")
    print(f"FLOPs: {flops} | Params: {params}")

    print("✅ Test complete: forward/backward/FLOPs/params all OK.")

if __name__ == "__main__":
    test_convIR_forward_backward()