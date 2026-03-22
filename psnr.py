import os
import torch
import numpy as np
import pandas as pd
from PIL import Image
import torch.nn.functional as f
from tqdm import tqdm
from pytorch_msssim import ssim
import torchvision.transforms as transforms


class Adder(object):
    def __init__(self):
        self.count = 0
        self.num = 0

    def __call__(self, num):
        self.count += num
        self.num += 1

    def reset(self):
        self.count = 0
        self.num = 0

    def average(self):
        return self.count / self.num if self.num != 0 else 0


def calculate_psnr_ssim_for_folders(pred_dir, gt_dir, output_csv_path):
    """
    计算预测文件夹和真值文件夹中图像的PSNR和SSIM
    使用与评估代码相同的计算方式

    参数:
        pred_dir: 预测图像文件夹路径
        gt_dir: 真值图像文件夹路径
        output_csv_path: 输出的CSV文件路径
    """

    # 获取所有预测图像文件
    pred_files = [f for f in os.listdir(pred_dir) if
                  f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'))]

    # 获取所有真值图像文件，并创建ID到文件路径的映射
    gt_files = [f for f in os.listdir(gt_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'))]
    gt_dict = {}

    for gt_file in gt_files:
        # 提取文件名（去掉扩展名）作为ID
        gt_id = os.path.splitext(gt_file)[0]
        gt_dict[gt_id] = os.path.join(gt_dir, gt_file)

    # 初始化数据结构
    results = []
    psnr_adder = Adder()
    ssim_adder = Adder()

    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 用于跟踪已处理的真值图像
    processed_gt_ids = set()

    print(f"开始计算PSNR和SSIM...")
    print(f"预测图像文件夹: {pred_dir}")
    print(f"真值图像文件夹: {gt_dir}")
    print(f"预测图像数量: {len(pred_files)}")
    print(f"真值图像数量: {len(gt_files)}")
    print(f"使用设备: {device}")

    # 图像转换
    to_tensor = transforms.ToTensor()

    # 遍历预测图像
    for pred_file in tqdm(pred_files, desc="处理图像"):
        pred_path = os.path.join(pred_dir, pred_file)

        # 提取ID（去掉扩展名，然后只取"_"前面的部分作为ID）
        base_name = os.path.splitext(pred_file)[0]
        # 如果文件名中有"_"，则只取"_"前面的部分作为ID
        if '_' in base_name:
            pred_id = base_name.split('_')[0]
        else:
            pred_id = base_name

        # 查找对应的真值图像
        if pred_id not in gt_dict:
            print(f"警告: 预测图像 '{pred_file}' 的ID '{pred_id}' 在真值图像文件夹中未找到对应图像")
            continue

        gt_path = gt_dict[pred_id]

        try:
            # 使用PIL读取图像
            pred_pil = Image.open(pred_path).convert('RGB')
            gt_pil = Image.open(gt_path).convert('RGB')

            # 转换为张量，并归一化到[0, 1]
            pred_tensor = to_tensor(pred_pil).unsqueeze(0)  # 添加batch维度
            gt_tensor = to_tensor(gt_pil).unsqueeze(0)

            # 移动到设备
            pred_tensor = pred_tensor.to(device)
            gt_tensor = gt_tensor.to(device)

            # 裁剪到相同尺寸（以防万一）
            h, w = pred_tensor.shape[2], pred_tensor.shape[3]
            H, W = gt_tensor.shape[2], gt_tensor.shape[3]

            if h != H or w != W:
                # 调整预测图像尺寸以匹配真值图像
                pred_tensor = f.interpolate(pred_tensor, size=(H, W), mode='bilinear', align_corners=False)
                print(f"调整图像尺寸: {pred_file} ({w}x{h}) -> {W}x{H}")

            # 裁剪到[0, 1]范围
            pred_clip = torch.clamp(pred_tensor, 0, 1)

            # 计算PSNR（使用评估代码相同的方式）
            psnr_val = 10 * torch.log10(1 / f.mse_loss(pred_clip, gt_tensor))
            psnr_val = psnr_val.item()  # 转换为标量

            # 计算SSIM（使用评估代码相同的方式，包括下采样）
            factor = 32
            h, w = H, W
            H_pad, W_pad = ((h + factor) // factor) * factor, ((w + factor) // factor * factor)

            # 计算下采样比例
            down_ratio = max(1, round(min(H_pad, W_pad) / 256))

            # 对图像进行下采样
            pred_down = f.adaptive_avg_pool2d(pred_clip, (int(H_pad / down_ratio), int(W_pad / down_ratio)))
            gt_down = f.adaptive_avg_pool2d(gt_tensor, (int(H_pad / down_ratio), int(W_pad / down_ratio)))

            # 计算SSIM
            ssim_val = ssim(pred_down, gt_down, data_range=1, size_average=False)
            ssim_val = ssim_val.item()  # 转换为标量

            # 记录结果
            results.append({
                '预测图像': pred_file,
                '真值图像': os.path.basename(gt_path),
                '图像ID': pred_id,
                'PSNR': round(psnr_val, 4),
                'SSIM': round(ssim_val, 6)
            })

            # 添加到累加器
            psnr_adder(psnr_val)
            ssim_adder(ssim_val)

            # 标记该真值图像已处理
            processed_gt_ids.add(pred_id)

        except Exception as e:
            print(f"处理图像 '{pred_file}' 时出错: {e}")
            continue

    # 检查是否有未处理的真值图像
    all_gt_ids = set(gt_dict.keys())
    unprocessed_gt_ids = all_gt_ids - processed_gt_ids
    if unprocessed_gt_ids:
        print(f"警告: 以下真值图像ID没有对应的预测图像: {sorted(unprocessed_gt_ids)[:10]}")  # 只显示前10个
        if len(unprocessed_gt_ids) > 10:
            print(f"... 还有 {len(unprocessed_gt_ids) - 10} 个未显示")

    # 如果没有处理任何图像，则退出
    if not results:
        print("错误: 没有成功处理任何图像")
        return

    # 创建DataFrame
    df = pd.DataFrame(results)

    # 添加平均值行
    avg_row = pd.DataFrame({
        '预测图像': ['平均值'],
        '真值图像': [''],
        '图像ID': [''],
        'PSNR': [round(psnr_adder.average(), 4)],
        'SSIM': [round(ssim_adder.average(), 6)]
    })

    df = pd.concat([df, avg_row], ignore_index=True)

    # 保存到CSV
    df.to_csv(output_csv_path, index=False, encoding='utf-8-sig')

    # 如果输出路径是Excel文件，也保存为Excel格式
    if output_csv_path.endswith('.xlsx'):
        try:
            df.to_excel(output_csv_path, index=False)
            print(f"结果已保存到Excel文件: {output_csv_path}")
        except Exception as e:
            print(f"保存到Excel文件时出错: {e}，已保存为CSV文件")

    # 打印统计信息
    print(f"\n{'=' * 50}")
    print(f"处理完成！")
    print(f"成功处理的图像数量: {len(results)}")
    print(f"平均PSNR: {psnr_adder.average():.2f} dB")
    print(f"平均SSIM: {ssim_adder.average():.4f}")
    print(f"结果已保存到: {output_csv_path}")
    print(f"{'=' * 50}")

    return df


if __name__ == "__main__":
    # 使用示例
    pred_dir = "F:/Datasets/SmokeBench/test/hazy/"  # 预测图像文件夹路径
    gt_dir = "F:/Datasets/SmokeBench/test/gt/"  # 真值图像文件夹路径
    output_csv_path = "C:/Users/zhaobaigan/Desktop/input.csv"  # 输出文件路径
    # 调用函数
    results_df = calculate_psnr_ssim_for_folders(pred_dir, gt_dir, output_csv_path)