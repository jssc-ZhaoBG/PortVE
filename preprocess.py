import os
from PIL import Image
from torchvision import transforms


def resize_small_images(data_dir, min_size=256):
    """遍历数据集，将所有小于min_size的图片调整到min_size"""
    for split in ['train', 'test']:
        for subdir in ['hazy', 'gt']:
            dir_path = os.path.join(data_dir, split, subdir)
            if not os.path.exists(dir_path):
                continue

            for filename in os.listdir(dir_path):
                if filename.endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(dir_path, filename)
                    img = Image.open(img_path).convert('RGB')

                    w, h = img.size
                    if w < min_size or h < min_size:
                        # 保持宽高比进行调整
                        if w < h:
                            new_w = min_size
                            new_h = int(h * (min_size / w))
                        else:
                            new_h = min_size
                            new_w = int(w * (min_size / h))

                        img = img.resize((new_w, new_h), Image.BILINEAR)
                        img.save(img_path)
                        print(f"Resized: {img_path} from {w}x{h} to {new_w}x{new_h}")


# 使用示例
if __name__ == "__main__":
    data_dir = "/data/coding/Smoke/"  # 修改为您的数据路径
    resize_small_images(data_dir, min_size=256)