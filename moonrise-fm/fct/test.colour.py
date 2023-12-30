import os
from PIL import Image


def get_image_colors(image_path):
    """获取图像中的所有颜色，并返回颜色列表"""
    with Image.open(image_path) as img:
        # 将图像转换为RGB模式（如果需要）
        img = img.convert("RGB")

        # 获取所有不同的颜色
        colors = img.getcolors(maxcolors=img.size[0] * img.size[1])
        return colors if colors is not None else []


def process_folder(folder_path):
    """遍历文件夹中的所有图像，打印颜色种类数和颜色列表"""
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
            image_path = os.path.join(folder_path, filename)
            colors = get_image_colors(image_path)
            color_count = len(colors)
            print(f"{filename}: 颜色种类数 = {color_count}")
            if colors:
                print("颜色列表:")
                for count, color in colors:
                    print(f"  {color} (数量: {count})")
            else:
                print("  太多颜色无法列出或无法识别颜色")


# 用实际的文件夹路径替换这里的 'path_to_folder'
folder_path = 'E:\Moonrise_1207\data\ACDC\masks'
process_folder(folder_path)
