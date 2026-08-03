#!/usr/bin/env python3
"""
HDF5 文件解析脚本
功能：
1. 递归打印文件内部的 Group / Dataset 层级结构
2. 显示每个 Dataset 的形状(shape)、数据类型(dtype)
3. 显示 Group / Dataset 上挂载的属性(attributes)
4. 可选：预览 Dataset 的部分数据内容

用法：s
    python parse_h5.py your_file.h5
    python parse_h5.py your_file.h5 --preview   # 同时预览数据内容
"""

import sys
import argparse
import h5py
import numpy as np


def format_attrs(obj):
    """格式化打印属性(attributes)"""
    lines = []
    for key, val in obj.attrs.items():
        lines.append(f"{key} = {val}")
    return lines


def preview_data(dataset, max_items=5):
    """预览数据集内容（只取前几个元素，避免大文件卡死）"""
    try:
        data = dataset[()]  # 读取全部（如果太大，可以改成 dataset[:max_items]）
        if isinstance(data, np.ndarray):
            flat = data.flatten()
            preview = flat[:max_items]
            suffix = " ..." if flat.size > max_items else ""
            return f"{preview}{suffix}"
        else:
            return str(data)
    except Exception as e:
        return f"<无法预览: {e}>"


def explore(name, obj, show_preview=False, indent_char="  "):
    """递归遍历并打印 HDF5 内部结构"""
    depth = name.count("/")
    indent = indent_char * depth
    short_name = name.split("/")[-1] or "/"

    if isinstance(obj, h5py.Group):
        print(f"{indent}📁 [Group] {short_name}")
    elif isinstance(obj, h5py.Dataset):
        print(f"{indent}📄 [Dataset] {short_name}")
        print(f"{indent}   shape={obj.shape}  dtype={obj.dtype}")
        if show_preview and obj.size > 0:
            print(f"{indent}   preview: {preview_data(obj)}")

    # 打印属性
    attrs = format_attrs(obj)
    if attrs:
        for line in attrs:
            print(f"{indent}   @attr {line}")


def parse_h5(filepath, show_preview=False):
    print(f"正在解析文件: {filepath}\n")
    with h5py.File(filepath, "r") as f:
        # 先打印根节点自身的属性
        root_attrs = format_attrs(f)
        if root_attrs:
            print("📁 [Root] /")
            for line in root_attrs:
                print(f"   @attr {line}")

        # 递归遍历文件内所有 group 和 dataset
        f.visititems(lambda name, obj: explore(name, obj, show_preview))

    print("\n解析完成。")


def main():
    parser = argparse.ArgumentParser(description="解析 HDF5 (.h5/.hdf5) 文件结构")
    parser.add_argument("filepath", help="HDF5 文件路径")
    parser.add_argument(
        "--preview", action="store_true", help="是否预览数据集的前几个元素"
    )
    args = parser.parse_args()

    parse_h5(args.filepath, show_preview=args.preview)


if __name__ == "__main__":
    main()