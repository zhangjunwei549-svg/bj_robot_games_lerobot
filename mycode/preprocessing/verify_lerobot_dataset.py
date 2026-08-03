#!/usr/bin/env python3
"""
verify_lerobot_dataset.py
用 LeRobot 自带的 LeRobotDataset 类加载转换后的数据集，做基本校验：
    1. 元数据是否正常（fps、features、episode数、总帧数）
    2. 随机抽几帧检查 state/action 的维度和数值范围
    3. 把某一帧的图像存成 png，肉眼看看画面对不对
    4. 检查 episode 边界（每个episode的起止frame index、task）是否合理

用法：
    python verify_lerobot_dataset.py \
        --repo_id my_org/wholebody_wam_stage1 \
        --root /mnt/pfs/lerobot_output \
        --episode 0
"""

import argparse
import os

import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_id", required=True)
    parser.add_argument("--root", required=True, help="转换脚本里 --root 传的那个目录")
    parser.add_argument("--episode", type=int, default=0, help="想重点检查的episode编号")
    parser.add_argument("--save_dir", default="./lerobot_check_out", help="抽帧图片存放目录")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    # ---------- 1. 加载数据集（纯本地，不联网） ----------
    # video_backend="pyav": 默认会优先尝试 torchcodec，但 torchcodec 需要
    # FFmpeg 的共享 DLL（"full-shared" 版），如果系统装的是 "essentials"
    # 静态版会加载失败。pyav 只依赖 ffmpeg 本身的解码能力，兼容性更好，
    # 而且这份数据集写入时用的编码后端本来就是 pyav（见 info.json 里的
    # "video.video_backend": "pyav"），用它读取是最匹配的。
    dataset = LeRobotDataset(repo_id=args.repo_id, root=args.root, video_backend="pyav")

    print("=" * 60)
    print("【元数据】")
    print(f"fps          : {dataset.fps}")
    print(f"num_episodes : {dataset.num_episodes}")
    print(f"num_frames   : {dataset.num_frames}")
    print(f"robot_type   : {dataset.meta.robot_type}")
    print("features:")
    for k, v in dataset.features.items():
        print(f"  - {k}: dtype={v.get('dtype')}, shape={v.get('shape')}")
    print("=" * 60)

    # ---------- 2. 检查 episode 边界 ----------
    if args.episode >= dataset.num_episodes:
        raise ValueError(f"episode 编号超界，该数据集共有 {dataset.num_episodes} 个episode")

    ep_meta = dataset.meta.episodes[args.episode]
    ep_start = ep_meta["dataset_from_index"]
    ep_end = ep_meta["dataset_to_index"]
    ep_len = ep_end - ep_start
    print(f"【episode {args.episode}】起止 index: [{ep_start}, {ep_end}) 共 {ep_len} 帧")

    # ---------- 3. 抽首帧、中间帧、末帧看 state/action ----------
    for offset_name, idx in [
        ("首帧", ep_start),
        ("中间帧", ep_start + ep_len // 2),
        ("末帧", ep_end - 1),
    ]:
        item = dataset[idx]
        state = item["observation.state"]
        action = item["action"]
        task = item.get("task", "N/A")
        print(f"\n--- {offset_name} (global idx={idx}) ---")
        print(f"task    : {task}")
        print(f"state   : shape={tuple(state.shape)}, "
              f"min={float(state.min()):.3f}, max={float(state.max()):.3f}")
        print(f"action  : shape={tuple(action.shape)}, "
              f"min={float(action.min()):.3f}, max={float(action.max()):.3f}")

        # 有没有 nan/inf，转换时最容易在这里出问题
        for name, arr in [("state", state), ("action", action)]:
            arr_np = arr.numpy() if hasattr(arr, "numpy") else np.asarray(arr)
            if np.isnan(arr_np).any() or np.isinf(arr_np).any():
                print(f"  ⚠️ 警告：{name} 里出现 NaN/Inf，检查一下HDF5源数据或字段拼接逻辑")

    # ---------- 4. 把中间帧的每一路相机图像存成png，肉眼检查 ----------
    mid_idx = ep_start + ep_len // 2
    item = dataset[mid_idx]
    img_keys = [k for k in item.keys() if k.startswith("observation.images.")]
    print(f"\n【保存抽帧图像】共 {len(img_keys)} 路相机 -> {args.save_dir}")
    for k in img_keys:
        img = item[k]
        img_np = img.numpy() if hasattr(img, "numpy") else np.asarray(img)
        # LeRobot 的图像张量通常是 CHW、float [0,1]，存图前转成 HWC uint8
        if img_np.ndim == 3 and img_np.shape[0] in (1, 3):
            img_np = np.transpose(img_np, (1, 2, 0))
        if img_np.dtype != np.uint8:
            img_np = np.clip(img_np * 255.0, 0, 255).astype(np.uint8)

        import cv2
        cam_name = k.replace("observation.images.", "")
        out_path = os.path.join(args.save_dir, f"ep{args.episode}_mid_{cam_name}.png")
        cv2.imwrite(out_path, cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))
        print(f"  - {cam_name}: {img_np.shape} -> {out_path}")

    print("\n检查完成！建议重点看：")
    print("  1) 上面打印的 state/action 维度是否和你预期一致（等于 STATE_FIELDS/ACTION_FIELDS 拼接后的总长度）")
    print("  2) 保存的png图像是否清晰、没有花屏/颜色反转（BGR/RGB搞反是最常见的坑）")
    print("  3) episode 帧数是否和源hdf5里对应文件的帧数一致")


if __name__ == "__main__":
    main()