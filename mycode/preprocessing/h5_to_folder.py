#!/usr/bin/env python3
"""
h5_to_folder.py
把人形机器人HDF5轨迹文件解压成一个"人类可直接查看"的文件夹：

    trajectory(1)_folder/
    ├── metadata.json                 <- 所有文字/标量元数据
    ├── camera_calibration.json       <- 相机标定参数（内参/外参/分辨率/颜色通道/坐标变换）
    ├── videos/
    │   ├── camera_camera.mp4         <- 主视角，499帧合成一个视频
    │   ├── camera_left.mp4
    │   └── camera_right.mp4
    ├── depth_images/
    │   ├── camera_camera/
    │   │   ├── frame_00000.png       <- 深度帧，16位PNG，保留原始深度值（不是视频，避免有损压缩破坏深度精度）
    │   │   └── ...
    │   ├── camera_left/
    │   └── camera_right/
    ├── is_intervene.csv              <- camera_observations/is_intervene，整段轨迹是否人工干预的标记
    ├── master/
    │   ├── arm_left_position_align.csv
    │   ├── arm_right_position_align.csv
    │   └── ...
    └── puppet/
        ├── arm_left_position_align.csv
        └── ...

依赖：
    pip install h5py opencv-python numpy --break-system-packages

用法：
    python h5_to_folder.py "trajectory(1).hdf5"
    python h5_to_folder.py "trajectory(1).hdf5" --outdir 自定义输出目录
"""

import argparse
import json
import os

import cv2
import h5py
import numpy as np


def decode_image(raw) -> np.ndarray:
    """把HDF5里存的单帧（JPEG字节流或原始像素）解码成 BGR uint8 图（cv2默认用BGR，直接写视频用这个更方便）"""
    if isinstance(raw, (bytes, bytearray, np.void)):
        raw = np.frombuffer(bytes(raw), dtype=np.uint8)

    if isinstance(raw, np.ndarray) and raw.dtype == np.uint8 and raw.ndim == 1:
        img = cv2.imdecode(raw, cv2.IMREAD_COLOR)  # 直接是BGR，写mp4正好不用转换
        if img is None:
            raise ValueError("图像解码失败：不是有效的JPEG/PNG字节流")
        return img
    elif isinstance(raw, np.ndarray) and raw.ndim >= 2:
        img = raw
        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        return img
    else:
        raise TypeError(f"未知的图像数据类型/形状: {type(raw)}")


def decode_depth_image(raw) -> np.ndarray:
    """把HDF5里存的单帧深度数据解码成单通道深度图（尽量保留原始精度，不强转8位）。

    深度数据常见的两种存储方式：
      1) 编码后的字节流（比如16位PNG编码），需要 cv2.imdecode 解码
      2) 已经是像素数组（比如 float32 米制深度，或 uint16 毫米深度），直接用
    """
    if isinstance(raw, (bytes, bytearray, np.void)):
        raw = np.frombuffer(bytes(raw), dtype=np.uint8)

    if isinstance(raw, np.ndarray) and raw.dtype == np.uint8 and raw.ndim == 1:
        # 字节流：优先用 IMREAD_UNCHANGED 保留位深（16位深度图常见）
        img = cv2.imdecode(raw, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError("深度图解码失败：不是有效的图像字节流")
        return img
    elif isinstance(raw, np.ndarray) and raw.ndim >= 2:
        return raw
    else:
        raise TypeError(f"未知的深度图数据类型/形状: {type(raw)}")


def estimate_fps(f: h5py.File, default_fps: float = 30.0) -> float:
    ts_path = "camera_observations/timestamp"
    if ts_path not in f:
        return default_fps
    ts = f[ts_path][:]
    if len(ts) < 2:
        return default_fps
    duration = float(ts[-1] - ts[0])
    if duration <= 0:
        return default_fps
    return round((len(ts) - 1) / duration, 2)


def export_videos(f: h5py.File, outdir: str, fps: float):
    """把 camera_observations/color_images 下每一路相机的499帧图片合成一个mp4"""
    color_group = "camera_observations/color_images"
    if color_group not in f:
        print("  [跳过视频] 找不到 camera_observations/color_images")
        return

    video_dir = os.path.join(outdir, "videos")
    os.makedirs(video_dir, exist_ok=True)

    for cam_name in f[color_group].keys():
        frames_dset = f[f"{color_group}/{cam_name}"]
        n_frames = frames_dset.shape[0]
        if n_frames == 0:
            continue

        first_img = decode_image(frames_dset[0])
        h, w = first_img.shape[:2]

        out_path = os.path.join(video_dir, f"{cam_name}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

        writer.write(first_img)
        for i in range(1, n_frames):
            img = decode_image(frames_dset[i])
            if img.shape[:2] != (h, w):
                img = cv2.resize(img, (w, h))
            writer.write(img)
        writer.release()
        print(f"  [视频] {cam_name}: {n_frames}帧 @ {fps}fps -> {out_path}")


def export_depth_images(f: h5py.File, outdir: str):
    """把 camera_observations/depth_images 下每一路相机的深度帧存成16位PNG序列。

    不用视频编码：视频编解码器（尤其是有损的 mp4v/H264/AV1）大多按8位RGB设计，
    直接拿去装深度值会丢精度甚至直接报错，所以这里逐帧存成PNG图片序列，
    cv2.imwrite 对 uint16 单通道数组会自动按16位PNG保存，精度不丢。
    """
    depth_group = "camera_observations/depth_images"
    if depth_group not in f:
        print("  [跳过深度图] 找不到 camera_observations/depth_images")
        return

    depth_root = os.path.join(outdir, "depth_images")

    for cam_name in f[depth_group].keys():
        frames_dset = f[f"{depth_group}/{cam_name}"]
        n_frames = frames_dset.shape[0]
        if n_frames == 0:
            continue

        cam_dir = os.path.join(depth_root, cam_name)
        os.makedirs(cam_dir, exist_ok=True)

        n_ok, n_fail = 0, 0
        for i in range(n_frames):
            try:
                depth = decode_depth_image(frames_dset[i])
                out_path = os.path.join(cam_dir, f"frame_{i:05d}.png")
                cv2.imwrite(out_path, depth)
                n_ok += 1
            except Exception as e:
                n_fail += 1
                if n_fail <= 3:  # 避免刷屏，只打印前几条
                    print(f"    [深度帧解码失败] {cam_name} 第{i}帧: {e}")

        print(f"  [深度图] {cam_name}: 成功{n_ok}帧 / 失败{n_fail}帧 -> {cam_dir}")


def export_is_intervene(f: h5py.File, outdir: str):
    """导出 camera_observations/is_intervene：整段轨迹每一帧是否处于人工干预状态"""
    key = "camera_observations/is_intervene"
    if key not in f:
        print("  [跳过is_intervene] 找不到 camera_observations/is_intervene")
        return
    data = f[key][:]
    out_path = os.path.join(outdir, "is_intervene.csv")
    np.savetxt(out_path, data.astype(int), delimiter=",", fmt="%d")
    print(f"  [is_intervene] {len(data)}帧 -> {out_path}")


def _h5_value_to_jsonable(val):
    """把 h5py 读出来的标量/小数组转换成能塞进 json 的 python 原生类型"""
    if isinstance(val, bytes):
        return val.decode("utf-8", errors="replace")
    if isinstance(val, np.generic):
        return val.item()
    if isinstance(val, np.ndarray):
        return val.tolist()
    return val


def export_camera_calibration(f: h5py.File, outdir: str):
    """导出相机标定相关的所有内容：内参、外参、分辨率、颜色通道、相机模型、
    机器人基座到相机的坐标变换。这些是"给算法用"的底层参数，体积很小，
    但对于核对标定是否正确、复现坐标变换很关键。"""
    calib = {}

    # 这些 group 下面通常是每路相机各一个 dataset（标量或小数组）
    per_camera_groups = [
        "camera_color_channel",
        "camera_color_resolution",
        "camera_depth_resolution",
        "camera_model",
        "camera_intrinsics",
        "camera_extrinsics",
    ]
    for group_name in per_camera_groups:
        if group_name not in f:
            continue
        group = f[group_name]
        if not isinstance(group, h5py.Group):
            continue
        cams = {}
        for cam_name in group.keys():
            item = group[cam_name]
            if isinstance(item, h5py.Dataset):
                cams[cam_name] = _h5_value_to_jsonable(item[()])
            elif isinstance(item, h5py.Group):
                # 有些标定字段可能是嵌套 group（比如 intrinsics 下面又分 fx/fy/cx/cy）
                sub = {}
                for sub_key in item.keys():
                    sub_item = item[sub_key]
                    if isinstance(sub_item, h5py.Dataset):
                        sub[sub_key] = _h5_value_to_jsonable(sub_item[()])
                cams[cam_name] = sub
        if cams:
            calib[group_name] = cams

    # base_to_robot_transformation：结构不确定（可能是空group、也可能内部有dataset），
    # 尽量兼容着导出
    if "base_to_robot_transformation" in f:
        item = f["base_to_robot_transformation"]
        if isinstance(item, h5py.Dataset):
            calib["base_to_robot_transformation"] = _h5_value_to_jsonable(item[()])
        elif isinstance(item, h5py.Group):
            sub = {}
            for key in item.keys():
                sub_item = item[key]
                if isinstance(sub_item, h5py.Dataset):
                    sub[key] = _h5_value_to_jsonable(sub_item[()])
            if sub:
                calib["base_to_robot_transformation"] = sub
            else:
                calib["base_to_robot_transformation"] = "（空group，文件里没有实际数据）"

    if not calib:
        print("  [跳过相机标定] 没有找到任何标定相关的group")
        return

    out_path = os.path.join(outdir, "camera_calibration.json")
    with open(out_path, "w", encoding="utf-8") as fp:
        json.dump(calib, fp, ensure_ascii=False, indent=2, default=str)
    print(f"  [标定参数] -> {out_path}")


def export_numeric_group(f: h5py.File, group_name: str, outdir: str):
    """把 master/ 或 puppet/ 下每个子字段的 data/is_intervene/timestamp 存成CSV"""
    if group_name not in f:
        return
    target_dir = os.path.join(outdir, group_name)
    os.makedirs(target_dir, exist_ok=True)

    for field_name in f[group_name].keys():
        field_group = f[f"{group_name}/{field_name}"]
        if not isinstance(field_group, h5py.Group):
            continue
        if "data" not in field_group:
            continue

        data = field_group["data"][:]
        csv_path = os.path.join(target_dir, f"{field_name}.csv")
        np.savetxt(csv_path, data, delimiter=",", fmt="%.6f")

        if "is_intervene" in field_group:
            is_intervene = field_group["is_intervene"][:]
            np.savetxt(
                os.path.join(target_dir, f"{field_name}_is_intervene.csv"),
                is_intervene.astype(int),
                delimiter=",",
                fmt="%d",
            )

        if "timestamp" in field_group:
            timestamp = field_group["timestamp"][:]
            np.savetxt(
                os.path.join(target_dir, f"{field_name}_timestamp.csv"),
                timestamp,
                delimiter=",",
                fmt="%.6f",
            )
    print(f"  [CSV] {group_name}/ 下 {len(list(f[group_name].keys()))} 个字段已导出")


def export_metadata(f: h5py.File, outdir: str):
    meta = {}
    if "metadata" in f:
        for key in f["metadata"].keys():
            val = f[f"metadata/{key}"][()]
            if isinstance(val, bytes):
                val = val.decode("utf-8")
            elif isinstance(val, np.generic):
                val = val.item()
            meta[key] = val

    if "camera_observations" in f and "timestamp" in f["camera_observations"]:
        ts = f["camera_observations/timestamp"][:]
        meta["estimated_fps"] = estimate_fps(f)
        meta["num_frames"] = int(len(ts))
        meta["duration_sec"] = float(ts[-1] - ts[0]) if len(ts) > 1 else 0.0

    meta_path = os.path.join(outdir, "metadata.json")
    with open(meta_path, "w", encoding="utf-8") as fp:
        json.dump(meta, fp, ensure_ascii=False, indent=2, default=str)
    print(f"  [META] -> {meta_path}")


def convert(h5_path: str, outdir: str):
    os.makedirs(outdir, exist_ok=True)
    print(f"正在处理: {h5_path}")
    print(f"输出到: {outdir}\n")

    with h5py.File(h5_path, "r") as f:
        fps = estimate_fps(f)

        export_metadata(f, outdir)
        export_camera_calibration(f, outdir)
        export_videos(f, outdir, fps)
        export_depth_images(f, outdir)
        export_is_intervene(f, outdir)
        export_numeric_group(f, "master", outdir)
        export_numeric_group(f, "puppet", outdir)

    print("\n完成。")


def main():
    parser = argparse.ArgumentParser(description="把HDF5轨迹文件导出为文件夹（图片合成视频，数值存CSV）")
    parser.add_argument("h5_path", help="HDF5文件路径")
    parser.add_argument("--outdir", default=None, help="输出目录，默认在原文件旁生成 *_folder 目录")
    args = parser.parse_args()

    outdir = args.outdir or (os.path.splitext(args.h5_path)[0] + "_folder")
    convert(args.h5_path, outdir)


if __name__ == "__main__":
    main()