#!/usr/bin/env python3
"""
h5_to_lerobot.py
将人形机器人自定义 HDF5 轨迹数据（master/puppet/camera_observations 结构）
批量转换为 LeRobot v3.0 格式数据集。

依赖：
    pip install lerobot h5py opencv-python numpy --break-system-packages

用法：
    python h5_to_lerobot.py \
        --src_dir /mnt/pfs/hdf5/my_robot_data \
        --repo_id my_org/wholebody_wam_stage1 \
        --root /mnt/pfs/lerobot_output

说明：
    - src_dir 下应包含若干个 .hdf5/.h5 轨迹文件（每个文件=一条episode）
    - 转换后的数据会写到 <root>/<repo_id>/ 目录下（不推送到HF Hub，纯本地）
    - state取自 puppet（机器人真实状态），action取自 master（操作者遥操作指令），
      这是ALOHA系列数据集的常见约定：state=follower当前状态，action=leader发出的指令。
      如果你想用别的字段组合，改下面 STATE_FIELDS / ACTION_FIELDS 即可。
"""

import argparse
import glob
import os

import cv2
import h5py
import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset

# ============ 按你的数据结构配置以下字段（可按需修改） ============

# 用哪几路相机
CAMERA_KEYS = ["camera_head", "camera_left", "camera_right"]
CAMERA_GROUP = "camera_observations/color_images"

# 拼成 observation.state 的字段（取自 puppet，即机器人真实状态）
STATE_FIELDS = [
    "puppet/arm_left_position_align/data",
    "puppet/arm_right_position_align/data",
    "puppet/end_effector_right_position_align/data",
    # "puppet/end_effector_left_position_align/data",
    "puppet/head_position_align/data",
    "puppet/waist_position_align/data",    
    "puppet/leg_position_align/data",
]

# 拼成 action 的字段（取自 master，即操作者遥操作指令）
ACTION_FIELDS = [
    "master/arm_left_position_align/data",
    "master/arm_right_position_align/data",
    "master/end_effector_right_position_align/data",
    "master/end_effector_left_position_align/data",
    "master/head_position_align/data",
    "master/waist_position_align/data",
    "master/body_height_align/data",
    "master/body_pitch_align/data",
]

DEFAULT_FPS = 30  # 当timestamp字段不存在或算不出来时的兜底值

# ================================================================


def estimate_fps(f: h5py.File) -> int:
    """用 camera_observations/timestamp 的首尾时间差估算真实帧率。
    注意：必须返回整数。LeRobotDataset 的视频编码依赖 Fraction 类型来设置
    time_base，传入 25.98 这类非整数浮点 fps 会导致
    "'float' object has no attribute 'numerator'" 报错，进而使该 episode
    的 save_episode() 失败，并污染内部缓冲区，连锁拖垮后续所有 episode。"""
    ts_path = "camera_observations/timestamp"
    if ts_path not in f:
        return DEFAULT_FPS
    ts = f[ts_path][:]
    if len(ts) < 2:
        return DEFAULT_FPS
    duration = float(ts[-1] - ts[0])
    if duration <= 0:
        return DEFAULT_FPS
    fps = (len(ts) - 1) / duration
    return int(round(fps))


def decode_image(raw) -> np.ndarray:
    """把HDF5里存的单帧数据解码成 HxWx3 uint8 的numpy数组（RGB）"""
    if isinstance(raw, (bytes, bytearray, np.void)):
        raw = np.frombuffer(bytes(raw), dtype=np.uint8)

    if isinstance(raw, np.ndarray) and raw.dtype == np.uint8 and raw.ndim == 1:
        # 一维uint8数组 = JPEG/PNG编码后的原始字节流，需要先解码
        img = cv2.imdecode(raw, cv2.IMREAD_COLOR)  # BGR
        if img is None:
            raise ValueError("图像解码失败：JPEG/PNG字节流解码为空，检查数据是否损坏")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    elif isinstance(raw, np.ndarray) and raw.ndim >= 2:
        # 已经是像素数组的情况（H,W,C）或（H,W）
        if raw.dtype != np.uint8:
            raw = np.clip(raw, 0, 255).astype(np.uint8)
        if raw.ndim == 2:
            raw = cv2.cvtColor(raw, cv2.COLOR_GRAY2RGB)
        return raw
    else:
        raise TypeError(f"未知的图像数据类型/形状: {type(raw)}, shape={getattr(raw, 'shape', None)}")


def read_concat_field(f: h5py.File, paths: list[str], frame_idx: int) -> np.ndarray:
    """从多个数据集路径里各取一帧，拼接成一维向量"""
    parts = []
    for p in paths:
        if p not in f:
            # 字段缺失直接报错，而不是跳过——跳过会导致向量维度变化，
            # 破坏 state/action 每一维和物理关节的固定对应关系
            raise KeyError(f"字段缺失: {p}，该文件字段不完整，无法保证维度对齐")
        val = f[p][frame_idx]
        parts.append(np.atleast_1d(np.asarray(val, dtype=np.float32)))
    if not parts:
        raise KeyError(f"一个字段都没读到，检查路径配置: {paths}")
    return np.concatenate(parts)


def get_num_frames(f: h5py.File) -> int:
    """用第一路相机的帧数作为该episode的总帧数"""
    cam0 = f[f"{CAMERA_GROUP}/{CAMERA_KEYS[0]}"]
    return cam0.shape[0]


def get_language_instruction(f: h5py.File) -> str:
    if "metadata/language_instruction" not in f:
        return "manipulate objects"
    val = f["metadata/language_instruction"][()]
    if isinstance(val, bytes):
        val = val.decode("utf-8")
    return str(val)


def build_dim_names(f: h5py.File, paths: list[str]) -> list[str]:
    """根据字段实际占的维度数，展开生成每一维的名字"""
    names = []
    for p in paths:
        # p 形如 "puppet/arm_left_position_align/data"，取中间这段当名字
        base_name = p.split("/")[1]
        dim = np.atleast_1d(np.asarray(f[p][0])).shape[0]
        if dim == 1:
            names.append(base_name)
        else:
            names.extend(f"{base_name}_{i}" for i in range(dim))
    return names


def build_features(sample_state_dim: int, sample_action_dim: int, sample_img_shape: tuple, state_names: list[str], action_names: list[str]):
    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (sample_state_dim,),
            "names": state_names,
        },
        "action": {
            "dtype": "float32",
            "shape": (sample_action_dim,),
            "names": action_names,
        },
    }
    for cam in CAMERA_KEYS:
        features[f"observation.images.{cam}"] = {
            "dtype": "video",
            "shape": sample_img_shape,  # (H, W, 3)
            # 之前这里漏了 names，导致 info.json 里视频类 feature 缺少
            # "names": ["height", "width", "channel"]，训练脚本解析
            # features schema 时依赖这个字段来知道 shape 每一维的语义，
            # 缺了就会报错。state/action 用 None 是对的（一维向量本来
            # 就不需要按维度命名），只有图像这种多维 shape 才需要写。
            "names": ["height", "width", "channel"],
        }
    return features


def convert(src_dir: str, repo_id: str, root: str, robot_type: str, resume: bool):
    h5_files = sorted(glob.glob(os.path.join(src_dir, "*.hdf5")) + glob.glob(os.path.join(src_dir, "*.h5")))
    if not h5_files:
        raise FileNotFoundError(f"{src_dir} 下没有找到 .hdf5/.h5 文件")
    print(f"共找到 {len(h5_files)} 个轨迹文件，准备转换到: {os.path.join(root, repo_id)}")

    # 用第一个文件的第一帧探测维度，用于定义数据集的 features schema
    with h5py.File(h5_files[0], "r") as f:
        state0 = read_concat_field(f, STATE_FIELDS, 0)
        action0 = read_concat_field(f, ACTION_FIELDS, 0)
        state_names = build_dim_names(f, STATE_FIELDS)
        action_names = build_dim_names(f, ACTION_FIELDS)
        img0 = decode_image(f[f"{CAMERA_GROUP}/{CAMERA_KEYS[0]}"][0])
        img_shape = img0.shape  # (H, W, 3)

        fps = estimate_fps(f)

    print(f"探测到 state维度={state0.shape[0]}, action维度={action0.shape[0]}, 图像尺寸={img_shape}, fps≈{fps}")

    features = build_features(state0.shape[0], action0.shape[0], img_shape, state_names, action_names)

    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=fps,
        root=root,
        features=features,
        robot_type=robot_type,
    )

    n_success, n_skipped = 0, 0

    for fi, h5_path in enumerate(h5_files):
        print(f"[{fi + 1}/{len(h5_files)}] 转换: {h5_path}")

        # 第一阶段：读取 h5 数据并 add_frame。这一步失败通常是"这一个文件本身
        # 数据有问题"（字段缺失、图像损坏等），跳过该文件、继续下一个是合理的。
        try:
            with h5py.File(h5_path, "r") as f:
                n_frames = get_num_frames(f)
                task = get_language_instruction(f)

                for t in range(n_frames):
                    frame = {
                        "observation.state": read_concat_field(f, STATE_FIELDS, t),
                        "action": read_concat_field(f, ACTION_FIELDS, t),
                        "task": task,
                    }
                    for cam in CAMERA_KEYS:
                        raw = f[f"{CAMERA_GROUP}/{cam}"][t]
                        frame[f"observation.images.{cam}"] = decode_image(raw)
                    dataset.add_frame(frame)
        except Exception as e:
            print(f"  [跳过] 读取该文件失败: {e}")
            n_skipped += 1
            # 清掉这一集已经攒进缓冲区但还没保存的帧，避免污染下一个 episode
            if hasattr(dataset, "clear_episode_buffer"):
                dataset.clear_episode_buffer()
            continue

        # 第二阶段：save_episode() 里会做视频编码（依赖 ffmpeg/av）。这一步失败
        # 通常是环境问题（ffmpeg 缺失、fps 格式不对、磁盘满等），不是单个文件
        # 的数据问题，而且失败后内部缓冲区状态可能已经不干净，继续跑后面的文件
        # 大概率会连锁出错，所以这里直接中止整个转换，而不是 continue。
        try:
            dataset.save_episode()
        except Exception as e:
            print(f"\n[致命错误] episode 保存/视频编码失败: {e}")
            print("这通常是环境问题（ffmpeg 未安装、fps 非整数等），而不是这个文件本身的数据问题。")
            print("请先解决该问题，再重新运行本脚本（建议先清空本次已生成的不完整输出目录）。")
            raise

        n_success += 1

    dataset.finalize()
    print("\n转换结束。")
    print(f"成功: {n_success} 个 | 跳过: {n_skipped} 个 | 共 {len(h5_files)} 个文件")
    print(f"输出目录: {os.path.join(root)}")
    if n_skipped > 0:
        print("⚠️ 有文件被跳过，请检查上面的日志确认是否需要处理。")


def main():
    parser = argparse.ArgumentParser(description="HDF5 -> LeRobot v3.0 数据集转换")
    parser.add_argument("--src_dir", required=True, help="存放.hdf5轨迹文件的目录")
    parser.add_argument("--repo_id", required=True, help="数据集逻辑名，如 my_org/my_dataset")
    parser.add_argument("--root", required=True, help="输出的本地根目录")
    parser.add_argument("--robot_type", default="humanoid_wam", help="机器人类型标识，写入元数据")
    parser.add_argument("--resume", action="store_true", help="预留参数：是否续跑（当前未启用）")
    args = parser.parse_args()

    convert(args.src_dir, args.repo_id, args.root, args.robot_type, args.resume)


if __name__ == "__main__":
    main()