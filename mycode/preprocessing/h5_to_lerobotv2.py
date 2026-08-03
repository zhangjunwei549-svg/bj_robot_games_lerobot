#!/usr/bin/env python3
"""
h5_to_lerobotv2.py (优化版)
将人形机器人自定义 HDF5 轨迹数据（master/puppet/camera_observations 结构）
批量转换为 LeRobot v3.0 格式数据集。

依赖：
    pip install lerobot h5py opencv-python numpy --break-system-packages

用法：
    python h5_to_lerobotv2.py \
        --src_dir /mnt/pfs/hdf5/my_robot_data \
        --repo_id my_org/wholebody_wam_stage1 \
        --root /mnt/pfs/lerobot_output \
        --num_decode_workers 6 \
        --image_writer_processes 4 \
        --image_writer_threads 4

说明：
    - src_dir 下应包含若干个 .hdf5/.h5 轨迹文件（每个文件=一条episode）
    - 转换后的数据会写到 <root>/<repo_id>/ 目录下（不推送到HF Hub，纯本地）
    - state取自 puppet（机器人真实状态），action取自 master（操作者遥操作指令）

本版相较原版的优化点：
    1. 不再逐帧调用 f[path][frame_idx] 做随机访问，而是把每个字段一次性
       整体读入内存（一次 I/O），循环内用 numpy 切片拼接——这是最大的
       性能瓶颈来源，HDF5 的逐元素随机访问开销远高于整块读取。
    2. 相机 dataset handle 在循环外缓存，避免每帧重复做 group 路径查找。
    3. 图像解码（cv2.imdecode，纯 CPU 计算）用线程池并行处理多路相机，
       cv2 的解码函数会释放 GIL，线程池对此类任务是有效的。
    4. LeRobotDataset.create() 传入 image_writer_processes/threads，
       让图片帧异步写盘，不阻塞主循环。
"""

import argparse
import glob
import os
from concurrent.futures import ThreadPoolExecutor

import cv2
import h5py
import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset

# ============ 按你的数据结构配置以下字段（可按需修改） ============

CAMERA_KEYS = ["camera_head", "camera_left", "camera_right"]
CAMERA_GROUP = "camera_observations/color_images"

STATE_FIELDS = [
    "puppet/arm_left_position_align/data",
    "puppet/arm_right_position_align/data",
    # "puppet/end_effector_left_position_align/data",
    "puppet/end_effector_right_position_align/data",
    # "puppet/head_position_align/data",
    # "puppet/waist_position_align/data",
    # "puppet/leg_position_align/data",
]

ACTION_FIELDS = [
    "master/arm_left_position_align/data",
    "master/arm_right_position_align/data",
    "master/end_effector_left_position_align/data",
    "master/end_effector_right_position_align/data",
    # "master/head_position_align/data",
    # "master/waist_position_align/data",
    # "master/body_height_align/data",
    # "master/body_pitch_align/data",
]

DEFAULT_FPS = 30  # 当timestamp字段不存在或算不出来时的兜底值

# ================================================================


def estimate_fps(f: h5py.File) -> int:
    """用 camera_observations/timestamp 的首尾时间差估算真实帧率。
    注意：必须返回整数。LeRobotDataset 的视频编码依赖 Fraction 类型来设置
    time_base，传入非整数浮点 fps 会导致 save_episode() 失败并污染内部
    缓冲区，连锁拖垮后续所有 episode。"""
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
        img = cv2.imdecode(raw, cv2.IMREAD_COLOR)  # BGR
        if img is None:
            raise ValueError("图像解码失败：JPEG/PNG字节流解码为空，检查数据是否损坏")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    elif isinstance(raw, np.ndarray) and raw.ndim >= 2:
        if raw.dtype != np.uint8:
            raw = np.clip(raw, 0, 255).astype(np.uint8)
        if raw.ndim == 2:
            raw = cv2.cvtColor(raw, cv2.COLOR_GRAY2RGB)
        return raw
    else:
        raise TypeError(f"未知的图像数据类型/形状: {type(raw)}, shape={getattr(raw, 'shape', None)}")


def read_fields_bulk(f: h5py.File, paths: list[str]) -> dict[str, np.ndarray]:
    """一次性把每个字段的全部帧读入内存，返回 {path: (n_frames, dim或标量) 数组}。
    这是本次优化的核心：避免对 h5py dataset 做逐帧随机索引 f[p][t]。"""
    out = {}
    for p in paths:
        if p not in f:
            raise KeyError(f"字段缺失: {p}，该文件字段不完整，无法保证维度对齐")
        arr = np.asarray(f[p][:], dtype=np.float32)
        if arr.ndim == 1:
            arr = arr[:, None]  # 统一成 (n_frames, dim)，标量字段 dim=1
        out[p] = arr
    return out


def concat_frame_from_bulk(bulk: dict[str, np.ndarray], paths: list[str], t: int) -> np.ndarray:
    """从预先整体读好的数组里取第 t 帧，拼接成一维向量（纯内存操作，无 I/O）"""
    return np.concatenate([bulk[p][t] for p in paths])


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
            "names": ["height", "width", "channel"],
        }
    return features


def decode_all_cameras(cam_datasets: dict, t: int, executor: ThreadPoolExecutor) -> dict:
    """并行解码同一帧下的多路相机图像。cv2.imdecode 会释放 GIL，
    线程池对这种 CPU 密集但可释放 GIL 的任务是有效的。"""
    futures = {cam: executor.submit(decode_image, ds[t]) for cam, ds in cam_datasets.items()}
    return {cam: fut.result() for cam, fut in futures.items()}


def convert(
    src_dir: str,
    repo_id: str,
    root: str,
    robot_type: str,
    resume: bool,
    num_decode_workers: int,
    image_writer_processes: int,
    image_writer_threads: int,
):
    h5_files = sorted(glob.glob(os.path.join(src_dir, "*.hdf5")) + glob.glob(os.path.join(src_dir, "*.h5")))
    if not h5_files:
        raise FileNotFoundError(f"{src_dir} 下没有找到 .hdf5/.h5 文件")
    print(f"共找到 {len(h5_files)} 个轨迹文件，准备转换到: {os.path.join(root)}")

    # 用第一个文件的第一帧探测维度，用于定义数据集的 features schema
    with h5py.File(h5_files[0], "r") as f:
        state_bulk0 = read_fields_bulk(f, STATE_FIELDS)
        action_bulk0 = read_fields_bulk(f, ACTION_FIELDS)
        state0 = concat_frame_from_bulk(state_bulk0, STATE_FIELDS, 0)
        action0 = concat_frame_from_bulk(action_bulk0, ACTION_FIELDS, 0)
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
        image_writer_processes=image_writer_processes,
        image_writer_threads=image_writer_threads,
    )

    n_success, n_skipped = 0, 0
    decode_executor = ThreadPoolExecutor(max_workers=num_decode_workers)

    try:
        for fi, h5_path in enumerate(h5_files):
            print(f"[{fi + 1}/{len(h5_files)}] 转换: {h5_path}")

            # 第一阶段：读取 h5 数据并 add_frame。这一步失败通常是"这一个文件本身
            # 数据有问题"（字段缺失、图像损坏等），跳过该文件、继续下一个是合理的。
            try:
                with h5py.File(h5_path, "r") as f:
                    n_frames = get_num_frames(f)
                    task = get_language_instruction(f)

                    # 关键优化：整个 episode 的 state/action 字段一次性读入内存，
                    # 循环内只做 numpy 切片，不再对 h5py dataset 做逐帧随机索引
                    state_bulk = read_fields_bulk(f, STATE_FIELDS)
                    action_bulk = read_fields_bulk(f, ACTION_FIELDS)

                    # 缓存相机 dataset handle，避免每帧重复做路径查找
                    cam_datasets = {cam: f[f"{CAMERA_GROUP}/{cam}"] for cam in CAMERA_KEYS}

                    for t in range(n_frames):
                        frame = {
                            "observation.state": concat_frame_from_bulk(state_bulk, STATE_FIELDS, t),
                            "action": concat_frame_from_bulk(action_bulk, ACTION_FIELDS, t),
                            "task": task,
                        }
                        images = decode_all_cameras(cam_datasets, t, decode_executor)
                        for cam, img in images.items():
                            frame[f"observation.images.{cam}"] = img
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
    finally:
        decode_executor.shutdown(wait=True)

    dataset.finalize()
    print("\n转换结束。")
    print(f"成功: {n_success} 个 | 跳过: {n_skipped} 个 | 共 {len(h5_files)} 个文件")
    print(f"输出目录: {os.path.join(root)}")
    if n_skipped > 0:
        print("⚠️ 有文件被跳过，请检查上面的日志确认是否需要处理。")


def main():
    parser = argparse.ArgumentParser(description="HDF5 -> LeRobot v3.0 数据集转换（优化版）")
    parser.add_argument("--src_dir", required=True, help="存放.hdf5轨迹文件的目录")
    parser.add_argument("--repo_id", required=True, help="数据集逻辑名，如 my_org/my_dataset")
    parser.add_argument("--root", required=True, help="输出的本地根目录")
    parser.add_argument("--robot_type", default="humanoid_wam", help="机器人类型标识，写入元数据")
    parser.add_argument("--resume", action="store_true", help="预留参数：是否续跑（当前未启用）")
    parser.add_argument(
        "--num_decode_workers",
        type=int,
        default=max(os.cpu_count() or 4, 4),
        help="并行解码图像的线程数，默认取 CPU 核数（至少4）",
    )
    parser.add_argument(
        "--image_writer_processes",
        type=int,
        default=4,
        help="LeRobotDataset 异步写图片帧用的进程数",
    )
    parser.add_argument(
        "--image_writer_threads",
        type=int,
        default=4,
        help="LeRobotDataset 异步写图片帧用的线程数",
    )
    args = parser.parse_args()

    convert(
        args.src_dir,
        args.repo_id,
        args.root,
        args.robot_type,
        args.resume,
        args.num_decode_workers,
        args.image_writer_processes,
        args.image_writer_threads,
    )


if __name__ == "__main__":
    main()