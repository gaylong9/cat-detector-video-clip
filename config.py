# config.py
from dataclasses import dataclass, field
from pathlib import Path
from typing import List


@dataclass
class Config:
    # 基本路径
    input_dir: Path = Path("D:/Desktop/dongdong")
    output_dir: Path = Path("D:/Desktop/dongdong")

    # YOLO / 检测
    model_path: str = "yolo11n.pt"  # 支持 yolo11 (n/s/m/l/x)
    confidence_threshold: float = 0.6
    cat_class_id: List[int] = field(default_factory=lambda: [15])  # yolo cat class id(s)
    min_continue_seconds: float = 1.0  # 连续出现猫的最小秒数，判定为有效片段

    # 跳帧系数：出现猫时与未出现猫时的帧间隔系数（你原来的设计）
    frame_step_coef_with_start: float = 1.0
    frame_step_coef_without_start: float = 0.25

    # 时间戳与临时/最终文件
    timestamps_name: str = "cat_timestamps.csv"  # CSV: video,start_sec,end_sec
    merged_timestamps_name: str = "cat_timestamps_merged.csv"
    processed_log_name: str = "processed_videos.txt"

    # 裁剪与拼接
    max_merge_gap_seconds: float = 5.0  # 合并相邻片段的最大间隔秒数
    final_video_name: str = "output.mp4"

    # 临时目录（在 output_dir 下）
    tmp_dir_name: str = ".tmp_catclipper"

    # ffmpeg behavior
    delete_temp_fragments: bool = True

    # 其它
    video_extensions: List[str] = field(default_factory=lambda: [".mp4", ".avi", ".mov", ".mkv"])
