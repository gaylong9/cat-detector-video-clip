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
    model_path: str = "yolo11s.pt"  # 支持 yolo11 (n/s/m/l/x)
    confidence_threshold: float = 0.6
    cat_class_id: List[int] = field(default_factory=lambda: [15])  # yolo cat class id(s)
    save_start_frame: bool = False  # 保存开始帧，观察检测结果

    # 检测步长：出现猫时与未出现猫时的检测步长，单位s
    detect_step_with_start: float = 1.0
    detect_step_without_start: float = 0.5

    # 时间戳与临时/最终文件
    timestamps_name: str = "cat_timestamps.csv"  # CSV: video,start_sec,end_sec
    merged_timestamps_name: str = "merged_cat_timestamps.csv"
    processed_log_name: str = "processed_videos.txt"

    # 裁剪与拼接
    max_merge_gap_seconds: float = 5.0  # 合并相邻片段的最大间隔秒数
    final_video_name: str = "output.mp4"
    start_expand_seconds: float = 2.0
    end_expand_seconds: float = 2.0

    # 临时目录（在 output_dir 下）
    tmp_dir_name: str = "tmp_catclipper"

    # 其它
    delete_temp_files: bool = True
    video_extensions: List[str] = field(default_factory=lambda: [".mp4", ".avi", ".mov", ".mkv"])
