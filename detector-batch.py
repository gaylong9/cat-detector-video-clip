# detector.py
import os

import torch
from ultralytics import YOLO
import cv2
from pathlib import Path
from typing import List, Tuple, Iterator
from config import Config
from utils import append_csv, format_seconds, logger
from tqdm import tqdm
import math


class Detector:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model = YOLO(cfg.model_path).to(self.device)
        logger.info(f"YOLO模型位于设备: {self.model.device}")

    def _iter_video_files(self) -> List[Path]:
        p = Path(self.cfg.input_dir)
        if not p.exists():
            raise FileNotFoundError(f"input_dir not found: {p}")
        files = sorted([f for f in p.iterdir() if f.suffix.lower() in self.cfg.video_extensions])
        return files

    def detect_all(self) -> None:
        """
        遍历 input_dir 中的视频，执行逐帧检测并将猫段写入 timestamps csv。
        返回检测到的片段列表（用于后续处理或测试）。
        """
        processed_log = Path(self.cfg.output_dir) / self.cfg.tmp_dir_name / self.cfg.processed_log_name
        timestamps_path = Path(self.cfg.output_dir) / self.cfg.tmp_dir_name/ self.cfg.timestamps_name
        detect_ok = Path(self.cfg.output_dir) / self.cfg.tmp_dir_name / 'detect.ok'

        # 若检测均完成，跳过
        if detect_ok.exists() and timestamps_path.exists():
            logger.info("发现detect.ok，检测均已完成，跳过本阶段")
            return

        # 已处理集合
        processed = set()
        if processed_log.exists():
            processed = set([l.strip() for l in processed_log.read_text(encoding='utf-8').splitlines() if l.strip()])

        found_fragment_num = 0

        video_files = self._iter_video_files()
        for video_path in tqdm(video_files, desc="扫描视频文件", unit="file"):
            if video_path.name in processed:
                logger.info("跳过已处理：%s", video_path.name)
                continue

            fragments = self.detect_video(video_path)
            rows = [[video_path.name, format_seconds(s), format_seconds(e)] for (s, e) in fragments]
            if rows:
                append_csv(timestamps_path, rows)
            # 追加 processed_log
            processed_log.parent.mkdir(parents=True, exist_ok=True)
            with processed_log.open("a", encoding='utf-8') as f:
                f.write(video_path.name + "\n")
                f.flush()

            found_fragment_num += len(fragments)

        # 检测完成，创建.ok文件
        detect_ok.touch()

        logger.info("检测完成，总片段数：%d 保存至：%s", found_fragment_num, timestamps_path)

    def detect_video(self, video_path: Path) -> List[Tuple[float, float]]:
        """
        对单个视频执行逐帧检测，返回[(start_sec, end_sec), ...]
        逻辑：
        - 逐帧读取并用 YOLO 检测（限制类 id）
        - 采用跳帧策略减少检测量：若当前处于“猫出现区间”使用较大步长，否则使用较小步长
        - 记录连续出现的帧区间，转换为秒（基于帧真实时间戳）
        """
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            logger.error("无法打开视频：%s", video_path)
            return []

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            logger.error("无法获取视频帧率：%s", video_path)
            raise Exception("无法获取视频帧率")
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if total_frames <= 0:
            logger.error("无法获取视频总帧数：%s", video_path)
            raise Exception("无法获取视频总帧数")
        duration = total_frames / fps
        frame_idx = 0

        fragments = []
        cat_start_time = None  # 用真实时间代替帧号

        # 用于控制跳帧
        frame_skip_remaining = 0

        pbar = tqdm(total=total_frames if total_frames > 0 else None,
                    desc=f"检测 {video_path.name}", unit="frame", leave=False)
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    # 视频结尾，若正在记录猫段，则结束
                    if cat_start_time is not None:
                        fragments.append((cat_start_time, duration))
                    break

                # 当前帧对应的视频时间（秒）
                current_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

                # 跳帧计数
                if frame_skip_remaining > 0:
                    frame_skip_remaining -= 1
                    frame_idx += 1
                    pbar.update(1)
                    continue

                # 模型推理
                results = self.model(frame,
                                     conf=self.cfg.confidence_threshold,
                                     classes=self.cfg.cat_class_id,
                                     verbose=False,
                                     device=self.device)
                has_cat = len(results[0].boxes) > 0

                if has_cat:
                    if cat_start_time is None:
                        cat_start_time = current_time
                        if self.cfg.save_detect_frame:
                            for idx, res in enumerate(results):
                                save_path = os.path.join(
                                    self.cfg.output_dir,
                                    self.cfg.tmp_dir_name,
                                    'frags',
                                    f"{video_path.name}-{current_time:.2f}-start-{idx}.jpg"
                                )
                                res.save(str(save_path))
                    # 当检测到猫时，下一次跳过的帧数按 with_start 系数
                    skip_frames = int(self.cfg.detect_step_with_start * fps)
                else:
                    if cat_start_time is not None:
                        # 猫消失，记录结束
                        cat_end_time = current_time
                        fragments.append((cat_start_time, cat_end_time))
                        cat_start_time = None
                        if self.cfg.save_detect_frame:
                            for idx, res in enumerate(results):
                                save_path = os.path.join(
                                    self.cfg.output_dir,
                                    self.cfg.tmp_dir_name,
                                    'frags',
                                    f"{video_path.name}-{current_time:.2f}-end-{idx}.jpg"
                                )
                                res.save(str(save_path))
                    # 未检测到猫时使用 without_start 系数跳帧
                    skip_frames = int(self.cfg.detect_step_without_start * fps)

                # 至少跳0帧确保每次前进
                frame_skip_remaining = max(0, skip_frames)
                frame_idx += 1
                pbar.update(1)

        finally:
            pbar.close()
            cap.release()

        return fragments
