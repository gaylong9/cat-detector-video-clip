# detector.py
import os

import torch
from ultralytics import YOLO
import cv2
from pathlib import Path
from typing import List, Tuple, Iterator

from ultralytics.engine.results import Results

import utils
from config import Config
from utils import append_csv, format_seconds, logger
from tqdm import tqdm
import math


class Detector:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.device = "cuda:0" if utils.support_cuda() else "cpu"
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
        timestamps_path = Path(self.cfg.output_dir) / self.cfg.tmp_dir_name/ self.cfg.timestamp_csv_name
        detect_ok = Path(self.cfg.output_dir) / self.cfg.tmp_dir_name / 'detect.ok'

        # 若检测均完成，跳过
        if detect_ok.exists() and timestamps_path.exists():
            logger.info("发现detect.ok，检测均已完成，跳过本阶段")
            return

        # 已处理集合
        processed = set()
        if processed_log.exists():
            processed = set([l.strip() for l in processed_log.read_text(encoding='utf-8').splitlines() if l.strip()])

        found_frame_num = 0

        video_files = self._iter_video_files()
        with tqdm(total=len(video_files), desc="扫描视频文件", unit="file") as pbar:
            for video_path in video_files:
                if video_path.name in processed:
                    logger.info("跳过已处理：%s", video_path.name)
                    pbar.update(1)  # 手动更新进度条
                    continue

                frame_times = self.detect_video(video_path)
                rows = [[video_path.name, format_seconds(t)] for t in frame_times]
                if rows:
                    append_csv(timestamps_path, rows, ['video_name', 'frame_time'])
                # 追加 processed_log
                processed_log.parent.mkdir(parents=True, exist_ok=True)
                with processed_log.open("a", encoding='utf-8') as f:
                    f.write(video_path.name + "\n")
                    f.flush()

                found_frame_num += len(frame_times)
                pbar.update(1)  # 手动更新进度条
        # 检测完成，创建.ok文件
        detect_ok.touch()

        logger.info("检测完成，总帧数：%d 保存至：%s", found_frame_num, timestamps_path)

    def detect_video(self, video_path: Path) -> List[float]:
        """
        对单个视频执行逐帧检测，返回[detect_sec, ...]
        逻辑：
        - 逐帧读取并用 YOLO 检测（限制类 id）
        - 采用跳帧策略减少检测量：
        - 基于帧真实时间戳，转换为秒
        :return: List[float] 有目标的时间的列表 s
        """
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            logger.error("无法打开视频：%s", video_path)
            return []

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            logger.error("无法获取视频帧率：%s", video_path)
            raise Exception("无法获取视频帧率")
        # skip_frame_num = int(self.cfg.detect_step * fps)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if total_frames <= 0:
            logger.error("无法获取视频总帧数：%s", video_path)
            raise Exception("无法获取视频总帧数")
        # duration = total_frames / fps
        frame_cnt = 1
        last_collect_data_time = -self.cfg.detect_step - 1

        detected_frame_times = []
        batch_data = []
        batch_time = []

        pbar = tqdm(total=total_frames if total_frames > 0 else None,
                    desc=f"检测 {video_path.name}", unit="frame", leave=False)
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                current_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

                if current_time - last_collect_data_time > self.cfg.detect_step:
                    last_collect_data_time = current_time
                    batch_data.append(frame)
                    batch_time.append(current_time)
                    if len(batch_data) == self.cfg.batch_size:
                        detected_frame_times.extend(
                            self.detect_batch_data(batch_data, batch_time, video_path)
                        )
                        batch_data = []
                        batch_time = []

                # 其余帧纯跳过
                frame_cnt += 1
                pbar.update(1)

            # 循环结束后，处理残余 batch
            if len(batch_data) > 0:
                detected_frame_times.extend(
                    self.detect_batch_data(batch_data, batch_time, video_path)
                )

        finally:
            pbar.close()
            cap.release()

        return detected_frame_times


    def detect_batch_data(self, data, data_times, video_path:Path) -> List[float]:
        # 模型推理
        detected_frame_times = []
        if utils.support_cuda():
            results = self.model(data,
                                 conf=self.cfg.confidence_threshold,
                                 classes=self.cfg.cat_class_id,
                                 verbose=False,
                                 half=utils.support_fp16(),
                                 device=self.device)
        else:
            results = self.model(data,
                                 conf=self.cfg.confidence_threshold,
                                 classes=self.cfg.cat_class_id,
                                 verbose=False,
                                 device=self.device)

        for res_time, res in zip(data_times, results):
            if len(res.boxes) > 0:
                detected_frame_times.append(res_time)
            if self.cfg.save_detect_frame:
                save_path = os.path.join(
                    self.cfg.output_dir,
                    self.cfg.tmp_dir_name,
                    'frags',
                    f"{video_path.name}-{res_time}.jpg"
                )
                res.save(str(save_path))

        return detected_frame_times