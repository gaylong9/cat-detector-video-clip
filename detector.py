# detector.py
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
        self.model = YOLO(cfg.model_path)
        # 如果需要，可设置 model.conf 或其他超参数
        # self.model.conf = cfg.confidence_threshold

    def _iter_video_files(self) -> List[Path]:
        p = Path(self.cfg.input_dir)
        if not p.exists():
            raise FileNotFoundError(f"input_dir not found: {p}")
        files = sorted([f for f in p.iterdir() if f.suffix.lower() in self.cfg.video_extensions])
        return files

    def detect_all(self, append: bool = True) -> List[Tuple[str, float, float]]:
        """
        遍历 input_dir 中的视频，执行逐帧检测并将猫段写入 timestamps csv。
        返回检测到的片段列表（用于后续处理或测试）。
        """
        processed_log = Path(self.cfg.output_dir) / self.cfg.processed_log_name
        timestamps_path = Path(self.cfg.output_dir) / self.cfg.timestamps_name

        # 已处理集合
        processed = set()
        if processed_log.exists():
            processed = set([l.strip() for l in processed_log.read_text(encoding='utf-8').splitlines() if l.strip()])

        found_fragments = []

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

            found_fragments.extend([(video_path.name, s, e) for s, e in fragments])

        logger.info("检测完成，总片段数：%d 保存至：%s", len(found_fragments), timestamps_path)
        return found_fragments

    def detect_video(self, video_path: Path) -> List[Tuple[float, float]]:
        """
        对单个视频执行逐帧检测，返回[(start_sec, end_sec), ...]
        逻辑：
        - 逐帧读取并用 YOLO 检测（限制类 id）
        - 采用跳帧策略减少检测量：若当前处于“猫出现区间”使用较大步长，否则使用较小步长
        - 记录连续出现的帧区间，转换为秒
        """
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            logger.error("无法打开视频：%s", video_path)
            return []

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        frame_idx = 0

        fragments = []
        cat_start_frame = None

        # 用于控制跳帧
        frame_skip_remaining = 0

        pbar = tqdm(total=total_frames if total_frames > 0 else None, desc=f"检测 {video_path.name}", unit="frame", leave=False)
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    # 视频结尾，若正在记录猫段，则结束
                    if cat_start_frame is not None:
                        cat_end_frame = frame_idx - 1
                        fragments.append((cat_start_frame / fps, cat_end_frame / fps))
                        cat_start_frame = None
                    break

                # 跳帧计数
                if frame_skip_remaining > 0:
                    frame_skip_remaining -= 1
                    frame_idx += 1
                    pbar.update(1)
                    continue

                # 模型推理（注意 ultralytics.YOLO: 可直接传 ndarray）
                results = self.model(frame, conf=self.cfg.confidence_threshold, classes=self.cfg.cat_class_id, verbose=False)
                has_cat = len(results[0].boxes) > 0

                if has_cat:
                    if cat_start_frame is None:
                        cat_start_frame = frame_idx
                    # 当检测到猫时，下一次跳过的帧数按 with_start 系数
                    skip_frames = int(self.cfg.frame_step_coef_with_start * self.cfg.min_continue_seconds * fps)
                else:
                    if cat_start_frame is not None:
                        # 猫消失，记录结束（frame_idx-1）
                        cat_end_frame = frame_idx - 1
                        # 仅记录满足最小持续时间的片段
                        duration = (cat_end_frame - cat_start_frame + 1) / fps
                        if duration >= self.cfg.min_continue_seconds:
                            fragments.append((cat_start_frame / fps, cat_end_frame / fps))
                        cat_start_frame = None
                    # 未检测到猫时使用 without_start 系数跳帧
                    skip_frames = int(self.cfg.frame_step_coef_without_start * self.cfg.min_continue_seconds * fps)

                # 至少跳0帧确保每次前进
                frame_skip_remaining = max(0, skip_frames)
                frame_idx += 1
                pbar.update(1)

        finally:
            pbar.close()
            cap.release()

        # 合并可能十分靠近的短片段（这里不做，交由 postprocess），直接返回
        return fragments
