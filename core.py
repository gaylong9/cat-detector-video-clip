# main.py
import argparse
from pathlib import Path
import logging
from typing import List

from tqdm import tqdm

from postprocess import postprocess
import utils
from config import Config
from detector import Detector
from clipper import Clipper
from utils import logger as utils_logger, find_ffmpeg

# 配置日志
logger = logging.getLogger("catclipper")
logger.setLevel(logging.INFO)
fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
ch = logging.StreamHandler()
ch.setFormatter(fmt)
logger.addHandler(ch)

# 将 utils 的 logger 也指向相同设置
utils_logger.setLevel(logging.INFO)
utils_logger.addHandler(ch)


def main(cfg:Config, progress_callback=None, cancel_flag=None):
    logger.info(f"配置：{cfg}")
    
    # 检查是否需要取消
    if cancel_flag and cancel_flag():
        logger.info("任务已取消")
        return

    # 检查 ffmpeg
    ffmpeg = find_ffmpeg()
    if not ffmpeg:
        logger.error("请先安装 ffmpeg 并确保在 PATH 中")
        return

    # 分组进行
    video_paths = utils.iter_video_files(cfg)
    video_groups = [video_paths[i:i + cfg.group_size] for i in range(0, len(video_paths), cfg.group_size)]
    with tqdm(total=len(video_groups), desc="处理视频组") as group_bar:
        for group_idx, group_paths in enumerate(video_groups):
            # Step 1: 检测（将结果追加到 timestamps CSV）
            detector = Detector(cfg, progress_callback, cancel_flag)
            if cfg.force:
                # 清空临时目录 强制全部重新检测
                temp_dir = Path(cfg.output_dir, cfg.tmp_dir_name)
                logger.info(f"force：清理临时文件 {temp_dir}，强制重新检测 ...")
                utils.remove_tree(temp_dir)
                logger.info("force：删除完成，开始强制重新检测所有视频")

            # 检查是否需要取消
            if cancel_flag and cancel_flag():
                logger.info("任务已取消")
                return

            (Path(cfg.output_dir) / cfg.tmp_dir_name).mkdir(exist_ok=True)
            tmp_path = Path(cfg.output_dir) / cfg.tmp_dir_name / ("group" + str(group_idx))
            tmp_path.mkdir(exist_ok=True)
            (tmp_path / 'frags').mkdir(exist_ok=True)

            detector.detect_all(video_paths=group_paths, tmp_path=tmp_path)

            # 检查是否需要取消
            if cancel_flag and cancel_flag():
                logger.info("任务已取消")
                return

            # Step 2: 后处理，帧前后扩展一段时间过渡，并合并靠近片段（此步从 timestamps.csv 读取解析）
            if progress_callback:
                progress_callback({
                    "current_video": "后处理",
                    "processed_videos": 0,
                    "total_videos": 0,
                    "current_video_progress": 0,
                    "current_video_total_frames": 0
                })
            timestamp_csv_path = tmp_path / cfg.timestamp_csv_name
            fragment_csv_path = tmp_path / cfg.fragment_csv_name
            postprocess(timestamp_csv_path, fragment_csv_path, cfg, cancel_flag)

            # 检查是否需要取消
            if cancel_flag and cancel_flag():
                logger.info("任务已取消")
                return

            # Step 3: 裁剪并拼接最终视频（按合并后顺序）
            if progress_callback:
                progress_callback({
                    "current_video": "视频裁剪拼接",
                    "processed_videos": 0,
                    "total_videos": 0,
                    "current_video_progress": 0,
                    "current_video_total_frames": 0
                })
            cutter = Clipper(tmp_dir=tmp_path, cfg=cfg)
            final_video = cutter.cut_and_concat(fragment_csv_path, cancel_flag)

            # 检查是否需要取消
            if cancel_flag and cancel_flag():
                logger.info("任务已取消")
                return

            # Step 4: 清理临时文件
            if cfg.delete_temp_files:
                temp_dir = Path(cfg.output_dir, cfg.tmp_dir_name)
                logger.info(f"清理临时文件 {temp_dir} ...")
                utils.remove_tree(temp_dir)

            logger.info("任务完成，最终视频：%s", final_video)

        group_bar.update(1)