# main.py
import argparse
from pathlib import Path
import logging
from config import Config
from detector import Detector
from postprocess import merge_close_fragments
from cutter import Cutter
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


def parse_args():
    # todo 阈值等超参加入arg
    p = argparse.ArgumentParser(description="Cat clipper: 用YOLO检测并拼接包含猫的视频片段")
    p.add_argument("--input_dir", type=str, help="监控视频目录（默认 config.input_dir）")
    p.add_argument("--output_dir", type=str, help="输出目录（默认同输入）")
    p.add_argument("--model", type=str, help="yolo 模型路径（覆盖 config）")
    p.add_argument("--force", action="store_true", help="强制重新检测（忽略 processed_videos.txt）")
    p.add_argument("--merge_gap", type=float, help="合并片段的最大间隔秒数（覆盖 config）")
    p.add_argument("--no_clean", action="store_true", help="拼接后不删除临时片段")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = Config()
    if args.input_dir:
        cfg.input_dir = Path(args.input_dir)
    if args.output_dir:
        cfg.output_dir = Path(args.output_dir)
    if args.model:
        cfg.model_path = args.model
    if args.merge_gap is not None:
        cfg.max_merge_gap_seconds = args.merge_gap
    if args.no_clean:
        cfg.delete_temp_fragments = False

    logger.info("配置：input=%s output=%s model=%s", cfg.input_dir, cfg.output_dir, cfg.model_path)

    # 检查 ffmpeg
    ffmpeg = find_ffmpeg()
    if not ffmpeg:
        logger.error("请先安装 ffmpeg 并确保在 PATH 中")
        return

    # Step 1: 检测（将结果追加到 timestamps CSV）
    detector = Detector(cfg)
    if args.force:
        # 删除 processed_videos.txt 强制全部重新检测
        proc_log = Path(cfg.output_dir) / cfg.processed_log_name
        if proc_log.exists():
            proc_log.unlink()
            logger.info("已删除 processed_videos.txt，开始强制重新检测所有视频")

    fragments_detected = detector.detect_all(append=True)

    # Step 2: 合并靠近片段
    timestamps_in = Path(cfg.output_dir) / cfg.timestamps_name
    timestamps_merged = Path(cfg.output_dir) / cfg.merged_timestamps_name
    merged = merge_close_fragments(cfg, timestamps_in, timestamps_merged, cfg.max_merge_gap_seconds)

    # Step 3: 裁剪并拼接最终视频（按合并后顺序）
    cutter = Cutter(cfg)
    # merged is list of tuples (video, s, e)
    if not merged:
        logger.info("没有任何片段需要拼接，退出")
        return

    final_video = cutter.cut_and_concat(merged)
    logger.info("任务完成，最终视频：%s", final_video)


if __name__ == "__main__":
    main()
