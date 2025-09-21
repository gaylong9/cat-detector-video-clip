# main.py
import argparse
from pathlib import Path
import logging

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


def parse_args():
    p = argparse.ArgumentParser(description="Cat clipper: 用YOLO检测并拼接包含猫的视频片段")
    p.add_argument("--input_dir", type=str, help="监控视频目录")
    p.add_argument("--output_dir", type=str, help="输出目录）")
    p.add_argument("--model", type=str, help="yolo 模型，默认 yolo11s.pt，支持n/s/m/l/x多种尺寸")
    p.add_argument('--confidence_threshold', type=float, help="检测阈值，0~1，模拟0.6")
    p.add_argument('--save_detect_frame', action="store_true", help="保存每个片段的开始帧，观察检测结果，默认关闭")
    p.add_argument("--force", action="store_true", help="检测阶段会记录进度以断点继续，可force强制重新检测")
    p.add_argument("--merge_gap", type=float, help="合并片段的最大间隔秒数，默认0.5s")
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
    if args.confidence_threshold:
        cfg.confidence_threshold = args.confidence_threshold
    if args.merge_gap:
        cfg.max_merge_gap_seconds = args.merge_gap
    if args.no_clean:
        cfg.delete_temp_files = False
    if args.save_detect_frame:
        cfg.save_detect_frame = True

    logger.info("配置：input=%s output=%s model=%s", cfg.input_dir, cfg.output_dir, cfg.model_path)

    # 检查 ffmpeg
    ffmpeg = find_ffmpeg()
    if not ffmpeg:
        logger.error("请先安装 ffmpeg 并确保在 PATH 中")
        return

    # Step 1: 检测（将结果追加到 timestamps CSV）
    detector = Detector(cfg)
    if args.force:
        # 清空临时目录 强制全部重新检测
        temp_dir = Path(cfg.output_dir, cfg.tmp_dir_name)
        logger.info(f"force：清理临时文件 {temp_dir}，强制重新检测 ...")
        utils.remove_tree(temp_dir)
        logger.info("force：删除完成，开始强制重新检测所有视频")

    (Path(cfg.output_dir) / cfg.tmp_dir_name).mkdir(exist_ok=True)
    (Path(cfg.output_dir) / cfg.tmp_dir_name / 'frags').mkdir(exist_ok=True)

    detector.detect_all()

    # Step 2: 后处理，片段前后扩展一段时间过渡，并合并靠近片段（此步从 timestamps.csv 读取解析）
    timestamps_in = Path(cfg.output_dir) / cfg.tmp_dir_name/ cfg.timestamps_name
    timestamps_merge = Path(cfg.output_dir) / cfg.tmp_dir_name/ cfg.merged_timestamps_name
    postprocess(timestamps_in, timestamps_merge, cfg)

    # Step 3: 裁剪并拼接最终视频（按合并后顺序）
    cutter = Clipper(cfg)
    # merged is list of tuples (video, s, e)
    final_video = cutter.cut_and_concat(timestamps_merge)

    # Step 4: 清理临时文件
    if cfg.delete_temp_files:
        temp_dir = Path(cfg.output_dir, cfg.tmp_dir_name)
        logger.info(f"清理临时文件 {temp_dir} ...")
        utils.remove_tree(temp_dir)

    logger.info("任务完成，最终视频：%s", final_video)


if __name__ == "__main__":
    main()
