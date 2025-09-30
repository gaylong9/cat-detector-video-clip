import argparse
from pathlib import Path

import core
from config import Config


def parse_args():
    p = argparse.ArgumentParser(description="Cat clipper: 用YOLO检测并拼接包含猫的视频片段")
    p.add_argument("--input_dir", type=str, help="监控视频目录")
    p.add_argument("--output_dir", type=str, help="输出目录）")
    p.add_argument("--model", type=str, help="yolo 模型，默认 yolo11s.pt，支持n/s/m/l/x多种尺寸")
    p.add_argument('--confidence_threshold', type=float, help="检测阈值，0~1，模拟0.6")
    p.add_argument('--save_detect_frame', action="store_true", help="保存每个片段的开始帧，观察检测结果，默认关闭")
    p.add_argument("--force", action="store_true", help="检测阶段会记录进度以断点继续，可force强制重新检测")
    p.add_argument("--no_clean", action="store_true", help="拼接后不删除临时片段")
    p.add_argument("--batch_size", type=int, help="检测阶段处理视频的批次帧大小，默认1，有GPU可尝试扩大")
    p.add_argument("--step", type=float, help="片段前后扩展时间，默认0.5秒")
    return p.parse_args()



if __name__ == '__main__':
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
    if args.no_clean:
        cfg.delete_temp_files = False
    if args.save_detect_frame:
        cfg.save_detect_frame = True
    if args.batch_size:
        cfg.batch_size = args.batch_size
    if args.step:
        cfg.detect_step = args.step
    if args.force:
        cfg.force = True

    core.main(cfg)
