# batch.py
"""Docker 批处理入口。

把原始视频目录（src）中的视频按批次处理：
  1. 取前 BATCH_SIZE 个视频，移动到 work 子目录
  2. 以 work 为 input_dir、des 为 output_dir 跑一遍完整流水线
  3. 删除已处理的这 BATCH_SIZE 个原始视频
  4. 重新取下一批，直到剩余视频不足 BATCH_SIZE（最后一轮按剩余数量处理）

目录约定（与 docker-compose 挂载对应）：
  /data/src   原始视频目录（挂载点）；内部会创建 .work 子目录暂存每批视频
  /data/des   拼接结果视频目录（挂载点）

用法：
  python batch.py --src /data/src --des /data/des [--batch-size 30] [--model yolo11n.pt] [--device intel:gpu]
"""

import argparse
import shutil
import sys
from pathlib import Path

from config import Config
from main import run_pipeline
from utils import logger as utils_logger
from main import logger


VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}


def list_videos(src: Path) -> list:
    """按文件名排序，返回 src 下的视频文件列表。"""
    if not src.exists():
        raise FileNotFoundError(f"原始视频目录不存在：{src}")
    files = [f for f in src.iterdir() if f.is_file() and f.suffix.lower() in VIDEO_EXTENSIONS]
    return sorted(files, key=lambda p: p.name)


def move_files(files: list, dest_dir: Path) -> None:
    """把文件移动到 dest_dir（同盘移动为 rename，跨盘为 copy+delete）。"""
    dest_dir.mkdir(parents=True, exist_ok=True)
    for f in files:
        target = dest_dir / f.name
        try:
            shutil.move(str(f), str(target))
        except OSError:
            # 跨文件系统 fallback
            shutil.copy2(str(f), str(target))
            f.unlink()


def process_one_batch(src: Path, des: Path, work: Path, batch_files: list, cfg: Config, batch_idx: int) -> None:
    """处理一批视频：move → pipeline → 清理。"""
    logger.info("=" * 60)
    logger.info("第 %d 批：共 %d 个视频", batch_idx, len(batch_files))
    for f in batch_files:
        logger.info("  - %s", f.name)

    # 1. 清空 work，把本批视频移入（work 位于 src 内部，同文件系统为纯 rename）
    if work.exists():
        shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True, exist_ok=True)
    move_files(batch_files, work)

    # 2. 设置本批的 input/output，跑流水线（force=True 确保每批干净）
    cfg.input_dir = work
    cfg.output_dir = des
    final_video = run_pipeline(cfg, force=True)

    # 3. 清空 work，删除本批已处理的原始视频
    shutil.rmtree(work, ignore_errors=True)
    logger.info("第 %d 批处理完成，结果：%s", batch_idx, final_video)


def main() -> None:
    parser = argparse.ArgumentParser(description="分批处理视频并拼接")
    parser.add_argument("--src", type=str, required=True, help="原始视频目录（如 /data/src）")
    parser.add_argument("--des", type=str, required=True, help="结果视频目录（如 /data/des）")
    parser.add_argument("--work", type=str, default=None, help="工作目录，默认 src 下的 .work 子目录")
    parser.add_argument("--batch-size", type=int, default=30, help="每批处理的视频数量，默认 30")
    parser.add_argument("--max-rounds", type=int, default=0, help="最多执行几轮；0 表示不限，直到剩余视频不足一个批次")
    parser.add_argument("--model", type=str, default="yolo26s.pt", help="YOLO 模型路径")
    parser.add_argument("--device", type=str, default="intel:gpu", help="推理设备：intel:gpu/intel:cpu/cpu/cuda")
    parser.add_argument("--confidence-threshold", type=float, default=0.5, help="检测阈值")
    parser.add_argument("--step", type=float, default=0.25, help="检测间隔（秒）")
    parser.add_argument("--batch-size-frames", type=int, default=1, help="检测阶段帧批次大小")
    parser.add_argument("--save-detect-frame", action="store_true", help="保存检测帧")
    args = parser.parse_args()

    src = Path(args.src)
    des = Path(args.des)
    work = Path(args.work) if args.work else src / ".work"

    des.mkdir(parents=True, exist_ok=True)

    # 构建配置
    cfg = Config()
    cfg.model_path = args.model
    cfg.device = args.device
    cfg.confidence_threshold = args.confidence_threshold
    cfg.detect_step = args.step
    cfg.batch_size = args.batch_size_frames
    cfg.save_detect_frame = args.save_detect_frame

    logger.info("批处理启动：src=%s des=%s batch_size=%d max_rounds=%d model=%s device=%s",
                src, des, args.batch_size, args.max_rounds, cfg.model_path, cfg.device)

    batch_idx = 0
    while True:
        # 达到最大轮数则提前停止
        if args.max_rounds > 0 and batch_idx >= args.max_rounds:
            logger.info("已达最大轮数 %d，停止。", args.max_rounds)
            break

        videos = list_videos(src)
        if not videos:
            logger.info("原始视频目录已无视频，处理结束。")
            break

        batch_files = videos[: args.batch_size]
        process_one_batch(src, des, work, batch_files, cfg, batch_idx + 1)
        batch_idx += 1

        # 若剩余视频不足一个 batch，下一轮会取到剩余全部，随后自然结束
        if len(videos) <= args.batch_size:
            logger.info("剩余视频不足一个批次，已处理完毕。")
            break

    logger.info("全部批次处理完成，共 %d 批。", batch_idx)


if __name__ == "__main__":
    main()