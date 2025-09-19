# postprocess.py
from pathlib import Path
from typing import List, Tuple
from config import Config
from utils import read_csv_rows, write_csv, logger, read_timestamp_csv


def expand_fragments(
        fragments: List[Tuple[str, float, float]],
        start_expand_seconds: float,
        end_expand_seconds: float
) -> List[Tuple[str, float, float]]:
    """在片段前后扩展一定时长"""
    expanded = []
    for video, s, e in fragments:
        new_s = max(0.0, s - start_expand_seconds)
        new_e = e + end_expand_seconds
        expanded.append((video, new_s, new_e))
    return expanded


def merge_fragments(
    fragments: List[Tuple[str, float, float]], max_gap: float
) -> List[Tuple[str, float, float]]:
    """对同一视频的片段按时间排序并合并间隔 <= max_gap 的片段"""
    video_groups = {}
    for video, s, e in fragments:
        video_groups.setdefault(video, []).append((s, e))

    merged_all = []
    for video in sorted(video_groups.keys()):
        frags = sorted(video_groups[video], key=lambda x: x[0])
        merged = []
        for s, e in frags:
            if not merged:
                merged.append([s, e])
                continue
            last_s, last_e = merged[-1]
            gap = s - last_e
            if gap <= max_gap:
                merged[-1][1] = max(last_e, e)  # 合并
            else:
                merged.append([s, e])
        for s, e in merged:
            merged_all.append((video, s, e))
    return merged_all


def postprocess(input_csv: Path, output_csv: Path, cfg: Config) -> List[Tuple[str, float, float]]:
    """
    后处理：扩展片段前后 -> 合并相邻片段 -> 写入输出 CSV
    """

    start_expand_seconds = cfg.start_expand_seconds
    end_expand_seconds = cfg.end_expand_seconds
    max_gap = cfg.max_merge_gap_seconds

    # 若存在ok文件，跳过后处理
    postprocess_ok = output_csv.parent / "postprocess.ok"
    if postprocess_ok.exists():
        logger.info("postprocess：发现postprocess.ok，跳过后处理")
        return []

    fragments = read_timestamp_csv(input_csv)

    # Step1: 扩展
    fragments = expand_fragments(fragments, start_expand_seconds, end_expand_seconds)

    # Step2: 合并
    merged_all = merge_fragments(fragments, max_gap)

    # 写输出
    write_csv(output_csv, [[v, f"{s:.2f}", f"{e:.2f}"] for v, s, e in merged_all])
    logger.info(
        "后处理完成：原片段=%d  扩展+合并后片段=%d  保存至=%s",
        len(fragments),
        len(merged_all),
        output_csv,
    )

    # postprocess.ok
    postprocess_ok.touch()

    return merged_all
