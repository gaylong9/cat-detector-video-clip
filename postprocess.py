# postprocess.py
from pathlib import Path
from typing import List, Tuple
from config import Config
from utils import read_csv_rows, write_csv, logger
import csv


def merge_close_fragments(cfg: Config, input_csv: Path, output_csv: Path, max_gap: float = None) -> List[Tuple[str, float, float]]:
    """
    读取 input_csv (video,start,end)，对同一视频的相邻片段按时间排序并合并间隔 <= max_gap 的片段，
    最终写入 output_csv 并返回合并后的片段列表。
    """
    if max_gap is None:
        max_gap = cfg.max_merge_gap_seconds

    rows = read_csv_rows(input_csv)
    video_groups = {}
    for r in rows:
        if len(r) < 3:
            continue
        video, s_str, e_str = r[0], r[1], r[2]
        try:
            s = float(s_str)
            e = float(e_str)
        except ValueError:
            logger.warning("跳过格式错误行：%s", r)
            continue
        if s >= e:
            logger.warning("跳过无效区间（start>=end）：%s", r)
            continue
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
                # 合并
                merged[-1][1] = max(last_e, e)
            else:
                merged.append([s, e])
        for s, e in merged:
            merged_all.append((video, s, e))

    # 写输出 CSV
    write_csv(output_csv, [[v, f"{s:.2f}", f"{e:.2f}"] for v, s, e in merged_all])
    logger.info("合并完成：原片段数=%d 合并后片段=%d 保存至=%s", len(rows), len(merged_all), output_csv)
    return merged_all
