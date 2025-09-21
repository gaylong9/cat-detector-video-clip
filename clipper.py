# clipper.py
from pathlib import Path
from typing import List, Tuple
from config import Config
from utils import run_cmd, find_ffmpeg, safe_make_tmp_dir, format_seconds, logger, read_csv_rows, read_timestamp_csv
from tqdm import tqdm
import os


class Clipper:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.ffmpeg = find_ffmpeg()
        if not self.ffmpeg:
            raise RuntimeError("ffmpeg is required but not found in PATH")
        self.output_dir = Path(self.cfg.output_dir)
        self.tmp_dir = self.output_dir / self.cfg.tmp_dir_name
        self.temp_frag_dir = self.tmp_dir / "frags"
        self.temp_frag_dir.mkdir(exist_ok=True)
        self.concat_list_path = self.tmp_dir / "concat_list.txt"


    def cut(self, fragments):
        concat_lines = []
        # 裁剪每个片段为独立 mp4
        for idx, (video_name, s, e) in enumerate(tqdm(fragments, desc="裁剪片段", unit="frag")):
            input_path = Path(self.cfg.input_dir) / video_name
            if not input_path.exists():
                logger.warning("源视频不存在，跳过：%s", input_path)
                continue
            frag_name = f"frag_{idx + 1:04d}.mp4"
            frag_path = self.temp_frag_dir / frag_name
            if frag_path.exists():
                concat_lines.append(f"file '{frag_path.resolve()}'")
                continue
            tmp_frag_path = frag_path.with_suffix('.tmp.mp4')

            # 转码方式虽然支持较短片段，但速度极慢体积极大，还是选用-c copy，前期调大片段时长
            # -ss在-i前可自动选择临近关键帧
            # logger.info(f"使用转码裁剪：{video_name} [{format_seconds(s)} - {format_seconds(e)}]")
            cmd_trans = [
                self.ffmpeg,
                "-y",
                "-ss", format_seconds(s),
                "-to", format_seconds(e),
                "-i", str(input_path),
                # "-c:v", "libx264",
                # "-c:a", "aac",
                "-c", "copy",
                "-crf", "23",
                "-preset", "medium",
                "-hide_banner",
                "-loglevel", "error",
                # "-avoid_negative_ts", "make_zero",
                str(tmp_frag_path)
            ]
            ret, out, err = run_cmd(cmd_trans)
            if ret != 0 or not tmp_frag_path.exists():
                logger.error("裁剪失败：%s (ret=%s err=%s)", tmp_frag_path, ret, err)
                continue

            tmp_frag_path.rename(frag_path)
            concat_lines.append(f"file '{frag_path.resolve()}'")

        if not concat_lines:
            raise RuntimeError("没有有效的临时片段可供拼接")

        self.concat_list_path.write_text("\n".join(concat_lines), encoding='utf-8')


    def concat(self, final_video_path):
        # 转码拼接
        cmd_transcat = [
            self.ffmpeg,
            "-y",
            "-f", "concat",
            "-safe", "0",
            " -fflags", "+genpts",
            "-i", str(self.concat_list_path),
            # "-c:v", "libx264",
            # "-c:a", "aac",
            "-c", "copy",
            "-crf", "23",
            "-preset", "medium",
            "-movflags", "+faststart",
            str(final_video_path)
        ]
        ret2, out2, err2 = run_cmd(cmd_transcat)
        if ret2 != 0 or not final_video_path.exists():
            raise RuntimeError(f"拼接失败：{err2}")


    def cut_and_concat(self, input_csv: Path) -> Path:
        """
        fragments: List of (video_name, start_sec, end_sec), already in desired order.
        返回 final_video_path
        """
        # 从csv读取片段时间数据
        fragments = read_timestamp_csv(input_csv)
        self.cut(fragments)
        final_video_path = self.output_dir / self.cfg.final_video_name
        self.concat(final_video_path)
        logger.info("最终拼接输出：%s", final_video_path)
        return final_video_path
