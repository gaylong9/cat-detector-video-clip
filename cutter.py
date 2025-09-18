# cutter.py
from pathlib import Path
from typing import List, Tuple
from config import Config
from utils import run_cmd, find_ffmpeg, safe_make_tmp_dir, format_seconds, logger
from tqdm import tqdm
import os


class Cutter:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.ffmpeg = find_ffmpeg()
        if not self.ffmpeg:
            raise RuntimeError("ffmpeg is required but not found in PATH")

    def cut_and_concat(self, fragments: List[Tuple[str, float, float]]) -> Path:
        """
        fragments: List of (video_name, start_sec, end_sec), already in desired order.
        返回 final_video_path
        """
        if not fragments:
            raise ValueError("no fragments to cut/concat")

        output_dir = Path(self.cfg.output_dir)
        tmp_dir = safe_make_tmp_dir(output_dir, self.cfg.tmp_dir_name)
        temp_frag_dir = tmp_dir / "frags"
        temp_frag_dir.mkdir(exist_ok=True)

        concat_list_path = tmp_dir / "concat_list.txt"
        final_video_path = output_dir / self.cfg.final_video_name

        concat_lines = []
        # 裁剪每个片段为独立 mp4
        for idx, (video_name, s, e) in enumerate(tqdm(fragments, desc="裁剪片段", unit="frag")):
            input_path = Path(self.cfg.input_dir) / video_name
            if not input_path.exists():
                logger.warning("源视频不存在，跳过：%s", input_path)
                continue
            frag_name = f"frag_{idx+1:04d}.mp4"
            frag_path = temp_frag_dir / frag_name

            # 先尝试使用 copy（快速），若失败则转码
            cmd_copy = [
                self.ffmpeg, "-y", "-i", str(input_path),
                "-ss", format_seconds(s), "-to", format_seconds(e),
                "-c:v", "copy", "-c:a", "copy",
                "-hide_banner", "-loglevel", "error", str(frag_path)
            ]
            ret, out, err = run_cmd(cmd_copy)
            if ret != 0 or not frag_path.exists():
                # fallback: transcode to h264/aac
                logger.info("使用转码裁剪：%s [%s - %s]", video_name, s, e)
                cmd_trans = [
                    self.ffmpeg, "-y", "-i", str(input_path),
                    "-ss", format_seconds(s), "-to", format_seconds(e),
                    "-c:v", "libx264", "-c:a", "aac", "-crf", "23", "-preset", "medium",
                    "-hide_banner", "-loglevel", "error", str(frag_path)
                ]
                ret2, out2, err2 = run_cmd(cmd_trans)
                if ret2 != 0 or not frag_path.exists():
                    logger.error("裁剪失败：%s (ret=%s err=%s)", frag_path, ret2, err2)
                    continue

            concat_lines.append(f"file '{frag_path.resolve()}'")

        if not concat_lines:
            raise RuntimeError("没有有效的临时片段可供拼接")

        concat_list_path.write_text("\n".join(concat_lines), encoding='utf-8')

        # 尝试快速拼接（copy）
        cmd_concat = [
            self.ffmpeg, "-y", "-f", "concat", "-safe", "0",
            "-i", str(concat_list_path), "-c", "copy", str(final_video_path)
        ]
        retc, outc, errc = run_cmd(cmd_concat)
        if retc != 0 or not final_video_path.exists():
            logger.warning("直接 copy 拼接失败，尝试转码拼接（统一编码）")
            cmd_transcat = [
                self.ffmpeg, "-y", "-f", "concat", "-safe", "0",
                "-i", str(concat_list_path),
                "-c:v", "libx264", "-c:a", "aac", "-crf", "23", "-preset", "medium",
                str(final_video_path)
            ]
            ret2, out2, err2 = run_cmd(cmd_transcat)
            if ret2 != 0 or not final_video_path.exists():
                raise RuntimeError(f"拼接失败：{err2}")

        # 可选择清理临时文件
        if self.cfg.delete_temp_fragments:
            for child in temp_frag_dir.iterdir():
                try:
                    child.unlink()
                except Exception:
                    logger.exception("删除碎片失败：%s", child)
            try:
                temp_frag_dir.rmdir()
                concat_list_path.unlink(missing_ok=True)
                # remove tmp_dir if empty
                if not any(tmp_dir.iterdir()):
                    tmp_dir.rmdir()
            except Exception:
                logger.exception("清理临时目录失败：%s", tmp_dir)

        logger.info("最终拼接输出：%s", final_video_path)
        return final_video_path
