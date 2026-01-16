# clipper.py
from pathlib import Path
from typing import List, Tuple
from config import Config
from utils import run_cmd, find_ffmpeg, safe_make_tmp_dir, format_seconds, logger, read_csv_rows, read_fragment_csv
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
                # "-crf", "23",
                # "-preset", "medium",
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


    def concat(self, final_video_path: Path):
        # 转码拼接
        logger.info(f"拼接中间文件...")
        tmp_concat_path = final_video_path.with_suffix('.tmp.mp4')
        cmd_transcat = [
            self.ffmpeg,
            "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", str(self.concat_list_path),
            # "-c:v", "libx264",
            # "-c:a", "aac",
            "-c", "copy",
            # "-crf", "23",
            # "-preset", "medium",
            # "-fflags", "+genpts",
            # "-movflags", "+faststart",
            str(tmp_concat_path)
        ]
        ret1, out1, err1 = run_cmd(cmd_transcat)
        if ret1 != 0 or not tmp_concat_path.exists():
            raise RuntimeError(f"拼接失败：{err1}")

        # 2) remux +genpts -> final
        logger.info(f"生成最终文件...")
        cmd2 = [
            self.ffmpeg,
            "-y",
            "-i", str(tmp_concat_path),
            "-c", "copy",
            "-fflags", "+genpts",
            "-movflags", "+faststart",
            str(final_video_path)
        ]
        ret2, out2, err2 = run_cmd(cmd2)
        if ret2 != 0 or not final_video_path.exists():
            raise RuntimeError(f"拼接后修复失败: {err2}")

        try:
            tmp_concat_path.unlink()
        except Exception as e:
            logger.warn(f"清理临时拼接文件失败 {e}")
            pass

    def generate_semantic_filename(self, fragments: List[Tuple[str, float, float]]) -> str:
        """
        根据片段信息生成语义化的文件名
        """
        if not fragments:
            logger.info("无片段信息，使用默认文件名")
            return self.cfg.final_video_name

        first_video_name = fragments[0][0]
        last_video_name = fragments[-1][0]

        # 尝试匹配包含起止时间的文件名模式 (两个yyyymmddhhmmss)
        # 查找文件名中是否有两个14位数字的时间戳
        time_pattern = r'\d{14}'
        first_times = re.findall(time_pattern, first_video_name)
        last_times = re.findall(time_pattern, last_video_name)

        if len(first_times) >= 2 and len(last_times) >= 2:
            # 情况1: 文件名包含至少两个时间戳，使用第一个文件的第一个时间戳和最后一个文件的第二个时间戳
            start_time = first_times[0]
            end_time = last_times[1] if len(last_times) > 1 else last_times[0]
            semantic_name = f"{start_time}_{end_time}.mp4"
            logger.info(f"检测到文件名包含多个时间戳，生成语义文件名: {semantic_name}")
            return semantic_name
        elif len(first_times) >= 1 and len(last_times) >= 1:
            # 情况2: 文件名包含单个时间戳，使用第一个和最后一个文件的时间
            start_time = first_times[0]
            end_time = last_times[0]
            semantic_name = f"{start_time}_{end_time}.mp4"
            logger.info(f"检测到文件名包含单个时间戳，生成语义文件名: {semantic_name}")
            return semantic_name

        # 情况3: 其他情况，尝试直接用首尾文件名拼接
        # 移除文件扩展名
        first_name_no_ext = Path(first_video_name).stem
        last_name_no_ext = Path(last_video_name).stem
        semantic_name = f"{first_name_no_ext}_{last_name_no_ext}.mp4"

        # 检查文件名长度
        if len(semantic_name) <= 255:  # 一般文件系统限制
            logger.info(f"使用首尾文件名拼接生成语义文件名: {semantic_name}")
            return semantic_name
        else:
            # 情况3.1: 文件名过长，使用默认命名
            logger.info(f"拼接后的文件名过长({len(semantic_name)}字符)，使用默认文件名")
            return self.generate_default_filename()

    def generate_default_filename(self) -> str:
        """
        生成默认文件名，如果文件已存在则添加递增数字后缀
        """
        base_name = Path(self.cfg.final_video_name).stem
        extension = Path(self.cfg.final_video_name).suffix
        final_video_path = self.output_dir / self.cfg.final_video_name

        # 情况3.1.1: 如果默认名文件已存在，添加递增数字后缀
        if final_video_path.exists():
            counter = 1
            while True:
                new_name = f"{base_name}_{counter}{extension}"
                new_path = self.output_dir / new_name
                if not new_path.exists():
                    logger.info(f"默认文件名已存在，生成新文件名: {new_name}")
                    return new_name
                counter += 1
        else:
            logger.info(f"使用默认文件名: {self.cfg.final_video_name}")
            return self.cfg.final_video_name

    def cut_and_concat(self, input_csv: Path, cancel_flag=None) -> Path:
        """
        fragments: List of (video_name, start_sec, end_sec), already in desired order.
        返回 final_video_path
        """
        # 从csv读取片段时间数据
        fragments = read_fragment_csv(input_csv)
        self.cut(fragments)
        # 生成语义化的文件名
        semantic_filename = self.generate_semantic_filename(fragments)
        final_video_path = self.output_dir / semantic_filename
        self.concat(final_video_path)
        logger.info("最终拼接输出：%s", final_video_path)
        return final_video_path
