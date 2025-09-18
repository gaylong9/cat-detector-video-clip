# cat-detector-video-clip

使用 [YOLO](https://github.com/ultralytics/ultralytics) 与 [FFmpeg](https://ffmpeg.org/) 从（监控）视频中检测出包含猫猫的片段，并整合为一个完整视频。  

---

## 环境准备

### 1. 创建 Conda 虚拟环境
本项目使用 Conda 管理依赖，请确保已安装 [Miniconda](https://docs.conda.io/en/latest/miniconda.html) 或 [Anaconda](https://www.anaconda.com/)。

在项目根目录执行：

```bash
conda env create -f environment.yml
```

激活环境：
```bash
conda activate catclip
```

### 2. 安装 FFmpeg

请根据操作系统安装 FFmpeg。

## 运行方式

执行检测脚本：
```bash
python main.py --input_dir "D:/Desktop/dongdong" --output_dir "D:/Desktop/dongdong"
```

### 可选参数

`--model`
模型文件，默认为 `yolo11n.pt`。表示nano最小尺寸。此外还有yolo11s.pt、yolo11m.pt、yolo11l.pt、yolo11x.pt

`--no_clean`
保留临时片段文件（默认会自动清理），适合调试时使用。

`--merge_gap`
检测出的片段，间隔过小会被合并，设置合并片段的最大间隔秒数

## 示例

假设桌面目录下有一个 dongdong 文件夹，里面存放监控视频：

```bash
python main.py --input_dir "D:/Desktop/dongdong" --output_dir "D:/Desktop/dongdong"
```

运行后，会在 dongdong 文件夹下生成最终视频（例如 output.mp4）。

