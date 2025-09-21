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

在项目目录执行检测脚本：
```bash
python main.py --input_dir "D:/Desktop/dongdong" --output_dir "D:/Desktop/dongdong"
```

### 参数
- `--input_dir` 指定存放原始监控视频文件的文件夹路径（支持 MP4、MOV、MKV 等格式），例如 `--input_dir "C:/Monitor/Videos"`
- `--output_dir`	输出目录，用于保存最终拼接视频、临时片段、时间戳文件等结果的文件夹，例如 `--output_dir "C:/CatVideo/Result"`，可以与输入目录一致
- `--model`	YOLO 模型选择，默认为 yolo11s.pt；支持更大尺寸模型以提升精度：
  - yolo11n.pt（nano，最小尺寸，速度快、占用资源少）
  - yolo11s.pt（small，平衡速度与精度）
  - yolo11m.pt（medium，中精度）
  - yolo11l.pt（large，高精度）
  - yolo11x.pt（xlarge，最高精度，速度较慢）
  示例：`--model "yolo11m.pt"`
- `--confidence_threshold`	检测置信度阈值，取值范围 0~1，仅当模型检测到目标的置信度高于该值时，才判定为有效目标（避免误检）。默认值 0.6。如 `--confidence_threshold 0.5`（降低阈值以减少漏检，可能增加误检）
- `--save_detect_frame`	开关参数，保存片段开始与结尾帧。默认关闭；开启后，会自动保存每个有效猫片段的 “第一帧”（带检测框标注）和末尾帧到输出目录，用于验证检测效果。若使用，无需传值，直接加参数即可：`--save_detect_frame`
- `--force`	开关参数，强制重新检测。工具默认会记录已处理的视频（断点续跑）；开启后，会忽略历史进度，强制重新检测所有视频。使用时直接加参数：`--force`
- `--no_clean`	开关参数，保留临时片段等中间数据。默认拼接完成后自动删除临时片段等中间数据（节省空间）；开启后，会保留所有裁剪后的独立片段（便于核验）。使用时直接加参数：`--no_clean`

## 示例

```bash
# 假设桌面目录下有一个 dongdong 文件夹，里面存放监控视频。运行后，会在 dongdong 文件夹下生成最终视频（例如 output.mp4）。
python main.py --input_dir "D:/Desktop/dongdong" --output_dir "D:/Desktop/dongdong"

# 基础用法：默认参数，指定输入/输出目录
python main.py --input_dir "C:/Monitor/202409" --output_dir "C:/CatResult"

# 高精度检测：用中尺寸模型，降低阈值减少漏检
python main.py --input_dir "C:/Monitor/202409" --output_dir "C:/CatResult" --model "yolo11m.pt" --confidence_threshold 0.5

# 保留临时片段+验证检测效果：开启保存开始帧+不清理临时文件
python main.py --input_dir "C:/Monitor/202409" --output_dir "C:/CatResult" --save_detect_frame --no_clean

# 强制重新检测：忽略历史进度
python main.py --input_dir "C:/Monitor/202409" --output_dir "C:/CatResult" --force
```
