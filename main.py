from ultralytics import YOLO
import cv2
import os

# todo: 间隔检测，提高速度 如出现时长大于1s才算的话，检测间隔也可扩大至1s；
# todo tqdm
# todo 并行

# 1. 配置参数
model = YOLO('yolo11n.pt')  # 加载YOLO模型 11n/s/m/l/x
input_dir: str = "D:/Desktop/dongdong"  # 例："C:/MonitorVideos"
confidence_threshold: float = 0.5  # 检测置信度（>0.5判定为猫，避免误检）
min_continue_seconds: int = 1  # 最少连续几秒判定为“有效猫片段”

# 2. 批量处理每个视频
output_timefile = "cat_timestamps.txt"  # 保存猫出现时间戳的文件
with open(output_timefile, 'w') as f:
    for video_name in os.listdir(input_dir):
        if not video_name.endswith(('.mp4', '.avi', '.mov')):  # 过滤非视频文件
            continue
        video_path = os.path.join(input_dir, video_name)
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)  # 获取视频帧率（计算时间戳用）
        min_continue_frames = int(min_continue_seconds * fps)
        frame_count = 0
        cat_start = None  # 猫出现的起始帧
        cat_end = None    # 猫消失的结束帧

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break  # 视频读取完毕

            # 3. 用YOLO检测当前帧是否有猫（class=15是YOLO预定义的“猫”类别）
            results = model(frame, conf=confidence_threshold, classes=[15], verbose=False)
            has_cat = len(results[0].boxes) > 0  # 判定当前帧是否有猫

            # 4. 记录“猫出现的连续帧区间”
            if has_cat and cat_start is None:
                cat_start = frame_count  # 开始出现猫，记录起始帧
            elif not has_cat and cat_start is not None:
                cat_end = frame_count - 1  # 猫消失，记录结束帧
                # 判定是否满足“最少连续帧”，满足则转换为时间戳（秒）
                if (cat_end - cat_start) >= min_continue_frames:
                    start_time = cat_start / fps
                    end_time = cat_end / fps
                    # 写入文件：视频名,起始时间(秒),结束时间(秒)
                    f.write(f"{video_name},{start_time:.2f},{end_time:.2f}\n")
                cat_start = None  # 重置，准备下一个猫片段

            frame_count += 1
        cap.release()
print(f"时间段检测完成，结果保存在 {output_timefile}")