import os

from ultralytics import YOLO
import cv2


def detect_single_image(image_path, model_path='yolo11n.pt', save_result=True):
    # 加载模型
    model = YOLO('yolo11n.pt')

    # 检测图片（关闭打印，可指定检测类别，如猫是15）
    results = model(image_path, classes=[15])

    # 处理结果
    for result in results:
        # 绘制检测框
        annotated_img = result.plot()  # 自动在图上画框和标签

        # 显示图片（按q键关闭）
        cv2.imshow('Detection Result', annotated_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # 保存结果图片
        if save_result:
            save_path = f"detected_{os.path.basename(image_path)}"
            cv2.imwrite(save_path, annotated_img)
            print(f"结果已保存至 {save_path}")

        # 打印检测到的目标信息
        if len(result.boxes) > 0:
            print(f"检测到 {len(result.boxes)} 个目标：")
            for box in result.boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                print(f"- 类别：{model.names[cls]}，置信度：{conf:.2f}")
        else:
            print("未检测到目标")


def video_detect(video_path):
    model = YOLO('yolo11n.pt')
    results = model.predict(source=video_path, conf=0.5, batch=4, vid_stride=20, classes=[15], verbose=False, stream=True)
    for result in results:
        if len(result.boxes) > 0:
            print(f"检测到 {len(result.boxes)} 个目标：")
        # annotated_img = result.plot()
        # cv2.imshow('Detection Result', annotated_img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()


def video_detect_own_batch(video_path):
    batch = 4
    data = []
    model = YOLO('yolo11n.pt')
    cap = cv2.VideoCapture(str(video_path))
    while True:
        ret, frame = cap.read()
        if len(data) < batch:
            data.append(frame)
        else:
            results = model.predict(source=data, conf=0.5, classes=[15], verbose=True)
            # for result in results:
            #     if len(result.boxes) > 0:
            #         print(f"检测到 {len(result.boxes)} 个目标：")
                # annotated_img = result.plot()
                # cv2.imshow('Detection Result', annotated_img)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()
            data = []

if __name__ == "__main__":
    # 替换为你的测试图片路径（如截图的猫的图片）
    test_image_path = "test_cat.png"
    # 可替换为其他模型，如'yolov8s.pt'、'yolov11s.pt'
    # detect_single_image(test_image_path, model_path='yolov11n.pt')
    # video_detect('D:\\Desktop\\dongdong\\video_0003_0_10_20250617154920_20250617202423.mp4')
    video_detect_own_batch('D:\\Desktop\\dongdong\\video_0003_0_10_20250617154920_20250617202423.mp4')