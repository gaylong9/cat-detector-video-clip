# verify_device.py
"""验证当前机器可用于 OpenVINO 推理，并做一次端到端测试：

1. 检测 OpenVINO 可用设备（尤其 Intel iGPU）
2. 将 yolo11n.pt 导出为 OpenVINO IR（首次，复用已有的 _openvino_model 目录则跳过）
3. 用 IR 模型对 test_cat.png 做一次猫识别

用法：
    python verify_device.py                    # 自动选择 intel:gpu > intel:cpu
    python verify_device.py --image cat.jpg    # 指定测试图片
"""

import argparse
import sys
from pathlib import Path

# 模型相关常量（与 detector.py 保持一致）
MODEL_PATH = "yolo11n.pt"
DEFAULT_TEST_IMAGE = "test_cat.png"
CAT_CLASS_ID = [15]          # COCO 中 cat 的类别 id
CONFIDENCE = 0.25            # 测试用置信度阈值


def check_openvino():
    """返回 core, devices；未安装则退出。"""
    try:
        import openvino as ov
    except ImportError:
        print("[X] 未安装 openvino，请先执行: pip install openvino")
        sys.exit(1)
    print(f"[OK] openvino 版本: {ov.__version__}")

    core = ov.Core()
    devices = list(core.available_devices)
    print(f"[OK] OpenVINO 可见设备: {devices}")
    return core, devices


def pick_device(devices):
    """返回 (ultralytics 设备字符串, 是否有 iGPU)。"""
    has_gpu = any(d == "GPU" or d.startswith("GPU.") for d in devices)
    has_cpu = any(d == "CPU" for d in devices)

    if has_gpu:
        print("[OK] 检测到 Intel iGPU，选用 intel:gpu")
        return "intel:gpu", True
    if has_cpu:
        print("[!] 未检测到 Intel iGPU，回退到 intel:cpu")
        print("     若需 iGPU 加速，请安装运行时（Linux: sudo apt install intel-opencl-icd intel-level-zero-gpu）")
        return "intel:cpu", False
    print("[X] OpenVINO 无可用设备")
    sys.exit(1)


def model_dir() -> Path:
    p = Path(__file__).parent / MODEL_PATH
    return p.with_name(p.stem + "_openvino_model")


def export_openvino_model():
    """若不存在 IR 模型，则从 yolo11n.pt 导出（与 detector._load_openvino_model 一致）。"""
    ov_dir = model_dir()
    if ov_dir.is_dir() and any(ov_dir.glob("*.xml")):
        print(f"[OK] 已存在 OpenVINO IR 模型：{ov_dir}")
        return ov_dir

    src = Path(__file__).parent / MODEL_PATH
    if not src.exists():
        print(f"[X] 未找到模型文件：{src}")
        sys.exit(1)

    print(f"[..] 正在将 {MODEL_PATH} 导出为 OpenVINO IR（首次较耗时）...")
    try:
        from ultralytics import YOLO
    except ImportError:
        print("[X] 未安装 ultralytics，请先执行: pip install ultralytics")
        sys.exit(1)

    YOLO(str(src)).export(format="openvino", half=True, verbose=False)

    if not (ov_dir.is_dir() and any(ov_dir.glob("*.xml"))):
        print(f"[X] OpenVINO 模型导出失败：{ov_dir}")
        sys.exit(1)
    print(f"[OK] 导出完成：{ov_dir}")
    return ov_dir


def run_inference(ov_dir: Path, device: str, image: Path):
    """用 IR 模型对图片做一次猫识别。"""
    print(f"[..] 使用设备 {device} 推理图片：{image.name}")
    try:
        from ultralytics import YOLO
    except ImportError:
        print("[X] 未安装 ultralytics")
        sys.exit(1)

    model = YOLO(str(ov_dir))
    results = model(str(image), device=device, conf=CONFIDENCE,
                    classes=CAT_CLASS_ID, verbose=False)

    detected = False
    for res in results:
        for box in res.boxes:
            detected = True
            cls_id = int(box.cls[0])
            name = res.names.get(cls_id, str(cls_id))
            conf = float(box.conf[0])
            xyxy = [round(float(v), 1) for v in box.xyxy[0].tolist()]
            print(f"    检出: {name}  conf={conf:.2f}  bbox={xyxy}")

    if detected:
        print("[OK] 端到端验证成功：已成功加载 IR 模型并识别到目标。")
    else:
        print("[!] 推理成功但未识别到目标。可能：图片无猫 / 阈值过高 / 类别 id 不匹配。")
    return detected


def main() -> None:
    parser = argparse.ArgumentParser(description="OpenVINO 设备验证 + 模型导出 + 图片识别")
    parser.add_argument("--image", type=str, default=DEFAULT_TEST_IMAGE,
                        help="测试图片路径，默认 test_cat.png")
    args = parser.parse_args()

    # 1. 设备检测
    core, devices = check_openvino()
    device, _ = pick_device(devices)

    # 2. 导出 OpenVINO IR
    ov_dir = export_openvino_model()

    # 3. 图片识别
    image = Path(__file__).parent / args.image
    if not image.exists():
        print(f"[!] 未找到测试图片：{image}，跳过识别步骤（可用 --image 指定）")
        return
    run_inference(ov_dir, device, image)


if __name__ == "__main__":
    main()
