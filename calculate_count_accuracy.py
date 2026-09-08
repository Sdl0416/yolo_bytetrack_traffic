import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="计算车辆计数准确率")
    parser.add_argument("--stats", default="outputs/stats.json")
    parser.add_argument(
        "--ground-truth",
        type=int,
        default=19,
        help="人工统计的真实车辆数",
    )
    args = parser.parse_args()

    data = json.loads(Path(args.stats).read_text(encoding="utf-8"))
    pred = int(data["total_vehicle_count"])
    gt = args.ground_truth

    if gt <= 0:
        raise ValueError("ground-truth 必须 > 0")

    absolute_error = abs(pred - gt)

    # 简化项目中使用的“计数准确率”口径：
    # 1 - |pred - gt| / gt
    accuracy = max(0.0, 1.0 - absolute_error / gt)

    print(f"人工车辆数: {gt}")
    print(f"系统车辆数: {pred}")
    print(f"绝对误差: {absolute_error}")
    print(f"计数准确率: {accuracy * 100:.2f}%")
    print(
        "\n注意：这是车辆总数计数误差的简化指标，"
        "不是目标检测模型的 mAP/Precision/Recall。"
    )


if __name__ == "__main__":
    main()
