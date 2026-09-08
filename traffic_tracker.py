from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


# COCO 中常见交通类别
# 0 person, 1 bicycle, 2 car, 3 motorcycle, 5 bus, 7 truck
TRACK_CLASS_IDS = [0, 1, 2, 3, 5, 7]
VEHICLE_CLASS_IDS = {1, 2, 3, 5, 7}


def parse_args():
    parser = argparse.ArgumentParser(
        description="YOLO + ByteTrack 交通目标检测、跟踪与越线计数"
    )
    parser.add_argument(
        "--source",
        default="videos/traffic.mp4",
        help="输入视频路径"
    )
    parser.add_argument(
        "--model",
        default="yolo11n.pt",
        help="Ultralytics YOLO 权重，例如 yolo11n.pt",
    )
    parser.add_argument(
        "--output",
        default="outputs/result.mp4",
        help="输出视频路径",
    )
    parser.add_argument(
        "--stats",
        default="outputs/stats.json",
        help="统计 JSON 输出路径",
    )
    parser.add_argument(
        "--line-y",
        type=int,
        default=-1,
        help="虚拟计数线 y 坐标；-1 表示画面高度的 60%%",
    )
    parser.add_argument("--conf", type=float, default=0.25, help="检测置信度阈值")
    parser.add_argument("--iou", type=float, default=0.5, help="检测 NMS IoU 阈值")
    parser.add_argument(
        "--show",
        action="store_true",
        help="运行时弹窗显示；服务器环境不要开启",
    )
    return parser.parse_args()


def crossed_line(prev_y: int, curr_y: int, line_y: int) -> str | None:
    """判断目标中心点是否跨过水平虚拟线。"""
    if prev_y < line_y <= curr_y:
        return "down"
    if prev_y > line_y >= curr_y:
        return "up"
    return None


def main():
    args = parse_args()

    source = Path(args.source)
    if not source.exists():
        raise FileNotFoundError(f"找不到输入视频: {source}")

    output = Path(args.output)
    stats_path = Path(args.stats)
    output.parent.mkdir(parents=True, exist_ok=True)
    stats_path.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {source}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    source_fps = cap.get(cv2.CAP_PROP_FPS)
    if source_fps <= 0:
        source_fps = 25.0

    line_y = args.line_y if args.line_y >= 0 else int(height * 0.60)
    line_y = max(0, min(height - 1, line_y))

    writer = cv2.VideoWriter(
        str(output),
        cv2.VideoWriter_fourcc(*"mp4v"),
        source_fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError("无法创建输出视频，请检查 OpenCV/编码器环境。")

    model = YOLO(args.model)

    # 每个 Track ID 的上一帧中心点 y
    last_center_y: dict[int, int] = {}

    # 防止同一 Track ID 多次计数：每个方向只记一次
    counted_ids = {"up": set(), "down": set()}

    total_by_direction = defaultdict(int)
    total_by_class = defaultdict(int)

    frame_count = 0
    processing_time = 0.0

    print(f"[INFO] input: {source}")
    print(f"[INFO] resolution: {width}x{height}")
    print(f"[INFO] source FPS: {source_fps:.2f}")
    print(f"[INFO] counting line y: {line_y}")
    print("[INFO] press Q to quit when --show is enabled")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        t0 = time.perf_counter()

        # persist=True：连续帧复用 tracker 状态，维持 Track ID
        result = model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            classes=TRACK_CLASS_IDS,
            conf=args.conf,
            iou=args.iou,
            verbose=False,
        )[0]

        annotated = frame.copy()

        boxes = result.boxes
        if boxes is not None and boxes.id is not None:
            xyxy = boxes.xyxy.cpu().numpy()
            track_ids = boxes.id.int().cpu().tolist()
            class_ids = boxes.cls.int().cpu().tolist()
            confidences = boxes.conf.cpu().tolist()

            for box, track_id, cls_id, conf in zip(
                xyxy, track_ids, class_ids, confidences
            ):
                x1, y1, x2, y2 = map(int, box)
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)

                class_name = result.names.get(cls_id, str(cls_id))

                # 画框 + ID
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"{class_name} ID:{track_id} {conf:.2f}"
                cv2.putText(
                    annotated,
                    label,
                    (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 255, 0),
                    2,
                )
                cv2.circle(annotated, (cx, cy), 4, (0, 0, 255), -1)

                # 只对车辆类别计数；person 不计入车辆总数
                if cls_id in VEHICLE_CLASS_IDS:
                    prev_y = last_center_y.get(track_id)
                    if prev_y is not None:
                        direction = crossed_line(prev_y, cy, line_y)
                        if (
                            direction is not None
                            and track_id not in counted_ids[direction]
                        ):
                            counted_ids[direction].add(track_id)
                            total_by_direction[direction] += 1
                            total_by_class[class_name] += 1

                    last_center_y[track_id] = cy

        # 虚拟计数线
        cv2.line(
            annotated,
            (0, line_y),
            (width, line_y),
            (0, 255, 255),
            2,
        )

        elapsed = time.perf_counter() - t0
        processing_time += elapsed
        frame_count += 1

        current_fps = 1.0 / elapsed if elapsed > 0 else 0.0
        total_vehicles = (
            total_by_direction["up"] + total_by_direction["down"]
        )

        cv2.putText(
            annotated,
            f"Vehicles: {total_vehicles}",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 0),
            2,
        )
        cv2.putText(
            annotated,
            f"Up: {total_by_direction['up']}  Down: {total_by_direction['down']}",
            (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 255, 0),
            2,
        )
        cv2.putText(
            annotated,
            f"FPS: {current_fps:.1f}",
            (20, 105),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 255, 0),
            2,
        )

        writer.write(annotated)

        if args.show:
            cv2.imshow("YOLO + ByteTrack Traffic", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    writer.release()
    cv2.destroyAllWindows()

    avg_fps = frame_count / processing_time if processing_time > 0 else 0.0
    video_seconds = frame_count / source_fps if source_fps > 0 else 0.0

    summary = {
        "source": str(source),
        "model": args.model,
        "resolution": [width, height],
        "source_fps": round(source_fps, 3),
        "processed_frames": frame_count,
        "video_seconds": round(video_seconds, 3),
        "counting_line_y": line_y,
        "confidence_threshold": args.conf,
        "iou_threshold": args.iou,
        "count_up": total_by_direction["up"],
        "count_down": total_by_direction["down"],
        "total_vehicle_count": (
            total_by_direction["up"] + total_by_direction["down"]
        ),
        "count_by_class": dict(total_by_class),
        "average_processing_fps": round(avg_fps, 3),
        "note": (
            "计数准确率需要你人工统计同一段视频中的真实车辆数后再计算。"
        ),
    }

    stats_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n===== DONE =====")
    print(f"output video: {output}")
    print(f"stats: {stats_path}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
