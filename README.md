# YOLO + ByteTrack 交通目标跟踪与车辆计数

一个基于 **Ultralytics YOLO11** 和 **ByteTrack** 的交通视频分析小项目，可对视频中的常见交通目标进行检测与持续跟踪，并通过虚拟水平计数线统计车辆的上下行数量。

项目同时提供一个简单的车辆总数计数准确率计算脚本，可将系统统计结果与人工统计结果进行对比。

## 功能特性

- 使用 YOLO11 进行交通目标检测
- 使用 ByteTrack 维持跨帧 Track ID
- 支持常见交通类别：
  - Person
  - Bicycle
  - Car
  - Motorcycle
  - Bus
  - Truck
- 对车辆进行水平虚拟线越线计数
- 分别统计 `Up` / `Down` 两个方向
- 按车辆类别统计通过数量
- 在输出视频中绘制：
  - 检测框
  - 类别名称
  - Track ID
  - 检测置信度
  - 目标中心点
  - 虚拟计数线
  - 实时车辆数量与处理 FPS
- 自动输出 JSON 统计结果
- 支持根据人工真值计算车辆总数计数准确率

> `person` 会参与检测与跟踪，但不会计入车辆总数。

---

## 项目结构

```text
.
├── traffic_tracker.py             # YOLO + ByteTrack 检测、跟踪与越线计数
├── calculate_count_accuracy.py    # 车辆总数计数准确率计算
├── requirements.txt               # Python 依赖
├── yolo11n.pt                     # YOLO11n 模型权重
├── .gitignore
└── .gitattributes
```

运行后默认会生成：

```text
outputs/
├── result.mp4     # 可视化检测与计数结果视频
└── stats.json     # 统计数据
```

默认输入视频路径为：

```text
videos/traffic.mp4
```

如果目录不存在，可以自行创建：

```bash
mkdir -p videos
```

然后将测试视频放入 `videos/` 目录。

---

## 环境安装

建议先创建独立 Python 虚拟环境。

### 1. 创建虚拟环境

```bash
python -m venv .venv
```

Linux / macOS：

```bash
source .venv/bin/activate
```

Windows PowerShell：

```powershell
.venv\Scripts\Activate.ps1
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

当前项目依赖：

```text
ultralytics>=8.3
opencv-python>=4.9
numpy>=1.26
```

---

## 快速开始

假设输入视频位于：

```text
videos/traffic.mp4
```

直接运行：

```bash
python traffic_tracker.py
```

程序默认使用：

```text
模型：yolo11n.pt
输入：videos/traffic.mp4
输出视频：outputs/result.mp4
统计文件：outputs/stats.json
计数线：画面高度的 60%
置信度阈值：0.25
IoU 阈值：0.5
```

处理完成后，可以查看：

```text
outputs/result.mp4
outputs/stats.json
```

---

## 常用运行方式

### 指定输入视频

```bash
python traffic_tracker.py --source path/to/traffic.mp4
```

### 指定输出文件

```bash
python traffic_tracker.py \
  --source videos/traffic.mp4 \
  --output outputs/demo.mp4 \
  --stats outputs/demo_stats.json
```

### 修改计数线位置

`--line-y` 表示水平计数线在图像中的 y 坐标。

例如：

```bash
python traffic_tracker.py --line-y 400
```

如果不传或设置为 `-1`，程序默认使用：

```text
画面高度 × 60%
```

### 调整检测阈值

```bash
python traffic_tracker.py \
  --conf 0.35 \
  --iou 0.5
```

### 运行时显示画面

```bash
python traffic_tracker.py --show
```

开启后可以按 `Q` 提前退出。

> 在没有图形界面的服务器环境中不要使用 `--show`。

---

## 参数说明

| 参数       | 默认值               | 说明                                       |
| ---------- | -------------------- | ------------------------------------------ |
| `--source` | `videos/traffic.mp4` | 输入视频路径                               |
| `--model`  | `yolo11n.pt`         | YOLO 模型权重路径                          |
| `--output` | `outputs/result.mp4` | 输出可视化视频路径                         |
| `--stats`  | `outputs/stats.json` | JSON 统计文件路径                          |
| `--line-y` | `-1`                 | 水平计数线 y 坐标，`-1` 表示画面高度的 60% |
| `--conf`   | `0.25`               | 检测置信度阈值                             |
| `--iou`    | `0.5`                | NMS IoU 阈值                               |
| `--show`   | 关闭                 | 是否实时显示处理画面                       |

---

## 计数逻辑

程序使用目标边界框中心点的 y 坐标判断是否穿过水平计数线。

### Down

目标中心点由计数线上方移动到计数线下方：

```text
prev_y < line_y <= curr_y
```

记为：

```text
down
```

### Up

目标中心点由计数线下方移动到计数线上方：

```text
prev_y > line_y >= curr_y
```

记为：

```text
up
```

为减少重复计数，同一个 Track ID 在同一个方向上最多计数一次。

当前参与跟踪的 COCO 类别 ID 为：

```python
[0, 1, 2, 3, 5, 7]
```

对应：

```text
0  person
1  bicycle
2  car
3  motorcycle
5  bus
7  truck
```

其中真正计入车辆数量的是：

```text
bicycle
car
motorcycle
bus
truck
```

---

## 统计结果

程序执行结束后，会将统计信息写入：

```text
outputs/stats.json
```

结果格式类似：

```json
{
  "source": "videos/traffic.mp4",
  "model": "yolo11n.pt",
  "resolution": [1920, 1080],
  "source_fps": 25.0,
  "processed_frames": 1000,
  "video_seconds": 40.0,
  "counting_line_y": 648,
  "confidence_threshold": 0.25,
  "iou_threshold": 0.5,
  "count_up": 8,
  "count_down": 11,
  "total_vehicle_count": 19,
  "count_by_class": {
    "car": 15,
    "truck": 2,
    "motorcycle": 2
  },
  "average_processing_fps": 30.5,
  "note": "计数准确率需要你人工统计同一段视频中的真实车辆数后再计算。"
}
```

其中：

```text
total_vehicle_count = count_up + count_down
```

---

## 计算车辆计数准确率

先运行交通跟踪程序生成：

```text
outputs/stats.json
```

然后人工统计同一段视频中的真实车辆总数。

例如人工统计结果为 `19`：

```bash
python calculate_count_accuracy.py \
  --stats outputs/stats.json \
  --ground-truth 19
```

程序会输出类似：

```text
人工车辆数: 19
系统车辆数: 18
绝对误差: 1
计数准确率: 94.74%
```

当前项目使用的简化计数准确率公式为：

```text
Accuracy = max(0, 1 - |Pred - GT| / GT)
```

其中：

- `Pred`：系统统计的车辆总数
- `GT`：人工统计的真实车辆总数

> 这个指标只用于评估“车辆总数计数误差”，并不是目标检测领域中的 mAP、Precision 或 Recall。

---

## 模型说明

当前项目使用：

```text
yolo11n.pt
```

也可以通过 `--model` 指定其他 Ultralytics YOLO 权重，例如：

```bash
python traffic_tracker.py --model path/to/model.pt
```

当前压缩包中已经包含 `yolo11n.pt`。

需要注意，项目的 `.gitignore` 中包含：

```text
*.pt
```

因此如果将项目上传到 Git 仓库，模型权重默认不会被 Git 跟踪。其他用户克隆项目后，需要自行准备对应的模型权重，或调整 `.gitignore`。

---

## 示例完整命令

```bash
python traffic_tracker.py \
  --source videos/traffic.mp4 \
  --model yolo11n.pt \
  --output outputs/result.mp4 \
  --stats outputs/stats.json \
  --line-y -1 \
  --conf 0.25 \
  --iou 0.5
```

处理完成后计算准确率：

```bash
python calculate_count_accuracy.py \
  --stats outputs/stats.json \
  --ground-truth 19
```

---

## 注意事项

1. 计数效果会受到摄像机角度、遮挡、目标尺寸、视频清晰度以及检测阈值影响。
2. 当前实现使用单条水平线，更适合车辆主要沿垂直方向穿过画面的场景。
3. ByteTrack 的 Track ID 稳定性会直接影响越线计数结果。
4. 同一个 Track ID 在同一个方向最多计数一次；如果同一目标先向下越线、之后又向上越线，则两个方向可以分别记录一次。
5. 如果车辆在计数线附近频繁抖动、发生严重遮挡或 Track ID 被重新分配，仍可能出现漏计或重复计数。
6. `average_processing_fps` 表示程序处理速度，不等同于输入视频本身的 FPS。

---

## 后续可扩展方向

可以在当前项目基础上继续加入：

- 自定义多条计数线
- 多车道分别计数
- 任意方向 / 多边形区域计数
- 车辆轨迹可视化
- 车流量按时间段统计
- 速度估计
- 拥堵程度分析
- CSV / Excel 数据导出
- 实时摄像头或 RTSP 视频流
- Web 可视化界面
- 检测 Precision / Recall / mAP 评估
