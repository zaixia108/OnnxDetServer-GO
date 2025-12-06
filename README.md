# OnnxDetServer

项目基于 Go 与 gocv，封装了一个使用 OnnxDet.dll（基于 ONNX Runtime）的目标检测推理服务（HTTP API）。服务在 Windows 下通过加载本地 DLL 调用本地推理库完成检测。

---

## 功能概览
- 初始化检测引擎并加载模型
- 支持 base64 图片与 multipart 文件上传两种推理方式
- 支持查看/销毁已创建的 Detector 实例
- 通过本地 DLL（OnnxDet.dll + onnxruntime.dll）进行高性能推理

---

## 目录结构（相关）
- main.go — HTTP 服务与路由实现
- engine/def.go — Detector 抽象与数据结构
- engine/onnxdet.go — DLL 加载与 C 风格接口封装
- src/ — 用于放置 OnnxDet.dll、onnxruntime.dll 等本地库（需与可执行文件同级）

---

## 依赖
- Go 1.18+（推荐最新稳定版）
- gocv（gocv.io/x/gocv）和 OpenCV（gocv 依赖 OpenCV，本地需先安装）
- 在 Windows 上需要 Visual C++ Redistributable（否则加载 DLL 可能失败）
- 本项目依赖的本地库：OnnxDet.dll、onnxruntime.dll（放在 ./src 或可执行文件同级目录）

gocv 安装参考：https://gocv.io/getting-started/

---

## Windows 特别说明
- 可执行文件附近应有 `src` 目录，里面放入 `OnnxDet.dll` 以及它依赖的 `onnxruntime.dll` 等动态库。
- 程序启动时会尝试把 `src` 目录加入加载路径（SetDllDirectory 或修改 PATH），若仍报错请确认：
  - DLL 文件确实存在于 `src`（或可执行文件同级）
  - 已安装对应的 Visual C++ 运行时
  - 程序有权限读取这些文件

---

## 构建与运行

开发时直接运行：
- 在项目根目录：
  - go run main.go
或先构建再运行：
  - go build -o OnnxDetServer.exe
  - ./OnnxDetServer.exe

默认监听端口：8080

确保启动前把 `./src` 放在可执行文件同级并包含所需 DLL。

---

## API 文档（重要）

1. 初始化引擎
- URL: GET /api/Engine/init/:engineType
- engineType: 使用常量 SingleThread=4097 (0x1001) 或 MultiThread=4098 (0x1002)。当前 MultiThread 未实现，建议使用 4097。
- 请求体(JSON):
  {
    "modelName": "模型文件路径或名称",
    "names": ["class1","class2", ...],
    "threads": 1,        // 可选，当前主要为占位
    "conf": 0.3,         // 可选，置信度阈值，默认 0.3
    "iou": 0.5,          // 可选，NMS IOU 阈值，默认 0.5
    "useGPU": false,     // 是否使用 GPU（由底层 DLL 支持）
    "description": "描述"
  }
- 返回示例:
  {
    "message": "Engine initialized successfully",
    "data": {
      "id": "<UUID>",
      "description": "...",
      "engineType": 4097
    }
  }

示例 curl（Linux/macOS shell）：
- 注意：modelName 需要是底层 DLL 能识别且可访问的路径。
curl -X GET "http://localhost:8080/api/Engine/init/4097" \
  -H "Content-Type: application/json" \
  -d '{"modelName":"C:\\models\\yolov5.onnx","names":["person","car"],"conf":0.3,"iou":0.5,"useGPU":false,"description":"test"}'

2. 基于 base64 的推理（旧接口）
- URL: POST /api/inference/:UUID
- 表单字段:
  - image: base64 字符串（支持带 data:image/...;base64, 的前缀或纯 base64）
- 返回: 包含检测结果的 JSON（RetData 结构）

示例（将图片编码为 base64 并 POST）：
IMAGE_B64=$(base64 -w0 test.jpg)
curl -X POST "http://localhost:8080/api/inference/<UUID>" -d "image=${IMAGE_B64}"

3. 文件上传推理（推荐）
- URL: POST /api/inference/modern/:UUID
- Multipart 表单:
  - image: 文件上传（表单 key 为 image）
- 示例：
curl -F "image=@./test.jpg" http://localhost:8080/api/inference/modern/<UUID>

4. 销毁指定 Detector
- URL: GET /api/Engine/destroy/:UUID
- 说明: 销毁并释放资源

5. 查看指定 Detector 状态
- URL: GET /api/Engine/check/:UUID
- 返回包含 description、engineType、modelPath、conf、iou、useGPU、state 等

6. 列出所有 Detector
- URL: GET /api/Engine/checkAll

7. 关闭服务（会销毁所有 Detector 并退出）
- URL: POST /api/Engine/shutdown

---

## 返回数据结构说明

服务层返回示例结构（engine.RetData）:
{
  "Success": true | false,
  "Data": <具体数据，检测成功时为 map<className, []result>>
}

result 结构（在最终 JSON 中会表现为对象）：
- Conf: float 浮点置信度
- Box: {
    LT: {X, Y},
    RT: {X, Y},
    RB: {X, Y},
    LB: {X, Y}
  }
- Center: {X, Y}

最终 detection 数据示例：
{
  "person": [
    {
      "Conf": 0.95,
      "Box": {
        "LT": {"X": 10, "Y": 20},
        "RT": {"X": 110, "Y": 20},
        "RB": {"X": 110, "Y": 220},
        "LB": {"X": 10, "Y": 220}
      },
      "Center": {"X": 60, "Y": 120}
    }
  ],
  "car": []
}

---

## 使用流程示例
1. 初始化引擎 -> 得到 UUID
2. 使用 /api/inference/modern/:UUID 上传图片进行推理
3. 查看 /api/Engine/check/:UUID 获取状态
4. 不再使用时调用 /api/Engine/destroy/:UUID

---

## 常见问题排查
- 服务启动报 DLL 加载错误：
  - 确认 ./src/OnnxDet.dll 与依赖（如 onnxruntime.dll）存在
  - 安装 Visual C++ Redistributable
  - 检查是否为 32/64 位不匹配（DLL 与 Go 可执行需位数一致）
- 图片解码失败（空 Mat）：
  - 检查 base64 是否正确（无换行或多余空白）
  - 使用文件上传接口验证图片是否可以被正确读取
- gocv 编译/链接问题：
  - 请先根据 gocv 官方文档安装并配置 OpenCV 与环境变量


---

