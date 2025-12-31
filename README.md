# OnnxDetServer-GO

OnnxDetServer-GO 是一个基于 Go 语言和 gocv 的高性能目标检测推理服务，专为 Windows 平台下本地部署和批量推理而设计。当前全部核心工作流通过 gRPC 实现，包括模型引擎初始化、推理和资源释放等，不再包含依赖 HTTP 接口管理 Detector 实例的功能。

---

## 功能概览

- 支持多模型本地加载和管理（gRPC）
- 全过程通过 gRPC 管理推理引擎与请求
- 通过 DLL（OnnxDet.dll + onnxruntime.dll）实现高性能、可选 GPU 的推理
- 标准化的推理数据返回格式，便于客户端后处理
- 支持图片推理、引擎状态查询与资源回收等常用操作

---

## 依赖与准备

- Go 1.18 及以上
- gocv（[gocv.io/x/gocv](https://gocv.io/)）
- Windows 64 位
- OnnxDet.dll、onnxruntime.dll 均需放置于可执行文件同级目录下的 `src/` 文件夹
- Microsoft Visual C++ Redistributable

---

## 主要目录结构与说明

- `OnnxDetServer-Go.go` —— gRPC 服务主入口
- `engine/` —— 核心推理封装（DLL 加载、Detector 定义与管理）
- `gRPC/` —— gRPC 接口与协议定义，完整的 Detector 生命周期管理
- `src/` —— 必须包含编译/下载好的 OnnxDet.dll、onnxruntime.dll
- `models/` —— 推荐用于存放 onnx 格式的模型文件
- `config.yaml` —— 服务参数配置
- `build.bat` —— 一站式打包和构建脚本

---

## 编译与启动

1. 准备好 Go 环境与依赖；
2. 放置 DLL 至 `src/` 文件夹（与主程序同级）；
3. 配置 `config.yaml`，包括 gRPC 端口、工作线程数等；
4. 编译并启动：

```shell
go build -o OnnxDetServer-Go.exe OnnxDetServer-Go.go
./OnnxDetServer-Go.exe
```

---

## gRPC 接口说明

所有操作建议通过 gRPC 客户端调用，下列为典型流程和接口。

### 1. 初始化推理引擎

- rpc 方法：`InitEngine(InitEngineRequest) returns (InitEngineResponse)`

```protobuf
message InitEngineRequest {
  string modelPath = 1;
  repeated string names = 2;
  float confidence = 3; // 默认 0.3
  float iou = 4;        // 默认 0.5
  bool useGpu = 5;
  int32 engineType = 6; // 推荐 4097 (SingleThread)
  string description = 7;
}
```
- 成功后获得引擎唯一 UUID。

### 2. 图片推理 Inference

- rpc 方法：`Inference(InferenceRequest) returns (InferenceResponse)`

```protobuf
message InferenceRequest {
  string id = 1;      // 引擎 UUID
  bytes imgData = 2;  // 图片内容（byte流，建议 jpg/png 原图）
}
```
- 返回标准化检测结果（含类别、置信度、边框、中心点）。

### 3. 资源释放

- rpc 方法：`DestroyEngine(DestroyEngineRequest) returns (DestroyEngineResponse)`

通过 UUID 释放对应 Detector 引擎占用资源。

### 4. 其他接口

- 引擎状态查询、批量检测处理等，详见 `gRPC/gRPCinterface.go` 或 proto 定义。

---

## 典型使用流程（gRPC）

1. `InitEngineRequest` 创建引擎，获得 UUID
2. 用 UUID 通过 `InferenceRequest` 实现图片推理
3. （可选）查状态
4. 推理结束或不再使用时，`DestroyEngine` 回收资源

---

## 返回结果示例

```json
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
```

---

## 常见问题

- 若 DLL 加载失败，请确认 DLL 路径是否正确，且 Visual C++ Redistributable 已安装
- 模型路径/类别必须与 onnxruntime 版本兼容，否则推理失败

---

## License

MIT

---

## 致谢

- gocv、onnxruntime

# Todo: OnnxDetServer-GO 项目企业级提升 

## ✅ 测试与质量保障

- [ ] 为 engine/、gRPC/、monitor/、logger/ 等模块添加单元测试（*_test.go）
- [ ] 添加集成测试，确保 gRPC 全流程可用
- [ ] 配置并维护覆盖率报表（推荐 `codecov`）

## ✅ 自动化与运维

- [ ] 添加 CI/CD 流水线（如 GitHub Actions），实现自动构建和测试
- [ ] 添加 Dockerfile，支持容器化部署
- [ ] 上传构建产物作为 Release 附件

## ✅ 配置与安全

- [ ] 支持从环境变量读取敏感配置，如 RegServerHost
- [ ] 增强配置校验逻辑
- [ ] gRPC 服务端增加 TLS 支持
- [ ] 实现 API Key/JWT 认证
- [ ] 加入请求频率限制功能（如 Rate Limiting）

## ✅ 文档与接口

- [ ] 完善 proto 文件注释和生成 gRPC 文档（如 grpc-docs、grpc-gateway）
- [ ] 添加接口使用范例和错误码说明
- [ ] 项目 README 增加部署与调优案例、容器启动指引

## ✅ 监控与可观测性

- [ ] 添加更多的 Prometheus 指标：推理延迟、模型加载耗时、当前服务引擎数量等
- [ ] 集成 OpenTelemetry，实现分布式链路追踪
- [ ] gRPC 日志中引入 TraceID

## ✅ 性能与可用性

- [ ] 增加健康检查接口（HealthCheck）
- [ ] 支持批量推理 RPC
- [ ] 支持模型热重载，无需重启即可切换模型
- [ ] 引擎满载时增加请求背压或优雅降级（如自动降级 CPU）

## ✅ 代码规范

- [ ] 目录名全部小写化（如 `gRPC/` 改为 `grpc/`）
- [ ] 统一代码风格及变量命名
- [ ] 集成 golangci-lint 进行静态检查

----

**完成这些 TODO，将极大提升 OnnxDetServer-GO 项目的企业级水平和可维护性。**