package proto

import (
	iface "OnnxDetServer/interface"
	"OnnxDetServer/monitor"
	"context"
	"fmt"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"gocv.io/x/gocv"
	"google.golang.org/grpc"
	"google.golang.org/protobuf/types/known/emptypb"
)

type MockBackend struct{}

func (m *MockBackend) LoadModel(modelPath string, names iface.NamesConf, conf float32, iou float32, useGPU bool) bool {
	fmt.Printf("Mock LoadModel called with modelPath: %s, names: %v, conf: %f, iou: %f, useGPU: %v\n", modelPath, names, conf, iou, useGPU)
	return true
}
func (m *MockBackend) Detect(mat gocv.Mat) iface.RetData {
	fakeResult := map[string][]iface.Result{}
	fmt.Println("AAAA - Mock Detect Called") // 添加标识以确认被调用
	fakeResult["mock"] = []iface.Result{
		{
			Conf: 0.99,
			Box: iface.Box{
				LT: iface.Position{X: 1, Y: 1},
				RT: iface.Position{X: 2, Y: 2},
				RB: iface.Position{X: 1, Y: 2},
				LB: iface.Position{X: 2, Y: 1},
			},
			Center: iface.Position{X: 2, Y: 2},
		},
	}
	return iface.RetData{Success: true, Data: fakeResult}
}
func (m *MockBackend) Destroy() {}
func (m *MockBackend) CheckConfig() iface.EngineConfig {
	return iface.EngineConfig{ModelPath: "mock", Names: iface.NamesConf{Data: []string{"mock"}}, Conf: 0.99, Iou: 0.5, UseGPU: false}
}
func (m *MockBackend) SetInputSize(size int)                    {}
func (m *MockBackend) SetBlobName(inputName, outputName string) {}

func TestMockEngine(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	// 注意：如果 monitor 已经在其他地方启动，这里可能会冲突，请确保端口未占用
	go monitor.StartMon(50052, ctx)

	backend := &MockBackend{}
	worker := &WorkerID{}
	DSequences = make(map[string]WorkerID)
	id := worker.add2Seq(backend, "mock_worker", 4097)

	server := StartGRPCServer(50051)
	defer server.GracefulStop()
	t.Log("Mock gRPC server started for testing")

	conn, err := grpc.NewClient("localhost:50051", grpc.WithInsecure())
	if err != nil {
		t.Fatalf("Failed to connect to gRPC server: %v", err)
	}
	defer conn.Close()

	client := NewDetectServiceClient(conn)
	JobQueue = make(chan JobPackage, 10)
	StartWorker(1)
	t.Log("Mock gRPC server start")
	time.Sleep(2 * time.Second) // 减少等待时间

	t.Run("Test Inference", func(t *testing.T) {
		MockImg := gocv.NewMatWithSize(224, 224, gocv.MatTypeCV8UC3)
		defer MockImg.Close()

		// 修正：使用 IMEncode 将 Mat 编码为 jpg 格式，以便服务器的 IMDecode 能正确解析
		buf, err := gocv.IMEncode(".jpg", MockImg)
		if err != nil {
			t.Fatalf("Failed to encode image: %v", err)
		}
		defer buf.Close()

		req := &InferenceRequest{
			Id:      id,
			ImgData: buf.GetBytes(), // 发送编码后的数据
		}
		t.Log("Inference request: ", req.Id)
		resp, err := client.Inference(context.Background(), req)
		if err != nil {
			t.Fatalf("Inference failed: %v", err)
		}
		fmt.Println("Results:", resp.Results)

		if assert.Len(t, resp.Results, 1) {
			r := resp.Results[0]
			assert.Equal(t, "mock", r.Name)
			assert.InDelta(t, 0.99, r.Confidence, 0.0001)

			assert.NotNil(t, r.Center)
			assert.Equal(t, int32(2), r.Center.X)
			assert.Equal(t, int32(2), r.Center.Y)

			if assert.Len(t, r.Box, 4) {
				assert.Equal(t, int32(1), r.Box[0].X)
				assert.Equal(t, int32(1), r.Box[0].Y)
			}
		}
	})

	t.Run("Test CheckEngine", func(t *testing.T) {
		req := &CheckEngineRequest{Id: id}
		resp, err := client.CheckEngine(context.Background(), req)
		if err != nil {
			t.Fatalf("CheckEngine failed: %v", err)
		}
		info := resp.EngineInfo
		assert.Equal(t, "mock", info.ModelPath)
		assert.InDelta(t, 0.99, info.Confidence, 0.0001)
		assert.Equal(t, []string{"mock"}, info.Names)
	})

	t.Run("Test CheckAllEngine", func(t *testing.T) {
		resp, err := client.CheckAllEngine(context.Background(), &emptypb.Empty{})
		if err != nil {
			t.Fatalf("CheckAllEngine failed: %v", err)
		}
		if assert.Len(t, resp.Engines, 1) {
			info := resp.Engines[0]
			assert.Equal(t, id, info.Id)
			assert.Equal(t, "mock", info.ModelPath)
		}
	})

	cancel()
}
