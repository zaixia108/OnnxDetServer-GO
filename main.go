package main

import (
	"OnnxDetServer/engine"
	iface "OnnxDetServer/interface"
	"context"
	"encoding/base64"
	"errors"
	"fmt"
	"log"
	"maps"
	"net"
	"os"
	"runtime"
	"strings"
	"sync"
	"time"

	pb "OnnxDetServer/gRPC"

	"github.com/gin-gonic/gin"
	"github.com/google/uuid"
	"gocv.io/x/gocv"
	"google.golang.org/grpc"
	"google.golang.org/protobuf/types/known/emptypb"
	"gopkg.in/yaml.v3"
)

type EngineInitRequest struct {
	ModelPath   string   `json:"modelName" binding:"required"`
	Names       []string `json:"names" binding:"required"`
	Threads     int      `json:"threads"`
	Conf        float32  `json:"conf"`
	Iou         float32  `json:"iou"`
	UseGPU      bool     `json:"useGPU"`
	Description string   `json:"description"`
}

type configStruct struct {
	Port       int `yaml:"port"`
	WorkersNum int `yaml:"workersNum"`
}

type backend interface {
	LoadModel(modelPath string, names iface.NamesConf, conf float32, iou float32, useGPU bool) bool
	Detect(image gocv.Mat) iface.RetData
	Destroy()
	CheckConfig() iface.EngineConfig
}

type WorkerID struct {
	detector    backend
	Description string
	EngineType  int
}

var (
	dSequences map[string]WorkerID
	seqMu      sync.Mutex
	mapMu      sync.RWMutex
)

func (d *WorkerID) add2Seq(detector *engine.Detector, description string, engineType int) string {
	d.detector = detector
	d.Description = description
	if engineType == engine.MultiThread {
		panic("Multi-threading is not supported yet")
	}
	d.EngineType = engineType
	UUID := uuid.New().String()
	dSequences[UUID] = *d
	fmt.Printf("Detector %s added with ID %s\n", description, UUID)
	return UUID
}

// Base64ToMat 将 base64 字符串（可带 data:image/... 前缀）转为 gocv.Mat
func Base64ToMat(b64 any) (gocv.Mat, error) {
	// 去掉可能的 data URL 前缀
	switch b64.(type) {
	case string:
		b64 := b64.(string)
		if i := strings.Index(b64, ","); i != -1 && strings.HasPrefix(b64, "data:") {
			b64 = b64[i+1:]
		}

		data, err := base64.StdEncoding.DecodeString(b64)
		if err != nil {
			return gocv.NewMat(), err
		}

		mat, _ := gocv.IMDecode(data, gocv.IMReadColor)
		if mat.Empty() {
			// IMDecode 返回空 Mat 表示解码失败
			err := mat.Close()
			if err != nil {
				return gocv.Mat{}, err
			}
			return gocv.NewMat(), errors.New("decoded image is empty or unsupported format")
		}
		return mat, nil
	case []byte:
		mat, _ := gocv.IMDecode(b64.([]byte), gocv.IMReadColor)
		if mat.Empty() {
			// IMDecode 返回空 Mat 表示解码失败
			err := mat.Close()
			if err != nil {
				return gocv.Mat{}, err
			}
			return gocv.NewMat(), errors.New("decoded image is empty or unsupported format")
		}
		return mat, nil
	default:
		return gocv.NewMat(), errors.New("input must be a base64 string or byte slice")
	}
}

type jobPackage struct {
	worker backend
	image  any
	Result chan jobResult
}

type Position struct {
	X, Y float32
}

type Box struct {
	LT Position
	RT Position
	RB Position
	LB Position
}

type result struct {
	Conf   float32
	Box    Box
	Center Position
}

type jobResult struct {
	Data iface.RetData
}

var jobQueue chan jobPackage

func startWorker(workerNum int) {
	for i := 0; i < workerNum; i++ {
		go runWorker(i)
	}
}

func runWorker(workerID int) {
	defer func() {
		if r := recover(); r != nil {
			fmt.Printf("Worker %d panic: %v. Restarting in 1s...\n", workerID, r)
			//重启这个 Worker
			time.Sleep(1 * time.Second)
			go runWorker(workerID)
		}
	}()
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	fmt.Printf("---Worker %d created\n", workerID)
	for job := range jobQueue {
		detector := job.worker
		image := job.image
		imgData, err := Base64ToMat(image)
		if err != nil {
			job.Result <- jobResult{Data: iface.RetData{}}
		} else {
			result := detector.Detect(imgData)
			job.Result <- jobResult{Data: result}
		}
		err = imgData.Close()
		if err != nil {
			fmt.Printf("⚠️ Worker %d: error closing imgData: %v\n", workerID, err)
		}
	}
}

type server struct {
	pb.UnimplementedDetectServiceServer
}

func (s *server) InitEngine(ctx context.Context, req *pb.InitEngineRequest) (*pb.InitEngineResponse, error) {
	detector := engine.Detector{}
	detector.New()
	names := iface.NamesConf{}
	names.IsFile = false
	names.Data = req.Names
	seqMu.Lock()
	detector.LoadModel(req.ModelPath, names, req.Confidence, req.Iou, req.UseGpu)
	seqMu.Unlock()
	seqdet := WorkerID{}
	seqdet.EngineType = int(req.EngineType)
	seqdet.Description = req.Description
	seqdet.detector = &detector
	mapMu.Lock()
	Id := seqdet.add2Seq(&detector, req.Description, int(req.EngineType))
	mapMu.Unlock()
	return &pb.InitEngineResponse{
		Success: true,
		Id:      Id,
		Message: "Successfully initialized engine",
	}, nil
}

func (s *server) Inference(ctx context.Context, req *pb.InferenceRequest) (*pb.InferenceResponse, error) {
	UUID := req.Id
	mapMu.RLock()
	detector, exists := dSequences[UUID]
	mapMu.RUnlock()
	if !exists {
		return nil, fmt.Errorf("detector with ID %s not found", UUID)
	}
	imageData := req.ImgData
	inferResult := make(chan jobResult)
	defer close(inferResult)
	job := jobPackage{
		image:  imageData,
		worker: detector.detector,
		Result: inferResult,
	}
	jobQueue <- job
	results := <-inferResult

	if results.Data.Data == nil {
		return &pb.InferenceResponse{
			Success: false,
			Results: make([]*pb.SingleResult, 0),
		}, nil
	}
	switch results.Data.Data.(type) {
	case string:
		{
			return &pb.InferenceResponse{
				Success: false,
				Results: make([]*pb.SingleResult, 0),
			}, nil
		}
	case map[string][]result:
		{
			detResults := results.Data.Data.(map[string][]result)
			singleResults := make([]*pb.SingleResult, 0, len(detResults))
			for class, resList := range detResults {
				for _, res := range resList {
					resBox := make([]*pb.Position, 4)
					resBox[0] = &pb.Position{X: int32(res.Box.LT.X), Y: int32(res.Box.LT.Y)}
					resBox[1] = &pb.Position{X: int32(res.Box.RT.X), Y: int32(res.Box.RT.Y)}
					resBox[2] = &pb.Position{X: int32(res.Box.RB.X), Y: int32(res.Box.RB.Y)}
					resBox[3] = &pb.Position{X: int32(res.Box.LB.X), Y: int32(res.Box.LB.Y)}
					singleResult := &pb.SingleResult{
						Name:       class,
						Confidence: res.Conf,
						Box:        resBox,
						Center:     &pb.Position{X: int32(res.Center.X), Y: int32(res.Center.Y)},
					}
					singleResults = append(singleResults, singleResult)
				}
			}
			return &pb.InferenceResponse{
				Success: true,
				Results: singleResults,
			}, nil
		}
	default:
		{
			return &pb.InferenceResponse{
				Success: false,
				Results: make([]*pb.SingleResult, 0),
			}, fmt.Errorf("unexpected data type in results: %T", results.Data.Data)
		}
	}

}

func (s *server) DestroyEngine(ctx context.Context, req *pb.DestroyEngineRequest) (*pb.DestroyEngineResponse, error) {
	UUID := req.Id
	mapMu.Lock()
	detector, exists := dSequences[UUID]
	if !exists {
		mapMu.Unlock()
		return nil, fmt.Errorf("detector with ID %s not found", UUID)
	}
	detector.detector.Destroy()
	delete(dSequences, UUID)
	mapMu.Unlock()
	return &pb.DestroyEngineResponse{
		Success: true,
		Message: "Detector destroyed successfully",
	}, nil
}

func (s *server) CheckEngine(ctx context.Context, req *pb.CheckEngineRequest) (*pb.CheckEngineResponse, error) {
	UUID := req.Id
	mapMu.RLock()
	detector, exists := dSequences[UUID]
	mapMu.RUnlock()
	if !exists {
		return nil, fmt.Errorf("detector with ID %s not found", UUID)
	}
	Dconfig := detector.detector.CheckConfig()
	names := make([]string, 0)
	switch Dconfig.Names.Data.(type) {
	case []string:
		names = Dconfig.Names.Data.([]string)
	case string:
		names = []string{}
		names = append(names, "From File")
	default:
		return nil, fmt.Errorf("unexpected type for names: %T", Dconfig.Names.Data)
	}
	ret := &pb.EngineInfo{
		Id:          UUID,
		Description: detector.Description,
		EngineType:  int32(detector.EngineType),
		ModelPath:   Dconfig.ModelPath,
		Names:       names,
		Confidence:  Dconfig.Conf,
		Iou:         Dconfig.Iou,
		UseGpu:      Dconfig.UseGPU,
	}
	return &pb.CheckEngineResponse{
		Success:    true,
		EngineInfo: ret,
		Message:    "Detector status retrieved successfully",
	}, nil
}

func (s *server) CheckAllEngine(ctx context.Context, req *emptypb.Empty) (*pb.CheckAllEngineResponse, error) {
	mapMu.RLock()
	allSeq := maps.Clone(dSequences)
	mapMu.RUnlock()
	engineInfos := make([]*pb.EngineInfo, 0, len(allSeq))
	for id, detector := range allSeq {
		Dconfig := detector.detector.CheckConfig()
		names := make([]string, 0, len(Dconfig.Names.Data.([]string)))
		switch Dconfig.Names.Data.(type) {
		case []string:
			names = Dconfig.Names.Data.([]string)
		case string:
			names = []string{}
			names = append(names, "From File")
		default:
			return nil, fmt.Errorf("unexpected type for names: %T", Dconfig.Names.Data)
		}
		engineInfo := &pb.EngineInfo{
			Id:          id,
			Description: detector.Description,
			EngineType:  int32(detector.EngineType),
			ModelPath:   Dconfig.ModelPath,
			Names:       names,
			Confidence:  Dconfig.Conf,
			Iou:         Dconfig.Iou,
			UseGpu:      Dconfig.UseGPU,
		}
		engineInfos = append(engineInfos, engineInfo)
	}
	return &pb.CheckAllEngineResponse{
		Success: true,
		Engines: engineInfos,
		Message: "All Detectors status retrieved successfully",
	}, nil
}

func (s *server) Shutdown(ctx context.Context, req *emptypb.Empty) (*emptypb.Empty, error) {
	go func() {
		mapMu.Lock()
		for id, detector := range dSequences {
			detector.detector.Destroy()
			delete(dSequences, id)
		}
		mapMu.Unlock()
		close(jobQueue)
		fmt.Println("Server shutting down in 1 second...")
		time.Sleep(1 * time.Second)
		os.Exit(0)
	}()
	return &emptypb.Empty{}, nil
}

func main() {
	gin.SetMode(gin.ReleaseMode)
	fmt.Println(strings.Repeat("#", 64))
	CPUNum := runtime.NumCPU()
	runtime.GOMAXPROCS(CPUNum)
	fmt.Printf("CPU Cores: %d\n", CPUNum)
	configData, err := os.ReadFile("config.yaml")
	if err != nil {
		fmt.Println("Failed to read config file:", err)
		return
	}
	config := configStruct{}
	err = yaml.Unmarshal(configData, &config)
	if err != nil {
		fmt.Println("Failed to parse config file:", err)
		return
	}
	fmt.Println("Server Port:", config.Port)
	fmt.Println("Configured Workers Num:", config.WorkersNum)
	fmt.Println(strings.Repeat("#", 64))
	fmt.Println("")
	if config.WorkersNum <= 0 {
		config.WorkersNum = 1
		fmt.Println(strings.Repeat("!", 64))
		fmt.Println("Invalid workersNum in config, defaulting to 1")
		fmt.Println(strings.Repeat("!", 64))
	} else if config.WorkersNum > CPUNum {
		fmt.Println(strings.Repeat("!", 64))
		fmt.Println("Please noted that workersNum exceeds CPU cores, which may lead to performance degradation.")
		fmt.Println(strings.Repeat("!", 64))
	}
	fmt.Println("")
	fmt.Println(strings.Repeat("#", 64))
	fmt.Println("If you need GPU acceleration, please make sure that your GPU has enough memory to handle multiple workers.")
	fmt.Println("for GPU memory usage, please refer to 1280*1280 Yolo v8s model requires about 0.5GB memory each.")
	fmt.Println(strings.Repeat("#", 64))
	fmt.Println("")
	jobQueue = make(chan jobPackage, config.WorkersNum)
	startWorker(config.WorkersNum)
	dSequences = make(map[string]WorkerID)

	port := ":50051"
	lis, err := net.Listen("tcp", port)
	if err != nil {
		fmt.Printf("Failed to listen on port %s: %v\n", port, err)
	}

	s := grpc.NewServer()
	pb.RegisterDetectServiceServer(s, &server{})
	log.Printf("server listening on port %s\n", port)
	if err := s.Serve(lis); err != nil {
		log.Fatalf("Failed to serve gRPC server: %v", err)
	}
}
