package proto

import (
	"OnnxDetServer/engine"
	iface "OnnxDetServer/interface"
	"OnnxDetServer/logger"
	"OnnxDetServer/monitor"
	"context"
	"errors"
	"fmt"
	"io"
	"log"
	"maps"
	"net"
	"os"
	"runtime"
	"sync"
	"time"

	"github.com/google/uuid"
	"go.uber.org/zap"
	"gocv.io/x/gocv"
	"google.golang.org/grpc"
	"google.golang.org/protobuf/types/known/emptypb"
)

type WorkerID struct {
	detector    iface.Backend
	Description string
	EngineType  int
}

var (
	DSequences map[string]WorkerID
	seqMu      sync.Mutex
	mapMu      sync.RWMutex
)

func (d *WorkerID) add2Seq(detector iface.Backend, description string, engineType int) string {
	d.detector = detector
	d.Description = description
	if engineType == engine.MultiThread {
		panic("Multi-threading is not supported yet")
	}
	d.EngineType = engineType
	UUID := uuid.New().String()
	DSequences[UUID] = *d
	output := fmt.Sprintf("Detector %s added with ID %s\n", description, UUID)
	logger.Log().Info(output)
	return UUID
}

// Byte64ToMat 将 base64 字符串（可带 data:image/... 前缀）转为 gocv.Mat
func Byte64ToMat(b64 []byte) (gocv.Mat, error) {
	// 去掉可能的 data URL 前缀
	mat, _ := gocv.IMDecode(b64, gocv.IMReadColor)
	if mat.Empty() {
		// IMDecode 返回空 Mat 表示解码失败
		err := mat.Close()
		if err != nil {
			return gocv.Mat{}, err
		}
		return gocv.NewMat(), errors.New("decoded image is empty or unsupported format")
	}
	return mat, nil
}

type JobPackage struct {
	worker iface.Backend
	image  []byte
	Result chan jobResult
}

type jobResult struct {
	Data iface.RetData
}

var JobQueue chan JobPackage

var CloseChannel chan bool

//var ServChan chan *grpc.Server

func StartWorker(workerNum int) {
	for i := 0; i < workerNum; i++ {
		go runWorker(i)
	}
}

func runWorker(workerID int) {
	defer func() {
		if r := recover(); r != nil {
			output := fmt.Sprintf("Worker %d panic: %v. Restarting in 1s...\n", workerID, r)
			logger.Log().Error(output)
			//重启这个 Worker
			time.Sleep(1 * time.Second)
			go runWorker(workerID)
		}
	}()
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	output := fmt.Sprintf("---Worker %d created\n", workerID)
	logger.Log().Info(output)
	for job := range JobQueue {
		imgData, err := Byte64ToMat(job.image)
		if err != nil {
			job.Result <- jobResult{Data: iface.RetData{}}
		} else {
			result := job.worker.Detect(imgData)
			fmt.Println(result)
			job.Result <- jobResult{Data: result}
		}
		err = imgData.Close()
		if err != nil {
			output = fmt.Sprintf("⚠️ Worker %d: error closing imgData: %v\n", workerID, err)
			logger.Log().Error(output)
		}
	}
}

type Server struct {
	UnimplementedDetectServiceServer
}

func (s *Server) InitEngine(ctx context.Context, req *InitEngineRequest) (*InitEngineResponse, error) {
	monitor.GRPCTotal.Inc()
	detector := engine.Detector{}
	detector.New()
	names := iface.NamesConf{}
	names.IsFile = false
	names.Data = req.Names
	if req.Iou > 1.0 || req.Iou < 0.0 {
		return nil, fmt.Errorf("IoU must be between 0.0 and 1.0, got %f", req.Iou)
	}
	if req.Confidence > 1.0 || req.Confidence < 0.0 {
		return nil, fmt.Errorf("confidence must be between 0.0 and 1.0, got %f", req.Confidence)
	}
	if req.ModelPath == "" {
		return nil, fmt.Errorf("model path cannot be empty")
	}
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
	logger.Log().Info("Initialized new engine", zap.String("ID", Id), zap.String("ModelPath", req.ModelPath), zap.Float32("Confidence", req.Confidence), zap.Float32("IoU", req.Iou), zap.Bool("UseGPU", req.UseGpu))
	return &InitEngineResponse{
		Success: true,
		Id:      Id,
		Message: "Successfully initialized engine",
	}, nil
}

func (s *Server) Inference(ctx context.Context, req *InferenceRequest) (*InferenceResponse, error) {
	monitor.GRPCTotal.Inc()
	UUID := req.Id
	mapMu.RLock()
	detector, exists := DSequences[UUID]
	mapMu.RUnlock()
	if !exists {
		return nil, fmt.Errorf("detector with ID %s not found", UUID)
	}
	imageData := req.ImgData
	inferResult := make(chan jobResult)
	defer close(inferResult)
	job := JobPackage{
		image:  imageData,
		worker: detector.detector,
		Result: inferResult,
	}
	JobQueue <- job
	results := <-inferResult
	if results.Data.Data == nil {
		logger.Log().Error("detector returned nil result")
		return &InferenceResponse{
			Success: false,
			Results: make([]*SingleResult, 0),
		}, nil
	}
	switch v := results.Data.Data.(type) {
	case string:
		{
			logger.Log().Error("detector returned not supported string result:", zap.String("data", v))
			return &InferenceResponse{
				Success: false,
				Results: make([]*SingleResult, 0),
			}, nil
		}
	case map[string][]iface.Result:
		{
			detResults := results.Data.Data.(map[string][]iface.Result)
			singleResults := make([]*SingleResult, 0, len(detResults))
			for class, resList := range detResults {
				for _, res := range resList {
					resBox := make([]*Position, 4)
					resBox[0] = &Position{X: int32(res.Box.LT.X), Y: int32(res.Box.LT.Y)}
					resBox[1] = &Position{X: int32(res.Box.RT.X), Y: int32(res.Box.RT.Y)}
					resBox[2] = &Position{X: int32(res.Box.RB.X), Y: int32(res.Box.RB.Y)}
					resBox[3] = &Position{X: int32(res.Box.LB.X), Y: int32(res.Box.LB.Y)}
					singleResult := &SingleResult{
						Name:       class,
						Confidence: res.Conf,
						Box:        resBox,
						Center:     &Position{X: int32(res.Center.X), Y: int32(res.Center.Y)},
					}
					singleResults = append(singleResults, singleResult)
				}
			}
			return &InferenceResponse{
				Success: true,
				Results: singleResults,
			}, nil
		}
	default:
		{
			output := fmt.Sprintf("Unknown type: %T", v)
			logger.Log().Error(output)
			return &InferenceResponse{
				Success: false,
				Results: make([]*SingleResult, 0),
			}, fmt.Errorf("unexpected data type in results: %T", results.Data.Data)
		}
	}

}

func (s *Server) DestroyEngine(ctx context.Context, req *DestroyEngineRequest) (*DestroyEngineResponse, error) {
	monitor.GRPCTotal.Inc()
	UUID := req.Id
	mapMu.Lock()
	detector, exists := DSequences[UUID]
	if !exists {
		mapMu.Unlock()
		logger.Log().Error("detector not found with ID", zap.String("ID", UUID))
		return nil, fmt.Errorf("detector with ID %s not found", UUID)
	}
	detector.detector.Destroy()
	delete(DSequences, UUID)
	mapMu.Unlock()
	logger.Log().Info("Destroyed engine", zap.String("ID", UUID))
	return &DestroyEngineResponse{
		Success: true,
		Message: "Detector destroyed successfully",
	}, nil
}

func (s *Server) CheckEngine(ctx context.Context, req *CheckEngineRequest) (*CheckEngineResponse, error) {
	monitor.GRPCTotal.Inc()
	UUID := req.Id
	mapMu.RLock()
	detector, exists := DSequences[UUID]
	mapMu.RUnlock()
	if !exists {
		return nil, fmt.Errorf("detector with ID %s not found", UUID)
	}
	Dconfig := detector.detector.CheckConfig()
	names := make([]string, 0)
	switch v := Dconfig.Names.Data.(type) {
	case []string:
		names = Dconfig.Names.Data.([]string)
	case string:
		names = []string{}
		names = append(names, "From File")
	default:
		output := fmt.Sprintf("Unknown type: %T", v)
		logger.Log().Error(output)
		return nil, fmt.Errorf("unexpected type for names: %T", Dconfig.Names.Data)
	}
	ret := &EngineInfo{
		Id:          UUID,
		Description: detector.Description,
		EngineType:  int32(detector.EngineType),
		ModelPath:   Dconfig.ModelPath,
		Names:       names,
		Confidence:  Dconfig.Conf,
		Iou:         Dconfig.Iou,
		UseGpu:      Dconfig.UseGPU,
	}
	return &CheckEngineResponse{
		Success:    true,
		EngineInfo: ret,
		Message:    "Detector status retrieved successfully",
	}, nil
}

func (s *Server) CheckAllEngine(ctx context.Context, req *emptypb.Empty) (*CheckAllEngineResponse, error) {
	monitor.GRPCTotal.Inc()
	mapMu.RLock()
	allSeq := maps.Clone(DSequences)
	mapMu.RUnlock()
	engineInfos := make([]*EngineInfo, 0, len(allSeq))
	for id, detector := range allSeq {
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
		engineInfo := &EngineInfo{
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
	return &CheckAllEngineResponse{
		Success: true,
		Engines: engineInfos,
		Message: "All Detectors status retrieved successfully",
	}, nil
}

func (s *Server) Shutdown(ctx context.Context, req *emptypb.Empty) (*emptypb.Empty, error) {
	monitor.GRPCTotal.Inc()
	go func() {
		time.Sleep(2 * time.Second)
		mapMu.Lock()
		for id, detector := range DSequences {
			detector.detector.Destroy()
			delete(DSequences, id)
		}
		mapMu.Unlock()
		close(JobQueue)
		fmt.Println("Server shutting down in 1 second...")
		time.Sleep(1 * time.Second)
	}()
	CloseChannel <- true
	logger.Log().Warn("Shutting down in 1 second...")
	close(CloseChannel)
	return &emptypb.Empty{}, nil
}

func (s *Server) UploadModel(stream DetectService_UploadModelServer) error {
	monitor.GRPCTotal.Inc()
	var outFile *os.File
	var fileSize int
	var filePath string

	for {
		req, err := stream.Recv()
		if err == io.EOF {
			if outFile != nil {
				outFile.Close()
			}
			return stream.SendAndClose(&UploadFileResponse{
				Success:  true,
				Message:  "File uploaded successfully",
				FilePath: filePath,
			})
		}
		if err != nil {
			return err
		}

		switch payload := req.Data.(type) {
		case *UploadFileRequest_FileInfo:
			fmt.Printf("Received UploadFileRequest_FileInfo: %+v\n", payload.FileInfo.Name)
			saveDir := "models/"
			fileName := payload.FileInfo.Name
			if fileName == "" {
				return fmt.Errorf("file name cannot be empty")
			}
			filePath = saveDir + fileName
			outFile, err = os.Create(filePath)
			if err != nil {
				return err
			}
		case *UploadFileRequest_ChunkData:
			if outFile == nil {
				return fmt.Errorf("file not opened, please send file info first")
			}
			n, writeErr := outFile.Write(payload.ChunkData)
			if writeErr != nil {
				return fmt.Errorf("failed to write chunk data: %v", writeErr)
			}
			fileSize += n
		}
	}
}

func StartGRPCServer(addr int) *grpc.Server {
	CloseChannel = make(chan bool)
	//ServChan = make(chan *grpc.Server)
	port := fmt.Sprintf(":%d", addr)
	lis, err := net.Listen("tcp", port)
	if err != nil {
		fmt.Printf("Failed to listen on port %s: %v\n", port, err)
	}
	s := grpc.NewServer()
	go func() {
		RegisterDetectServiceServer(s, &Server{})
		log.Printf("server listening on port %s\n", port)
		if err := s.Serve(lis); err != nil {
			log.Fatalf("Failed to serve gRPC server: %v", err)
		}
	}()
	return s
}
