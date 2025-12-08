package main

import (
	"OnnxDetServer/engine"
	iface "OnnxDetServer/interface"
	"encoding/base64"
	"errors"
	"fmt"
	"io"
	"maps"
	"mime/multipart"
	"net/http"
	"os"
	"runtime"
	"strings"
	"sync"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/google/uuid"
	"gocv.io/x/gocv"
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

	r := gin.New()
	r.Use(gin.Logger(), gin.Recovery())

	r.GET("/api/Engine/init/:engineType", func(c *gin.Context) {
		engineTypeStr := c.Param("engineType")
		var engineType int
		_, err := fmt.Sscanf(engineTypeStr, "%d", &engineType)
		if err != nil {
			c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
			return
		}
		if engineType != engine.SingleThread && engineType != engine.MultiThread {
			c.JSON(http.StatusBadRequest, gin.H{"error": "Invalid engine type"})
			return
		}

		req := new(EngineInitRequest)
		if err := c.ShouldBindJSON(&req); err != nil {
			c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
			return
		}
		if req.Threads <= 0 {
			req.Threads = 1 // 默认单线程
		}
		if req.Conf <= 0 {
			req.Conf = 0.3 // 默认置信度
		}
		if req.Iou <= 0 {
			req.Iou = 0.5 // 默认IOU
		}
		if req.Description == "" {
			req.Description = "Default Detector"
		}

		detector := engine.Detector{}
		detector.New()
		names := iface.NamesConf{}
		names.IsFile = false
		names.Data = req.Names
		seqMu.Lock()
		detector.LoadModel(req.ModelPath, names, req.Conf, req.Iou, req.UseGPU)
		seqMu.Unlock()
		seqdet := WorkerID{}
		seqdet.EngineType = engineType
		seqdet.Description = req.Description
		seqdet.detector = &detector
		mapMu.Lock()
		Id := seqdet.add2Seq(&detector, req.Description, engineType)
		mapMu.Unlock()
		c.JSON(http.StatusOK, gin.H{
			"message": "Engine initialized successfully",
			"data": gin.H{
				"id":          Id,
				"description": seqdet.Description,
				"engineType":  seqdet.EngineType,
			},
		})
	})
	r.POST("/api/inference/:UUID", func(c *gin.Context) {
		UUID := c.Param("UUID")
		mapMu.RLock()
		detector, exists := dSequences[UUID]
		mapMu.RUnlock()
		if !exists {
			c.JSON(http.StatusNotFound, gin.H{"error": "Detector not found"})
			return
		}
		imageBase64 := c.PostForm("image")
		if imageBase64 == "" {
			c.JSON(http.StatusBadRequest, gin.H{"error": "Image data is required"})
			return
		}
		inferResult := make(chan jobResult)
		defer close(inferResult)
		job := jobPackage{
			image:  imageBase64,
			worker: detector.detector,
			Result: inferResult,
		}
		jobQueue <- job
		results := <-inferResult
		c.JSON(http.StatusOK, gin.H{
			"message": "Inference completed successfully",
			"data":    results,
		})
	})
	r.POST("/api/inference/modern/:UUID", func(c *gin.Context) {
		UUID := c.Param("UUID")
		mapMu.RLock()
		detector, exists := dSequences[UUID]
		mapMu.RUnlock()
		if !exists {
			c.JSON(http.StatusNotFound, gin.H{"error": "Detector not found"})
			return
		}
		fileHeader, err := c.FormFile("image")
		if err != nil {
			c.JSON(http.StatusBadRequest, gin.H{"error": "Image file is required"})
			fmt.Println(err.Error())
			return
		}
		file, err := fileHeader.Open()
		if err != nil {
			c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to open image file"})
			return
		}
		defer func(file multipart.File) {
			err := file.Close()
			if err != nil {

			}
		}(file)
		limitedReader := io.LimitReader(file, 10*1024*1024)
		fileByte, err := io.ReadAll(limitedReader)
		if err != nil {
			c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to read image file"})
			return
		}
		inferResult := make(chan jobResult)
		defer close(inferResult)
		job := jobPackage{
			image:  fileByte,
			worker: detector.detector,
			Result: inferResult,
		}
		jobQueue <- job
		results := <-inferResult
		//fmt.Println(results.Data.Data)
		c.JSON(http.StatusOK, gin.H{
			"message": "Inference completed successfully",
			"data":    results.Data.Data,
		})

	})
	r.GET("/api/Engine/destroy/:UUID", func(c *gin.Context) {
		UUID := c.Param("UUID")
		mapMu.Lock()
		detector, exists := dSequences[UUID]
		if !exists {
			mapMu.Unlock()
			c.JSON(http.StatusNotFound, gin.H{"error": "Detector not found"})
			return
		}
		detector.detector.Destroy()
		delete(dSequences, UUID)
		mapMu.Unlock()
		c.JSON(http.StatusOK, gin.H{
			"message": "Detector destroyed successfully",
		})
	})
	r.GET("/api/Engine/check/:UUID", func(c *gin.Context) {
		UUID := c.Param("UUID")
		mapMu.Lock()
		detector, exists := dSequences[UUID]
		mapMu.Unlock()
		if !exists {
			c.JSON(http.StatusNotFound, gin.H{"error": "Detector not found"})
			return
		}
		var detectorState map[string]any
		detectorState = make(map[string]any)
		engineConfig := detector.detector.CheckConfig()
		detectorState["description"] = detector.Description
		detectorState["engineType"] = detector.EngineType
		detectorState["modelPath"] = engineConfig.ModelPath
		detectorState["conf"] = engineConfig.Conf
		detectorState["iou"] = engineConfig.Iou
		detectorState["useGPU"] = engineConfig.UseGPU
		c.JSON(http.StatusOK, gin.H{
			"message": "Detector status retrieved successfully",
			"data":    detectorState,
		})
	})
	r.GET("/api/Engine/checkAll", func(c *gin.Context) {
		mapMu.RLock()
		allSeq := maps.Clone(dSequences)
		mapMu.RUnlock()
		c.JSON(http.StatusOK, gin.H{"message": "All States", "data": allSeq})
	})
	r.POST("/api/Engine/shutdown", func(c *gin.Context) {
		c.JSON(http.StatusOK, gin.H{"message": "Shutting down..."})
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
	})
	addr := fmt.Sprintf(":%d", config.Port)
	err = r.Run(addr)
	if err != nil {
		fmt.Println(err)
	}
}
