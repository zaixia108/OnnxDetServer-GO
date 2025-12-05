package main

import (
	"OnnxDetServer/engine"
	"encoding/base64"
	"errors"
	"fmt"
	"maps"
	"net/http"
	"strings"
	"sync"

	"github.com/gin-gonic/gin"
	"github.com/google/uuid"
	"gocv.io/x/gocv"
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

type WorkerID struct {
	detector    *engine.Detector
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
	if dSequences == nil {
		dSequences = make(map[string]WorkerID)
	}
	UUID := uuid.New().String()
	dSequences[UUID] = *d
	return UUID
}

// Base64ToMat 将 base64 字符串（可带 data:image/... 前缀）转为 gocv.Mat
func Base64ToMat(b64 string) (gocv.Mat, error) {
	// 去掉可能的 data URL 前缀
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
}

type jobPackage struct {
	worker *engine.Detector
	image  string
	Result chan jobResult
}

type jobResult struct {
	Data engine.RetData
}

var jobQueue = make(chan jobPackage, 4)

func startWorker(workerNum int) {
	for i := 0; i < workerNum; i++ {
		go func(workerID int) {
			fmt.Printf("Worker %d started\n", workerID)
			for job := range jobQueue {
				detector := job.worker
				image := job.image
				imgData, err := Base64ToMat(image)
				if err != nil {
					job.Result <- jobResult{Data: engine.RetData{}}
				} else {
					result := detector.Detect(imgData)
					job.Result <- jobResult{Data: result}
				}
				err = imgData.Close()
			}
			fmt.Printf("Worker %d finished\n", workerID)
		}(i)
	}
}

func main() {
	startWorker(4)
	r := gin.Default()
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
		names := engine.NamesConf{}
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
		mapMu.Lock()
		detector, exists := dSequences[UUID]
		mapMu.Unlock()
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
	r.GET("/api/Engine/destroy/:UUID", func(c *gin.Context) {
		UUID := c.Param("UUID")
		mapMu.Lock()
		detector, exists := dSequences[UUID]
		mapMu.Unlock()
		if !exists {
			c.JSON(http.StatusNotFound, gin.H{"error": "Detector not found"})
			return
		}
		detector.detector.Destroy()
		mapMu.Lock()
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
		detectorState["description"] = detector.Description
		detectorState["engineType"] = detector.EngineType
		detectorState["modelPath"] = detector.detector.ModelPath
		detectorState["conf"] = detector.detector.Conf
		detectorState["iou"] = detector.detector.Iou
		detectorState["useGPU"] = detector.detector.UseGPU
		detectorState["state"] = detector.detector.State
		c.JSON(http.StatusOK, gin.H{
			"message": "Detector status retrieved successfully",
			"data":    detectorState,
		})
	})
	r.GET("/api/Engine/checkAll", func(c *gin.Context) {
		mapMu.Lock()
		allSeq := maps.Clone(dSequences)
		mapMu.Unlock()
		c.JSON(http.StatusOK, gin.H{"message": "All States", "data": allSeq})
	})
	err := r.Run(":8080")
	if err != nil {
		fmt.Println(err)
	}
}
