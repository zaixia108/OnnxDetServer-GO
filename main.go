package main

import (
	"OnnxDetServer/engine"
	"encoding/base64"
	"errors"
	"fmt"
	"net/http"
	"strings"
	"sync"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/google/uuid"
	"github.com/gorilla/websocket"
	"gocv.io/x/gocv"
)

const (
	IDLE = 0x1001
	BUSY = 0x1002
)

type EngineParam struct {
	ModelPath   string
	Names       []string
	Conf        float32
	Iou         float32
	UseGPU      bool
	State       int
	Description string
}

type worker struct {
	mu          sync.Mutex
	State       int
	Description string
	EngineType  int
	detector    *engine.Detector
}

type instance struct {
	id          string
	worker      *worker
	lastActive  time.Time
	conn        *websocket.Conn
	closeOnce   sync.Once
	cancelTimer chan struct{}
	cancelOnce  sync.Once
}

var (
	seqMu     sync.RWMutex
	workers   = map[string]*worker{}
	sessionMu sync.RWMutex
	sessions  = map[string]*instance{}
	upgrader  = websocket.Upgrader{
		ReadBufferSize:  1024,
		WriteBufferSize: 1024,
		CheckOrigin: func(r *http.Request) bool {
			return true
		},
	}
	idleTimeout = 1000 * time.Millisecond
)

func addWorker(description string, engineType int, param EngineParam) string {
	detector := &engine.Detector{}
	detector.New()
	names := engine.NamesConf{
		IsFile: false,
		Data:   param.Names,
	}
	detector.LoadModel(param.ModelPath, names, param.Conf, param.Iou, param.UseGPU)
	w := &worker{
		State:       IDLE,
		Description: description,
		EngineType:  engineType,
		detector:    detector,
	}
	id := uuid.New().String()
	seqMu.Lock()
	workers[id] = w
	seqMu.Unlock()
	if param.UseGPU {
		fmt.Println("Using GPU ,Warming up for worker", id)
		warmMat := gocv.NewMatWithSize(32, 32, gocv.MatTypeCV8UC3) // 小黑图，非空
		for i1 := 0; i1 < 3; i1++ {
			// 防止 Detect 内部 panic 导致服务崩溃，保护性调用
			func() {
				defer func() {
					if r := recover(); r != nil {
						fmt.Println("panic during warmup detect:", r)
					}
				}()
				_ = w.detector.Detect(warmMat)
			}()
		}
		_ = warmMat.Close()
		fmt.Println("Warm up Finished for worker", id)
	}

	return id
}

func allocInstance() (string, string, error) {
	seqMu.RLock()
	var chosenID string
	var chosen *worker
	for id, w := range workers {
		w.mu.Lock()
		if w.State == IDLE {
			w.State = BUSY
			chosenID = id
			chosen = w
			w.mu.Unlock()
			break
		}
		w.mu.Unlock()
	}
	seqMu.RUnlock()
	if chosen == nil {
		return "", "", errors.New("no available workers")
	}

	sessionID := uuid.New().String()
	inst := &instance{
		id:          sessionID,
		worker:      chosen,
		lastActive:  time.Now(),
		cancelTimer: make(chan struct{}),
	}

	sessionMu.Lock()
	sessions[sessionID] = inst
	sessionMu.Unlock()

	return sessionID, chosenID, nil
}

func releaseInstance(sessionID string) bool {
	sessionMu.Lock()
	inst, ok := sessions[sessionID]
	if ok {
		delete(sessions, sessionID)
	}
	sessionMu.Unlock()
	if !ok {
		return false
	}

	inst.closeOnce.Do(func() {
		if inst.conn != nil {
			_ = inst.conn.WriteMessage(websocket.CloseMessage, websocket.FormatCloseMessage(websocket.CloseNormalClosure, "1000 ms not active, released"))
			_ = inst.conn.Close()
		}
	})
	inst.cancelOnce.Do(func() {
		close(inst.cancelTimer)
	})
	inst.worker.mu.Lock()
	inst.worker.State = IDLE
	inst.worker.mu.Unlock()
	return true
}

func startIdleMonitor(inst *instance) {
	go func() {
		ticker := time.NewTicker(50 * time.Millisecond)
		defer ticker.Stop()
		for {
			select {
			case <-inst.cancelTimer:
				return
			case <-ticker.C:
				if time.Since(inst.lastActive) > idleTimeout {
					_ = releaseInstance(inst.id)
					fmt.Println("IdleMonitor timed out")
					return
				}
			}
		}
	}()
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

	mat, err := gocv.IMDecode(data, gocv.IMReadColor)
	if err != nil {
		return gocv.NewMat(), err
	}
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

func main() {
	r := gin.Default()
	r.GET("/api/ping", func(c *gin.Context) {
		c.JSON(http.StatusOK, gin.H{"message": "pong"})
	})
	r.POST("/api/workers/init/:count", func(c *gin.Context) {
		countStr := c.Param("count")
		var count int
		_, err := fmt.Sscanf(countStr, "%d", &count)
		if err != nil || count <= 0 {
			c.JSON(http.StatusBadRequest, gin.H{"error": "Invalid count"})
			return
		}

		var initParam EngineParam
		if err := c.ShouldBindJSON(&initParam); err != nil {
			c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
			return
		}

		// 设置默认值（如果客户端未提供）
		if initParam.Conf == 0 {
			initParam.Conf = 0.5
		}
		if initParam.Iou == 0 {
			initParam.Iou = 0.5
		}
		if initParam.Names == nil {
			initParam.Names = []string{}
		}

		fmt.Println("Creating", count, "workers with param:", initParam)
		ids := make([]string, count)
		for i := 0; i < count; i++ {
			id := addWorker("Sample Worker", engine.SingleThread, initParam)
			ids[i] = id
			seqMu.Lock()
			workers[id].Description = initParam.Description
			seqMu.Unlock()
		}

		c.JSON(http.StatusOK, gin.H{"data": ids})
	})
	r.GET("/api/workers/check/:id", func(c *gin.Context) {
		id := c.Param("id")
		seqMu.RLock()
		w, exists := workers[id]
		seqMu.RUnlock()
		if !exists {
			c.JSON(404, gin.H{"error": "Worker not found"})
			return
		}
		w.mu.Lock()
		state := w.State
		description := w.Description
		engineType := w.EngineType
		w.mu.Unlock()
		retData := map[string]any{
			"state":       state,
			"description": description,
			"engineType":  engineType,
		}
		c.JSON(200, gin.H{"data": retData})
	})
	r.POST("/api/workers/alloc", func(c *gin.Context) {
		sessionID, workerID, err := allocInstance()
		if err != nil {
			c.JSON(http.StatusBadRequest, gin.H{"error": "All workers are busy"})
			return
		}
		c.JSON(http.StatusOK, gin.H{
			"sessionID": sessionID,
			"workerID":  workerID,
			"wsURL":     fmt.Sprintf("ws://%s/ws/%s", c.Request.Host, sessionID),
			"timeoutMs": idleTimeout.Milliseconds(),
		})
	})
	r.POST("/api/workers/:sessionID/release", func(c *gin.Context) {
		sessionID := c.Param("sessionID")
		if !releaseInstance(sessionID) {
			c.JSON(404, gin.H{"error": "Session not found"})
			return
		}
		c.JSON(200, gin.H{"data": "Session released"})
	})
	r.GET("/ws/:sessionID", func(c *gin.Context) {
		sessionID := c.Param("sessionID")
		// 在升级前检查会话是否存在
		sessionMu.RLock()
		inst, exists := sessions[sessionID]
		sessionMu.RUnlock()
		if !exists {
			c.JSON(404, gin.H{"error": "Session not found"})
			return
		}

		conn, err := upgrader.Upgrade(c.Writer, c.Request, nil)
		if err != nil {
			// 升级失败，不要再写 JSON
			return
		}
		inst.conn = conn
		conn.SetReadLimit(20 * 1024 * 1024)

		startIdleMonitor(inst)
		for {
			mt, msg, err := conn.ReadMessage()
			if err != nil {
				// 客户端断开或读取错误，释放实例
				releaseInstance(sessionID)
				fmt.Println("Connection closed for session:", sessionID, "error:", err)
				return
			}
			inst.lastActive = time.Now()
			switch mt {
			case websocket.TextMessage:
				// 文本消息：base64 图像
				mat, err := Base64ToMat(string(msg))
				// save to local for debug
				gocv.IMWrite(fmt.Sprintf("./debug_%s.jpg", sessionID), mat)
				if err != nil {
					_ = conn.WriteMessage(websocket.TextMessage, []byte(fmt.Sprintf("invalid image: %v", err)))
					continue
				}
				result := inst.worker.detector.Detect(mat)
				_ = mat.Close()
				if !result.Success {
					_ = conn.WriteMessage(websocket.TextMessage, []byte(fmt.Sprintf("inference error: %v", result.Data)))
					continue
				}
				_ = conn.WriteMessage(websocket.TextMessage, []byte(fmt.Sprintf("inference result: %v", result.Data)))
			default:
				_ = conn.WriteMessage(websocket.TextMessage, []byte("unsupported message type"))
			}
		}
	})
	r.POST("/api/models/upload", func(c *gin.Context) {
		file, err := c.FormFile("file")
		if err != nil {
			c.JSON(http.StatusBadRequest, gin.H{"error": "File upload failed: " + err.Error()})
			return
		}

		modelPath := fmt.Sprintf("./models/%s", file.Filename)
		err = c.SaveUploadedFile(file, modelPath)
		if err != nil {
			c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to save file: " + err.Error()})
			return
		}
		c.JSON(http.StatusOK, gin.H{"data": modelPath})
	})
	_ = r.Run(":8080")
}
