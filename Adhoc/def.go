package Adhoc

import (
	"OnnxDetServer/logger"
	"context"
	"fmt"
	"runtime"
	"sync"
	"time"

	"github.com/go-resty/resty/v2"
	"github.com/google/uuid"
)

const (
	DmlInstance    = 0x2001
	CpuInstance    = 0x2002
	CudaInstance   = 0x2003
	RocmInstance   = 0x2004
	TimeOutSeconds = 5
)

type RegisterRequest struct {
	Id            string `json:"id"`
	IP            string `json:"ip"`
	Port          int    `json:"port"`
	InstanceClass int    `json:"instanceClass"`
	TimeStamp     int64  `json:"timestamp"`
}

type RegisterResponse struct {
	Id      string `json:"id"`
	Success bool   `json:"success"`
}

type RegServerConfig struct {
	Port int
	Addr string
}

func (reg *RegServerConfig) SetAddress(addr string, port int) {
	reg.Addr = addr
	reg.Port = port
}

var RegServerCfg RegServerConfig

func SendAliveMessage(CCIP string, CCPort int, instanceClass int, ctx context.Context, wg *sync.WaitGroup) {
	addr := fmt.Sprintf("%s:%d", RegServerCfg.Addr, RegServerCfg.Port)
	defer wg.Done()
	ticker := time.NewTicker(TimeOutSeconds * time.Second)
	defer ticker.Stop()
	client := resty.New().SetTimeout(TimeOutSeconds * time.Second) // 总超时
	// 构造请求体
	id := uuid.NewString()
	safeDoRequest := func() {
		defer func() {
			if r := recover(); r != nil {
				logger.Log().Error(fmt.Sprintf("SendAliveMessage panic recovered: %v", r))
			}
		}()
		runtime.LockOSThread()
		defer runtime.UnlockOSThread()
		var respBody RegisterResponse
		url := fmt.Sprintf("http://%s/api/register", addr)
		reqBody := RegisterRequest{
			Id:            id,
			IP:            CCIP,
			Port:          CCPort,
			InstanceClass: instanceClass,
			TimeStamp:     time.Now().Unix(),
		}
		resp, err := client.R().
			SetContext(ctx).
			SetHeader("Content-Type", "application/json").
			SetBody(reqBody).     // 可以直接传 struct，resty 会 JSON 编码
			SetResult(&respBody). // 2xx 自动反序列化到 respBody
			Post(url)
		if err != nil {
			logger.Log().Error(fmt.Sprintf("request error: %v", err))
		}
		// 检查 HTTP 状态码
		if resp.IsError() {
			logger.Log().Error(fmt.Sprintf("server returned error: %s, body: %s", resp.Status(), resp.String()))
		}
		time.Sleep(time.Duration(TimeOutSeconds) * time.Second)
	}
	safeDoRequest()
	for {
		select {
		case <-ctx.Done():
			logger.Log().Info("SendAliveMessage context cancelled, exiting goroutine.")
			return
		case <-ticker.C:
			safeDoRequest()
		}
	}
}
