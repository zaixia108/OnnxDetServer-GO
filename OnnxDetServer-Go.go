package main

import (
	adhoc "OnnxDetServer/Adhoc"
	"OnnxDetServer/engine"
	backend "OnnxDetServer/gRPC"
	"OnnxDetServer/logger"
	"OnnxDetServer/monitor"
	"context"
	"fmt"
	"net"
	"os"
	"runtime"
	"strings"
	"sync"

	"gopkg.in/yaml.v3"
)

type configStruct struct {
	RPCPort          int    `yaml:"RPCPort"`
	AdhocPort        int    `yaml:"AdhocPort"`
	WorkersNum       int    `yaml:"workersNum"`
	InstanceClass    string `yaml:"instanceClass"`
	UseRegServer     bool   `yaml:"UseRegServer"`
	RegServerPort    int    `yaml:"RegServerPort"`
	RegServerHost    string `yaml:"RegServerHost"`
	InferenceBackend string `yaml:"InferenceBackend"`
}

func GetOutboundIP() (string, error) {
	// 8.8.8.8 是 Google DNS，这里只是为了建立路由路径得到本地出口 IP
	// 实际并没有真正的物理连接，所以不需要联网也可以（只要有路由表）
	conn, err := net.Dial("udp", "8.8.8.8:80")
	if err != nil {
		return "", err
	}
	defer conn.Close()

	localAddr := conn.LocalAddr().(*net.UDPAddr)

	return localAddr.IP.String(), nil
}

func main() {
	ip, err := GetOutboundIP()
	if err != nil {
		fmt.Println("Failed to get outbound IP:", err)
		return
	} else {
		fmt.Println("Outbound IP:", ip)
	}
	var wg sync.WaitGroup
	err = logger.InitProduction()
	if err != nil {
		return
	}
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
	fmt.Println(" gRPC  Port:", config.RPCPort)
	fmt.Println(" Adhoc Port:", config.AdhocPort)
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

	engine.LoadEngine(config.InferenceBackend)

	var InstanceClass int
	switch config.InstanceClass {
	case "Dml":
		InstanceClass = adhoc.DmlInstance
	case "Cuda":
		InstanceClass = adhoc.CudaInstance
	case "Rocm":
		InstanceClass = adhoc.RocmInstance
	case "Cpu":
		InstanceClass = adhoc.CpuInstance
	default:
		fmt.Println("Invalid instanceClass in config, defaulting to Cpu")
		InstanceClass = adhoc.CpuInstance
	}
	adhoc.RegServerCfg = adhoc.RegServerConfig{}
	adhoc.RegServerCfg.SetAddress(config.RegServerHost, config.RegServerPort)
	backend.JobQueue = make(chan backend.JobPackage, config.WorkersNum)
	backend.StartWorker(config.WorkersNum)
	backend.DSequences = make(map[string]backend.WorkerID)
	//Adhoc server setup
	ctx, cancel := context.WithCancel(context.Background())
	wg.Add(1)
	if config.UseRegServer {
		go adhoc.SendAliveMessage(ip, config.RPCPort, InstanceClass, ctx, &wg)
	} else {
		fmt.Println("UseRegServer is set to false, skipping registration")
		wg.Done()
	}
	//gRPC server setup
	fmt.Println("Starting gRPC Server")
	server := backend.StartGRPCServer(config.RPCPort)
	go monitor.StartMon(config.AdhocPort, ctx)
	<-backend.CloseChannel
	cancel()
	server.GracefulStop()
	fmt.Println("Done")
	wg.Wait()
	fmt.Println("Safely exited")
}
