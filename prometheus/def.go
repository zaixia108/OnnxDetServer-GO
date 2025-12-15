package main

import (
	"fmt"
	"io"
	"net/http"
	"strconv"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promhttp"
	"github.com/shirou/gopsutil/process"
)

var (
	PID      process.Process
	memUsage prometheus.Gauge
	cpuUsage prometheus.Gauge
)

func prom() {
	registry := prometheus.NewRegistry()

	memUsage = prometheus.NewGauge(prometheus.GaugeOpts{
		Name: "memory_usage_Megabytes",
		Help: "Memory usage in Megabytes",
	})

	cpuUsage = prometheus.NewGauge(prometheus.GaugeOpts{
		Name: "cpu_usage_percent",
		Help: "CPU usage in percent",
	})

	registry.MustRegister(memUsage, cpuUsage)

	http.Handle("/metrics", promhttp.HandlerFor(registry, promhttp.HandlerOpts{Registry: registry}))
	http.ListenAndServe(":50053", nil)
}

func CheckProcessInfo() {
	for {
		running, err := PID.IsRunning()
		if err != nil {
			panic(err)
		}
		if !running {
			fmt.Println("Main Process has exited.")
			time.Sleep(3 * time.Second)
			break
		} else {
			MemInfo, _ := PID.MemoryInfo()
			var MemMB = MemInfo.RSS / 1024 / 1024
			CPUPercent, _ := PID.CPUPercent()
			CPUPercentFloat, _ := strconv.ParseFloat(fmt.Sprintf("%.2f", CPUPercent), 64)
			memUsage.Set(float64(MemMB))
			cpuUsage.Set(CPUPercentFloat)
			time.Sleep(500 * time.Millisecond)
		}
	}

}

func GotPID() {
	resp, err := http.Get("http://127.0.0.1:50054/getpid")
	if err != nil {
		panic(err)
	}
	defer resp.Body.Close()
	content, _ := io.ReadAll(resp.Body)
	fmt.Printf("Got PID: %s\n", string(content))
	var pid int32
	fmt.Sscanf(string(content), "%d", &pid)
	PID.Pid = pid
}

func main() {
	PID = process.Process{}
	GotPID()
	go prom()
	time.Sleep(1 * time.Second)
	CheckProcessInfo()
}
