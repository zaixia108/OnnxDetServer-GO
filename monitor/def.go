package monitor

import (
	"context"
	"errors"
	"fmt"
	"math"
	"net/http"
	"os"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promhttp"
	"github.com/shirou/gopsutil/v4/process"
)

var (
	PID       process.Process
	memUsage  prometheus.Gauge
	cpuUsage  prometheus.Gauge
	GRPCTotal prometheus.Counter
)

var srv *http.Server

func prom(port int) {
	registry := prometheus.NewRegistry()
	memUsage = prometheus.NewGauge(prometheus.GaugeOpts{
		Name: "memory_usage_Megabytes",
		Help: "Memory usage in Megabytes",
	})

	cpuUsage = prometheus.NewGauge(prometheus.GaugeOpts{
		Name: "cpu_usage_percent",
		Help: "CPU usage in percent",
	})

	GRPCTotal = prometheus.NewCounter(prometheus.CounterOpts{
		Name: "grpc_requests_total",
		Help: "Total number of gRPC requests processed",
	})

	registry.MustRegister(memUsage, cpuUsage, GRPCTotal)
	http.Handle("/metrics", promhttp.HandlerFor(registry, promhttp.HandlerOpts{Registry: registry}))
	srv = &http.Server{
		Addr:    fmt.Sprintf(":%d", port),
		Handler: nil,
	}
	go func() {
		if err := srv.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
			fmt.Printf("Prometheus server ListenAndServe error: %v\n", err)
		}
	}()
}

func CheckProcessInfo() {
	MemInfo, _ := PID.MemoryInfo()
	var MemMB = MemInfo.RSS / 1024 / 1024
	CPUPercent, _ := PID.CPUPercent()
	CPUPercentFloat := math.Round(CPUPercent*100) / 100
	memUsage.Set(float64(MemMB))
	cpuUsage.Set(CPUPercentFloat)
}

func GotPID() {
	pid := os.Getpid()
	i32Pid := int32(pid)
	PID.Pid = i32Pid
}

func StartMon(port int, ctx context.Context) {
	PID = process.Process{}
	GotPID()
	go prom(port)
	ticker := time.NewTicker(500 * time.Millisecond)
	defer ticker.Stop()
checkPcs:
	for {
		select {
		case <-ctx.Done():
			break checkPcs
		case <-ticker.C:
			CheckProcessInfo()
		}
	}
	if err := srv.Shutdown(ctx); err != nil {
		fmt.Printf("Prometheus server Shutdown error: %v\n", err)
	}
}
