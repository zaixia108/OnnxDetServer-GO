package engine

import (
	iface "OnnxDetServer/interface"
	"fmt"
)

type BackendInstance struct {
	Instance iface.Backend
}

func (d *BackendInstance) LoadBackend(instance iface.Backend) {
	d.Instance = instance
}

func (d *BackendInstance) UnloadBackend() {
	d.Instance.Destroy()
	d.Instance = nil
}

var Instance BackendInstance

func LoadEngine(engineType string) {
	fmt.Printf("Initializing engine of type: %s\n", engineType)
	switch engineType {
	case "onnx":
		fmt.Println("ONNX engine selected.")
		InitDll("src/onnx", "OnnxDet.dll")
		newInstance := Detector{}
		Instance.LoadBackend(&newInstance)
		// Initialize ONNX engine here
	case "ncnn":
		fmt.Println("NCNN engine selected.")
		InitDll("src/ncnn", "NcnnDet.dll")
		// Initialize NCNN engine here
		// backend.LoadBackend(&ncnn.Detector{})
		// Assuming ncnn.Detector implements iface.Backend
		newInstance := Detector{}
		Instance.LoadBackend(&newInstance)
	default:
		fmt.Println("Not Support Now. Defaulting to ONNX.")
		InitDll("src/onnx", "OnnxDet.dll")
		// Initialize default engine here
		newInstance := Detector{}
		Instance.LoadBackend(&newInstance)
	}
}
