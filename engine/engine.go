package engine

import (
	iface "OnnxDetServer/interface"
	"fmt"
	"os"
	"reflect"
	"strings"
	"unsafe"
)

const UNREGISTERED = 0x0001
const REGISTERED = 0x0002
const IDLE = 0x0003
const BUSY = 0x0004
const SingleThread = 0x1001
const MultiThread = 0x1002

func ReadLinesReadFile(path string) ([]string, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	// 支持 Windows CRLF，去掉尾部的 '\r'
	raw := strings.Split(string(b), "\n")
	for i := range raw {
		raw[i] = strings.TrimRight(raw[i], "\r")
	}
	//TIP 如果文件末尾有空行，可以按需过滤：
	var lines []string
	for _, l := range raw {
		if l != "" {
			lines = append(lines, l)
		}
	}
	return raw, nil
}

type Detector struct {
	ModelPath    string
	Names        []string
	Conf         float32
	Iou          float32
	UseGPU       bool
	Instance     unsafe.Pointer
	State        int
	ErrorMessage string
}

func (d *Detector) New() bool {
	d.Instance = CreateDetector()
	d.State = REGISTERED
	return d.Instance != nil
}

func (d *Detector) CheckConfig() iface.EngineConfig {
	retConfig := iface.EngineConfig{}
	retConfig.ModelPath = d.ModelPath
	retConfig.Conf = d.Conf
	retConfig.Iou = d.Iou
	retConfig.UseGPU = d.UseGPU
	retConfig.Names = iface.NamesConf{
		IsFile: false,
		Data:   d.Names,
	}
	return retConfig
}

func (d *Detector) LoadModel(modelPath string, names iface.NamesConf, conf float32, iou float32, useGPU bool) (bool, error) {
	if names.IsFile {
		d.Names, _ = ReadLinesReadFile(names.Data.(string))
	} else {
		rv := reflect.ValueOf(names.Data)
		if rv.Kind() != reflect.Slice {
			panic("names must be a slice or a file path")
		} else {
			n := rv.Len()
			d.Names = make([]string, n)
			for i := 0; i < n; i++ {
				d.Names[i] = rv.Index(i).Interface().(string)
			}
		}
	}
	d.ModelPath = modelPath
	switch backendCfg.UseBackend {
	case "ncnn":
		if d.ModelPath[len(d.ModelPath)-5:] != ".param" {
			return false, fmt.Errorf("ncnn.LoadModel only supports .param")
		}
	case "onnx":
		if d.ModelPath[len(d.ModelPath)-5:] != ".onnx" {
			return false, fmt.Errorf("onnx.LoadModel only supports .onnx")
		}
	default:
		return false, fmt.Errorf("unsupported backend: %s", backendCfg.UseBackend)
	}
	d.Conf = conf
	d.Iou = iou
	d.UseGPU = useGPU
	d.State = IDLE
	state := InitDetector(d.Instance, d.ModelPath, d.Conf, d.Iou, d.UseGPU)
	return state, nil
}

func (d *Detector) Destroy() {
	DestroyDetector(d.Instance)
	d.ModelPath = ""
	d.Conf = 0
	d.Iou = 0
	d.UseGPU = false
	d.Instance = nil
	d.State = UNREGISTERED
}

func (d *Detector) Detect(img iface.ImageData) iface.RetData {
	switch d.State {
	case UNREGISTERED:
		return iface.RetData{Success: false, Data: "Detector not registered"}
	case REGISTERED:
		return iface.RetData{Success: false, Data: "Model not loaded"}
	case BUSY:
		return iface.RetData{Success: false, Data: "Detector is busy"}
	}
	d.State = BUSY
	imgData := img.Data
	width := img.Width
	height := img.Height
	channels := img.Channels

	resultDict := make(map[string][]iface.Result)
	boxes, scores, classes, _, ok := Detect(d.Instance, imgData, int(width), int(height), int(channels))
	if !ok {
		d.State = IDLE
		return iface.RetData{Success: false, Data: resultDict}
	}
	for item := range d.Names {
		resultDict[d.Names[item]] = []iface.Result{}
	}
	for i := 0; i < len(classes); i++ {
		classIdx := int(classes[i])
		conf := scores[i]
		box := iface.Box{
			LT: iface.Position{X: boxes[i*4], Y: boxes[i*4+1]},
			RT: iface.Position{X: boxes[i*4+2], Y: boxes[i*4+1]},
			RB: iface.Position{X: boxes[i*4+2], Y: boxes[i*4+3]},
			LB: iface.Position{X: boxes[i*4], Y: boxes[i*4+3]},
		}
		center := iface.Position{
			X: (box.LT.X + box.RB.X) / 2,
			Y: (box.LT.Y + box.RB.Y) / 2,
		}
		res := iface.Result{
			Conf:   conf,
			Box:    box,
			Center: center,
		}
		className := d.Names[classIdx]
		resultDict[className] = append(resultDict[className], res)
	}
	d.State = IDLE
	return iface.RetData{Success: true, Data: resultDict}
}

func (d *Detector) SetInputSize(size int) {
	SetInputSize(d.Instance, size)
}

func (d *Detector) SetBlobName(inputName, outputName string) {
	SetBlobName(d.Instance, inputName, outputName)
}
