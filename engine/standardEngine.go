package engine

import (
	iface "OnnxDetServer/interface"
	"unsafe"

	"gocv.io/x/gocv"
)

type Detector struct {
	ModelPath string
	Names     []string
	Conf      float32
	Iou       float32
	UseGPU    bool
	UseFp16   bool
	Instance  unsafe.Pointer
}

func (d *Detector) New() bool {
	d.Instance = CreateDetector()
	return d.Instance != nil
}

func (d *Detector) CheckConfig() iface.EngineConfig {
	retConfig := iface.EngineConfig{}
	retConfig.ModelPath = d.ModelPath
	retConfig.Conf = d.Conf
	retConfig.Iou = d.Iou
	retConfig.UseGPU = d.UseGPU
	retConfig.Names = d.Names
	return retConfig
}

func (d *Detector) LoadModel(modelPath string, names []string, conf float32, iou float32, useGPU bool, useFp16 bool) bool {
	d.Names = names
	d.ModelPath = modelPath
	d.Conf = conf
	d.Iou = iou
	d.UseGPU = useGPU
	d.UseFp16 = useFp16
	state := InitDetector(d.Instance, d.ModelPath, d.Conf, d.Iou, d.UseGPU, d.UseFp16)
	return state
}

func (d *Detector) Destroy() {
	DestroyDetector(d.Instance)
	d.ModelPath = ""
	d.Conf = 0
	d.Iou = 0
	d.UseGPU = false
	d.Instance = nil
}

func (d *Detector) Detect(img gocv.Mat) iface.RetData {
	imgData := img.ToBytes()
	width := img.Cols()
	height := img.Rows()
	channels := img.Channels()

	resultDict := make(map[string][]iface.Result)

	boxes, scores, classes, _, ok := Detect(d.Instance, imgData, width, height, channels)
	if !ok {
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
	return iface.RetData{Success: true, Data: resultDict}
}

func (d *Detector) SetInputSize(size int) {
	SetDefaultInputSize(size)
}
