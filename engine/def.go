package engine

import (
	"os"
	"reflect"
	"strings"
	"unsafe"

	"gocv.io/x/gocv"
)

const UNREGISTERED = 0x0001
const REGISTERED = 0x0002
const IDLE = 0x0003
const BUSY = 0x0004
const ERROR = 0x0005
const SingleThread = 0x1001
const MultiThread = 0x1002

type Position struct {
	X, Y float32
}

type Box struct {
	LT Position
	RT Position
	RB Position
	LB Position
}

type result struct {
	conf   float32
	box    Box
	center Position
}

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
	names        []string
	Conf         float32
	Iou          float32
	UseGPU       bool
	instance     unsafe.Pointer
	State        int
	ErrorMessage string
}

func (d *Detector) New() bool {
	d.instance = CreateDetector()
	d.State = REGISTERED
	return d.instance != nil
}

type NamesConf struct {
	IsFile bool
	Data   any
}

func (d *Detector) LoadModel(modelPath string, names NamesConf, conf float32, iou float32, useGPU bool) bool {
	if names.IsFile {
		d.names, _ = ReadLinesReadFile(names.Data.(string))
	} else {
		rv := reflect.ValueOf(names.Data)
		if rv.Kind() != reflect.Slice {
			panic("names must be a slice or a file path")
		} else {
			n := rv.Len()
			d.names = make([]string, n)
			for i := 0; i < n; i++ {
				d.names[i] = rv.Index(i).Interface().(string)
			}
		}
	}
	d.ModelPath = modelPath
	d.Conf = conf
	d.Iou = iou
	d.UseGPU = useGPU
	d.State = IDLE
	state := InitDetector(d.instance, d.ModelPath, d.Conf, d.Iou, d.UseGPU)
	return state
}

func (d *Detector) Destroy() {
	DestroyDetector(d.instance)
	d.ModelPath = ""
	d.Conf = 0
	d.Iou = 0
	d.UseGPU = false
	d.instance = nil
	d.State = UNREGISTERED
}

type RetData struct {
	success bool
	data    any
}

func (d *Detector) Detect(img gocv.Mat) RetData {
	switch d.State {
	case UNREGISTERED:
		return RetData{success: false, data: "Detector not registered"}
	case REGISTERED:
		return RetData{success: false, data: "Model not loaded"}
	case BUSY:
		return RetData{success: false, data: "Detector is busy"}
	}
	d.State = BUSY
	imgData := img.ToBytes()
	width := img.Cols()
	height := img.Rows()
	channels := img.Channels()

	boxes, scores, classes, _, ok := Detect(d.instance, imgData, width, height, channels)
	if !ok {
		d.State = IDLE
		return RetData{success: false, data: "Detection failed"}
	}
	resultDict := make(map[string][]result)
	for item := range d.names {
		resultDict[d.names[item]] = []result{}
	}
	for i := 0; i < len(classes); i++ {
		classIdx := int(classes[i])
		conf := scores[i]
		box := Box{
			LT: Position{X: boxes[i*4], Y: boxes[i*4+1]},
			RT: Position{X: boxes[i*4+2], Y: boxes[i*4+1]},
			RB: Position{X: boxes[i*4+2], Y: boxes[i*4+3]},
			LB: Position{X: boxes[i*4], Y: boxes[i*4+3]},
		}
		center := Position{
			X: (box.LT.X + box.RB.X) / 2,
			Y: (box.LT.Y + box.RB.Y) / 2,
		}
		res := result{
			conf:   conf,
			box:    box,
			center: center,
		}
		className := d.names[classIdx]
		resultDict[className] = append(resultDict[className], res)
	}
	d.State = IDLE
	return RetData{success: true, data: resultDict}
}
