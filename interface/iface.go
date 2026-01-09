package iface

import "gocv.io/x/gocv"

type NamesConf struct {
	IsFile bool
	Data   any
}

type RetData struct {
	Success bool
	Data    any
}

type EngineConfig struct {
	UseGPU    bool
	ModelPath string
	Names     NamesConf
	Conf      float32
	Iou       float32
}

type Position struct {
	X, Y float32
}

type Box struct {
	LT Position
	RT Position
	RB Position
	LB Position
}

type Result struct {
	Conf   float32
	Box    Box
	Center Position
}

type Backend interface {
	LoadModel(modelPath string, names NamesConf, conf float32, iou float32, useGPU bool) bool
	Detect(image gocv.Mat) RetData
	Destroy()
	CheckConfig() EngineConfig
	SetInputSize(size int)
	SetBlobName(inputName, outputName string)
}
