package iface

import "gocv.io/x/gocv"

type RetData struct {
	Success bool
	Data    map[string][]Result
}

type EngineConfig struct {
	ModelPath string
	Names     []string
	Conf      float32
	Iou       float32
	UseGPU    bool
	UseFp16   bool
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
	LoadModel(modelPath string, names []string, conf float32, iou float32, useGPU bool, useFp16 bool) bool
	Detect(image gocv.Mat) RetData
	Destroy()
	CheckConfig() EngineConfig
	SetInputSize(size int)
}
