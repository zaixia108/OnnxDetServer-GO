package iface

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
