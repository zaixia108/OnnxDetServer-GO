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
