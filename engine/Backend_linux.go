package engine

/*
#CGO_ENABLED=1
#cgo CFLAGS: -Isrc
#cgo LDFLAGS: -Lsrc -lNcnnDet
#include "src/NcnnDet.h"
*/
import "C"
import "unsafe"

func CreateDetector() unsafe.Pointer {
	detector := C.CreateDetector()
	return unsafe.Pointer(detector)
}

func DestroyDetector(p unsafe.Pointer) {
	if p == nil {
		return
	}
	C.DestroyDetector((*C.Detector)(p))
}

func InitDetector(p unsafe.Pointer, modelPath string, conf, iou float32, useGPU bool) bool {
	if p == nil {
		return false
	}
	cModelPath := C.CString(modelPath)
	defer C.free(unsafe.Pointer(cModelPath))
	var ug C.bool
	if useGPU {
		ug = C.bool(true)
	} else {
		ug = C.bool(false)
	}
	ret := C.InitDetector((*C.Detector)(p), cModelPath, C.float(conf), C.float(iou), ug)
	return bool(ret)
}

func Detect(detector unsafe.Pointer, imageData []byte, width, height, channels int) (boxes []float32, scores []float32, classes []int32, count int32, ok bool) {
	if detector == nil || len(imageData) == 0 {
		return
	}
	var outBoxesPtr *C.float
	var outScoresPtr *C.float
	var outClassesPtr *C.int
	var outCount C.int

	ret := C.Detect(
		(*C.Detector)(detector),
		(*C.uchar)(unsafe.Pointer(&imageData[0])),
		C.int(width),
		C.int(height),
		C.int(channels),
		&outBoxesPtr,
		&outScoresPtr,
		&outClassesPtr,
		&outCount,
	)
	ok = bool(ret)
	count = int32(outCount)
	if !ok || count == 0 {
		return
	}

	boxes = make([]float32, count*4)
	scores = make([]float32, count)
	classes = make([]int32, count)

	boxesSlice := unsafe.Slice((*C.float)(unsafe.Pointer(outBoxesPtr)), int(count*4))
	scoresSlice := unsafe.Slice((*C.float)(unsafe.Pointer(outScoresPtr)), int(count))
	classesSlice := unsafe.Slice((*C.int)(unsafe.Pointer(outClassesPtr)), int(count))

	for i := 0; i < int(count); i++ {
		scores[i] = float32(scoresSlice[i])
		classes[i] = int32(classesSlice[i])
		for j := 0; j < 4; j++ {
			boxes[i*4+j] = float32(boxesSlice[i*4+j])
		}
	}

	// 释放 C 语言分配的内存
	C.FreeMemory(unsafe.Pointer(outBoxesPtr))
	C.FreeMemory(unsafe.Pointer(outScoresPtr))
	C.FreeMemory(unsafe.Pointer(outClassesPtr))

	return
}

func SetInputSize(detector unsafe.Pointer, size int) {
	if detector == nil {
		return
	}
	C.SetInputSize((*C.Detector)(detector), C.int(size))
	return
}

func SetBlobName(detector unsafe.Pointer, inputName, outputName string) {
	if detector == nil {
		return
	}
	cInputName := C.CString(inputName)
	defer C.free(unsafe.Pointer(cInputName))
	cOutputName := C.CString(outputName)
	defer C.free(unsafe.Pointer(cOutputName))
	C.SetBlobName((*C.Detector)(detector), cInputName, cOutputName)
	return
}
