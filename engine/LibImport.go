package engine

/*
#cgo CFLAGS: -Isrc
#cgo LDFLAGS: -L/usr/loacl/lib -lNcnnDet
#include "NcnnDet.h"
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
	// 修改点 1: 去掉 (*C.Detector) 转换，直接传 unsafe.Pointer
	C.DestroyDetector(p)
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
	// 修改点 1: 去掉 (*C.Detector) 转换
	ret := C.InitDetector(p, cModelPath, C.float(conf), C.float(iou), ug)
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

	// 修改点 1: 去掉 (*C.Detector) 转换
	ret := C.Detect(
		detector,
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

	// 修改点 2: FreeMemory -> ReleaseResults
	// 注意：ReleaseResults 需要三个参数，根据头文件定义：
	// void ReleaseResults(float* boxes, float* scores, int* classes);
	C.ReleaseResults(outBoxesPtr, outScoresPtr, outClassesPtr)

	return
}

func SetInputSize(detector unsafe.Pointer, size int) {
	if detector == nil {
		return
	}
	// 修改点 1: 去掉 (*C.Detector) 转换
	C.SetInputSize(detector, C.int(size))
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

	// 修改点 1: 去掉 (*C.Detector) 转换
	// 修改点 3: SetBlobName -> SetBlobNames (复数)
	C.SetBlobNames(detector, cInputName, cOutputName)
	return
}
