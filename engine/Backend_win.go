package engine

import (
	"errors"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"runtime"
	"syscall"
	"time"
	"unsafe"

	"gopkg.in/yaml.v3"
)

type BackendConfig struct {
	UseBackend     string `yaml:"useBackend"`
	BackendDir     string `yaml:"backendDir"`
	BackendLibName string `yaml:"backendLibName"`
}

var backendCfg BackendConfig

func loadBackendWithDepsWin64(dllDir, dllName string) (*syscall.LazyDLL, error) {
	k32 := syscall.NewLazyDLL("kernel32.dll")
	procSetDllDirectoryW := k32.NewProc("SetDllDirectoryW")
	ptr, err := syscall.UTF16PtrFromString(dllDir)
	if err != nil {
		return nil, err
	}
	ret, _, callErr := procSetDllDirectoryW.Call(uintptr(unsafe.Pointer(ptr)))
	if ret == 0 {
		old := os.Getenv("PATH")
		_ = os.Setenv("PATH", dllDir+";"+old)
		if callErr != nil && !errors.Is(callErr, syscall.Errno(0)) {
			return nil, fmt.Errorf("SetDllDirectoryW failed: %v", callErr)
		}
	}
	dllPath := filepath.Join(dllDir, dllName)
	mod := syscall.NewLazyDLL(dllPath)
	if err := mod.Load(); err != nil {
		return nil, fmt.Errorf("load %s failed: %w", dllPath, err)
	}
	return mod, nil
}

var (
	mod                *syscall.LazyDLL
	procCreate         *syscall.LazyProc
	procDestroy        *syscall.LazyProc
	procInit           *syscall.LazyProc
	procDetect         *syscall.LazyProc
	procReleaseResults *syscall.LazyProc
	procSetInputSize   *syscall.LazyProc
	procSetBlobName    *syscall.LazyProc
)

func detArch(system, arch string) string {
	switch arch {
	case "amd64":
		{
			return fmt.Sprintf("%s-%s", system, "x64")
		}
	case "386":
		{
			return fmt.Sprintf("%s-%s", system, "x86")
		}
	case "arm64":
		{
			return fmt.Sprintf("%s-%s", system, "arm64")
		}
	default:
		panic(fmt.Sprintf("Architecture %s not supported", arch))
	}
}

func getPlatform() string {
	system := runtime.GOOS
	arch := runtime.GOARCH
	switch system {
	case "windows":
		return detArch(system, arch)
	case "darwin":
		panic("MacOS is not supported")
	case "linux":
		return detArch(system, arch)
	default:
		panic(fmt.Sprintf("Operating system %s not supported", system))
	}
}

func init() {
	var err error
	Platform := getPlatform()
	// 获取可执行文件的路径
	configData, err := os.ReadFile("src/backend.yaml")
	if err != nil {
		fmt.Fprintf(os.Stderr, "failed to read backend.yaml: %v\n", err)
		time.Sleep(5 * time.Second)
		os.Exit(1)
	}
	err = yaml.Unmarshal(configData, &backendCfg)
	if err != nil {
		fmt.Fprintf(os.Stderr, "failed to parse backend.yaml: %v\n", err)
		time.Sleep(5 * time.Second)
		os.Exit(1)
	}
	exePath, err := os.Executable()
	if err != nil {
		fmt.Fprintf(os.Stderr, "failed to get executable path: %v\n", err)
		os.Exit(1)
	}
	// 基于可执行文件路径构建 'src' 目录的绝对路径
	exeDir := filepath.Dir(exePath)
	dllDir := filepath.Join(exeDir, backendCfg.BackendDir)
	if Platform == "linux-x64" {
		panic("Linux is not supported yet, developing...")
	} else {
		mod, err = loadBackendWithDepsWin64(dllDir, backendCfg.BackendLibName)
		if err != nil {
			fmt.Fprintf(os.Stderr, "failed to load DLLs from '%s': %v\nEnsure `src` directory with DLLs exists next to the executable, and install Visual C++ Redistributable.\n", dllDir, err)
			os.Exit(1)
		}
	}
	fmt.Println("Lib Loaded...")
	procCreate = mod.NewProc("CreateDetector")
	procDestroy = mod.NewProc("DestroyDetector")
	procInit = mod.NewProc("InitDetector")
	procDetect = mod.NewProc("Detect")
	procReleaseResults = mod.NewProc("ReleaseResults")
	if backendCfg.UseBackend == "ncnn" {
		procSetInputSize = mod.NewProc("SetInputSize")
		procSetBlobName = mod.NewProc("SetBlobName")
	}
}

func CreateDetector() unsafe.Pointer {
	if procCreate == nil {
		return nil
	}
	r, _, _ := procCreate.Call()
	return unsafe.Pointer(r)
}

func DestroyDetector(p unsafe.Pointer) {
	if p == nil || procDestroy == nil {
		return
	}
	procDestroy.Call(uintptr(p))
}

func InitDetector(p unsafe.Pointer, modelPath string, conf, iou float32, useGPU bool) bool {
	if p == nil || procInit == nil {
		return false
	}
	mp, _ := syscall.BytePtrFromString(modelPath)
	var ug uintptr
	if useGPU {
		ug = 1
	}
	r, _, _ := procInit.Call(
		uintptr(p),
		uintptr(unsafe.Pointer(mp)),
		uintptr(math.Float32bits(conf)),
		uintptr(math.Float32bits(iou)),
		ug,
	)
	return r != 0
}

func Detect(detector unsafe.Pointer, imageData []byte, width, height, channels int) (boxes []float32, scores []float32, classes []int32, count int32, ok bool) {
	if detector == nil || len(imageData) == 0 || procDetect == nil {
		return
	}

	var outBoxesPtr, outScoresPtr, outClassesPtr uintptr
	var outCount int32

	r, _, _ := procDetect.Call(
		uintptr(detector),
		uintptr(unsafe.Pointer(&imageData[0])),
		uintptr(width),
		uintptr(height),
		uintptr(channels),
		uintptr(unsafe.Pointer(&outBoxesPtr)),
		uintptr(unsafe.Pointer(&outScoresPtr)),
		uintptr(unsafe.Pointer(&outClassesPtr)),
		uintptr(unsafe.Pointer(&outCount)),
	)
	ok = r != 0
	count = outCount
	if !ok || count == 0 {
		return
	}

	tmpBoxes := unsafe.Slice((*float32)(unsafe.Pointer(outBoxesPtr)), int(count*4))
	tmpScores := unsafe.Slice((*float32)(unsafe.Pointer(outScoresPtr)), int(count))
	tmpClasses := unsafe.Slice((*int32)(unsafe.Pointer(outClassesPtr)), int(count))

	boxes = append([]float32(nil), tmpBoxes...)
	scores = append([]float32(nil), tmpScores...)
	classes = append([]int32(nil), tmpClasses...)

	if procReleaseResults != nil {
		procReleaseResults.Call(outBoxesPtr, outScoresPtr, outClassesPtr)
	}
	return
}

func SetInputSize(detector unsafe.Pointer, size int) {
	if detector == nil || procSetInputSize == nil {
		return
	}
	_, _, _ = procSetInputSize.Call(
		uintptr(detector),
		uintptr(size),
	)
	return
}

func SetBlobName(detector unsafe.Pointer, inputBlobName, outputBlobName string) {
	if detector == nil || procSetBlobName == nil {
		return
	}
	inBlobPtr, _ := syscall.BytePtrFromString(inputBlobName)
	outBlobPtr, _ := syscall.BytePtrFromString(outputBlobName)
	_, _, _ = procSetBlobName.Call(
		uintptr(detector),
		uintptr(unsafe.Pointer(inBlobPtr)),
		uintptr(unsafe.Pointer(outBlobPtr)),
	)
	return
}
