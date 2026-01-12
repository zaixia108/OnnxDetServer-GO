package engine

import (
	iface "OnnxDetServer/interface"
	"testing"
	"unsafe"

	"github.com/stretchr/testify/assert"
)

func TestDetector_All(t *testing.T) {
	model_path := "model/test_model.param"
	names := iface.NamesConf{
		IsFile: false,
		Data:   []string{"person", "car", "bicycle"},
	}
	conf := float32(0.5)
	iou := float32(0.4)

	d := &Detector{}

	t.Run("Test New", func(t *testing.T) {
		if !d.New() {
			t.Errorf("Detector.New() failed, expected true, got false")
		}
	})

	t.Run("Test LoadModel", func(t *testing.T) {
		state, err := d.LoadModel(
			model_path,
			names,
			conf,
			iou,
			false,
		)
		if err != nil {
			t.Errorf("Detector.LoadModel() returned an error: %v", err)
		}
		assert.Equal(t, state, true)
	})

	t.Run("Test CheckModel", func(t *testing.T) {
		config := d.CheckConfig()
		assert.Equal(t, IDLE, d.State)
		assert.Equal(t, model_path, config.ModelPath)
		assert.Equal(t, conf, config.Conf)
		assert.Equal(t, iou, config.Iou)
		assert.Equal(t, false, config.UseGPU)
		assert.Equal(t, false, config.Names.IsFile)
		assert.Equal(t, []string{"person", "car", "bicycle"}, config.Names.Data)
	})

	t.Run("Test Detect", func(t *testing.T) {
		d.SetInputSize(1280)
		//读取本地文件并转换为字节流

		//fmt.Println(result)
		//assert.Equal(t, true, result.Success)
	})

	t.Run("Test Destroy", func(t *testing.T) {
		d.Destroy()
		assert.Equal(t, d.ModelPath, "")
		assert.Equal(t, d.Conf, float32(0))
		assert.Equal(t, d.Iou, float32(0))
		assert.Equal(t, d.UseGPU, false)
		assert.Equal(t, d.Instance, unsafe.Pointer(nil))
		assert.Equal(t, d.State, UNREGISTERED)
	})
}
