mkdir .release

go build -o .release/launcher.exe .\launcher.go
go build -o .release/OnnxDetServer-Go.exe .\OnnxDetServer-Go.go
go build -o .release/mon-onnx.exe .\prometheus\def.go

xcopy "src" ".release\src" /E /I /H /Y
xcopy "models" ".release\models" /E /I /H /Y
copy /Y "config.yaml" ".release\config.yaml"