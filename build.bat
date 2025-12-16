mkdir .release

go build -o .release/launcher.exe .\launcher.go
go build -o .release/OnnxDetServer-Go.exe .\OnnxDetServer-Go.go

xcopy "src" ".release\src" /E /I /H /Y
xcopy "models" ".release\models" /E /I /H /Y
copy /Y "config.yaml" ".release\config.yaml"