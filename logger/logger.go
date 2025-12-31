package logger

import (
	"sync"

	"go.uber.org/zap"
	"go.uber.org/zap/zapcore"
)

var (
	logMu sync.RWMutex
	log   *zap.Logger
	sugar *zap.SugaredLogger
)

// InitProduction 初始化一个 production logger（供 main 调用）
func InitProduction() error {
	cfg := zap.NewProductionConfig()
	cfg.EncoderConfig.TimeKey = "timestamp"
	cfg.EncoderConfig.EncodeTime = zapcore.ISO8601TimeEncoder
	l, err := cfg.Build()
	if err != nil {
		return err
	}
	setLogger(l)
	return nil
}

// InitDevelopment 初始化一个 development logger（更友好地输出到控制台）
func InitDevelopment() error {
	cfg := zap.NewDevelopmentConfig()
	cfg.EncoderConfig.TimeKey = "timestamp"
	cfg.EncoderConfig.EncodeTime = zapcore.ISO8601TimeEncoder
	l, err := cfg.Build()
	if err != nil {
		return err
	}
	setLogger(l)
	return nil
}

// setLogger 内部设置并替换 zap 全局 logger
func setLogger(l *zap.Logger) {
	logMu.Lock()
	defer logMu.Unlock()
	// 替换 zap 全局（可使 zap.L()/zap.S() 返回相同实例）
	zap.ReplaceGlobals(l)
	// 保存实例以便通过本包访问
	if log != nil {
		_ = log.Sync()
	}
	log = l
	sugar = l.Sugar()
}

// Log 返回 *zap.Logger（非 nil）
func Log() *zap.Logger {
	logMu.RLock()
	defer logMu.RUnlock()
	if log != nil {
		return log
	}
	// 如果还没初始化，返回 zap 的全局（可能是 noop）
	return zap.L()
}

// S 返回 *zap.SugaredLogger（非 nil）
func S() *zap.SugaredLogger {
	logMu.RLock()
	defer logMu.RUnlock()
	if sugar != nil {
		return sugar
	}
	return zap.S()
}

// Sync flush logs
func Sync() {
	logMu.RLock()
	defer logMu.RUnlock()
	if log != nil {
		_ = log.Sync()
	}
}
