package main

import (
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"syscall"
	"time"
	"unsafe"
)

func setDllDirectory(dir string) error {
	kernel32 := syscall.NewLazyDLL("kernel32.dll")
	proc := kernel32.NewProc("SetDllDirectoryW")
	ptr, err := syscall.UTF16PtrFromString(dir)
	if err != nil {
		return err
	}
	ret, _, callErr := proc.Call(uintptr(unsafe.Pointer(ptr)))
	if ret == 0 {
		if callErr != nil && callErr != syscall.Errno(0) {
			return callErr
		}
		return fmt.Errorf("SetDllDirectoryW returned 0")
	}
	return nil
}

// searchLocations tries to find the target executable in a set of sensible places:
// - the directory containing the launcher executable
// - the current working directory
// - ".dist" sibling/child directories
// - ascending parent directories (up to a limit)
// It returns the full path if found, or an error with diagnostic info.
func searchLocations(preferred string) (string, error) {
	var tried []string

	// 1) exe dir
	exePath, err := os.Executable()
	if err == nil {
		exeDir := filepath.Dir(exePath)
		tried = append(tried, exeDir)
		if p := filepath.Join(exeDir, preferred); fileExists(p) {
			return p, nil
		}
		// try glob in exeDir
		if m := globFirst(exeDir, "OnnxDetServer*.exe"); m != "" {
			return m, nil
		}
		// try exeDir/.dist
		dist := filepath.Join(exeDir, ".dist")
		tried = append(tried, dist)
		if p := filepath.Join(dist, preferred); fileExists(p) {
			return p, nil
		}
		if m := globFirst(dist, "OnnxDetServer*.exe"); m != "" {
			return m, nil
		}
	}

	// 2) cwd
	if cwd, err := os.Getwd(); err == nil {
		tried = append(tried, cwd)
		if p := filepath.Join(cwd, preferred); fileExists(p) {
			return p, nil
		}
		if m := globFirst(cwd, "OnnxDetServer*.exe"); m != "" {
			return m, nil
		}
		dist := filepath.Join(cwd, ".dist")
		tried = append(tried, dist)
		if p := filepath.Join(dist, preferred); fileExists(p) {
			return p, nil
		}
		if m := globFirst(dist, "OnnxDetServer*.exe"); m != "" {
			return m, nil
		}
	}

	// 3) ascend parents from cwd and exeDir
	parentsChecked := make(map[string]bool)
	checkAscend := func(start string) (string, bool) {
		cur := start
		for i := 0; i < 10; i++ {
			if cur == "" || parentsChecked[cur] {
				break
			}
			parentsChecked[cur] = true
			tried = append(tried, cur)
			if p := filepath.Join(cur, preferred); fileExists(p) {
				return p, true
			}
			if m := globFirst(cur, "OnnxDetServer*.exe"); m != "" {
				return m, true
			}
			// also check cur/.dist
			if p := filepath.Join(cur, ".dist", preferred); fileExists(p) {
				return p, true
			}
			if m := globFirst(filepath.Join(cur, ".dist"), "OnnxDetServer*.exe"); m != "" {
				return m, true
			}
			parent := filepath.Dir(cur)
			if parent == cur {
				break
			}
			cur = parent
		}
		return "", false
	}

	if exePath, err := os.Executable(); err == nil {
		if res, ok := checkAscend(filepath.Dir(exePath)); ok {
			return res, nil
		}
	}
	if cwd, err := os.Getwd(); err == nil {
		if res, ok := checkAscend(cwd); ok {
			return res, nil
		}
	}

	// not found — prepare diagnostic message listing tried locations and directory contents for the launcher dir and .dist if present
	diag := "Tried locations:\n"
	for _, t := range tried {
		diag += "  - " + t + "\n"
	}
	// include listing of likely dirs to help debug
	diag += "\nDirectory listings (if available):\n"
	addListing := func(d string) {
		diag += "Listing: " + d + "\n"
		if entries, err := os.ReadDir(d); err == nil {
			for _, e := range entries {
				info, _ := e.Info()
				diag += fmt.Sprintf("  %s (dir=%v size=%d)\n", e.Name(), e.IsDir(), info.Size())
			}
		} else {
			diag += "  (cannot read or not exists)\n"
		}
	}
	// attempt to list launcher exe dir and cwd
	if exePath, err := os.Executable(); err == nil {
		addListing(filepath.Dir(exePath))
	}
	if cwd, err := os.Getwd(); err == nil {
		addListing(cwd)
		if d := filepath.Join(cwd, ".dist"); d != "" {
			addListing(d)
		}
	}
	return "", fmt.Errorf("executable %q not found. %s", preferred, diag)
}

func fileExists(p string) bool {
	if p == "" {
		return false
	}
	info, err := os.Stat(p)
	return err == nil && !info.IsDir()
}

func globFirst(dir, pat string) string {
	if dir == "" {
		return ""
	}
	glob := filepath.Join(dir, pat)
	ms, err := filepath.Glob(glob)
	if err != nil || len(ms) == 0 {
		return ""
	}
	return ms[0]
}

func main() {
	preferredName := "OnnxDetServer-GO.exe"

	// Print basic environment info
	if exePath, err := os.Executable(); err == nil {
		fmt.Printf("Launcher executable: %s\n", exePath)
	} else {
		fmt.Printf("Launcher executable: (error: %v)\n", err)
	}
	if cwd, err := os.Getwd(); err == nil {
		fmt.Printf("Current working directory: %s\n", cwd)
	}

	// Determine and set DLL directory (we'll try exeDir/src and cwd\src)
	exeDir := ""
	if exePath, err := os.Executable(); err == nil {
		exeDir = filepath.Dir(exePath)
	}
	var dllCandidates []string
	if exeDir != "" {
		dllCandidates = append(dllCandidates, filepath.Join(exeDir, "src"))
		dllCandidates = append(dllCandidates, filepath.Join(exeDir, ".dist", "src"))
	}
	if cwd, err := os.Getwd(); err == nil {
		dllCandidates = append(dllCandidates, filepath.Join(cwd, "src"))
		dllCandidates = append(dllCandidates, filepath.Join(cwd, ".dist", "src"))
	}

	foundDllDir := ""
	for _, d := range dllCandidates {
		if d == "" {
			continue
		}
		if info, err := os.Stat(d); err == nil && info.IsDir() {
			foundDllDir = d
			break
		}
	}
	if foundDllDir != "" {
		fmt.Printf("Attempting to set DLL directory to: %s\n", foundDllDir)
		if err := setDllDirectory(foundDllDir); err != nil {
			fmt.Fprintf(os.Stderr, "warning: SetDllDirectory failed: %v\n", err)
		} else {
			fmt.Println("SetDllDirectory succeeded")
		}
	} else {
		fmt.Println("No src Dll directory found in candidate locations; will continue without SetDllDirectory")
	}

	// Find target exe
	target, err := searchLocations(preferredName)
	if err != nil {
		fmt.Fprintf(os.Stderr, "failed to locate target executable: %v\n", err)
		os.Exit(1)
	}
	fmt.Printf("Starting target: %s\n", target)

	// Exec the target
	cmdOnnx := exec.Command(target, os.Args[1:]...)
	cmdOnnx.Stdout = os.Stdout
	cmdOnnx.Stderr = os.Stderr
	cmdOnnx.Stdin = os.Stdin

	if err := cmdOnnx.Run(); err != nil {
		if exitErr, ok := err.(*exec.ExitError); ok {
			os.Exit(exitErr.ExitCode())
		}
		fmt.Fprintf(os.Stderr, "failed to start target: %v\n", err)
		time.Sleep(10 * time.Second) // give some time to see the error
		os.Exit(1)
	}
}
