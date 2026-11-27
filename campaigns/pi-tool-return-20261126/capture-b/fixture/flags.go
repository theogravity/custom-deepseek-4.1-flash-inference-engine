package compose

import (
	"os"
	"strconv"
	"strings"
)

func extractFlagStr(cmdLine, flag string) string {
	if cmdLine == "" {
		return ""
	}
	idx := strings.Index(cmdLine, flag)
	if idx < 0 {
		return ""
	}
	rest := cmdLine[idx+len(flag):]
	rest = strings.TrimSpace(rest)
	if rest == "" {
		return ""
	}
	val := rest
	if rest[0] == '"' {
		if end := strings.Index(rest[1:], `"`); end >= 0 {
			val = rest[1 : 1+end]
		}
	} else if rest[0] == '\'' {
		if end := strings.Index(rest[1:], `'`); end >= 0 {
			val = rest[1 : 1+end]
		}
	} else {
		parts := strings.Fields(rest)
		if len(parts) == 0 {
			return ""
		}
		val = parts[0]
	}
	val = strings.TrimSpace(val)
	return os.ExpandEnv(val)
}

// extractFlagInt extracts an integer flag value from a command line string.
func extractFlagInt(cmdLine, flag string, defaultVal int) int {
	s := extractFlagStr(cmdLine, flag)
	if s == "" {
		return defaultVal
	}
	if n, err := strconv.Atoi(s); err == nil {
		return n
	}
	return defaultVal
}
