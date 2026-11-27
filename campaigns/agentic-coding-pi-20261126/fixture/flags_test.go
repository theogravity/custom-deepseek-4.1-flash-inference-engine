package compose

import "testing"

func TestFlagValues(t *testing.T) {
	t.Setenv("MODEL_PORT", "8001")
	for _, tc := range []struct{ cmd, flag, want string }{
		{"--port 8001 --name demo", "--port", "8001"},
		{"--name='Local Flash' --port 8001", "--name", "Local Flash"},
		{"--port-file /tmp/port --port=8001", "--port", "8001"},
		{"--name \"Local Flash\" --port 8001", "--name", "Local Flash"},
		{"--port $MODEL_PORT", "--port", "8001"},
	} {
		if got := extractFlagStr(tc.cmd, tc.flag); got != tc.want {
			t.Errorf("extractFlagStr(%q,%q) = %q, want %q", tc.cmd, tc.flag, got, tc.want)
		}
	}
	if got := extractFlagInt("--port=8010", "--port", 1); got != 8010 {
		t.Errorf("port = %d, want 8010", got)
	}
}
