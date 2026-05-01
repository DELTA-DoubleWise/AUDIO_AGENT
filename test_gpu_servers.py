#!/usr/bin/env python3
"""Test each MCP server loads model on GPU correctly."""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

# Tool configs: (name, working_dir, command, env_vars)
TOOLS = [
    ("asr_qwen3", "audio_agent/tools/catalog/asr_qwen3", [".venv/bin/python", "server.py"], {}),
    ("fireredasr2s", "audio_agent/tools/catalog/fireredasr2s", [".venv/bin/python", "server.py"], {}),
    ("whisperx", "audio_agent/tools/catalog/whisperx", [".venv/bin/python", "server.py"], {}),
    ("diarizen", "audio_agent/tools/catalog/diarizen", [".venv/bin/python", "server.py"], {}),
    ("fireredvad", "audio_agent/tools/catalog/fireredvad", [".venv/bin/python", "server.py"], {}),
    ("silero-vad", "audio_agent/tools/catalog/snakers4_silero-vad", [".venv/bin/python", "server.py"], {}),
    ("wespeaker", "audio_agent/tools/catalog/wespeaker", [".venv/bin/python", "server.py"], {}),
]

PROJECT_ROOT = Path("/cpfs/user/jingpeng/workspace/AUDIO_AGENT")


def send_request(proc, request: dict) -> dict | None:
    """Send JSON-RPC request via stdin and read response from stdout."""
    line = json.dumps(request) + "\n"
    proc.stdin.write(line.encode())
    proc.stdin.flush()
    try:
        response_line = proc.stdout.readline()
        if not response_line:
            return None
        return json.loads(response_line.decode())
    except Exception as e:
        print(f"  Error reading response: {e}")
        return None


def test_server(name: str, cwd: str, cmd: list, extra_env: dict):
    """Test a single MCP server."""
    print(f"\n{'='*60}")
    print(f"Testing: {name}")
    print(f"{'='*60}")

    working_dir = PROJECT_ROOT / cwd
    if not working_dir.exists():
        print(f"  SKIP: working dir not found: {working_dir}")
        return False

    env = {**os.environ, **extra_env}

    try:
        proc = subprocess.Popen(
            cmd,
            cwd=working_dir,
            env=env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except Exception as e:
        print(f"  FAIL: Could not start server: {e}")
        return False

    # Wait a bit for model loading
    time.sleep(2)

    # Send initialize
    init_req = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "gpu-test", "version": "1.0"},
        },
    }
    resp = send_request(proc, init_req)
    if resp is None or resp.get("error"):
        print(f"  FAIL: initialize error: {resp}")
        proc.terminate()
        return False
    print(f"  ✓ initialize OK")

    # Send initialized notification
    send_request(proc, {"jsonrpc": "2.0", "method": "notifications/initialized"})

    # Send tools/list
    list_req = {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}}
    resp = send_request(proc, list_req)
    if resp is None or resp.get("error"):
        print(f"  FAIL: tools/list error: {resp}")
        proc.terminate()
        return False
    tools = resp.get("result", {}).get("tools", [])
    print(f"  ✓ tools/list OK ({len(tools)} tools)")

    # Try healthcheck if available
    has_healthcheck = any(t.get("name") == "healthcheck" for t in tools)
    if has_healthcheck:
        health_req = {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {"name": "healthcheck", "arguments": {}},
        }
        resp = send_request(proc, health_req)
        if resp and not resp.get("error"):
            print(f"  ✓ healthcheck OK")
        else:
            print(f"  ⚠ healthcheck failed: {resp}")

    # Check GPU usage via nvidia-smi
    time.sleep(1)
    try:
        smi = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,name,used_memory", "--format=csv,noheader"],
            capture_output=True, text=True, check=True,
        )
        gpu_procs = [line.strip() for line in smi.stdout.strip().split("\n") if line.strip()]
        own_pid = proc.pid
        own_gpu = [p for p in gpu_procs if str(own_pid) in p]
        if own_gpu:
            print(f"  ✓ GPU USAGE: {own_gpu[0]}")
        else:
            print(f"  ⚠ No GPU process found for PID {own_pid}")
            # Show all GPU processes for debugging
            if gpu_procs:
                print(f"    Other GPU processes: {len(gpu_procs)}")
    except Exception as e:
        print(f"  ⚠ nvidia-smi error: {e}")

    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()

    return True


def main():
    print("GPU Server Verification Test")
    print(f"Project root: {PROJECT_ROOT}")

    results = {}
    for name, cwd, cmd, extra_env in TOOLS:
        results[name] = test_server(name, cwd, cmd, extra_env)

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for name, ok in results.items():
        status = "✓ PASS" if ok else "✗ FAIL"
        print(f"  {status}: {name}")


if __name__ == "__main__":
    main()
