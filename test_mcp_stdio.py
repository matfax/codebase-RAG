#!/usr/bin/env python3
"""Test MCP server via stdio communication."""

import json
import subprocess
import sys


def send_mcp_request(request):
    """Send a JSON-RPC request to the MCP server."""
    process = subprocess.Popen(
        [".venv/bin/python", "src/run_mcp.py"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    try:
        stdout, stderr = process.communicate(input=json.dumps(request) + "\n", timeout=10)
        if stderr:
            print(f"STDERR: {stderr}", file=sys.stderr)
        return json.loads(stdout.strip()) if stdout.strip() else None
    except subprocess.TimeoutExpired:
        process.kill()
        return {"error": "timeout"}
    except json.JSONDecodeError as e:
        print(f"JSON decode error: {e}", file=sys.stderr)
        print(f"Raw output: {stdout}", file=sys.stderr)
        return {"error": "json_decode_error"}


def test_mcp_communication():
    """Test complete MCP server communication flow."""
    print("Testing MCP server stdio communication...")

    # Step 1: Initialize
    print("\n1. Initializing MCP session...")
    init_request = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "1.0.0",
            "capabilities": {},
            "clientInfo": {"name": "test-client", "version": "1.0.0"},
        },
    }

    response = send_mcp_request(init_request)
    print(f"Initialize response: {json.dumps(response, indent=2)}")

    if not response or "error" in response:
        print("❌ Initialization failed")
        return False

    print("✅ MCP server initialized successfully")
    return True


if __name__ == "__main__":
    success = test_mcp_communication()
    sys.exit(0 if success else 1)
