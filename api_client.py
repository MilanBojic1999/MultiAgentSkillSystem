#!/usr/bin/env python3
"""
CLI client for the Agent Skills Pipeline API.

Zero external dependencies — uses only Python stdlib (urllib + json).

Usage:
    # Interactive mode — no arguments, prompts for tasks in a REPL loop
    python api_client.py

    # Run a task through the pipeline
    python api_client.py "Calculate sin(pi/4) + cos(pi/4) and explain the result"

    # Check if the API is alive
    python api_client.py --health

    # Specify a custom server URL
    python api_client.py --url http://192.168.1.100:8000 "Research quantum computing"

    # Async mode: start a task and poll until it finishes
    python api_client.py --async "Research the history of machine learning"

    # Stream mode: stream tokens from /run-stream as they arrive
    python api_client.py --stream "Write a haiku about neural networks"

Examples:
    python api_client.py                           # interactive REPL
    python api_client.py "What is 2 + 2?"
    python api_client.py --health
    python api_client.py --stream "Tell me a joke"
    python api_client.py --url http://localhost:9000 "Plot sin(x) from -pi to pi"
"""

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from typing import Any


DEFAULT_URL = "http://localhost:8999"
POLL_INTERVAL = 2  # seconds between async status checks


# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------


def _request(method: str, path: str, base_url: str, body: dict | None = None) -> dict[str, Any]:
    """Send an HTTP request and return the parsed JSON response."""
    url = f"{base_url.rstrip('/')}{path}"
    data = json.dumps(body).encode("utf-8") if body else None

    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Content-Type", "application/json")
    req.add_header("Accept", "application/json")

    try:
        with urllib.request.urlopen(req) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")
        try:
            detail = json.loads(error_body).get("detail", error_body)
        except json.JSONDecodeError:
            detail = error_body
        return {"error": True, "status_code": exc.code, "detail": detail}
    except urllib.error.URLError as exc:
        return {"error": True, "detail": str(exc.reason)}


def check_health(base_url: str) -> bool:
    """Return True if the API is reachable and healthy."""
    print(f"🔍 Checking API health at {base_url} ...")
    result = _request("GET", "/health", base_url)
    if result.get("status") == "ok":
        print("✅ API is healthy.")
        return True
    print(f"❌ API health check failed: {result}")
    return False


def run_task(task: str, base_url: str) -> str | None:
    """Run a task synchronously and return the final output."""
    print(f"🚀 Running task:\n   {task}\n")
    print(f"📡 POST {base_url}/run ...")

    result = _request("POST", "/run", base_url, body={"task": task})

    if result.get("error"):
        print(f"❌ Error: {result.get('detail', 'Unknown error')}")
        return None

    return result.get("final_output", "")


def run_task_async(task: str, base_url: str) -> str | None:
    """Start an async task and poll until completion."""
    print(f"🚀 Starting async task:\n   {task}\n")
    print(f"📡 POST {base_url}/run-async ...")

    start_result = _request("POST", "/run-async", base_url, body={"task": task})
    if start_result.get("error"):
        print(f"❌ Error starting task: {start_result.get('detail', 'Unknown error')}")
        return None

    task_id = start_result.get("task_id")
    if not task_id:
        print("❌ No task_id returned from server.")
        return None

    print(f"📋 Task ID: {task_id}")
    print(f"⏳ Polling {base_url}/status/{task_id} every {POLL_INTERVAL}s ...\n")

    dots = 0
    while True:
        time.sleep(POLL_INTERVAL)
        status = _request("GET", f"/status/{task_id}", base_url)

        if status.get("error"):
            print(f"\n❌ Error checking status: {status.get('detail', 'Unknown')}")
            return None

        task_status = status.get("status")

        if task_status == "completed":
            print(f"\n✅ Task completed!")
            return status.get("final_output", "")

        if task_status == "failed":
            print(f"\n❌ Task failed: {status.get('error', 'Unknown error')}")
            return None

        # Still running — show a spinner
        dots = (dots + 1) % 4
        print(f"\r   Running{'.' * dots}{' ' * (3 - dots)}", end="", flush=True)


def run_task_stream(task: str, base_url: str) -> str | None:
    """Run a task via the /run-stream SSE endpoint and print tokens as they arrive.

    Returns the full assembled output, or None on error.
    """
    print(f"🚀 Starting stream task:\n   {task}\n")
    print(f"📡 POST {base_url}/run-stream ...\n")

    url = f"{base_url.rstrip('/')}/run-stream"
    data = json.dumps({"task": task}).encode("utf-8")

    req = urllib.request.Request(url, data=data, method="POST")
    req.add_header("Content-Type", "application/json")
    req.add_header("Accept", "text/event-stream")

    full_output: list[str] = []

    try:
        with urllib.request.urlopen(req) as resp:
            # Read SSE line by line
            for raw_line in resp:
                line = raw_line.decode("utf-8", errors="replace")
                if not line:
                    continue  # skip empty lines (SSE framing)
                # if not line.startswith("data: "):
                #     continue

                payload = line.replace("data: ", "", 1).rstrip("\n\n")
                if payload == "<stop>":
                    break
                if payload.startswith("[error] "):
                    error_msg = payload.removeprefix("[error] ")
                    print(f"\n❌ Stream error: {error_msg}")
                    return None

                # Normal token — print in-place and accumulate
                full_output.append(payload)
                # print(repr(payload), end="", flush=True)
                print(payload, end="", flush=True)
                if payload == "<thinking_step>":
                    print("\n")
                if payload == "<think>" or payload == "<non_think>":
                    print("")

        print()  # final newline after stream
        assembled = "".join(full_output)
        return assembled if assembled else None

    except urllib.error.HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")
        try:
            detail = json.loads(error_body).get("detail", error_body)
        except json.JSONDecodeError:
            detail = error_body
        print(f"❌ Error: {detail}")
        return None
    except urllib.error.URLError as exc:
        print(f"❌ Connection error: {exc.reason}")
        return None


# ---------------------------------------------------------------------------
# Interactive REPL
# ---------------------------------------------------------------------------

HELP_TEXT = """\
Commands:
  <any text>     Run the text as a task through the pipeline
  :health        Check if the API server is healthy
  :url <URL>     Change the server URL (current: {url})
  :async         Toggle async mode (currently: {async_mode})
  :stream        Toggle stream mode (currently: {stream_mode})
  :help, :?      Show this help
  :quit, :q      Exit the client

You can also paste multi-line input — press Enter twice on an empty line to submit.\
"""


def interactive_repl(base_url: str) -> None:
    """Run an interactive REPL loop for submitting tasks."""
    print()
    print("╔══════════════════════════════════════════════════╗")
    print("║     Agent Skills Pipeline — Interactive CLI      ║")
    print("╚══════════════════════════════════════════════════╝")
    print()
    print(f"Server: {base_url}")
    print("Type a task to run it, or :help for commands.")
    print()

    async_mode = False
    stream_mode = False

    while True:
        try:
            if stream_mode:
                prompt = "🌊"
            elif async_mode:
                prompt = "⏳"
            else:
                prompt = "⚡"
            line = input(f"{prompt} ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n👋 Goodbye!")
            break

        # Empty input — skip
        if not line:
            continue

        # Commands
        if line.startswith(":"):
            cmd, *rest = line[1:].split(maxsplit=1)
            arg = rest[0] if rest else ""

            if cmd in ("quit", "q"):
                print("👋 Goodbye!")
                break
            elif cmd in ("help", "?"):
                print(HELP_TEXT.format(url=base_url, async_mode=async_mode, stream_mode=stream_mode))
            elif cmd == "health":
                check_health(base_url)
            elif cmd == "url":
                if arg:
                    base_url = arg.rstrip("/")
                    print(f"✅ Server URL set to {base_url}")
                else:
                    print(f"Current URL: {base_url}")
            elif cmd == "async":
                async_mode = not async_mode
                stream_mode = False  # mutually exclusive
                state = "ON" if async_mode else "OFF"
                print(f"✅ Async mode: {state}")
            elif cmd == "stream":
                stream_mode = not stream_mode
                async_mode = False  # mutually exclusive
                state = "ON" if stream_mode else "OFF"
                print(f"✅ Stream mode: {state}")
            else:
                print(f"❌ Unknown command: :{cmd}  (type :help for help)")
            continue

        # Run the task
        if stream_mode:
            output = run_task_stream(line, base_url)
        elif async_mode:
            output = run_task_async(line, base_url)
        else:
            output = run_task(line, base_url)

        # if output is not None:
        #     print("\n" + "─" * 60)
        #     print(output)
        #     print("─" * 60)
        print()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Agent Skills Pipeline — CLI client",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python api_client.py                           # interactive REPL
  python api_client.py "What is 2 + 2?"
  python api_client.py --health
  python api_client.py --stream "Tell me a joke"
  python api_client.py --async "Research the history of AI"
  python api_client.py --url http://192.168.1.100:8000 "Plot sin(x)"
        """.strip(),
    )
    parser.add_argument(
        "task",
        nargs="?",
        help="The task to run through the pipeline (wrap in quotes). Omit to enter interactive mode.",
    )
    parser.add_argument(
        "--url", "-u",
        default=DEFAULT_URL,
        help=f"Base URL of the API server (default: {DEFAULT_URL})",
    )
    parser.add_argument(
        "--health",
        action="store_true",
        help="Check if the API server is healthy and exit",
    )
    parser.add_argument(
        "--async",
        dest="async_mode",
        action="store_true",
        help="Run the task asynchronously (start and poll until done)",
    )
    parser.add_argument(
        "--stream", "-s",
        dest="stream_mode",
        action="store_true",
        help="Run the task via the /run-stream SSE endpoint (streams tokens live)",
    )

    args = parser.parse_args()

    # --health mode (works with or without a task)
    if args.health:
        ok = check_health(args.url)
        sys.exit(0 if ok else 1)

    # No task and no --health → interactive REPL
    if not args.task:
        interactive_repl(args.url)
        return

    # Task supplied — run once
    if args.stream_mode:
        output = run_task_stream(args.task, args.url)
    elif args.async_mode:
        output = run_task_async(args.task, args.url)
    else:
        output = run_task(args.task, args.url)

    if output is None:
        sys.exit(1)

    print("\n" + "=" * 60)
    print(output)
    print("=" * 60)


if __name__ == "__main__":
    main()
