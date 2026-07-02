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

import base64
import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from typing import Any


DEFAULT_URL = "http://localhost:8999"
POLL_INTERVAL = 2  # seconds between async status checks
DEFAULT_MAX_FILE_CHARS = 50_000  # ~12.5k tokens by the 4-char/token heuristic
TOKEN_ESTIMATE_CHARS = 4         # rough chars-per-token for English text


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


def _read_files(file_paths: list[str], max_chars: int = DEFAULT_MAX_FILE_CHARS) -> list[dict[str, str]]:
    """Read each file path and return a list of {filename, content, encoding?} dicts.

    Tries UTF-8 text first; falls back to base64-encoding for binary files
    (PDFs, images, etc.).  Content is truncated to ``max_chars`` characters
    with a visible notice to avoid burning the LLM context window.

    Prints a warning and skips unreadable files rather than aborting.
    """
    files: list[dict[str, str]] = []
    for path in file_paths:
        if not os.path.isfile(path):
            print(f"⚠️  Skipping '{path}' — not a file or doesn't exist.")
            continue
        try:
            # Attempt UTF-8 text first
            with open(path, "r", encoding="utf-8") as fh:
                content = fh.read()
            if len(content) > max_chars:
                est_tokens = len(content) // TOKEN_ESTIMATE_CHARS
                trunc_tokens = max_chars // TOKEN_ESTIMATE_CHARS
                print(f"⚠️  {os.path.basename(path)}: {len(content):,} chars "
                      f"(~{est_tokens:,} tokens) exceeds limit "
                      f"({max_chars:,} chars / ~{trunc_tokens:,} tokens) — truncating")
                half = max_chars // 2
                content = (
                    content[:half]
                    + f"\n\n... [TRUNCATED — {len(content) - max_chars:,} chars omitted] ...\n\n"
                    + content[-half:]
                )
            files.append({"filename": os.path.basename(path), "content": content})
        except (UnicodeDecodeError, UnicodeError):
            # Binary file — read raw bytes and base64-encode
            try:
                with open(path, "rb") as fh:
                    raw = fh.read()
                # Truncate raw bytes before encoding (base64 blows up by ~33%)
                max_raw = (max_chars * 3) // 4  # inverse of base64 expansion
                if len(raw) > max_raw:
                    print(f"⚠️  {os.path.basename(path)}: {len(raw):,} raw bytes "
                          f"exceeds limit — truncating to ~{max_raw:,} bytes")
                    raw = raw[:max_raw]
                encoded = base64.b64encode(raw).decode("ascii")
                files.append({
                    "filename": os.path.basename(path),
                    "content": encoded,
                    "encoding": "base64",
                })
                print(f"📎 {os.path.basename(path)} — binary file, base64-encoded "
                      f"({len(raw)} bytes → {len(encoded)} chars)")
            except Exception as exc:
                print(f"⚠️  Could not read binary file '{path}': {exc}")
        except Exception as exc:
            print(f"⚠️  Could not read '{path}': {exc}")
    return files


def check_health(base_url: str) -> bool:
    """Return True if the API is reachable and healthy."""
    print(f"🔍 Checking API health at {base_url} ...")
    result = _request("GET", "/health", base_url)
    if result.get("status") == "ok":
        print("✅ API is healthy.")
        return True
    print(f"❌ API health check failed: {result}")
    return False


def run_task(task: str, base_url: str, files: list[dict] | None = None) -> str | None:
    """Run a task synchronously and return the final output."""
    print(f"🚀 Running task:\n   {task}\n")
    if files:
        print(f"📎 Attached files: {', '.join(f['filename'] for f in files)}")
    print(f"📡 POST {base_url}/run ...")

    body: dict[str, Any] = {"task": task}
    if files:
        body["files"] = files

    result = _request("POST", "/run", base_url, body=body)

    if result.get("error"):
        print(f"❌ Error: {result.get('detail', 'Unknown error')}")
        return None

    return result.get("final_output", "")


def run_task_async(task: str, base_url: str, files: list[dict] | None = None) -> str | None:
    """Start an async task and poll until completion."""
    print(f"🚀 Starting async task:\n   {task}\n")
    if files:
        print(f"📎 Attached files: {', '.join(f['filename'] for f in files)}")
    print(f"📡 POST {base_url}/run-async ...")

    body: dict[str, Any] = {"task": task}
    if files:
        body["files"] = files

    start_result = _request("POST", "/run-async", base_url, body=body)
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


def run_task_stream(task: str, base_url: str, files: list[dict] | None = None) -> str | None:
    """Run a task via the /run-stream SSE endpoint and print tokens as they arrive.

    Returns the full assembled output, or None on error.
    """
    print(f"🚀 Starting stream task:\n   {task}\n")
    if files:
        print(f"📎 Attached files: {', '.join(f['filename'] for f in files)}")
    print(f"📡 POST {base_url}/run-stream ...\n")

    url = f"{base_url.rstrip('/')}/run-stream"
    body: dict[str, Any] = {"task": task}
    if files:
        body["files"] = files
    data = json.dumps(body).encode("utf-8")

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
  :file <PATH>   Attach a file as input context for the next task
  :files         List currently attached files
  :clear-files   Remove all attached files
  :max-chars [N] Show/set max chars per file (current: {max_chars})
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
    attached_files: list[str] = []  # paths of files to send with the next task
    max_file_chars = DEFAULT_MAX_FILE_CHARS

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
                print(HELP_TEXT.format(url=base_url, async_mode=async_mode, stream_mode=stream_mode, max_chars=f"{max_file_chars:,}"))
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
            elif cmd == "file":
                if arg:
                    attached_files.append(arg)
                    print(f"📎 Attached: {arg}")
                else:
                    print("Usage: :file <PATH>")
            elif cmd == "files":
                if attached_files:
                    print("📎 Attached files:")
                    for i, f in enumerate(attached_files, 1):
                        print(f"   {i}. {f}")
                else:
                    print("No files attached.")
            elif cmd in ("clear-files", "cf"):
                count = len(attached_files)
                attached_files.clear()
                print(f"🗑️  Cleared {count} attached file(s).")
            elif cmd == "max-chars":
                if arg:
                    try:
                        max_file_chars = int(arg)
                        est_tokens = max_file_chars // TOKEN_ESTIMATE_CHARS
                        print(f"✅ Max file chars set to {max_file_chars:,} "
                              f"(~{est_tokens:,} tokens)")
                    except ValueError:
                        print(f"❌ Invalid number: {arg}")
                else:
                    est_tokens = max_file_chars // TOKEN_ESTIMATE_CHARS
                    print(f"Max file chars: {max_file_chars:,} "
                          f"(~{est_tokens:,} tokens)")
            else:
                print(f"❌ Unknown command: :{cmd}  (type :help for help)")
            continue

        # Run the task
        files = _read_files(attached_files, max_file_chars) if attached_files else None
        if stream_mode:
            output = run_task_stream(line, base_url, files)
        elif async_mode:
            output = run_task_async(line, base_url, files)
        else:
            output = run_task(line, base_url, files)
        # Clear attached files after each task (one-shot behaviour)
        attached_files.clear()

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
    parser.add_argument(
        "--file", "-f",
        dest="files",
        action="append",
        default=[],
        metavar="PATH",
        help="Attach a file as input context (can be used multiple times)",
    )
    parser.add_argument(
        "--max-file-chars",
        type=int,
        default=DEFAULT_MAX_FILE_CHARS,
        metavar="N",
        help=f"Max characters per attached file (default: {DEFAULT_MAX_FILE_CHARS:,})",
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
    files = _read_files(args.files, args.max_file_chars) if args.files else None
    if args.stream_mode:
        output = run_task_stream(args.task, args.url, files)
    elif args.async_mode:
        output = run_task_async(args.task, args.url, files)
    else:
        output = run_task(args.task, args.url, files)

    if output is None:
        sys.exit(1)

    print("\n" + "=" * 60)
    print(output)
    print("=" * 60)


if __name__ == "__main__":
    main()
