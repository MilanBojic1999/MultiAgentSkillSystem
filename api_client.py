#!/usr/bin/env python3
"""
CLI client for the Agent Skills Pipeline API.

Zero external dependencies - uses only Python stdlib (urllib + json).

Usage:
    # Run a task through the pipeline
    python api_client.py "Calculate sin(pi/4) + cos(pi/4) and explain the result"

    # Check if the API is alive
    python api_client.py --health

    # Specify a custom server URL
    python api_client.py --url http://192.168.1.100:8000 "Research quantum computing"

    # Async mode: start a task and poll until it finishes
    python api_client.py --async "Research the history of machine learning"

Examples:
    python api_client.py "What is 2 + 2?"
    python api_client.py --health
    python api_client.py --url http://localhost:9000 "Plot sin(x) from -pi to pi"
"""

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from typing import Any, Optional


DEFAULT_URL = "http://localhost:8000"
POLL_INTERVAL = 2  # seconds between async status checks


# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------


def _request(method: str, path: str, base_url: str, body: Optional[dict] = None) -> dict[str, Any]:
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


def list_graphs(base_url: str) -> bool:
    """Print the graphs the server discovered in its graphs/ directory."""
    result = _request("GET", "/graphs", base_url)
    if result.get("error"):
        print(f"❌ Error listing graphs: {result.get('detail', 'Unknown error')}")
        return False

    for entry in result.get("graphs", []):
        marker = " (default)" if entry.get("default") else ""
        description = entry.get("description") or ""
        print(f"  {entry['name']}{marker}" + (f" — {description}" if description else ""))
    return True


def run_task(task: str, base_url: str, graph: Optional[str] = None) -> Optional[str]:
    """Run a task synchronously and return the final output."""
    print(f"🚀 Running task:\n   {task}\n")
    print(f"📡 POST {base_url}/run ...")

    result = _request("POST", "/run", base_url, body=_run_body(task, graph))

    if result.get("error"):
        print(f"❌ Error: {result.get('detail', 'Unknown error')}")
        return None

    status = result.get("status", "completed")
    if status == "partial":
        failed = len(result.get("failed_steps", []))
        print(f"⚠️  Pipeline finished with status 'partial' ({failed} step(s) failed).")
    else:
        print(f"✅ Pipeline finished with status '{status}'.")
    return result.get("final_output", "")


def _run_body(task: str, graph: Optional[str]) -> dict[str, Any]:
    """Request body for /run and /run-async; omit `graph` to take the server default."""
    body: dict[str, Any] = {"task": task}
    if graph:
        body["graph"] = graph
    return body


def run_task_async(task: str, base_url: str, graph: Optional[str] = None) -> Optional[str]:
    """Start an async task and poll until completion."""
    print(f"🚀 Starting async task:\n   {task}\n")
    print(f"📡 POST {base_url}/run-async ...")

    start_result = _request("POST", "/run-async", base_url, body=_run_body(task, graph))
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

        if task_status == "partial":
            failed = len(status.get("failed_steps", []))
            print(f"\n⚠️  Task finished with status 'partial' ({failed} step(s) failed).")
            return status.get("final_output", "")

        if task_status == "failed":
            print(f"\n❌ Task failed: {status.get('error', 'Unknown error')}")
            return None

        # Still running — show a spinner
        dots = (dots + 1) % 4
        print(f"\r   Running{'.' * dots}{' ' * (3 - dots)}", end="", flush=True)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Agent Skills Pipeline — CLI client",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python api_client.py "What is 2 + 2?"
  python api_client.py --health
  python api_client.py --async "Research the history of AI"
  python api_client.py --url http://192.168.1.100:8000 "Plot sin(x)"
        """.strip(),
    )
    parser.add_argument(
        "task",
        nargs="?",
        help="The task to run through the pipeline (wrap in quotes)",
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
        "--graph", "-g",
        help="Which graph the server should run (default: the server's own default)",
    )
    parser.add_argument(
        "--list-graphs",
        action="store_true",
        help="List the graphs available on the server and exit",
    )
    parser.add_argument(
        "--async",
        dest="async_mode",
        action="store_true",
        help="Run the task asynchronously (start and poll until done)",
    )

    args = parser.parse_args()

    # --health mode
    if args.health:
        ok = check_health(args.url)
        sys.exit(0 if ok else 1)

    # --list-graphs mode
    if args.list_graphs:
        sys.exit(0 if list_graphs(args.url) else 1)

    # Task required for run modes
    if not args.task:
        parser.error("A task string is required (unless using --health).")

    # Run
    if args.async_mode:
        output = run_task_async(args.task, args.url, args.graph)
    else:
        output = run_task(args.task, args.url, args.graph)

    if output is None:
        sys.exit(1)

    print("\n" + "=" * 60)
    print(output)
    print("=" * 60)


if __name__ == "__main__":
    main()
