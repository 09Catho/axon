"""One-command AXON demo: start server, wait for readiness, open browser."""

import subprocess
import sys
import time
import urllib.parse
import urllib.request
import webbrowser

HOST = "127.0.0.1"
PORT = 8000
HEALTH_URL = f"http://{HOST}:{PORT}/health"
DEMO_PROMPT = "The history of human civilization began when"
READY_TIMEOUT_S = 180  # cold-cache model download can take a while


def wait_ready(timeout: float = READY_TIMEOUT_S) -> bool:
    start = time.time()
    while time.time() - start < timeout:
        try:
            with urllib.request.urlopen(HEALTH_URL, timeout=2) as r:
                if r.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(1)
    return False


def main():
    print("Starting AXON server ...")
    proc = subprocess.Popen([sys.executable, "-u", "server.py"])

    try:
        print("Waiting for /health ...")
        if not wait_ready():
            print(f"Server did not become ready within {READY_TIMEOUT_S}s.", file=sys.stderr)
            proc.terminate()
            sys.exit(1)
        print("Server is ready.")

        query = urllib.parse.urlencode({"autorun": "1", "prompt": DEMO_PROMPT})
        url = f"http://{HOST}:{PORT}/static/index.html?{query}"
        print(f"Opening {url}")
        webbrowser.open(url)

        print("\nAXON is running. Press Ctrl+C to stop.")
        proc.wait()
    except KeyboardInterrupt:
        print("\nShutting down ...")
    finally:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()


if __name__ == "__main__":
    main()
