"""python -m dashboard [--port 8765] [--host 127.0.0.1] [--no-browser] [--data DIR]"""
import argparse
import socket
import threading
import time
import webbrowser


def _free_port(host, want):
    for p in range(want, want + 30):
        with socket.socket() as s:
            if s.connect_ex((host, p)) != 0:
                return p
    return want


def main():
    ap = argparse.ArgumentParser(description="TuftView dashboard")
    ap.add_argument("--host", default="127.0.0.1", help="use 0.0.0.0 to let other computers on the network open it")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-browser", action="store_true")
    ap.add_argument("--data", default=None, help="where analyses are stored (default: dashboard_data/)")
    a = ap.parse_args()

    from .server import create_app
    app = create_app(a.data)
    port = _free_port(a.host, a.port)
    url = f"http://{'127.0.0.1' if a.host == '0.0.0.0' else a.host}:{port}"
    print(f"\n  TuftView is running at {url}\n  Close this window (or press Ctrl+C) to stop it.\n")
    if not a.no_browser:
        threading.Thread(target=lambda: (time.sleep(1.0), webbrowser.open(url)), daemon=True).start()
    try:
        from waitress import serve
        serve(app, host=a.host, port=port, threads=8, max_request_body_size=8 * 1024 ** 3)
    except ImportError:
        app.run(host=a.host, port=port, threaded=True, debug=False)


if __name__ == "__main__":
    main()
