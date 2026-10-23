"""Flask app: serves the dashboard page and a small JSON API. Local only (127.0.0.1) by default."""
import gzip
import io
import json
import os
import threading
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from flask import Flask, Response, abort, jsonify, request, send_file, send_from_directory

from coordinates import CoordinateError, CoordinateSystem, LocationError, TuftLocations

from . import ROOT, __version__, analysis, tables
from .jobs import JobStore

STATIC = Path(__file__).parent / "static"


def create_app(data_dir=None):
    data_dir = Path(data_dir or os.environ.get("TUFTVIEW_DATA") or ROOT / "dashboard_data")
    store = JobStore(data_dir)
    app = Flask(__name__, static_folder=None)
    app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 ** 3
    app.store = store
    cache = {}
    lock = threading.Lock()

    # ------------------------------------------------------------ helpers
    def job_or_404(jid):
        m = store.get(jid)
        if m is None:
            abort(404, "That analysis no longer exists.")
        return m

    def load_results(jid):
        p = store.dir(jid) / "results.json.gz"
        if not p.is_file():
            abort(409, "This analysis has no results yet.")
        key = ("res", jid, p.stat().st_mtime_ns)
        with lock:
            if key not in cache:
                cache.clear() if len(cache) > 12 else None
                cache[key] = analysis.load_results(p)
            return cache[key]

    def load_setup(jid):
        p = store.dir(jid) / "setup.json"
        try:
            return json.loads(p.read_text("utf8"))
        except (OSError, ValueError):
            return tables.default_setup()

    def save_setup(jid, setup):
        p = store.dir(jid) / "setup.json"
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(setup), "utf8")
        tmp.replace(p)

    def history(jid):
        res, setup = load_results(jid), load_setup(jid)
        p = store.dir(jid) / "setup.json"
        key = ("hist", jid, p.stat().st_mtime_ns if p.exists() else 0, (store.dir(jid) / "results.json.gz").stat().st_mtime_ns)
        with lock:
            if key not in cache:
                cache[key] = tables.history_frame(res, setup)
            return cache[key]

    def csv_response(df, name):
        return Response(df.to_csv(index=False), mimetype="text/csv",
                        headers={"Content-Disposition": f'attachment; filename="{name}"'})

    def clean(df):
        return json.loads(df.to_json(orient="records", double_precision=4))

    @app.errorhandler(400)
    @app.errorhandler(404)
    @app.errorhandler(409)
    def _err(e):
        return jsonify(error=getattr(e, "description", str(e))), e.code

    # ------------------------------------------------------------ pages
    @app.get("/")
    def index():
        return send_from_directory(STATIC, "index.html", max_age=0)

    @app.get("/static/<path:name>")
    def static_files(name):
        return send_from_directory(STATIC, name, max_age=0)

    @app.get("/api/status")
    def status():
        engine, weights, note = analysis.pick_engine(analysis.Options())
        return jsonify(version=__version__, engine=note, unet=engine == "unet",
                       ffmpeg=bool(__import__("dashboard.media", fromlist=["x"]).ffmpeg_exe()))

    # ------------------------------------------------------------ jobs
    @app.get("/api/jobs")
    def jobs():
        return jsonify(store.list())

    @app.post("/api/upload")
    def upload():
        files = request.files.getlist("files")
        if not files:
            abort(400, "No file was received.")
        made, errors = [], []
        for f in files:
            try:
                made.append(store.create(f.filename or "upload", lambda p, f=f: f.save(p)))
            except ValueError as exc:
                errors.append({"name": f.filename, "error": str(exc)})
        code = 200 if made else 400
        return jsonify(jobs=made, errors=errors), code

    @app.post("/api/demo")
    def demo():
        from .demo import make_demo_video

        def writer(p):
            make_demo_video(p)
        return jsonify(store.create("demo-wind-tunnel.mp4", writer))

    @app.get("/api/jobs/<jid>")
    def job(jid):
        return jsonify(job_or_404(jid))

    @app.delete("/api/jobs/<jid>")
    def delete(jid):
        job_or_404(jid)
        store.delete(jid)
        return jsonify(ok=True)

    @app.post("/api/jobs/<jid>/cancel")
    def cancel(jid):
        job_or_404(jid)
        return jsonify(ok=store.request_cancel(jid))

    @app.post("/api/jobs/<jid>/reanalyze")
    def reanalyze(jid):
        job_or_404(jid)
        try:
            return jsonify(store.reanalyze(jid, request.get_json(silent=True) or {}))
        except ValueError as exc:
            abort(409, str(exc))

    @app.get("/api/jobs/<jid>/preview.jpg")
    def preview(jid):
        job_or_404(jid)
        d = store.dir(jid)
        p = d / "preview.jpg" if (d / "preview.jpg").exists() else d / "first_frame.jpg"
        return send_file(p, mimetype="image/jpeg", max_age=0)

    @app.get("/api/jobs/<jid>/first_frame.jpg")
    def first_frame(jid):
        job_or_404(jid)
        return send_file(store.dir(jid) / "first_frame.jpg", mimetype="image/jpeg", max_age=60)

    @app.get("/api/jobs/<jid>/media")
    def media(jid):
        m = job_or_404(jid)
        d = store.dir(jid)
        p = d / ("annotated.mp4" if m["kind"] == "video" else "annotated.jpg")
        if not p.exists():
            abort(409, "No annotated output yet.")
        mt = "video/mp4" if p.suffix == ".mp4" else "image/jpeg"
        return send_file(p, mimetype=mt, conditional=True, max_age=0)

    @app.get("/api/jobs/<jid>/original")
    def original(jid):
        m = job_or_404(jid)
        return send_file(store.dir(jid) / m["input"], conditional=True, max_age=3600)

    @app.get("/api/jobs/<jid>/results.json")
    def results_json(jid):
        job_or_404(jid)
        p = store.dir(jid) / "results.json.gz"
        if not p.is_file():
            abort(409, "This analysis has no results yet.")
        r = Response(p.read_bytes(), mimetype="application/json")
        r.headers["Content-Encoding"] = "gzip"
        r.headers["Cache-Control"] = "no-cache"
        return r

    # ------------------------------------------------------------ setup (coordinates, positions)
    @app.get("/api/jobs/<jid>/setup")
    def get_setup(jid):
        job_or_404(jid)
        s = load_setup(jid)
        cs = CoordinateSystem.from_dict(s.get("coords"))
        return jsonify(setup=s, residuals=cs.residuals(), upp=cs.units_per_pixel() if cs.defined else None)

    @app.put("/api/jobs/<jid>/coords")
    def put_coords(jid):
        job_or_404(jid)
        b = request.get_json(silent=True) or {}
        setup = load_setup(jid)
        try:
            if b.get("clear"):
                cs = CoordinateSystem()
            elif b.get("method") == "origin_axis":
                cs = CoordinateSystem.from_origin_axis(b["origin"], b["axis"], b["distance"], b.get("units", "mm"),
                                                       y_up=b.get("y_up", True), z_plane=b.get("z_plane", 0.0))
            else:
                cs = CoordinateSystem.from_points(b.get("points", []), b.get("units", "mm"), b.get("z_plane", 0.0))
        except (CoordinateError, KeyError, TypeError, ValueError) as exc:
            abort(400, str(exc) if isinstance(exc, CoordinateError) else "Some of the numbers are missing or not valid.")
        setup["coords"] = cs.to_dict()
        if "estimate_lift" in b:
            setup["estimate_lift"] = bool(b["estimate_lift"])
        save_setup(jid, setup)
        return jsonify(ok=True, residuals=cs.residuals(), upp=cs.units_per_pixel() if cs.defined else None)

    @app.put("/api/jobs/<jid>/lift")
    def put_lift(jid):
        job_or_404(jid)
        setup = load_setup(jid)
        setup["estimate_lift"] = bool((request.get_json(silent=True) or {}).get("on"))
        save_setup(jid, setup)
        return jsonify(ok=True)

    @app.put("/api/jobs/<jid>/locations/<int:tid>")
    def put_location(jid, tid):
        job_or_404(jid)
        b = request.get_json(silent=True) or {}
        setup = load_setup(jid)
        locs = TuftLocations.from_dict(setup.get("locations"))
        try:
            locs.set(tid, b.get("x"), b.get("y"), b.get("z"))
        except LocationError as exc:
            abort(400, str(exc))
        setup["locations"] = locs.to_dict()
        save_setup(jid, setup)
        return jsonify(ok=True, count=len(locs))

    @app.post("/api/jobs/<jid>/locations.csv")
    def import_locations(jid):
        job_or_404(jid)
        f = request.files.get("file")
        text = f.read().decode("utf-8-sig", "replace") if f else request.get_data(as_text=True)
        try:
            new = TuftLocations.from_csv(text)
        except LocationError as exc:
            abort(400, str(exc))
        setup = load_setup(jid)
        locs = TuftLocations.from_dict(setup.get("locations"))
        locs.items.update(new.items)
        setup["locations"] = locs.to_dict()
        save_setup(jid, setup)
        return jsonify(ok=True, imported=len(new), count=len(locs))

    @app.delete("/api/jobs/<jid>/locations")
    def clear_locations(jid):
        job_or_404(jid)
        setup = load_setup(jid)
        setup["locations"] = {}
        save_setup(jid, setup)
        return jsonify(ok=True)

    @app.post("/api/jobs/<jid>/rerender")
    def rerender(jid):
        m = job_or_404(jid)
        if m["kind"] != "video" or m["status"] != "done":
            abort(409, "Nothing to redraw yet.")
        res = load_results(jid)
        cs = CoordinateSystem.from_dict(load_setup(jid).get("coords"))

        def go():
            store.update(jid, render="running")
            try:
                analysis.rerender_video(store.dir(jid) / m["input"], store.dir(jid), res, cs)
                store.update(jid, render="done", render_ver=(store.get(jid) or {}).get("render_ver", 0) + 1)
            except Exception as exc:
                store.update(jid, render="error", error=str(exc))
        threading.Thread(target=go, daemon=True).start()
        return jsonify(ok=True)

    # ------------------------------------------------------------ tables
    @app.get("/api/jobs/<jid>/overview.json")
    def overview(jid):
        job_or_404(jid)
        return jsonify(tables.overview(load_results(jid), load_setup(jid)))

    @app.get("/api/jobs/<jid>/summary.json")
    def summary(jid):
        job_or_404(jid)
        s = tables.summary_frame(load_results(jid), load_setup(jid), history(jid))
        return jsonify(rows=clean(s), units=tables.effective_cs(load_setup(jid), load_results(jid)["height"]).units)

    @app.get("/api/jobs/<jid>/tuft/<int:tid>.json")
    def tuft(jid, tid):
        job_or_404(jid)
        h = history(jid)
        g = h[h["tuft_id"] == tid]
        if g.empty:
            abort(404, f"Tuft {tid} was not found.")
        s = tables.summary_frame(load_results(jid), load_setup(jid), g)
        return jsonify(summary=clean(s)[0], rows=clean(g.drop(columns=["tuft_id"])))

    @app.get("/api/jobs/<jid>/history.json")
    def history_json(jid):
        job_or_404(jid)
        h = history(jid)
        try:
            off = max(0, int(request.args.get("offset", 0)))
            lim = min(2000, max(1, int(request.args.get("limit", 200))))
        except ValueError:
            abort(400, "offset and limit must be numbers")
        if request.args.get("tuft"):
            h = h[h["tuft_id"] == int(request.args["tuft"])]
        cols = ["time_s", "tuft_id", "x", "y", "z", "vector_x", "vector_y", "vector_z", "angle_deg",
                "flow_state", "length", "confidence"]
        return jsonify(total=int(len(h)), rows=clean(h.iloc[off:off + lim][cols]))

    # ------------------------------------------------------------ downloads
    @app.get("/api/jobs/<jid>/download/<what>")
    def download(jid, what):
        m = job_or_404(jid)
        res, setup = load_results(jid), load_setup(jid)
        stem = Path(m["name"]).stem
        h = history(jid)
        if what == "vector_time_history.csv":
            return csv_response(h, f"{stem}_vector_time_history.csv")
        if what == "tuft_summary.csv":
            return csv_response(tables.summary_frame(res, setup, h), f"{stem}_tuft_summary.csv")
        if what == "tuft_locations.csv":
            s = tables.locations_frame(res, setup)
            return csv_response(s[["tuft_id", "x", "y", "z"]], f"{stem}_tuft_locations.csv")
        if what == "results.xlsx":
            return Response(tables.excel_bytes(res, setup, m["name"]),
                            mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            headers={"Content-Disposition": f'attachment; filename="{stem}_results.xlsx"'})
        if what in ("annotated.mp4", "annotated.jpg"):
            p = store.dir(jid) / what
            if not p.exists():
                abort(404, "Not available for this file.")
            return send_file(p, as_attachment=True, download_name=f"{stem}_annotated{p.suffix}")
        if what == "all.zip":
            buf = io.BytesIO()
            with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
                z.writestr(f"{stem}_vector_time_history.csv", h.to_csv(index=False))
                z.writestr(f"{stem}_tuft_summary.csv", tables.summary_frame(res, setup, h).to_csv(index=False))
                z.writestr(f"{stem}_results.xlsx", tables.excel_bytes(res, setup, m["name"]))
                for n in ("annotated.mp4", "annotated.jpg"):
                    p = store.dir(jid) / n
                    if p.exists():
                        z.write(p, f"{stem}_annotated{p.suffix}")
            buf.seek(0)
            return send_file(buf, mimetype="application/zip", as_attachment=True, download_name=f"{stem}_results.zip")
        abort(404, "Unknown download.")

    return app
