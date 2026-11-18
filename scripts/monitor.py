"""
Monitor de simulaciones en marcha: lee .jobs/*.job (los escribe scripts/track.sh) y muestra, refrescando cada pocos
segundos, el tiempo transcurrido y la última línea del log de cada trabajo, más el uso de la GPU.  Un trabajo
desaparece al terminar (track.sh borra su registro) o si su proceso ya no existe.
Uso:  python scripts/monitor.py [segundos]        (Ctrl+C para salir;  --once: una sola vez, sin refrescar)
"""
import os, re, subprocess, sys, time

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
JOBS = os.path.join(ROOT, ".jobs")


def last_line(path, nbytes=8192):
    try:
        with open(path, "rb") as f:
            f.seek(0, 2); f.seek(max(0, f.tell() - nbytes))
            txt = f.read().decode("utf-8", "replace")
    except OSError:
        return "(sin log todavía)"
    lines = [l.strip() for l in re.split(r"[\r\n]", txt) if l.strip()]
    return lines[-1] if lines else "(log vacío)"


def alive(pid):
    try:
        os.kill(pid, 0); return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def fmt(sec):
    h, r = divmod(int(sec), 3600); m, s = divmod(r, 60)
    return f"{h}h{m:02d}m" if h else f"{m}m{s:02d}s"


def gpu():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5).stdout
        u, m, t = [x.strip() for x in out.strip().split(",")]
        return f"GPU {u}%  {int(m) / 1024:.1f}/{int(t) / 1024:.1f} GB"
    except Exception:
        return "GPU: (nvidia-smi no disponible)"


def snapshot(width):
    rows = []
    for fn in sorted(os.listdir(JOBS)) if os.path.isdir(JOBS) else []:
        if not fn.endswith(".job"):
            continue
        p = os.path.join(JOBS, fn)
        try:
            d = dict(l.split("=", 1) for l in open(p).read().splitlines() if "=" in l)
        except OSError:
            continue
        if not alive(int(d.get("pid", 0))):
            os.remove(p); continue
        log = os.path.join(ROOT, d["log"])
        rows.append((fn[:-4], fmt(time.time() - int(d["start"])), last_line(log)[: width - 4]))
    out = [f"{time.strftime('%H:%M:%S')}  {gpu()}  |  {len(rows)} en marcha", "─" * min(width, 120)]
    if not rows:
        out.append("Nada corriendo.")
    for lab, el, ll in rows:
        out += [f"● {lab}  ({el})", f"    {ll}"]
    fin = os.path.join(JOBS, "finished.log")
    if os.path.exists(fin):
        last = open(fin).read().splitlines()[-3:]
        if last:
            out += ["─" * min(width, 120), "Últimos terminados:"] + [f"    {l}" for l in last]
    return "\n".join(out)


if __name__ == "__main__":
    once = "--once" in sys.argv
    args = [x for x in sys.argv[1:] if x != "--once"]
    dt = float(args[0]) if args else 5.0
    try:
        while True:
            w = os.get_terminal_size().columns if sys.stdout.isatty() else 160
            txt = snapshot(w)
            if once:
                print(txt); break
            print("\033[2J\033[H" + txt, flush=True)
            time.sleep(dt)
    except KeyboardInterrupt:
        pass
