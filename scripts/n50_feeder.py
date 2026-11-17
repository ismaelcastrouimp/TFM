"""
n50_feeder.py
=============
Alimentador local del barrido N=50 en el CESGA (Claude, 12-11).  T = 0.0..4.0 paso 0.1 x {tl, sw} con
slurm/claude_n50.slurm.  Corre en local (bucle cada 5 min, una sola conexion ssh por vuelta), es REANUDABLE (estado en
data/gap_arch/n50/feeder/state.json) y nunca cancela nada.

Logica:
  - Piloto (tl_T3.0, sw_T3.0, ya enviados): el metodo m no envia nada hasta que su piloto este DONE; si falla, m se detiene.
  - Anclas T in {4.0, 2.0, 1.0, 0.5} (y 3.0 = piloto) desde in/X5_to_N50.msgpack.
  - Resto: desde params del ancla MAS CERCANA del mismo metodo que ya tenga params en el CESGA (empate -> la T mayor).
  - Limite: trabajos del usuario en short (Ismael + Claude) <= LIMIT.
  - Fin de un trabajo: DONE si existe obs_<label>.json; PARTIAL si solo params; FAILED si nada.  Reintento (1) solo si
    sacct dice FAILED o NODE_FAIL (no TIMEOUT / OUT_OF_MEMORY / CANCELLED).
  - rsync de out/n50 y logs/cl_n50_* a data/cesga/n50 en cada vuelta con novedades; al final rsync --remove-source-files
    (borra del CESGA lo de Claude una vez copiado).
Eventos (una linea por evento) en data/gap_arch/n50/feeder/events.log; estado legible en data/gap_arch/n50/STATUS.txt.
Uso: python3 scripts/n50_feeder.py   (run_in_background)
"""
import json, os, subprocess, sys, time, datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SSH = ["ssh", "-F", os.path.expanduser("~/.config/claude-ssh/config"), "ft3"]
RSH = "ssh -F " + os.path.expanduser("~/.config/claude-ssh/config")
CDIR = "/home/csic/atm/icl/Documentos/TFM/claude"
FD = os.path.join(ROOT, "data/gap_arch/n50/feeder"); os.makedirs(FD, exist_ok=True)
LOC = os.path.join(ROOT, "data/cesga/n50"); os.makedirs(os.path.join(LOC, "logs"), exist_ok=True)
STATE, EV, STATUS = os.path.join(FD, "state.json"), os.path.join(FD, "events.log"), os.path.join(ROOT, "data/gap_arch/n50/STATUS.txt")
LIMIT, PERIOD = 50, 300
ANCH = [4.0, 3.0, 2.0, 1.0, 0.5]
TS = [round(0.1 * i, 1) for i in range(41)]
X5 = "in/X5_to_N50.msgpack:8"
assert "/data/0" not in LOC


def now(): return datetime.datetime.now().strftime("%d-%m %H:%M")
def ev(msg):
    with open(EV, "a") as f: f.write(f"[{now()}] {msg}\n")
    print(f"[{now()}] {msg}", flush=True)
def ssh(cmd, timeout=120):
    r = subprocess.run(SSH + [cmd], capture_output=True, text=True, timeout=timeout)
    if r.returncode != 0: raise RuntimeError(r.stderr.strip()[-300:])
    return r.stdout
def lab(m, T): return f"{m}_T{T:.1f}"


def load():
    if os.path.exists(STATE): return json.load(open(STATE))
    st = {}
    for m in ("tl", "sw"):
        for T in TS:
            st[lab(m, T)] = dict(m=m, T=T, job=None, status="TODO", tries=0, init=None)
    st["tl_T3.0"].update(job="11598617", status="SUBMITTED", tries=1, init=X5)
    st["sw_T3.0"].update(job="11598618", status="SUBMITTED", tries=1, init=X5)
    return st
def save(st):
    json.dump(st, open(STATE + ".tmp", "w"), indent=1); os.replace(STATE + ".tmp", STATE)


def poll():
    """Una conexion: trabajos del usuario en cola (id particion), ficheros de out/n50."""
    out = ssh(f"squeue -u csatmicl -h -o '%i %P'; echo '@@'; ls {CDIR}/out/n50 2>/dev/null; echo '@@END'")
    q, files = out.split("@@\n", 1)
    jobs = {l.split()[0]: l.split()[1] for l in q.strip().splitlines() if l.strip()}
    files = set(files.replace("@@END", "").split())
    return jobs, files


def pick_init(st, m, T, files):
    if T in ANCH: return X5
    cands = [A for A in ANCH if f"params_{lab(m, A)}.msgpack" in files and st[lab(m, A)]["status"] != "SUBMITTED"]
    if not cands:  # ninguna ancla con params: si ya no queda ninguna por acabar, desde X5
        return X5 if all(st[lab(m, A)]["status"] not in ("TODO", "SUBMITTED") for A in ANCH) else None
    A = min(cands, key=lambda A: (abs(A - T), -A))
    # solo si el ancla mas cercana posible (de las 5) ya esta; si no, esperar a que acabe (salvo que haya fallado)
    best = min(ANCH, key=lambda A: (abs(A - T), -A))
    if A != best and st[lab(m, best)]["status"] in ("TODO", "SUBMITTED"): return None
    return f"out/n50/params_{lab(m, A)}.msgpack:8"


def submit(st, L, init):
    e = st[L]
    jid = ssh(f"cd {CDIR} && sbatch --parsable --job-name=cl_n50_{L} "
              f"--export=ALL,METHOD={e['m']},T={e['T']:.1f},INIT={init},LABEL={L} ../slurm/claude_n50.slurm").strip().split(";")[0]
    e.update(job=jid, status="SUBMITTED", tries=e["tries"] + 1, init=init)
    ev(f"ENVIADO {L} job {jid} init {init}")


def rsync(remove=False):
    extra = ["--remove-source-files"] if remove else []
    for src, dst in ((f"ft3:{CDIR}/out/n50/", LOC + "/"), (f"ft3:{CDIR}/logs/cl_n50_*", LOC + "/logs/")):
        subprocess.run(["rsync", "-a", *extra, "-e", RSH, src, dst], capture_output=True, timeout=1800)


def write_status(st, jobs):
    c = {}
    for e in st.values(): c[e["status"]] = c.get(e["status"], 0) + 1
    run = [L for L, e in st.items() if e["status"] == "SUBMITTED"]
    with open(STATUS, "w") as f:
        f.write(f"Barrido N=50 (alimentador Claude) - actualizado {now()}\n{c}\n")
        f.write(f"Trabajos del usuario en cola: {len(jobs)} (short: {sum(p == 'short' for p in jobs.values())})\n")
        f.write("En cola/ejecucion: " + " ".join(sorted(run)) + "\n")
        for s in ("DONE", "PARTIAL", "FAILED"):
            f.write(f"{s}: " + " ".join(sorted(L for L, e in st.items() if e["status"] == s)) + "\n")
        f.write("Tabla: data/gap_arch/n50/summary.txt   Eventos: data/gap_arch/n50/feeder/events.log\n")


def main():
    st = load(); ev("alimentador arrancado"); fails = 0
    while True:
        try:
            jobs, files = poll(); fails = 0
        except Exception as ex:
            fails += 1; ev(f"aviso: fallo de ssh ({fails}): {ex}")
            if fails >= 36: ev("PARADA: 3 h sin conexion con el CESGA"); sys.exit(3)
            time.sleep(PERIOD); continue
        changed = False
        # 1) trabajos terminados
        for L, e in st.items():
            if e["status"] != "SUBMITTED" or e["job"] in jobs: continue
            try: sa = ssh(f"sacct -j {e['job']} -n -X -o State").split()
            except Exception: continue
            sa = sa[0] if sa else "?"
            if f"obs_{L}.json" in files: e["status"] = "DONE"
            elif f"params_{L}.msgpack" in files: e["status"] = "PARTIAL"
            else: e["status"] = "FAILED"
            ev(f"{e['status']} {L} job {e['job']} (sacct {sa})"); changed = True
            if e["status"] == "FAILED" and sa in ("FAILED", "NODE_FAIL") and e["tries"] < 2:
                e["status"] = "TODO"; ev(f"reintento programado {L}")
            if L in ("tl_T3.0", "sw_T3.0") and e["status"] != "DONE":
                ev(f"PILOTO {L} NO OK ({e['status']}): metodo {e['m']} detenido")
        save(st)
        # 2) envios
        nshort = sum(p == "short" for p in jobs.values())
        todo = sorted((L for L, e in st.items() if e["status"] == "TODO" and st[lab(e["m"], 3.0)]["status"] == "DONE"),
                      key=lambda L: (st[L]["T"] not in ANCH, min(abs(st[L]["T"] - A) for A in ANCH), -st[L]["T"], L))
        for L in todo:
            if nshort >= LIMIT: break
            init = pick_init(st, st[L]["m"], st[L]["T"], files)
            if init is None: continue
            try: submit(st, L, init); nshort += 1; save(st); changed = True
            except Exception as ex: ev(f"aviso: sbatch {L} fallo: {ex}"); break
        if changed:
            rsync()
            subprocess.run([sys.executable, os.path.join(ROOT, "scripts/n50_summary.py")], capture_output=True, timeout=300)
        write_status(st, jobs)
        # 3) fin
        pend = [L for L, e in st.items() if e["status"] in ("SUBMITTED",) or
                (e["status"] == "TODO" and st[lab(e["m"], 3.0)]["status"] in ("DONE", "SUBMITTED", "TODO"))]
        if not pend:
            rsync(); rsync(remove=True)
            subprocess.run([sys.executable, os.path.join(ROOT, "scripts/n50_summary.py")], capture_output=True, timeout=300)
            ev("FIN DEL BARRIDO: todo copiado a data/cesga/n50 y borrado del CESGA"); write_status(st, {}); return
        time.sleep(PERIOD)


if __name__ == "__main__":
    main()
