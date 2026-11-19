"""measure.py - what runs did and measured, as rows that can be searched, joined and plotted: one SQLite file (<data>/measure.db).
The run folders stay the truth. Every row here comes from a run's plan.json, run.json or journal.jsonl and from nothing else, so the file can be deleted: it fills again from the runs that still exist. A run that is removed keeps its rows (measurements outlive the folders they came from); reading the runs again never drops them.
Tables:
  runs     one per run: id, top (the run it belongs to: itself, or the run whose step made it), plan, name, grp, owner, state, created, ended, error, bound (JSON {role: {provider, asset}}), inputs (JSON: the numbers and short texts it was started with), seen (how far its journal has been read, in bytes)
  work     one per step that ended: run, t (when it ended), step, of (the each-step it is an item of), seconds, error, role, capability, provider, asset (the last four only for work done through a role)
  figures  one per number of that piece of work: run, t, step, name, value. Names: "seconds"; whatever the provider reported, under its own names (tps, gen_n, prompt_n, ttft_ms, ...); the item values of an each-step as "item.<name>" (the prompt length this item was at)
Nothing here knows what a figure means. It is filled by sync(), which every question below does first: a run whose journal grew or whose state changed since it was last read is read from where reading stopped."""
import json, time, sqlite3
from pathlib import Path

SCHEMA = """CREATE TABLE IF NOT EXISTS runs (id TEXT PRIMARY KEY, top TEXT, plan TEXT, name TEXT, grp TEXT, owner TEXT, state TEXT, created REAL, ended REAL, error TEXT, bound TEXT, inputs TEXT, seen INTEGER DEFAULT 0);
CREATE INDEX IF NOT EXISTS runs_plan ON runs (plan, created);
CREATE TABLE IF NOT EXISTS work (run TEXT, t REAL, step TEXT, of TEXT, seconds REAL, error TEXT, role TEXT, capability TEXT, provider TEXT, asset TEXT, PRIMARY KEY (run, t, step));
CREATE INDEX IF NOT EXISTS work_t ON work (t);
CREATE INDEX IF NOT EXISTS work_asset ON work (asset, t);
CREATE TABLE IF NOT EXISTS figures (run TEXT, t REAL, step TEXT, name TEXT, value REAL, PRIMARY KEY (run, t, step, name));
CREATE INDEX IF NOT EXISTS figures_name ON figures (name);"""
BY = {"asset": "w.asset", "provider": "w.provider", "run": "r.top", "plan": "r.plan", "capability": "w.capability", "step": "(CASE WHEN w.of != '' THEN w.of ELSE w.step END)", "owner": "r.owner"}   # what points can be told apart by: the only column names a caller's choice ever becomes

class Measure:
    BY = BY
    def __init__(self, path: Path, director):
        self.path, self.director = Path(path), director
        with self._con() as c: c.executescript(SCHEMA)

    def _con(self) -> sqlite3.Connection:
        c = sqlite3.connect(self.path, timeout=30)
        c.row_factory = sqlite3.Row
        return c

    def q(self, sql: str, args: tuple = ()) -> list:
        with self._con() as c: return [dict(r) for r in c.execute(sql, args)]

    # --- Filling: from the run folders, a run at a time, from where its journal was last read ---

    def sync(self) -> int:
        """Reads what is new in every run the director holds. Answers how many runs were read."""
        seen, n = {r["id"]: r for r in self.q("SELECT id, state, seen FROM runs")}, 0
        with self._con() as c:
            for run in list(self.director.runs.values()):
                size, old = (run.dir / "journal.jsonl").stat().st_size if (run.dir / "journal.jsonl").exists() else 0, seen.get(run.id)
                if old and old["seen"] == size and old["state"] == run.meta["state"]: continue
                self._read(c, run, old["seen"] if old else 0)
                n += 1
        return n

    def _read(self, c, run, start: int):
        steps = {s["id"]: s for s in run.plan["steps"]}
        raw = (run.dir / "journal.jsonl").read_bytes()[start:] if (run.dir / "journal.jsonl").exists() else b""
        raw, at, work = raw[:raw.rfind(b"\n") + 1], start, {}   # whole lines only: a line being written is read next time
        for line in raw.splitlines(keepends=True):
            ev = json.loads(line)
            if ev["what"] == "work": work[ev["step"]] = (at, ev)   # what a step did through a role, written just before its end
            elif ev["what"] == "end":
                w, s = work.pop(ev["step"], (0, {}))[1], steps.get(ev["step"], {})
                c.execute("INSERT OR REPLACE INTO work VALUES (?,?,?,?,?,?,?,?,?,?)", (run.id, ev["t"], ev["step"], s.get("of", ""), ev.get("seconds"), ev.get("error", ""), w.get("role", ""), w.get("capability", ""), w.get("provider", ""), w.get("asset", "")))
                figs = {"seconds": ev.get("seconds"), **(w.get("usage") or {}), **{f"item.{k}": v for k, v in (s.get("item") or {}).items()}}
                c.executemany("INSERT OR REPLACE INTO figures VALUES (?,?,?,?,?)", [(run.id, ev["t"], ev["step"], k, float(v)) for k, v in figs.items() if isinstance(v, (int, float)) and not isinstance(v, bool)])
            at += len(line)
        if work: at = min(a for a, _ in work.values())   # a step's work without its end yet: read again from there next time
        m, short = run.meta, lambda v: isinstance(v, (int, float, bool)) or (isinstance(v, str) and len(v) <= 200)
        inputs = {k: run.data[k] for k in (run.plan.get("params") or {}) if k in run.data and short(run.data[k])}
        c.execute("INSERT OR REPLACE INTO runs VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)", (run.id, run.id.split("~")[0], run.plan.get("id", ""), run.plan.get("name", ""), run.plan.get("group", ""), m.get("owner", ""), m["state"], m.get("created"), m.get("ended"), m.get("error", ""), json.dumps({r: {k: b.get(k) for k in ("provider", "asset")} for r, b in (m.get("bound") or {}).items()}), json.dumps(inputs, ensure_ascii=False), at))

    def rebuild(self) -> int:
        """Every run that still has its folder is read again from the start. Rows of runs whose folders are gone are kept: there is nothing left to read them from."""
        with self._con() as c:
            for table, col in (("figures", "run"), ("work", "run"), ("runs", "id")): c.executemany(f"DELETE FROM {table} WHERE {col} = ?", [(rid,) for rid in self.director.runs])
        return self.sync()

    # --- Questions ---

    def names(self, plan: str = "") -> list:
        """The figure names there are, most frequent first, for all work or for one plan's (its runs and the runs their steps made)."""
        self.sync()
        return [r["name"] for r in self.q("SELECT f.name, COUNT(*) n FROM figures f JOIN runs r ON r.id = f.run WHERE (? = '' OR r.plan = ? OR r.top IN (SELECT id FROM runs WHERE plan = ?)) GROUP BY f.name ORDER BY n DESC, f.name", (plan, plan, plan))]

    def plans(self) -> list:
        """[(plan id, name)] of the plans whose runs have rows, the latest first. A run made by another run's step counts for that run's plan."""
        self.sync()
        return [(r["plan"], r["name"]) for r in self.q("SELECT plan, name, MAX(created) c FROM runs WHERE id = top AND plan != '' GROUP BY plan ORDER BY c DESC")]

    def points(self, x: str, y: str, by: str = "asset", plan: str = "", since: float = 0, owner: str = "", split: str = "") -> list:
        """One row per piece of work that has both figures: {"x", "y", "by": what tells it apart (see BY), "split": a third figure's value where split names one (the pass an item was, ...), "run", "step", "t", "asset", "provider"}. x may be "t": when it ended. owner: only that person's runs."""
        self.sync()
        fig = lambda a: f"JOIN figures {a} ON {a}.run = w.run AND {a}.t = w.t AND {a}.step = w.step AND {a}.name = ?"
        return self.q(f"""SELECT {"w.t" if x == "t" else "fx.value"} x, fy.value y, {BY[by]} by, {"fs.value" if split else "NULL"} split, r.top run, w.step step, w.t t, w.asset asset, w.provider provider FROM work w JOIN runs r ON r.id = w.run {fig("fy")} {"" if x == "t" else fig("fx")} {"LEFT " + fig("fs") if split else ""} WHERE w.t >= ? AND (? = '' OR r.owner = ?) AND (? = '' OR r.plan = ? OR r.top IN (SELECT id FROM runs WHERE plan = ?)) ORDER BY w.t""", (y, *(() if x == "t" else (x,)), *((split,) if split else ()), since, owner, owner, plan, plan, plan))

    def usage(self, since: float = 0, owner: str = "") -> list:
        """What was used since then, per provider, asset and capability: {"provider", "asset", "capability", "n", "seconds", "failed", "figures": {name: {"sum", "avg"}}}, the most used first."""
        self.sync()
        rows = {(r["provider"], r["asset"], r["capability"]): {**r, "figures": {}} for r in self.q("SELECT w.provider, w.asset, w.capability, COUNT(*) n, SUM(w.seconds) seconds, SUM(w.error != '') failed FROM work w JOIN runs r ON r.id = w.run WHERE w.capability != '' AND w.t >= ? AND (? = '' OR r.owner = ?) GROUP BY w.provider, w.asset, w.capability ORDER BY seconds DESC", (since, owner, owner))}
        for f in self.q("SELECT w.provider, w.asset, w.capability, f.name, SUM(f.value) s, AVG(f.value) a FROM figures f JOIN work w ON w.run = f.run AND w.t = f.t AND w.step = f.step JOIN runs r ON r.id = w.run WHERE w.capability != '' AND w.t >= ? AND (? = '' OR r.owner = ?) AND f.name NOT LIKE 'item.%' GROUP BY w.provider, w.asset, w.capability, f.name", (since, owner, owner)): rows[(f["provider"], f["asset"], f["capability"])]["figures"][f["name"]] = {"sum": f["s"], "avg": f["a"]}
        return list(rows.values())

    def counts(self) -> dict:
        self.sync()
        return {t: self.q(f"SELECT COUNT(*) n FROM {t}")[0]["n"] for t in ("runs", "work", "figures")} | {"since": (self.q("SELECT MIN(created) c FROM runs")[0]["c"] or time.time())}
