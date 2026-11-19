"""resources.py - what is available, as it really is (decision D-05): groupings, their parts, models and where their copies are, and every figure with what it is said to be beside what it was measured to be.
One SQLite file (resources.db beside the library):
  groups(id, kind, name, provider)        a machine, a site, an outside service - a grouping of parts, for naming and entering them; the planner reads parts and paths
  parts(id, grp, kind, name, attrs)       cpu, ram, gpu, storage (one per place the machine keeps things), link (a path between two groups: "a~b", measured from a, to_mb_s is a to b)
  models(id, name, facts)                 one row per distinct model file, whatever it is called where (identity: architecture and exact size; name and size where the header cannot be read); facts are its header's
  copies(model, grp, name, location, size, t)   where a copy of it is, under the name that machine uses
  figures(subject, key, stated, measured, n, unit, source, t)   one figure of a part, a model, or a model on a group ("<model>@<group>": decode_tps, prefill_tps, ttft_s, first_s, ram_gb_loaded, gpu_gb_loaded): stated is the label or a value computed from labels (a first guess); measured is the running mean of tests, n how many; confidence = n / (n + 2)
Sync rebuilds groups, parts, models and copies from what the providers report; figures are kept and only added to. Nothing here decides placement: the planner and Asio read it.
estimate(): a figure not measured on one group, guessed from what was measured elsewhere (decode speed from the group's measured memory rate per byte of the model), marked as an estimate."""
import json, time, sqlite3, asyncio, ipaddress, urllib.parse
from pathlib import Path
from . import providers

SCHEMA = """
CREATE TABLE IF NOT EXISTS groups(id TEXT PRIMARY KEY, kind TEXT, name TEXT, provider TEXT);
CREATE TABLE IF NOT EXISTS parts(id TEXT PRIMARY KEY, grp TEXT, kind TEXT, name TEXT, attrs TEXT);
CREATE TABLE IF NOT EXISTS models(id TEXT PRIMARY KEY, name TEXT, facts TEXT);
CREATE TABLE IF NOT EXISTS copies(model TEXT, grp TEXT, name TEXT, location TEXT, size INTEGER, t REAL, PRIMARY KEY(model, grp, name));
CREATE TABLE IF NOT EXISTS figures(subject TEXT, key TEXT, stated REAL, measured REAL, n INTEGER DEFAULT 0, unit TEXT, source TEXT, t REAL, PRIMARY KEY(subject, key));
"""
PORTAL = "portal"   # the group every carried file passes through

def conf(n: int) -> float: return round(n / (n + 2), 2)

class Resources:
    def __init__(self, path: Path, lib):
        self.path, self.lib = path, lib
        with self._db() as db: db.executescript(SCHEMA)

    def _db(self): return sqlite3.connect(self.path)

    # --- Figures ---

    def state(self, subject: str, key: str, value, unit: str = "", source: str = "label"):
        """What a part or model says it is (or what follows from that by calculation). Measurements are never touched."""
        if value is None: return
        with self._db() as db: db.execute("INSERT INTO figures(subject, key, stated, unit, source, t) VALUES (?,?,?,?,?,?) ON CONFLICT(subject, key) DO UPDATE SET stated=excluded.stated, unit=excluded.unit, t=excluded.t", (subject, key, float(value), unit, source, time.time()))

    def measure(self, subject: str, key: str, value: float, unit: str = "", source: str = "test"):
        """One measurement folded into the running mean."""
        with self._db() as db:
            row = db.execute("SELECT measured, n FROM figures WHERE subject=? AND key=?", (subject, key)).fetchone()
            m, n = (row or (None, 0))[0], (row or (None, 0))[1] or 0
            new = value if m is None else (m * n + value) / (n + 1)
            db.execute("INSERT INTO figures(subject, key, measured, n, unit, source, t) VALUES (?,?,?,?,?,?,?) ON CONFLICT(subject, key) DO UPDATE SET measured=?, n=?, source=excluded.source, t=excluded.t, unit=COALESCE(NULLIF(excluded.unit, ''), figures.unit)", (subject, key, new, n + 1, unit, source, time.time(), new, n + 1))

    def figures(self, subject: str = "", prefix: bool = False) -> dict:
        """{subject: {key: {stated, measured, n, conf, unit, source, t, best}}}; best is measured where there is a measurement, else stated."""
        with self._db() as db: rows = db.execute("SELECT subject, key, stated, measured, n, unit, source, t FROM figures" + (" WHERE subject LIKE ?" if prefix else " WHERE subject=?" if subject else ""), (subject + "%",) if prefix else (subject,) if subject else ()).fetchall()
        out = {}
        for s, k, st, me, n, u, src, t in rows: out.setdefault(s, {})[k] = {"stated": st, "measured": me, "n": n or 0, "conf": conf(n or 0), "unit": u, "source": src, "t": t, "best": me if me is not None else st}
        return out

    def estimate(self, model: str, grp: str) -> dict:
        """Decode speed of a model on a group where it was not measured: the group's effective memory rate (measured decode speed x bytes read per token, over the models measured there; else its memory test) over this model's bytes per token. Marked as an estimate; None when nothing is known."""
        f, m = self.figures(prefix=True), self.model(model)
        if not m: return {}
        per_tok = lambda mid: (self.model(mid) or {}).get("size", 0) * min(1.0, ((self.model(mid) or {}).get("facts", {}).get("experts_used") or 1) / ((self.model(mid) or {}).get("facts", {}).get("experts") or 1))   # bytes read per token: the whole file, or the used share of experts (rough: shared weights not counted apart)
        rates = [d["decode_tps"]["measured"] * per_tok(s.split("@")[0]) for s, d in f.items() if s.endswith("@" + grp) and d.get("decode_tps", {}).get("measured")]
        rate = sum(rates) / len(rates) if rates else (f.get(f"{grp}/ram", {}).get("mem_gb_s", {}).get("best") or 0) * 1e9 * 0.6   # a memory test reads faster than inference does; 0.6 is a first guess, replaced as soon as one model is measured there
        return {"decode_tps": round(rate / per_tok(model), 1), "from": f"{len(rates)} model(s) measured here" if rates else "the memory test only", "estimate": True} if rate and per_tok(model) else {}

    def speed(self, name: str, grp: str) -> float:
        """Decode speed of the copy called name on group grp: measured there, else estimated, else 0 (nothing known). What placement sorts by among things that fit."""
        with self._db() as db: r = db.execute("SELECT model FROM copies WHERE name=? AND grp=?", (name, grp)).fetchone()
        if not r: return 0.0
        f = self.figures(f"{r[0]}@{grp}").get(f"{r[0]}@{grp}", {}).get("decode_tps", {})
        return f.get("measured") or self.estimate(r[0], grp).get("decode_tps") or 0.0

    # --- Groups, parts, models, copies ---

    def groups(self) -> list:
        with self._db() as db: return [dict(zip(("id", "kind", "name", "provider"), r)) for r in db.execute("SELECT id, kind, name, provider FROM groups ORDER BY name")]

    def parts(self, grp: str = "") -> list:
        with self._db() as db: return [{**dict(zip(("id", "grp", "kind", "name"), r[:4])), "attrs": json.loads(r[4] or "{}")} for r in db.execute("SELECT id, grp, kind, name, attrs FROM parts" + (" WHERE grp=?" if grp else ""), (grp,) if grp else ())]

    def model(self, mid: str) -> dict:
        with self._db() as db: r = db.execute("SELECT id, name, facts FROM models WHERE id=?", (mid,)).fetchone()
        return {"id": r[0], "name": r[1], "facts": json.loads(r[2] or "{}"), "size": int(mid.rsplit("-", 1)[-1]) if mid.rsplit("-", 1)[-1].isdigit() else 0} if r else {}

    def models(self) -> list:
        with self._db() as db:
            ms = [{"id": r[0], "name": r[1], "facts": json.loads(r[2] or "{}")} for r in db.execute("SELECT id, name, facts FROM models ORDER BY name")]
            cs = db.execute("SELECT model, grp, name, location, size FROM copies").fetchall()
        return [m for m in ({**m, "size": int(m["id"].rsplit("-", 1)[-1]), "copies": [dict(zip(("grp", "name", "location", "size"), c[1:])) for c in cs if c[0] == m["id"]]} for m in ms) if m["copies"]]   # a model no machine holds a copy of any more (removed, or once listed only as passed on) is not a model of the set-up; its measured figures stay stored under its id

    def _put(self, db, table: str, row: dict): db.execute(f"""INSERT OR REPLACE INTO {table}({", ".join(row)}) VALUES ({", ".join("?" * len(row))})""", [json.dumps(v) if isinstance(v, (dict, list)) else v for v in row.values()])

    async def _sync_one(self, p: dict) -> dict:
        """One provider's machine as a group of parts, its models and copies. Answers what was found, or the error."""
        g = p["id"]
        try: node, locs, ms = await asyncio.gather(providers.machine(p, "GET", "/node", timeout=10.0), providers.machine(p, "GET", "/locations", timeout=10.0), providers.machine(p, "GET", "/models", timeout=20.0))
        except providers.ProviderError as e: return {"group": g, "error": str(e)}
        hw = node.get("hardware") or {}
        with self._db() as db:
            db.execute("DELETE FROM parts WHERE grp=?", (g,))
            db.execute("DELETE FROM copies WHERE grp=?", (g,))
            self._put(db, "groups", {"id": g, "kind": "machine", "name": p.get("name") or g, "provider": g})
            self._put(db, "parts", {"id": f"{g}/cpu", "grp": g, "kind": "cpu", "name": hw.get("cpu", ""), "attrs": {k: hw.get(k) for k in ("cores", "threads", "arch", "cpu_features")}})
            self._put(db, "parts", {"id": f"{g}/ram", "grp": g, "kind": "ram", "name": "memory", "attrs": {"swap_gb": hw.get("swap_gb"), "limit_gb": hw.get("cgroup_mem_limit_gb")}})
            for i, gpu in enumerate(hw.get("gpus") or []): self._put(db, "parts", {"id": f"{g}/gpu{i}", "grp": g, "kind": "gpu", "name": str(gpu.get("name") if isinstance(gpu, dict) else gpu), "attrs": gpu if isinstance(gpu, dict) else {}})
            for loc in locs.get("locations", []): self._put(db, "parts", {"id": f"""{g}/storage/{loc["name"]}""", "grp": g, "kind": "storage", "name": loc["name"], "attrs": {"path": loc["path"], **(loc.get("disk") or {})}})
            for n in node.get("net") or []: self._put(db, "parts", {"id": f"""{g}/net/{n["name"]}""", "grp": g, "kind": "net", "name": n["name"], "attrs": n})   # an older node reports no interfaces
        for n in node.get("net") or []: self.state(f"""{g}/net/{n["name"]}""", "link_mbit", n.get("speed_mbit"), "Mbit/s", "negotiated link, as the driver reports it")
        self.state(f"{g}/cpu", "threads", hw.get("threads"), "threads")
        self.state(f"{g}/ram", "total_gb", hw.get("ram_gb"), "GB")
        for loc in locs.get("locations", []): self.state(f"""{g}/storage/{loc["name"]}""", "total_gb", (loc.get("disk") or {}).get("total_gb"), "GB", "reported by the machine")
        found = 0
        for m in [m for m in ms.get("models", []) if m.get("size") and m.get("adapter") != "remote"]:   # a model the machine only passes on from another server (its "remote" program: <source>/<model>) is not a copy it holds; it is listed where it is kept
            facts = {}
            if "chat" in m.get("capabilities", []) or "embed" in m.get("capabilities", []):
                try: facts = await providers.machine(p, "GET", f"""/api/model/facts?model={m["name"]}""", timeout=20.0)
                except providers.ProviderError as e: facts = {"unread": str(e)[:300]}   # a program without header facts (or an older node): the model is still listed, by size, with why its header was not read
            mid = f"""{facts["general.architecture"]}-{m["size"]}""" if facts.get("general.architecture") else f"""{m["name"].replace("-", "_")}-{m["size"]}"""   # the same file under other names is one model; without its header, a name and size are all there is to tell files apart
            with self._db() as db:
                if facts or not db.execute("SELECT 1 FROM models WHERE id=?", (mid,)).fetchone(): self._put(db, "models", {"id": mid, "name": m["name"], "facts": {**{k: v for k, v in facts.items() if k not in ("name", "size", "source")}, "capabilities": m.get("capabilities", [])}})
                self._put(db, "copies", {"model": mid, "grp": g, "name": m["name"], "location": m.get("standby") and f"""{m["standby"]} (off)""" or "", "size": m["size"], "t": time.time()})
            for k, unit in (("params_b", "B params"), ("block_count", "layers"), ("kv_bytes_per_token_f16", "bytes/token (f16 cache)"), ("context_length", "tokens"), ("bpw", "bits/weight")): self.state(mid, k, facts.get(k), unit, "file header" if k != "kv_bytes_per_token_f16" else "computed from the header (to be verified)")
            found += 1
        return {"group": g, "models": found, "parts": len(self.parts(g))}

    async def _sync_server(self, p: dict) -> dict:
        """A server that does not speak the node contract (speaks: ollama, ...) as a group of kind "server": the models it lists, each a copy kept there by the server itself. It says nothing of its machine, and its files are its own to move or remove, so it has no parts and its copies only show where a model is."""
        g = p["id"]
        try: ms = await providers.assets(p, timeout=20.0)
        except providers.ProviderError as e: return {"group": g, "error": str(e)}
        with self._db() as db:
            db.execute("DELETE FROM copies WHERE grp=?", (g,))
            self._put(db, "groups", {"id": g, "kind": "server", "name": p.get("name") or g, "provider": g})
            for m in ms:
                mid = f"""{m["name"].replace("-", "_")}-{int(m.get("size") or 0)}"""
                if not db.execute("SELECT 1 FROM models WHERE id=?", (mid,)).fetchone(): self._put(db, "models", {"id": mid, "name": m["name"], "facts": {"capabilities": m.get("capabilities", [])}})
                self._put(db, "copies", {"model": mid, "grp": g, "name": m["name"], "location": f"""kept by the {p["speaks"]} server itself""", "size": int(m.get("size") or 0), "t": time.time()})
        return {"group": g, "models": len(ms), "parts": 0}

    async def sync(self) -> list:
        """Every provider read again: Magpie nodes as machines with their parts, servers that are not Magpie (speaks: ollama) as groups with the models they list."""
        with self._db() as db: self._put(db, "groups", {"id": PORTAL, "kind": "portal", "name": "this portal", "provider": ""})
        return await asyncio.gather(*((self._sync_server if p.get("speaks") else self._sync_one)(p) for p in self.lib.list("provider")))

    # --- Models as movable resources (D-06): copy, move, remove a copy ---

    def copies(self, mid: str) -> list: return [c for m in self.models() if m["id"] == mid for c in m["copies"]]

    async def files_of(self, p: dict, name: str) -> list:
        """Where a model's file(s) are on a machine ([{location, path, size}]), as its program says. Empty when it cannot say (a model an outside server keeps)."""
        try: return (await providers.machine(p, "GET", f"/api/model/facts?model={name}", timeout=20.0)).get("files") or []
        except providers.ProviderError as e:
            if " 404" in str(e) or "answered 404" in str(e): return []   # no program there says where this model's files are (a model an outside server keeps)
            raise

    async def copy(self, mid: str, src: dict, name: str, dst: dict, note=lambda s: None) -> dict:
        """A copy of the model on dst, sent by src straight to dst (dst makes a ticket for the one folder, so neither holds the other's key). Kept under gguf_local/<name>/ in dst's models place; checked by size and header once there."""
        files, folder = await self.files_of(src, name), f"""gguf_local/{name.replace(":", "_").replace("/", "_")}"""
        if not files: raise providers.ProviderError(f"""{src.get("name") or src["id"]} cannot say where {name}'s files are, so it cannot be copied from there (a model an outside server keeps is copied by that server)""")
        t = await providers.machine(dst, "POST", "/files/ticket", {"location": "models", "under": folder, "seconds": 6 * 3600}, timeout=10.0)
        for f in files:
            to = f"""{folder}/{f["path"].rsplit("/", 1)[-1]}""" + ("" if f["path"].endswith(".gguf") else ".gguf")   # an Ollama store blob has no extension; as a file of its own it is a GGUF
            note(f"""sending {f["path"].rsplit("/", 1)[-1]} ({f["size"] / 1e9:.1f} GB)""")
            await providers.machine(src, "POST", "/files/send", {"from": {"location": f["location"], "path": f["path"]}, "to": {"url": dst["url"], "ticket": t["ticket"], "location": "models", "path": to}}, timeout=24 * 3600.0)
        await self._sync_one(dst)
        here = [c for c in self.copies(mid) if c["grp"] == dst["id"]]
        note(f"""arrived: {"the same model (size and header agree)" if here else "a file arrived but it does not read as the same model - check it before removing anything"}""")
        return {"copied": bool(here), "to": dst["id"], "folder": folder}

    async def pull(self, p: dict, name: str) -> dict:
        """A model brought onto a server that fetches its own (its translator's "manage.pull": an Ollama pulls from the registry it is set to use - an outside server unless it was pointed elsewhere). Answers when the server is done, which for a large model is long."""
        got = await providers.server_do(p, "pull", name, timeout=6 * 3600.0)
        await self._sync_server(p)
        return got

    async def remove(self, mid: str, p: dict, name: str, last: bool = False) -> dict:
        """One copy's files removed from a machine. The last copy is never removed unless last is said; a model an outside server keeps is removed through that server."""
        if len(self.copies(mid)) <= 1 and not last: return {"removed": False, "why": "this is the only copy: removing it is said separately"}
        if p.get("speaks"):   # kept by a server that is not a node: removed by that server's own act (its translator's "manage.remove"); what it frees it does not say
            await providers.server_do(p, "remove", name)
            await self._sync_server(p)
            return {"removed": True, "freed_gb": 0.0}
        files = await self.files_of(p, name)
        if not files: return {"removed": False, "why": f"""{p.get("name") or p["id"]} cannot say where its files are: a model an outside server keeps is removed by that server"""}
        if any(f["path"].startswith("blobs/") for f in files): return {"removed": False, "why": "it is in an Ollama store (blobs shared between models): remove it with Ollama"}
        freed = sum((await providers.machine(p, "POST", "/files/delete", {"location": f["location"], "path": f["path"], "confirm": True}, timeout=60.0)).get("freed_bytes", 0) for f in files)
        await self._sync_one(p)
        return {"removed": True, "freed_gb": round(freed / 1e9, 2)}

    # --- Tests (the characterizer: Asio's measured side) ---

    PROMPT = "Write a short paragraph about how rivers shape valleys over long spans of time, then list three examples of famous river valleys."   # the same text every time, so runs compare

    async def test_model(self, p: dict, name: str, mid: str, rounds: int = 2) -> dict:
        """A model's prefill and decode speed and time to first token on one machine, from its own figures. Cold against warm: whether it was already loaded is asked first; the first round's whole time is kept as cold_s (not loaded before) or first_s (it was), the later rounds' as warm_s, and their difference as load_s when it started cold."""
        got, s, wall = [], f"""{mid}@{p["id"]}""", []
        cold, watts = not any(x.get("model") == name for x in await self._running(p)), []
        async def sample():   # the machine's draw once a second while a warm round runs
            while True:
                if (w := await self._power(p)) is not None: watts.append(w)
                await asyncio.sleep(1.0)
        for i in range(rounds):
            t0 = time.time()
            watch = asyncio.create_task(sample()) if i == rounds - 1 and rounds > 1 else None
            try: out = await providers.run(p, "chat", {"model": name, "messages": [{"role": "user", "content": self.PROMPT}], "options": {"num_predict": 96}, "stream": False}, f"""characterize.{p["id"]}.{int(time.time())}.{i}""")
            finally:
                if watch: watch.cancel()
            wall.append(time.time() - t0)
            u = out.get("usage") or {}
            if i == 0 and rounds > 1: self.measure(s, "cold_s" if cold else "first_s", wall[0], "s (asked to answer, not loaded before: loading included)" if cold else "s (first run, it was loaded already)")
            else: got.append(u)
        if rounds > 1:
            warm = sorted(wall[1:])[len(wall[1:]) // 2]
            self.measure(s, "warm_s", warm, "s (the same request once loaded)")
            if cold: self.measure(s, "load_s", max(wall[0] - warm, 0.0), "s (cold minus warm: what starting it costs)")
            if watts and (u := got[-1] if got else {}).get("gen_n"):
                self.measure(s, "power_w_busy", sum(watts) / len(watts), "W (its own sensors, while answering)")
                self.measure(s, "joules_per_token", sum(watts) / len(watts) * wall[-1] / ((u.get("gen_n") or 0) + (u.get("prompt_n") or 0) or 1), "J per token read or written (its own sensors: the chip, not the outlet)")
        await self._loaded(p, name, s)
        for u in got:
            if u.get("gen_ms") and u.get("gen_n"): self.measure(s, "decode_tps", u["gen_n"] / u["gen_ms"] * 1000, "tokens/s")
            if u.get("prompt_ms") and u.get("prompt_n"): self.measure(s, "prefill_tps", u["prompt_n"] / u["prompt_ms"] * 1000, "tokens/s")
            if u.get("ttft_ms"): self.measure(s, "ttft_s", u["ttft_ms"] / 1000, "s")
        return {s: {k: v["measured"] for k, v in self.figures(s).get(s, {}).items()}}

    async def _power(self, p: dict) -> float:
        """What the machine's own sensors say it draws now (watts, summed over its sensors; GET /node "now"); None when it has none (or is an older node)."""
        w = ((await providers.machine(p, "GET", "/node", timeout=10.0)).get("now") or {}).get("power_w") or {}
        return round(sum(w.values()), 1) if w else None

    async def _running(self, p: dict) -> list:
        """The machine's programs that are up now, each with the model it holds (its own account, GET /node)."""
        return [x for x in (await providers.machine(p, "GET", "/node", timeout=10.0)).get("services", []) if x.get("status") == "up"]

    async def _loaded(self, p: dict, name: str, s: str):
        """What the model holds once loaded, read from the machine's own account of its running programs (measured there after loading; resident memory otherwise). The figure depends on the context size it was loaded with, which is kept in the source."""
        sv = next((x for x in (await providers.machine(p, "GET", "/node", timeout=10.0)).get("services", []) if x.get("model") == name), None)
        if not sv: return
        lease, how = sv.get("lease") or {}, f"""loaded with context {sv.get("ctx") or "default"}, cache {sv.get("kv") or "default"}"""
        self.measure(s, "ram_gb_loaded", lease["ram_gb"] if lease.get("measured") else sv.get("rss_gb") or 0, "GB", how)
        if lease.get("gpu_gb"): self.measure(s, "gpu_gb_loaded", lease["gpu_gb"], "GB", how)

    async def test_machine(self, p: dict) -> dict:
        """The machine's own speed test (processor, memory, the models place's disk), as measurements of its parts."""
        sp, g = await providers.machine(p, "POST", "/node/speed", timeout=300.0), p["id"]
        if not await self._running(p) and (w := await self._power(p)) is not None: self.measure(f"{g}/cpu", "power_w_idle", w, "W (its own sensors, nothing loaded)")   # at rest only when no program is up
        for k, v in sp.items():
            if k in ("sig", "t") or not v: continue
            self.measure(f"""{g}/{"ram" if k.startswith("mem") else "storage/models" if k.startswith("disk") else "cpu"}""", k, v, "GB/s" if "gb_s" in k else "MB/s", "the machine's speed test")
        return sp

    async def test_link(self, p: dict, mb: int = 32) -> dict:
        """The path between this portal and one machine, both ways, by writing and reading back a file of mb megabytes in its vault (one file, written over by the next test)."""
        data, rel, g = bytes(range(256)) * (mb * 4096), "_characterize/link.bin", p["id"]
        t0 = time.time()
        await providers.put(p, "vault", rel, data, overwrite=True)
        up = mb / (time.time() - t0)
        t0 = time.time()
        await providers.fetch(p, "vault", rel)
        down = mb / (time.time() - t0)
        lid = f"{PORTAL}~{g}"
        with self._db() as db: self._put(db, "parts", {"id": lid, "grp": PORTAL, "kind": "link", "name": f"""this portal ~ {p.get("name") or g}""", "attrs": {"a": PORTAL, "b": g}})
        self.measure(lid, "to_mb_s", up, "MB/s")
        self.measure(lid, "from_mb_s", down, "MB/s")
        trips = []
        for _ in range(10):   # the delay alone: small requests to its open /health
            t0 = time.time()
            await providers.machine(p, "GET", "/health", timeout=10.0)
            trips.append((time.time() - t0) * 1000)
        self.measure(lid, "rtt_ms", sorted(trips)[len(trips) // 2], "ms (median round trip)")
        return {"to_mb_s": round(up, 1), "from_mb_s": round(down, 1), "rtt_ms": round(sorted(trips)[len(trips) // 2], 2)}

    async def test_link_between(self, a: dict, b: dict, mb: int = 32) -> dict:
        """The path between two machines, both ways, measured by a itself (b makes a short-lived ticket, so neither holds the other's key). Kept as the link part "a~b": to_mb_s is a to b."""
        t = await providers.machine(b, "POST", "/node/linktest/ticket", timeout=10.0)
        got, lid = await providers.machine(a, "POST", "/node/linktest", {"url": b["url"], "ticket": t["ticket"], "mb": mb}, timeout=600.0), f"""{a["id"]}~{b["id"]}"""
        with self._db() as db: self._put(db, "parts", {"id": lid, "grp": a["id"], "kind": "link", "name": f"""{a.get("name") or a["id"]} ~ {b.get("name") or b["id"]}""", "attrs": {"a": a["id"], "b": b["id"]}})
        self.measure(lid, "to_mb_s", got["to_mb_s"], "MB/s")
        self.measure(lid, "from_mb_s", got["from_mb_s"], "MB/s")
        if got.get("rtt_ms") is not None: self.measure(lid, "rtt_ms", got["rtt_ms"], "ms (median round trip)")   # an older node does not measure it
        return got

    async def lanes(self, a: dict, b: dict) -> list:
        """Every way a can reach b: for each address of b, the address of a on the same network (from each machine's interfaces as it reports them now). [{"a_if", "a_addr", "b_if", "b_addr", "url"}]; the address in b's own url is one of them when it is on a shared network. An older node reports no interfaces: no lanes."""
        na, nb = await asyncio.gather(providers.machine(a, "GET", "/node", timeout=10.0), providers.machine(b, "GET", "/node", timeout=10.0))
        u, out = urllib.parse.urlsplit(b["url"]), []
        mine = [(i["name"], ipaddress.ip_interface(x)) for i in na.get("net") or [] if i.get("state") != "down" for x in i.get("addrs") or []]
        for i in [i for i in nb.get("net") or [] if i.get("state") != "down"]:
            for x in i.get("addrs") or []:
                there = ipaddress.ip_interface(x)
                if (src := next(((n, m) for n, m in mine if there.ip in m.network), None)): out.append({"a_if": src[0], "a_addr": str(src[1].ip), "b_if": i["name"], "b_addr": str(there.ip), "url": f"""{u.scheme}://{there.ip}:{u.port or (443 if u.scheme == "https" else 80)}"""})
        return out

    async def test_lanes(self, a: dict, b: dict, mb: int = 32) -> list:
        """Each lane from a to b measured on its own (a sends from its address on that network, to b's address there), kept as the link part "a~b@a_if-b_if". What b's url uses is the main path (test_link_between); this shows whether another network between the same two machines is faster, or which one is slow. A lane that fails is answered with its error."""
        out = []
        for ln in await self.lanes(a, b):
            lid = f"""{a["id"]}~{b["id"]}@{ln["a_if"]}-{ln["b_if"]}"""
            try:
                t = await providers.machine(b, "POST", "/node/linktest/ticket", timeout=10.0)
                got = await providers.machine(a, "POST", "/node/linktest", {"url": ln["url"], "ticket": t["ticket"], "mb": mb, "from": ln["a_addr"]}, timeout=600.0)
            except providers.ProviderError as e:
                out.append({**ln, "error": str(e)[:300]})
                continue
            with self._db() as db: self._put(db, "parts", {"id": lid, "grp": a["id"], "kind": "lane", "name": f"""{a.get("name") or a["id"]} {ln["a_if"]} ~ {b.get("name") or b["id"]} {ln["b_if"]}""", "attrs": {"a": a["id"], "b": b["id"], **ln}})
            for k, unit in (("to_mb_s", "MB/s"), ("from_mb_s", "MB/s"), ("rtt_ms", "ms (median round trip)")):
                if got.get(k) is not None: self.measure(lid, k, got[k], unit)
            out.append({**ln, **got})
        return out

    def others(self, p: dict) -> list:
        """The other machines a path test from p goes to: every provider that speaks the node contract, but p."""
        return [q for q in self.lib.list("provider") if not q.get("speaks") and q["id"] != p["id"]]

    async def characterize(self, p: dict, models: list, note=lambda s: None, among: list = None) -> dict:
        """The tests in order: the machine, the path to it from this portal, the paths to the other machines and each lane to them (each network they share; only those in among, provider ids, when given), then each model named ([(name, id)]). Each result is kept as it comes, so a stop keeps what was done; a path a machine cannot test (an older node, one that is off) is noted and passed."""
        out = {"machine": await self.test_machine(p)}
        note("machine tested")
        out["link"] = await self.test_link(p)
        note("path from this portal tested")
        for q in [q for q in self.others(p) if among is None or q["id"] in among]:
            try: out[f"""link~{q["id"]}"""] = await self.test_link_between(p, q)
            except providers.ProviderError as e: out[f"""link~{q["id"]}"""] = {"error": str(e)}
            try: out[f"""lanes~{q["id"]}"""] = await self.test_lanes(p, q)
            except providers.ProviderError as e: out[f"""lanes~{q["id"]}"""] = {"error": str(e)}   # either machine did not answer for its interfaces
            note(f"""path to {q.get("name") or q["id"]}: {"not tested" if "error" in out[f'link~{q["id"]}'] else "tested"}""")
        for name, mid in models:
            try: out[name] = await self.test_model(p, name, mid)
            except providers.ProviderError as e: out[name] = {"error": str(e)}
            note(f"{name} tested")
        return out
