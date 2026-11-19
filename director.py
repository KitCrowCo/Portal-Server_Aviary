"""director.py - carries out plans (plan.py). The plan says what; the director decides how: which steps run together and which one after another, across every run it holds, by what is free and what matters most.
The director keeps no state of its own that a run depends on. A run is a folder: the plan as it was started, the keys its steps chose to keep, and a journal of what happened. Whatever stops (a step, this process, the machine), a director that opens the folder works out what is left from the keys that exist (plan.wanted) and goes on from there - or from the start, when the plan kept nothing.
Step types are functions given to the director: async fn(config, inputs, ctx) -> {key: value}. The director knows none of them.
A warm run (run.meta["warm"]) walks a plan without doing it, to bring up ahead what a real run with the same inputs would use: step types marked warm (fn.warm: they only read, or know a warm run of their own) run as usual, every other step gives None under each key it gives, and a step that fails gives None too, so every step the inputs lead to is reached and its roles placed. Nothing of it is kept.
Lanes are how steps share something scarce: lane(run, step) names what a step would occupy (a model on a machine, a repository) and caps says how many steps a lane takes at once (1 unless given). Steps in different lanes, or in none, run together. This is where a plan's graph becomes a list when the means require it, without the plan changing.
Files per run, under <root>/<run id>/: plan.json, run.json (state), kept/<key>.json, live/<key>.json (a step's work so far), once/<step id> (a started once-step), journal.jsonl."""
import shutil, json, time, uuid, asyncio
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import quote, unquote
from . import plan as P

LIVE = ("running", "stopping")

class Run:
    def __init__(self, folder: Path):
        self.dir, self.id, self.plan, self.meta = folder, folder.name, json.loads((folder / "plan.json").read_text()), json.loads((folder / "run.json").read_text())
        self.data, self.busy = {unquote(f.stem): json.loads(f.read_text()) for f in (folder / "kept").glob("*.json")}, {}   # busy: step id -> its task
        self.live = {unquote(f.stem): json.loads(f.read_text()) for f in (folder / "live").glob("*.json")}   # what a step has made so far while it works (a transcript as it grows): for whoever watches, and for the step itself to go on from after a stop. Never an input of another step
        self.watchers, self.rest = [], asyncio.Event()
        if self.meta["state"] not in LIVE: self.rest.set()   # watchers: fn(step, event) told what a step reports while it works (nothing of it is kept); rest: set whenever the run is not under way

    def note(self, what: str, **more):
        with open(self.dir / "journal.jsonl", "a") as f: f.write(json.dumps({"t": round(time.time(), 3), "what": what, **more}, ensure_ascii=False) + "\n")

    def emit(self, step: dict, event: dict):
        for fn in self.watchers: fn(step, event)

    def state(self, state: str, **more):
        self.meta.update(state=state, **more)
        self.rest.clear() if state in LIVE else self.rest.set()
        (self.dir / "run.json").write_text(json.dumps(self.meta, indent=1))
        self.note("state", state=state, **more)

    def keep(self, values: dict):
        """Written so that a stop leaves each key either as it was or complete."""
        for k, v in values.items():
            tmp = self.dir / "kept" / f"""{quote(k, safe="")}.tmp"""
            tmp.write_text(json.dumps(v, ensure_ascii=False))
            tmp.replace(tmp.with_suffix(".json"))

    def keep_live(self, key: str, value):
        """A step's work so far under one of its keys (Run.live), written whole each time so a stop leaves the last complete value."""
        (self.dir / "live").mkdir(exist_ok=True)
        tmp = self.dir / "live" / f"""{quote(key, safe="")}.tmp"""
        tmp.write_text(json.dumps(value, ensure_ascii=False))
        tmp.replace(tmp.with_suffix(".json"))
        self.live[key] = value

    def view(self) -> dict:
        have = set(self.data)
        return {**self.meta, "id": self.id, "name": self.plan.get("name", ""), "done": [s["id"] for s in self.plan["steps"] if P.mark(s) in have], "busy": sorted(self.busy), "course": P.course(self.plan, have), "unreached": P.unreached(self.plan, have)}

class Director:
    def __init__(self, root: Path, types: dict, lane=None, caps: dict = None, on_restart: str = "ask", place=None):
        self.root, self.types, self.lane, self.caps, self.runs, self.held = Path(root), types, lane or (lambda run, step: None), caps or {}, {}, {}   # held: lane -> {(run id, step id)}
        self.place, self.placing = place, set()   # place: async fn(run, roles) that decides who fills roles a step is about to use; placing: (run id, step id) waiting on it
        self.on_restart = on_restart   # what happens to a run that was under way when its director stopped, unless its plan says otherwise: "ask" (it waits) or "continue"
        self.root.mkdir(parents=True, exist_ok=True)

    # --- Runs ---

    def create(self, plan: dict, inputs: dict = None, owner: str = "", priority: int = 0, rid: str = "") -> str:
        """A run of this plan, not yet started. Its inputs are kept: without them there is nothing to pick up. rid: the run's name when its maker must find it again by that name (a run made by a step of another run)."""
        inputs = P.start_keys(plan, inputs)
        if miss := sorted(P.given(plan) - set(inputs)): raise ValueError(f"this plan needs: {miss}")
        plan = P.expand(plan, inputs, {name: fn.verbatim for name, fn in self.types.items() if hasattr(fn, "verbatim")})
        P.check(plan, self.types)
        folder = self.root / (rid or f"""{time.strftime("%Y%m%d-%H%M%S")}-{uuid.uuid4().hex[:4]}""")
        for d in ("kept", "once"): (folder / d).mkdir(parents=True)
        (folder / "plan.json").write_text(json.dumps(plan, indent=1))
        (folder / "run.json").write_text(json.dumps({"state": "new", "owner": owner, "priority": int(priority), "created": time.time()}, indent=1))
        self.runs[folder.name] = run = Run(folder)
        run.keep(inputs)
        run.data.update(inputs)
        run.note("created", owner=owner, plan=plan.get("id", ""))
        return run.id

    def open(self) -> list:
        """Reads every run folder (at start). A run that was under way when its director stopped is "interrupted" until someone says go. Answers the ids that are to go on without being asked: the plan's "on_restart" ("continue" | "ask") decides, else this director's own setting. The caller says go for those, after whatever it has to prepare."""
        on = []
        for folder in sorted(d for d in self.root.iterdir() if (d / "run.json").exists()):
            self.runs[folder.name] = run = Run(folder)
            if run.meta["state"] not in LIVE: continue
            run.state("interrupted")
            if run.plan.get("on_restart", self.on_restart) == "continue" and not run.meta.get("parent"): on.append(run.id)   # a run made by another run's step goes on when that step does
        return on

    def go(self, rid: str):
        """Starts a run, or picks one up from whatever exists of it."""
        run = self.runs[rid]
        for k in ("error", "held_at"): run.meta.pop(k, None)
        run.state("running")
        self._fill()

    async def wait(self, rid: str) -> str:
        """Answers the run's state once it is no longer under way (done, failed, held, stopped, ...)."""
        await self.runs[rid].rest.wait()
        return self.runs[rid].meta["state"]

    def stop(self, rid: str, now: bool = False):
        """No further steps start. now: the running ones are cancelled as well; what they had not kept is done again at the next go."""
        run = self.runs[rid]
        run.state("stopping" if run.busy else "stopped")
        if now: [t.cancel() for t in run.busy.values()]

    def remove(self, rid: str):
        """Deletes a run that is not under way, with the runs its steps made: its folder, what it kept and its journal."""
        if self.runs[rid].meta["state"] in LIVE: raise ValueError(f"{rid} is under way: stop it first")
        for r in [r for r in self.runs if r == rid or r.startswith(rid + "~")]: shutil.rmtree(self.runs.pop(r).dir)

    def decide(self, rid: str, step_id: str, again: bool = False, values: dict = None):
        """For a once-step that was started and has no kept end: again = it may run once more; values = it did happen, and these are what it gave (kept)."""
        run = self.runs[rid]
        step = next(s for s in run.plan["steps"] if s["id"] == step_id)
        if again: (run.dir / "once" / step_id).unlink()
        else:
            out = {**(values or {}), P.mark(step): True}
            run.keep(out)
            run.data.update(out)
        run.note("decided", step=step_id, again=again)

    # --- Enactment ---

    def _fill(self):
        """Starts every step that can start now, over all running runs: the more important run first, then the older one; a full lane makes a step wait, not its run."""
        for run in sorted((r for r in self.runs.values() if r.meta["state"] == "running"), key=lambda r: (-r.meta["priority"], r.meta["created"])):
            ready = P.ready(run.plan, set(run.data), run.busy)
            for step in ready:
                if step.get("once") and (run.dir / "once" / step["id"]).exists():
                    run.state("held", held_at=step["id"], error=f"""{step["id"]} was started before and its end was not kept: decide whether it may run again""")
                    break
                if self.place and (todo := P.roles_of(run.plan, step) - set(run.meta.setdefault("bound", {}))):   # a role first used now is placed now (a branch not taken never needs its machine); the step starts once it is
                    if (run.id, step["id"]) not in self.placing:
                        self.placing.add((run.id, step["id"]))
                        asyncio.ensure_future(self._placed(run, step, todo))
                    continue
                lane = self.lane(run, step)
                if lane and len(self.held.setdefault(lane, set())) >= self.caps.get(lane, 1): continue
                if lane: self.held[lane].add((run.id, step["id"]))
                if step.get("once"): (run.dir / "once" / step["id"]).touch()
                run.busy[step["id"]] = asyncio.ensure_future(self._step(run, step, lane))
            if run.meta["state"] == "running" and not run.busy and not ready: run.state("done", ended=time.time(), unreached=P.unreached(run.plan, set(run.data)))

    async def lend(self, run: Run, step: dict, coro):
        """Awaits coro with the step's lane given back meanwhile, then takes it again once there is room: a step that waits on work it started (an agent's sub-agent on the same model) must not hold the model that work needs."""
        lane = self.lane(run, step)
        if not lane or (run.id, step["id"]) not in self.held.get(lane, set()): return await coro
        self.held[lane].discard((run.id, step["id"]))
        self._fill()
        try: return await coro
        finally:
            while len(self.held.setdefault(lane, set())) >= self.caps.get(lane, 1): await asyncio.sleep(0.2)
            self.held[lane].add((run.id, step["id"]))

    async def _placed(self, run: Run, step: dict, todo: set):
        """The roles a step is about to use, placed; a role that cannot be filled leaves the run "unplaced" with the reason (go again once a binding or a machine has changed)."""
        try: await self.place(run, todo)
        except Exception as e:   # ValueError (nothing fits) or a provider fault, said where the run shows it
            if run.meta["state"] == "running" and not run.busy: run.state("unplaced", error=f"""{step["id"]}: {e}""")
            elif run.meta["state"] == "running": run.meta["error"] = f"""{step["id"]}: {e}"""
        self.placing.discard((run.id, step["id"]))
        self._fill()

    async def _step(self, run: Run, step: dict, lane):
        run.note("start", step=step["id"], lane=lane)
        t0, out, err = time.time(), None, None
        took = {**{k: run.data[k] for k in P.needs(step)}, **{name: [run.data[k] for k in ks] if isinstance(ks, list) else run.data[ks] for name, ks in step.get("from", {}).items()}}
        warm, fn = run.meta.get("warm"), self.types[step["type"]]
        try: out = {**({k: None for k in P.gives(step)} if warm and not getattr(fn, "warm", False) else {step.get("as", {}).get(k, k): v for k, v in (await fn(step.get("config") or {}, took, SimpleNamespace(run=run, step=step, director=self)) or {}).items()}), P.mark(step): True}   # a warm run walks the plan without doing it: a type not marked warm (one that changes something) is not run
        except asyncio.CancelledError: err = "stopped"
        except Exception as e:
            err = f"{type(e).__name__}: {e}"
            if warm: out, err = {**{k: None for k in P.gives(step)}, P.mark(step): True}, None   # a warm run goes on past a step that fails on the empty values before it
            if warm: run.note("warm_skip", step=step["id"], why=f"{type(e).__name__}: {e}"[:300])
        if lane: self.held[lane].discard((run.id, step["id"]))
        del run.busy[step["id"]]
        if out:
            if P.keeps(run.plan, step) and not warm: run.keep(out)
            run.data.update(out)
        run.note("end", step=step["id"], seconds=round(time.time() - t0, 3), kept=bool(out) and P.keeps(run.plan, step), **({"error": err} if err else {}))
        if err and err != "stopped" and run.meta["state"] == "running": run.state("failed", error=f"""{step["id"]}: {err}""")
        if run.meta["state"] == "stopping" and not run.busy: run.state("stopped")
        self._fill()
