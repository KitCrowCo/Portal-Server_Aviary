"""aviary.py - Aviary, the portal's AI work tool: carries out plans for anyone in the portal that wants AI work done, and has no pages of its own.
What it holds: a library (pipelines, binding profiles, providers, anything else kept as JSON objects), a director for the runs, the step types, and what the runs measured as rows (AVIARY.measure, measure.py). Other tools and modules add step types (AVIARY.types) and shipped library folders (AVIARY.lib.shipped).
Use: rid = await AVIARY.start("pipeline id" or a plan, inputs, owner=..., profiles=[binding profile ids], pins={role: binding}); AVIARY.view(rid); AVIARY.director.stop / decide; await AVIARY.go(rid)."""
import json
from pathlib import Path
from .providers import ProviderError
from . import plan as P, roles, steps, providers, access, agent_loop, agents, code_index, seen   # providers, seen: also for callers that show what a provider offers, and what machines last said; agents: AI profiles, applied by the agent loop once imported
from .library import Library
from .director import Director
from .measure import Measure
from .schedule import Schedule
from .resources import Resources

TOOL_META = {"label": "Aviary", "icon": "&#x2699;", "description": "Carries out pipelines: plans as data, roles bound to providers, runs that can be picked up"}
AVIARY = None

class Aviary:
    def __init__(self, data: Path, shipped: list = (), on_restart: str = "ask"):
        self.data, self.lib, self.types = Path(data), Library(Path(data) / "library", shipped), dict(steps.TYPES)
        self.types["agent"], agent_loop.SPACES_DIR, code_index.INDEX_DIR = agent_loop.agent, self.data / "agent_spaces", self.data / "code_index"   # the agent loop (agent_loop.py) and where its own spaces and code maps are kept
        self.director = Director(Path(data) / "runs", self.types, lane=self._lane, on_restart=on_restart, place=self.bind)
        providers.translator = lambda tid: self.lib.get("translator", tid)
        self.director.aviary, self.bi = self, None   # bi: the portal's built-in parts (file manager, proposed-change store), set when Aviary runs inside a portal
        self.access = access   # pools: named sets of what providers hold, and who may use what (access.py); roles.place asks it for every run
        self.schedule = Schedule(self, Path(data))   # saved set-ups started by the clock (schedule.py)
        self.seen = seen.Seen(Path(data) / "seen.json")   # what each machine last said, for showing without waiting on it (seen.py)
        self.resources = Resources(Path(data) / "resources.db", self.lib)   # what is available as it really is: parts, links, models and copies, stated beside measured (resources.py)
        roles.SPEED = self.resources.speed   # placement prefers, among what fits, what was measured faster
        self.measure = Measure(Path(data) / "measure.db", self.director)   # the runs and what their steps measured, as rows: filled from the run folders, asked by whoever shows or plans (measure.py)

    def status(self, kind: str, oid: str) -> str: return self.schedule.status(oid) if kind == "schedule" else ""   # what a library object is doing now, for the kinds that do something by themselves

    def _lane(self, run, step: dict): return ((run.meta.get("bound") or {}).get((step.get("config") or {}).get("role")) or {}).get("lane")

    async def bind(self, run, todo: set):
        """Decides who fills the roles in todo, now (when a step first uses them): a pin if the run has one, else placement over what the profiles allow, as things stand. A role marked "optional" with neither stays empty (None): its steps give nothing and the rest goes on.
        Every go forgets what was decided (go), since what is free and what fits changes; a branch the run does not take never places its roles."""
        start, bound = {k: run.data[k] for k in P.given(run.plan) if k in run.data}, run.meta.setdefault("bound", {})
        for role in sorted(todo):
            need = run.plan["roles"][role]
            cands, pin = roles.candidates(self.lib, role, run.meta.get("profiles", [])), (run.meta.get("pins") or {}).get(role)
            if need.get("optional") and not cands and not pin: bound[role] = None
            else: bound[role] = await roles.place(self.lib, role, need, cands, pin, roles.facts(need, start, run.meta.get("case")), owner=run.meta.get("owner", ""))
            self.director.caps.update({x["lane"]: int(x.get("streams", 1)) for x in ((bound[role], *bound[role].get("also", [])) if bound[role] else ())})
        run.meta["bound"] = bound

    async def start(self, plan, inputs: dict = None, owner: str = "", profiles: list = (), pins: dict = None, priority: int = 0, watch=None, may: list = (), case: dict = None) -> str:
        """watch: fn(step, event), told what the run's steps report while they work, from its first step on. may: what the portal's owner allows this run that is not allowed by default (see policy), as answered by the caller for the person starting it. case: what the caller says of this piece of work (speed wanted, how private, ...), for candidates' "when"."""
        rid = self.director.create(self.lib.get("pipeline", plan) if isinstance(plan, str) else plan, inputs, owner, priority)
        self.director.runs[rid].meta.update(profiles=list(profiles), pins=pins or {}, may=list(may), case=case or {})
        if watch: self.director.runs[rid].watchers.append(watch)
        await self.go(rid)
        return rid

    async def warm(self, plan, inputs: dict = None, owner: str = "", profiles: list = (), pins: dict = None, roles: list = (), may: list = (), case: dict = None) -> dict:
        """What a run of this plan with these inputs would use, brought up now where placement puts it - the plan walked without doing it (director.py), so a branch the inputs do not take warms nothing - and how that went, per role: {role: {"provider", "asset", "capability", "seconds", "read_ahead"?, "how", "error"?}}, plus "_unplaced": why, when a role could not be placed.
        roles: only these are warmed (the walk still places the others it reaches). The walk is a run like any other while it goes (it is listed), and is removed once it rests."""
        rid = self.director.create(self.lib.get("pipeline", plan) if isinstance(plan, str) else plan, inputs, owner)
        run = self.director.runs[rid]
        run.meta.update(profiles=list(profiles), pins=pins or {}, may=list(may), case=case or {}, warm=True, warm_roles=list(roles))
        try:
            await self.go(rid)
            await self.director.wait(rid)
            return {**run.meta.get("warmed", {}), **({"_unplaced": run.meta["error"]} if run.meta.get("error") else {})}
        finally:
            if run.meta["state"] in ("running", "stopping"): self.director.stop(rid, now=True)
            if run.meta["state"] not in ("running", "stopping"): self.director.remove(rid)

    async def sub(self, parent, rid: str, plan: dict, inputs: dict, roles: dict = None) -> str:
        """A run made by a step of another run, under a name that step can find again: started, or - when it exists from before a stop - gone on with from what it kept. Answers its state once it rests.
        It belongs to the parent's owner, uses the parent's profiles and pins, and reports to whoever watches the parent. roles: {its role: the parent's role that fills it}, where the names differ."""
        if rid not in self.director.runs:
            self.director.create(plan, inputs, parent.meta.get("owner", ""), parent.meta.get("priority", 0), rid=rid)
            same = {r: {k: parent.meta["bound"][mine][k] for k in ("provider", "asset", "also") if k in parent.meta["bound"][mine]} for r, mine in (roles or {}).items()}
            self.director.runs[rid].meta.update(profiles=parent.meta.get("profiles", []), pins={**parent.meta.get("pins", {}), **same}, parent=parent.id, may=parent.meta.get("may", []), case=parent.meta.get("case", {}))
        child = self.director.runs[rid]
        if parent.emit not in child.watchers: child.watchers.append(parent.emit)
        if child.meta["state"] != "done": await self.go(rid)
        return await self.director.wait(rid)

    def policy(self) -> dict:
        """The owner's policy as it holds: the shipped decisions, with the owner's own (mine) over them key by key, so a decision added in a later version has its shipped value until the owner sets it."""
        return {k: v for root in self.lib.shipped if (root / "policy" / "aviary.json").exists() for k, v in json.loads((root / "policy" / "aviary.json").read_text()).items()} | (self.lib.get("policy", "aviary") if self.lib.origin("policy", "aviary") == "mine" else {})

    def allows(self, what: str, user: str, admin: bool) -> bool:
        """The owner's decision on something runs may not do by default (library object policy "aviary": {what: "nobody" | "admins" | "everyone" | [user names]}). A portal admin is the owner here."""
        rule = self.policy().get(what, "nobody")
        return rule == "everyone" or (rule == "admins" and admin) or (isinstance(rule, list) and user in rule)

    def space(self, sid: str) -> tuple:
        """A space is a folder work may read, and write into only as proposals (library kind "space": {"id", "name", "root"}). Answers its file manager and its store of proposed changes."""
        root, store = Path(self.lib.get("space", sid)["root"]), self.data / "proposals" / sid   # kept beside Aviary's own data, not inside the space, so a picker or a folder listing of the space never shows them
        if (root / "_shadow").is_dir() and not store.exists():   # once, for proposals made while they were kept inside the space
            store.parent.mkdir(parents=True, exist_ok=True)
            (root / "_shadow").rename(store)
        fm = self.bi.FileManager(root)
        return fm, self.bi.ShadowStore(fm, store)

    async def go(self, rid: str):
        """Starts a run or picks it up. A role that cannot be filled leaves the run "unplaced" with the reason; go again once a binding or a provider has changed."""
        run = self.director.runs[rid]
        run.meta["bound"] = {r: b for r, b in run.meta.get("bound", {}).items() if r in (run.meta.get("pins") or {})}   # decided again when next used; a pin stays
        first = {r for st in P.ready(run.plan, set(run.data)) for r in P.roles_of(run.plan, st)}   # the roles the steps that start now use: placed before it goes, so a run that cannot start says so at once
        try: await self.bind(run, first - set(run.meta["bound"]))
        except (ValueError, ProviderError) as e: return run.state("unplaced", error=str(e))
        run.meta.pop("error", None)
        self.director.go(rid)

    async def open(self):
        """At start: reads the runs that exist and goes on with those that are to continue by themselves."""
        for rid in [r for r in self.director.open() if not self.director.runs[r].meta.get("warm")]: await self.go(rid)
        for rid in [r for r, x in self.director.runs.items() if x.meta.get("warm")]:   # a warm walk cut off by a stop is not worth picking up: its machines may have changed since
            self.director.runs[rid].state("stopped")
            self.director.remove(rid)

    def view(self, rid: str) -> dict: return self.director.runs[rid].view()

def _old_data(data: Path):
    """One time, for a portal that ran this tool under its earlier name (engine): its data folder becomes this one, and the owner's policy object takes the new name. Nothing is copied or deleted."""
    old = data.parent / "engine"
    if not old.is_dir() or any(data.iterdir()): return
    data.rmdir()
    old.rename(data)
    if (data / "library" / "policy" / "engine.json").exists(): (data / "library" / "policy" / "engine.json").rename(data / "library" / "policy" / "aviary.json")
    print(f"--- aviary: data moved from {old} to {data} ---")

def init_module(env):
    global AVIARY
    _old_data(Path(env["data_dir"]))
    AVIARY = Aviary(Path(env["data_dir"]), [Path(__file__).parent / "library"])
    AVIARY.bi = env["tools"]["built_ins"]
    env["register_startup"](AVIARY.open)
    env["register_startup"](AVIARY.schedule.loop)
    print(f"""[aviary] ready | shipped pipelines: {sorted(f.stem for f in (Path(__file__).parent / "library" / "pipeline").glob("*.json")) or "NONE - expected in " + str(Path(__file__).parent / "library" / "pipeline")}""")
    for f in AVIARY.lib.stale(): print(f"--- aviary: {f} is retired (no longer shipped) and ignored; it can be deleted ---")
