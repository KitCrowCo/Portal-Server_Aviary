"""roles.py - a plan names roles, never machines or models. A role is what a plan needs: {"needs": [capabilities], "about", "facts": {name: expression over the run's starting keys, or {"is": that expression, "label": what it means to a person, "kind": "number" | "yes/no"}}}.
What fills a role is a list of candidates, in order of preference: [{"asset", "provider", "when": expression, "body": {request defaults}, "streams": n, "note"}]. A candidate may name a "pool" (access.py) instead of an asset: any member of that set that fits, so the choice among machines and models is made when the work starts, from what is free. "also": further candidates used together with it, for a role that takes several assets at once. "when" says the cases a candidate is for, over the role's facts and the caller's own (the run's "case": speed wanted, how private, ...); none = always. One role can so be a small quick model by default, a larger one for a long prompt and another when images come with it.
Candidates come from profiles (library kind "bindings": {"id", "inherits", "roles": {role: [candidates] or one candidate}}); a later profile's list for a role replaces an earlier one's. A pin ({role: candidate}, from the run) stands above all of it: it is used or the run is not placed, nothing is switched.
Without a pin the first candidate whose case holds, which a provider offers and which fits there (the provider's own plan) is taken, and the ones passed over are recorded with the reason. Nothing is guessed: a role with nothing said and several possible assets is refused with the choices."""
import asyncio
from . import providers, access

_SAFE = {"len": len, "str": str, "int": int, "float": float, "bool": bool, "min": min, "max": max, "sum": sum, "any": any, "all": all, "isinstance": isinstance}

def chain(lib, ids: list) -> list:
    """The profiles in the order they apply: each one's "inherits" before itself."""
    out = []
    for i in ids:
        prof = lib.get("bindings", i)
        out += [*chain(lib, [prof["inherits"]] if prof.get("inherits") else []), prof]
    return out

def candidates(lib, role: str, profiles: list) -> list:
    """The role's candidates as the profiles leave them (the last profile that speaks of the role decides)."""
    said = [p["roles"][role] for p in chain(lib, profiles) if p.get("roles", {}).get(role)]
    return [c for c in (said[-1] if isinstance(said[-1], list) else [said[-1]]) if c.get("asset") or c.get("provider") or c.get("pool")] if said else []

def facts(need: dict, start: dict, case: dict = None) -> dict:
    """The case a role is filled for: the role's own facts worked out from the run's starting keys, then what the caller said of the case."""
    return {**{k: eval(x if isinstance(x, str) else x["is"], {"__builtins__": _SAFE, **start}) for k, x in (need.get("facts") or {}).items()}, **(case or {})}

def holds(when: str, case: dict) -> bool:
    """A name the case does not have makes the condition not hold (a candidate for "images" is not for a run that says nothing of images)."""
    try: return bool(eval(when, {"__builtins__": _SAFE, **case})) if when else True
    except NameError: return False

SPEED = lambda name, provider: 0.0   # decode speed of a model on a provider, measured or estimated (set by the engine to its resources' speed); among things that fit equally, the faster is taken
ANY = {"id": "*", "name": "anything that fits", "members": ["*|*"]}   # the pool of everything: a role filled by whatever can do it now, among what the run's owner may use

CAN = ("yes", "wait")   # a machine's plan that needs nothing changed: it starts the work at once, or holds it until its own work in progress ends and then makes room by itself
def pl_wait(pl: dict) -> bool: return pl.get("fits") == "wait"   # among machines that can, one that starts at once goes before one that waits

async def _from_pool(lib, caps: list, c: dict, pool: dict, found: list) -> dict:
    """A pool's member for the role: of those offered that fit as things stand, one already running first, then the fastest as measured (or estimated) in the resources, then the pool's own order; a pool that says "prefer": "listed" keeps its own order alone."""
    opts = sorted(((p, n) for p, names in found for n in names), key=lambda o: access.rank(pool, o[0]["id"], o[1]))[:24]
    plans = await asyncio.gather(*(providers.ask(p, "plan", body={"capability": caps[0], "model": n}) for p, n in opts), return_exceptions=True) if caps else [{"fits": "yes"}] * len(opts)
    fit = sorted((i for i, pl in enumerate(plans) if isinstance(pl, dict) and pl.get("fits") in CAN), key=lambda i: (pl_wait(plans[i]), pool.get("prefer", "running") == "running" and not plans[i].get("running"), pool.get("prefer") != "listed" and -SPEED(opts[i][1], opts[i][0]["id"]), i))
    if not fit: raise ValueError(f"""nothing in the pool {pool.get("name") or pool["id"]} fits as things are: """ + "; ".join(f"""{n} on {p.get("name") or p["id"]}: {pl.get("will", "") if isinstance(pl, dict) else pl}""" for (p, n), pl in zip(opts, plans)))
    (p, n), pl = opts[fit[0]], plans[fit[0]]
    return {**c, "provider": p["id"], "asset": n, "fits": pl["fits"], "will": pl.get("will", ""), "lane": f"""{p["id"]}/{n}""", "running": bool(pl.get("running")), "speed": SPEED(n, p["id"]), "chosen_from": [f"""{q.get("name") or q["id"]}: {m}""" for q, m in opts], "fit_count": len(fit)}

async def _fit(lib, caps: list, c: dict, owner: str = "") -> dict:
    """One candidate completed (provider decided, the provider's plan, its lane), or ValueError with why not. A candidate names an asset, or a pool whose members it may be any of. Assets a pool keeps from the run's owner are passed over as if not offered, and said when one was asked for by name."""
    found, plans, pool = [], [], ANY if c.get("pool") == "*" else lib.get("pool", c["pool"]) if c.get("pool") else None   # found: (provider, asset names it offers for every capability the role needs)
    for p in ([lib.get("provider", c["provider"])] if c.get("provider") else lib.list("provider")):
        try: per = [{a["name"] for a in await providers.assets(p, cap)} for cap in caps]
        except providers.ProviderError as e:
            if c.get("provider"): raise ValueError(str(e))
            continue
        names = sorted(set.intersection(*per)) if per else []
        if c.get("asset") in names and not access.allowed(lib, owner, p["id"], c["asset"]): raise ValueError(access.why_not(lib, owner, p["id"], c["asset"]))
        names = [n for n in names if access.allowed(lib, owner, p["id"], n) and (not pool or access.holds(pool, p["id"], n))]
        if (c.get("asset") in names) if c.get("asset") else names: found.append((p, names))
    if not found: raise ValueError(f"""no provider offers {c.get("asset") or ("a member of the pool " + c["pool"] if pool else "anything")} for {", ".join(caps)}""" + (f" that {owner or 'work without an owner'} may use" if lib.list("pool") else ""))
    if pool and not c.get("asset"): return await _from_pool(lib, caps, c, pool, found)
    if not c.get("asset") and (len(found) > 1 or len(found[0][1]) > 1): return await _from_pool(lib, caps, c, ANY, found)   # nothing chosen: as "anything that fits" - every offer is weighed, none has to be set up first
    asset = c.get("asset") or found[0][1][0]
    for p, _ in found:
        pl = await providers.ask(p, "plan", body={"capability": caps[0], "model": asset}) if caps else {"fits": "yes"}
        plans.append((p, pl))
        if pl["fits"] == "yes": return {**c, "provider": p["id"], "asset": asset, "fits": "yes", "will": pl.get("will", ""), "lane": f"""{p["id"]}/{asset}"""}
    if (later := next(((p, pl) for p, pl in plans if pl["fits"] == "wait"), None)): return {**c, "provider": later[0]["id"], "asset": asset, "fits": "wait", "will": later[1].get("will", ""), "lane": f"""{later[0]["id"]}/{asset}"""}   # none can start it at once; this one gets there by itself when its work in progress ends
    raise ValueError(f"""{asset} fits on none of the {len(plans)} machine(s) that have it: """ + " | ".join(_verdict(p, pl) for p, pl in plans))

def _verdict(p: dict, pl: dict) -> str:
    """One machine's answer in full: what it says it would need, what it has, and the ways it offers (each with the setting it changes)."""
    h, ways = pl.get("have") or {}, [w for w in pl.get("ways", []) if w.get("fits")]
    room = f""" It would need about {pl["need_gb"]} GB; free now {h.get("free_gb")} GB, {h.get("free_if_alone_gb")} GB if everything else stopped (of {h.get("ram_total_gb")} GB, {h.get("reserve_gb")} GB kept for the system).""" if pl.get("need_gb") is not None else ""
    plan = pl.get("needs") or {}
    how = (f""" Its parts ({plan.get("source", "estimated")}): {", ".join(f"{k} {v} GB" for k, v in plan["parts"].items())}.""" if plan.get("parts") else f""" ({plan.get("source")})""" if plan.get("source") else "") + "".join(f" {n}" for n in plan.get("notes") or [])   # the node's own breakdown (weights, KV for the context it planned, buffers), so a figure that looks wrong can be traced
    said = "; ".join(w["label"] + (" (setting " + ", ".join(f"{k} = {v}" for k, v in w["set"].items()) + ")" if w.get("set") else "") for w in ways) or "none"   # each way with the setting it changes: Settings > Machines > Its settings
    return f"""{p.get("name") or p["id"]}: {pl.get("will", "")}{room}{how} Ways: {said}"""

async def _fit_all(lib, caps: list, c: dict, owner: str) -> dict:
    """A candidate with those that go with it ("also": further candidates for the same role, where a role takes several assets at once - knowledge groups asked together). Every one has to fit."""
    return {**await _fit(lib, caps, {k: v for k, v in c.items() if k != "also"}, owner), **({"also": [await _fit(lib, caps, a, owner) for a in c["also"]]} if c.get("also") else {})}

async def place(lib, role: str, need: dict, cands: list, pin: dict = None, case: dict = None, owner: str = "") -> dict:
    """Who fills the role now: {"provider", "asset", "lane", "fits", "will", "pinned", "passed": [{"asset", "why"}], "also": [the same for each further asset], ...the candidate's own fields}. owner: whose work it is - what a pool keeps from that person is not used. ValueError says what stands in the way."""
    caps = need.get("needs") or []
    if pin:
        try: return {**await _fit_all(lib, caps, pin, owner), "pinned": True, "passed": []}
        except ValueError as e: raise ValueError(f"role {role!r} is pinned to {pin.get('asset')}: {e}")
    passed = []
    for c in cands or [{"pool": "*"}]:   # nothing said for the role: placement over everything that can do it (D-01); a profile's candidates only narrow it
        if not holds(c.get("when", ""), case or {}):
            passed.append({"asset": c.get("asset", ""), "why": f"""not its case ({c["when"]})"""})
            continue
        try: return {**await _fit_all(lib, caps, c, owner), "pinned": False, "passed": passed}
        except ValueError as e: passed.append({"asset": c.get("asset") or c.get("pool", ""), "why": str(e)})
    why = "; ".join((x["asset"] + ": " if x["asset"] else "") + x["why"] for x in passed)
    raise ValueError(f"""role `{role}` ({", ".join(f"`{c}`" for c in caps)}) is not filled: {why}""")
