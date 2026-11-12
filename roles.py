"""roles.py - a plan names roles, never machines or models. A role is what a plan needs: {"needs": [capabilities], "about", "facts": {name: expression over the run's starting keys, or {"is": that expression, "label": what it means to a person, "kind": "number" | "yes/no"}}}.
What fills a role is a list of candidates, in order of preference: [{"asset", "provider", "when": expression, "body": {request defaults}, "streams": n, "note"}]. "when" says the cases a candidate is for, over the role's facts and the caller's own (the run's "case": speed wanted, how private, ...); none = always. One role can so be a small quick model by default, a larger one for a long prompt and another when images come with it.
Candidates come from profiles (library kind "bindings": {"id", "inherits", "roles": {role: [candidates] or one candidate}}); a later profile's list for a role replaces an earlier one's. A pin ({role: candidate}, from the run) stands above all of it: it is used or the run is not placed, nothing is switched.
Without a pin the first candidate whose case holds, which a provider offers and which fits there (the provider's own plan) is taken, and the ones passed over are recorded with the reason. Nothing is guessed: a role with nothing said and several possible assets is refused with the choices."""
from . import providers

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
    return [c for c in (said[-1] if isinstance(said[-1], list) else [said[-1]]) if c.get("asset") or c.get("provider")] if said else []

def facts(need: dict, start: dict, case: dict = None) -> dict:
    """The case a role is filled for: the role's own facts worked out from the run's starting keys, then what the caller said of the case."""
    return {**{k: eval(x if isinstance(x, str) else x["is"], {"__builtins__": _SAFE, **start}) for k, x in (need.get("facts") or {}).items()}, **(case or {})}

def holds(when: str, case: dict) -> bool:
    """A name the case does not have makes the condition not hold (a candidate for "images" is not for a run that says nothing of images)."""
    try: return bool(eval(when, {"__builtins__": _SAFE, **case})) if when else True
    except NameError: return False

async def _fit(lib, caps: list, c: dict) -> dict:
    """One candidate completed (provider decided, the provider's plan, its lane), or ValueError with why not."""
    found, plans = [], []   # found: (provider, asset names it offers for every capability the role needs)
    for p in ([lib.get("provider", c["provider"])] if c.get("provider") else lib.list("provider")):
        try: per = [{a["name"] for a in await providers.assets(p, cap)} for cap in caps]
        except providers.ProviderError as e:
            if c.get("provider"): raise ValueError(str(e))
            continue
        names = sorted(set.intersection(*per)) if per else []
        if (c.get("asset") in names) if c.get("asset") else names: found.append((p, names))
    if not found: raise ValueError(f"""no provider offers {c.get("asset") or "anything"} for {", ".join(caps)}""")
    choices = "; ".join((p.get("name") or p["id"]) + ": " + ", ".join(n) for p, n in found)
    if not c.get("asset") and (len(found) > 1 or len(found[0][1]) > 1): raise ValueError(f"nothing is chosen and there is more than one choice: {choices}")
    asset = c.get("asset") or found[0][1][0]
    for p, _ in found:
        pl = await providers.ask(p, "plan", body={"capability": caps[0], "model": asset}) if caps else {"fits": "yes"}
        plans.append((p, pl))
        if pl["fits"] == "yes": return {**c, "provider": p["id"], "asset": asset, "fits": "yes", "will": pl.get("will", ""), "lane": f"""{p["id"]}/{asset}"""}
    p, pl = plans[0]
    ways = "; ".join(w["label"] for w in pl.get("ways", []) if w.get("fits")) or "none"
    raise ValueError(f"""{asset} does not fit on {p.get("name") or p["id"]} as it is set: {pl.get("will", "")} Ways: {ways}""")

async def place(lib, role: str, need: dict, cands: list, pin: dict = None, case: dict = None) -> dict:
    """Who fills the role now: {"provider", "asset", "lane", "fits", "will", "pinned", "passed": [{"asset", "why"}], ...the candidate's own fields}. ValueError says what stands in the way."""
    caps = need.get("needs") or []
    if pin:
        try: return {**await _fit(lib, caps, pin), "pinned": True, "passed": []}
        except ValueError as e: raise ValueError(f"role {role!r} is pinned to {pin.get('asset')}: {e}")
    passed = []
    for c in cands or [{}]:
        if not holds(c.get("when", ""), case or {}):
            passed.append({"asset": c.get("asset", ""), "why": f"""not its case ({c["when"]})"""})
            continue
        try: return {**await _fit(lib, caps, c), "pinned": False, "passed": passed}
        except ValueError as e: passed.append({"asset": c.get("asset", ""), "why": str(e)})
    why = "; ".join((x["asset"] + ": " if x["asset"] else "") + x["why"] for x in passed)
    raise ValueError(f"""role `{role}` ({", ".join(f"`{c}`" for c in caps)}) is not filled: {why}""")
