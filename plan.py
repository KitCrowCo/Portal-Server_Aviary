"""plan.py - a plan is data: what is to be done and what each step needs and gives. Nothing here runs anything, reads a file or knows a machine.
How a plan gets carried out (in what order, how many steps at once, where, alongside which other plans) is the director's business (director.py) and may differ from one enactment to the next; the plan does not change.
A plan: {"id", "name", "objective", "params": {name: {...}}, "keep": bool, "on_restart": "ask" | "continue", "steps": [step]}
A step: {"id", "type", "config", "needs": [keys], "gives": [keys], "keep": bool, "once": bool}
  needs / gives  keys of the run's shared data. They are the only wiring: a step can run when every key it needs is there. A step may give fewer keys than it lists (a branch): what waits on the rest is never reached.
  keep           what this step gives is written down as it ends, so it is still there after a stop of any kind. Default: the plan's "keep", else false. Keeping is the plan's choice per step: cheap work that is easy to do again needs none, and a step whose only purpose is to make something durable (store a model's state, write a checkpoint) is an ordinary step that keeps.
  once           doing it twice would be wrong (it sends, pays, deletes). If it was started and there is no kept end, nobody but a person or the plan's owner decides what happens next.
  each           {"var": [values] or "{name of a list}"}: the step stands for one step per item (per combination, the first variable outermost). "in_turn": true makes each item wait for the one before it, for work whose order is part of what it measures.
  from / as      {name the step type uses: key or keys in the run's data}: lets one step type serve under any key names; expand() writes them for each-steps.
"data" in a plan: keys every run of it starts with (fixed texts, lists). {name} in a step's config is filled from the run's parameters and data, and from the step's own item for an each-step.
What can be picked up after a stop follows from this alone: a step is done when its mark (@<id>) is among the keys that exist; everything else is worked out again from what exists."""

import re, itertools

def mark(step: dict) -> str: return f"""@{step["id"]}"""
def needs(step: dict) -> set: return set(step.get("needs", []))
def gives(step: dict) -> set: return set(step.get("gives", []))
def keeps(plan: dict, step: dict) -> bool: return bool(step.get("keep", plan.get("keep", False)))
def given(plan: dict) -> set: return {*(plan.get("params") or {}), *(plan.get("data") or {})}

def render(node, env: dict):
    """{name} filled from env, through objects and lists. A text that is exactly one {name} becomes the value itself, its type kept; a name env does not have stays as written."""
    if isinstance(node, dict): return {k: render(v, env) for k, v in node.items()}
    if isinstance(node, list): return [render(v, env) for v in node]
    if not isinstance(node, str): return node
    if (m := re.fullmatch(r"\{([\w.@]+)\}", node)) and m.group(1) in env: return env[m.group(1)]
    return re.sub(r"\{([\w.@]+)\}", lambda m: str(env[m.group(1)]) if m.group(1) in env else m.group(0), node)

def start_keys(plan: dict, inputs: dict = None) -> dict:
    """What a run starts with: the plan's data, each parameter's default, then what was given."""
    return {**(plan.get("data") or {}), **{k: v["default"] for k, v in (plan.get("params") or {}).items() if "default" in v}, **(inputs or {})}

def expand(plan: dict, inputs: dict = None, verbatim: dict = None) -> dict:
    """The plan with every each-step written out for these inputs. Item n of step s is step "s.n" and gives "<key>.n" for each key s gives, with its item's values filled into its config. An each-step may hold "steps" instead of a type: then the whole group is written out per item ("s.n.<inner id>"), wired among themselves within the item; what the group itself needs, each item's first step waits for.
    A step outside that needs such a key needs every item's. The step types see none of this: "from" and "as" carry the names. Parameters and data are filled into every step's config here as well; what is left in {braces} is a key a step fills in when it runs. verbatim: {step type: config keys that are never filled in}."""
    env, steps, fam = start_keys(plan, inputs), [], {}
    def fill(t, *envs):   # a step's config with each env filled in, in order; the keys its type takes as they are (verbatim: {type: keys} - code, a plan written inside a step) are not touched
        asis, cfg = (verbatim or {}).get(t.get("type"), ()), t.get("config") or {}
        out = {k: v for k, v in cfg.items() if k not in asis}
        for e in envs: out = render(out, e)
        return {**out, **{k: cfg[k] for k in asis if k in cfg}}
    for s in plan["steps"]:
        if "each" not in s:
            steps.append({**s, "config": fill(s, env)})
            continue
        inner = s.get("steps") or [{**{k: v for k, v in s.items() if k not in ("each", "in_turn")}, "id": ""}]
        own, combos = {k for t in inner for k in t.get("gives", [])}, list(itertools.product(*(render(v, env) for v in s["each"].values())))
        name = lambda n, t: ".".join(x for x in (s["id"], str(n), t["id"]) if x)
        for n, combo in enumerate(combos):
            item = dict(zip(s["each"], combo))
            for j, t in enumerate(inner):
                wait = [] if j else [*(s.get("needs", []) if "steps" in s else []), *([f"@{name(n - 1, inner[-1])}"] if s.get("in_turn") and n else [])]
                steps.append({**t, "id": name(n, t), "of": s["id"], "item": item, "config": fill(t, item, env), "needs": [f"{k}.{n}" if k in own else k for k in t.get("needs", [])] + wait, "gives": [f"{k}.{n}" for k in t.get("gives", [])], "as": {k: f"{k}.{n}" for k in t.get("gives", [])}, "from": {k: f"{k}.{n}" for k in t.get("needs", []) if k in own}})
        fam.update({k: [f"{k}.{n}" for n in range(len(combos))] for k in own})
    for t in steps:
        if hit := [k for k in t.get("needs", []) if k in fam]: t.update(needs=[x for k in t["needs"] for x in (fam[k] if k in fam else [k])], **{"from": {**t.get("from", {}), **{k: fam[k] for k in hit}}})
    return {**plan, "steps": steps}

def check(plan: dict, types=None):
    """ValueError naming everything wrong: ids used twice, a needed key that no step gives and no parameter supplies, a step type nobody offers."""
    ids, made = [s["id"] for s in plan["steps"]], given(plan).union(*(gives(s) | {mark(s)} for s in plan["steps"]))
    wrong = [f"step id {i!r} is used twice" for i in sorted({i for i in ids if ids.count(i) > 1})]
    wrong += [f"""step {s["id"]!r} needs {sorted(needs(s) - made)}, which no step gives and no parameter supplies""" for s in plan["steps"] if needs(s) - made]
    wrong += [f"""step {s["id"]!r}: no step type {s["type"]!r}""" for s in plan["steps"] if types is not None and s["type"] not in types]
    if wrong: raise ValueError("; ".join(wrong))

def wanted(plan: dict, have: set) -> set:
    """Ids of the steps that still have to run, given the keys that exist. A step that is not done is wanted when nothing uses what it gives (it is an end in itself), or when a wanted step lacks a key it gives.
    So after a stop, work goes back only as far as the kept keys require: a step whose results were not kept is not done again when everything that used them is already done."""
    todo, want = [s for s in plan["steps"] if mark(s) not in have], set()
    while True:
        new = {s["id"] for s in todo if s["id"] not in want and (not any(needs(t) & gives(s) for t in plan["steps"] if t is not s) or any(t["id"] in want and (needs(t) & gives(s)) - have for t in todo if t is not s))}   # a step that gives a key it also needs (it renews a value) is not its own user
        if not new: return want
        want |= new

def ready(plan: dict, have: set, busy=()) -> list:
    """The steps that can start now."""
    want = wanted(plan, have)
    return [s for s in plan["steps"] if s["id"] in want and s["id"] not in busy and needs(s) <= have]

def course(plan: dict, have: set) -> list:
    """What is left, as waves of step ids, if every step gives all it lists: [[can start now], [after those], ...]. One step per wave is a list, several is a graph; which of the two an enactment really uses is the director's choice. Steps that can never start are left out (see unreached)."""
    have, waves = set(have), []
    while (wave := ready(plan, have)):
        waves.append([s["id"] for s in wave])
        have = have.union(*(gives(s) | {mark(s)} for s in wave))
    return waves

def unreached(plan: dict, have: set) -> list: return sorted(wanted(plan, have) - {i for w in course(plan, have) for i in w})
