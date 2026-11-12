"""steps.py - the step types Aviary ships. Each is one function: async fn(config, inputs, ctx) -> {name: value}; config is the step's own data, inputs are the keys it needs, by the names it uses.
None of them knows a machine, a model or a program: work on a provider goes through the role the step names (ctx.run.meta["bound"][role], decided by roles.place)."""
import re, sys, json, time, base64, asyncio, fnmatch, hashlib
from . import plan as P, providers, text

def _dig(d, path: str):
    for k in path.split("."): d = d.get(k) if isinstance(d, dict) else None
    return d

def _bound(ctx, role: str) -> tuple:
    b = ctx.run.meta["bound"][role]
    return b, ctx.director.aviary.lib.get("provider", b["provider"])

def _empty(ctx, config: dict) -> bool:
    """An optional role nobody filled: the step gives nothing (None under each name) instead of failing the run."""
    return config["role"] not in ctx.run.meta.get("bound", {}) and bool((ctx.run.plan.get("roles") or {}).get(config["role"], {}).get("optional"))

async def capability(config: dict, inputs: dict, ctx) -> dict:
    """Any capability, through a role. config: {"role", "capability", "body": the request ({key} filled from inputs), "give": {name: part of the answer}}.
    Parts: text, thinking, tool_calls, result (and result.<field>), usage (and usage.<field>), seconds, job. While it works, what the provider reports is passed to whoever watches the run (ctx.run.emit). The binding's own body values come first, the step's over them; the asset is the binding's.
    "detached": true - the work goes on at the provider even if Aviary stops, and a run picked up later joins it there instead of asking twice. For long work; a short answer someone is waiting on does not need it."""
    if _empty(ctx, config): return {name: None for name in config.get("give") or {"text": "text"}}
    b, p = _bound(ctx, config["role"])
    t0 = time.time()
    rid, body = f"""{ctx.run.id}.{ctx.step["id"]}""", {**b.get("body", {}), **P.render(config.get("body") or {}, inputs), **({"model": b["asset"]} if b.get("asset") and config.get("model", True) else {})}   # "model": false - asked of the role's provider without naming its asset (a program that is not a model, like a prompt encoder beside an image model)
    out = await (providers.run_detached(p, config["capability"], body, rid) if config.get("detached") else providers.run(p, config["capability"], body, rid, on_event=lambda ev: ctx.run.emit(ctx.step, ev)))
    out["seconds"] = round(time.time() - t0, 3)
    return {name: _dig(out, part) for name, part in (config.get("give") or {"text": "text"}).items()}

async def describe(config: dict, inputs: dict, ctx) -> dict:
    """What a role's provider says about itself. config: {"role", "verbs": [describe | state | memory | modules | offer | assets | interface], "give": name}."""
    _, p = _bound(ctx, config["role"])
    return {config.get("give", "card"): {v: await providers.ask(p, v) for v in config["verbs"]}}

async def fill(config: dict, inputs: dict, ctx) -> dict:
    """A text of about config["tokens"] tokens (4 characters each) from config["lines"], numbered and repeated, then config["then"]: the same text every time for the same config. "line" shapes each one ({n}, {line})."""
    lines, want, out = config["lines"], int(config["tokens"]) * 4 - 200, []
    while sum(map(len, out)) < want: out.append(config.get("line", "{line}").replace("{n}", str(len(out) + 1)).replace("{line}", lines[len(out) % len(lines)]))
    return {config.get("give", "text"): "\n".join(out) + config.get("then", "")}

async def digest(config: dict, inputs: dict, ctx) -> dict:
    """sha256 over inputs, to compare two runs byte for byte. config: {"of": [input names, each a list of equal length], "give": {"each": name, "all": name}}: one hash per item (its parts joined by a zero byte) and one over those in order."""
    each = [hashlib.sha256("\x00".join(str(part or "") for part in item).encode()).hexdigest() for item in zip(*(inputs[k] for k in config["of"]))]
    return {config["give"]["each"]: each, config["give"]["all"]: hashlib.sha256("".join(each).encode()).hexdigest()}

async def shape(config: dict, inputs: dict, ctx) -> dict:
    """Values built from inputs, no work done. config: {"set": {name: template}}."""
    return P.render(config["set"], inputs)

_SAFE = {"len": len, "str": str, "int": int, "float": float, "bool": bool, "min": min, "max": max, "abs": abs, "sum": sum, "any": any, "all": all, "sorted": sorted, "round": round, "list": list, "dict": dict, "tuple": tuple, "zip": zip, "range": range, "enumerate": enumerate, "reversed": reversed, "isinstance": isinstance, "next": next, "iter": iter, "set": set, "map": map, "filter": filter}

async def expr(config: dict, inputs: dict, ctx) -> dict:
    """A value worked out from the inputs by one Python expression (an expression only, with a short list of plain builtins; the inputs are its names). config: {"expr", "give": name}.
    With "routes": {value as text: name} (and "else": name) it is how a plan branches: the value is given under the one name it picks, and the steps waiting on the other names are never reached."""
    value = eval(config["expr"], {"__builtins__": _SAFE, **inputs})   # inputs as globals: a comprehension cannot see eval's locals before Python 3.12
    if "routes" not in config: return {config.get("give", "value"): value}
    if not (name := config["routes"].get(str(value), config.get("else"))): raise ValueError(f"""no route for {value!r}: {sorted(config["routes"])}""")
    return {name: value}

async def regex(config: dict, inputs: dict, ctx) -> dict:
    """config: {"of": input name, "pattern", "dotall", "give": name} gives every match as {"text", "groups"}; with "replace" ({0}, {1}.. are the groups, {match} the whole match) it gives the text with every match replaced."""
    source, flags = str(inputs[config["of"]]), re.DOTALL if config.get("dotall") else 0
    if "replace" not in config: return {config.get("give", "matches"): [{"text": m.group(0), "groups": list(m.groups())} for m in re.finditer(config["pattern"], source, flags)]}
    return {config.get("give", "text"): re.sub(config["pattern"], lambda m: P.render(config["replace"], {"match": m.group(0), **{str(i): g or "" for i, g in enumerate(m.groups())}}), source, flags=flags)}

async def chunk(config: dict, inputs: dict, ctx) -> dict:
    """A text cut into pieces (text.py). config: {"of": input name, "chars", "overlap", "code": true for source files, "path": the file's name for a code piece's heading, "give": name}."""
    source, chars = str(inputs[config["of"]]), int(config.get("chars", 4000))
    return {config.get("give", "chunks"): text.code(source, P.render(config.get("path", ""), inputs), chars) if config.get("code") else text.paragraphs(source, chars, int(config.get("overlap", 0)))}

async def pipeline(config: dict, inputs: dict, ctx) -> dict:
    """A whole pipeline as one step: its runs are real runs (named after this step, so a run picked up later goes on with them from what they kept).
    config: {"pipeline": id | "plan": a plan written here, "inputs": {its parameter: template over this step's inputs}, "give": {name here: key there}, "roles": {its role: this run's role}}.
    "each": input name of a list - one run per item ({item} and {n} in "inputs"), their gives as lists. "carry": [keys] - the items run in turn, each given what the one before left under those keys, and the last values are given. "in_turn": true - one after the other without carrying anything. Otherwise the items run as the means allow."""
    eng, me, give = ctx.director.aviary, ctx.run, config.get("give") or {}
    plan, items, carry = config.get("plan") or eng.lib.get("pipeline", config["pipeline"]), inputs[config["each"]] if config.get("each") else [None], {k: inputs[k] for k in config.get("carry", []) if k in inputs}
    name = lambda n: f"""{me.id}~{ctx.step["id"]}""" + (f"~{n}" if config.get("each") else "")
    async def one(n, item, carried):
        given = {k: v for k, v in P.render(config.get("inputs") or {}, {**inputs, **carried, "item": item, "n": n}).items() if not (isinstance(v, str) and re.fullmatch(r"\{[\w.@]+\}", v))}   # a value nothing has filled yet (a carried key before the first item) is left to the pipeline's own default
        if (state := await eng.sub(me, name(n), plan, given, config.get("roles"))) != "done": raise RuntimeError(f"""{name(n)} is {state}: {eng.director.runs[name(n)].meta.get("error", "")}""")
        return eng.director.runs[name(n)].data
    try:
        if config.get("carry") or config.get("in_turn"):
            outs = []
            for n, item in enumerate(items):
                outs.append(await one(n, item, carry))
                carry = {k: outs[-1].get(k, carry.get(k)) for k in config.get("carry", [])}
        else: outs = list(await asyncio.gather(*(one(n, item, {}) for n, item in enumerate(items))))
    except asyncio.CancelledError:
        for n in range(len(items)):
            if name(n) in eng.director.runs and eng.director.runs[name(n)].meta["state"] == "running": eng.director.stop(name(n), now=True)
        raise
    return {**({k: [o.get(v) for o in outs] for k, v in give.items()} if config.get("each") else {k: outs[0].get(v) for k, v in give.items()}), **carry}

pipeline.verbatim = ("plan",)   # a plan written inside the step is that pipeline's own text: this run's parameters reach it only through "inputs"

# --- Python: code run as its own process. Whether a run may is the portal owner's decision (policy), never Aviary's ---

async def python(config: dict, inputs: dict, ctx) -> dict:
    """Runs Python code as a process of its own, with every right the portal itself has: files, network, other programs. Nothing here limits what the code does.
    Whether a run may do this at all is decided by the portal's owner (library object policy "aviary") and answered by whoever starts the run; this step only checks that the answer was yes (the run's "may"). Two decisions: "python_fixed" for code written into the pipeline itself ("script": only someone who may change the pipeline can change it), "python" for code of the starter's own choosing (given as an input, or read from a file).
    config: one of {"script": the code, written in the pipeline} | {"from": input name holding the code} | {"space", "path": a file in a space}; "arg": a template given as the one argument; "timeout_s" (600).
    The step's inputs arrive as one JSON object on standard input. The last line it prints is its result: a JSON object gives those names, anything else is given as "value". It runs in the run's own scratch folder."""
    need = "python_fixed" if config.get("script") else "python"
    if need not in ctx.run.meta.get("may", []): raise PermissionError(f"""this run may not run {"the Python code written into its pipeline" if need == "python_fixed" else "Python code of its starter's choosing"}. The portal's owner decides who may (policy "aviary", key "{need}"); it was started by {ctx.run.meta.get("owner") or "nobody"}""")
    code = config.get("script") or (inputs[config["from"]] if config.get("from") else ctx.director.aviary.space(config["space"])[0].read(P.render(config["path"], inputs)))
    f = ctx.run.dir / "scratch" / f"""{ctx.step["id"]}.py"""
    f.parent.mkdir(exist_ok=True)
    f.write_text(code)
    proc = await asyncio.create_subprocess_exec(sys.executable, str(f.resolve()), str(P.render(config.get("arg", ""), inputs)), stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE, cwd=f.parent)
    try: out, err = await asyncio.wait_for(proc.communicate(json.dumps(inputs, default=str).encode()), float(config.get("timeout_s", 600)))
    except (asyncio.TimeoutError, asyncio.CancelledError):
        proc.kill()
        await proc.wait()
        raise
    if proc.returncode: raise RuntimeError(f"""exit {proc.returncode}: {err.decode(errors="replace")[-1000:]}""")
    last = (out.decode(errors="replace").strip().splitlines() or [""])[-1]
    try: result = json.loads(last)
    except ValueError: result = last
    return result if isinstance(result, dict) else {config.get("give", "value"): result}
python.verbatim = ("script",)   # code is taken as written: its own braces are not placeholders

# --- Files: read from a space, write into it only as a proposal its people accept or reject ---

async def file_read(config: dict, inputs: dict, ctx) -> dict:
    """config: {"space", "path" (template), "give": name, "base64": true for the file as it is (a PDF, an image), base64, instead of its text}."""
    fm, path = ctx.director.aviary.space(config["space"])[0], P.render(config["path"], inputs)
    return {config.get("give", "text"): base64.b64encode(fm.resolve(path).read_bytes()).decode() if config.get("base64") else fm.read(path)}

async def file_list(config: dict, inputs: dict, ctx) -> dict:
    """Paths of the files in a space, relative to it. config: {"space", "under": one folder, or "paths": picked files and folders (a folder stands for every file in it), "extensions": [".md", ...], "skip": [patterns], "give": name}. Hidden names and __pycache__ are left out."""
    fm = ctx.director.aviary.space(config["space"])[0]
    base, picked = fm.resolve(""), P.render(config["paths"], inputs) if "paths" in config else [P.render(config.get("under", ""), inputs)]
    skip = P.render(config.get("skip", []), inputs)   # fnmatch patterns: "folder/" anywhere in the path, else matched against the path and the file name
    def skipped(rel: str) -> bool: return any(("/" + x) in ("/" + rel) if x.endswith("/") else fnmatch.fnmatch(rel, x) or fnmatch.fnmatch(rel.rsplit("/", 1)[-1], x) for x in skip)
    ok = lambda f: f.is_file() and not any(part.startswith(".") or part == "__pycache__" for part in f.relative_to(base).parts) and (not config.get("extensions") or f.suffix.lower() in config["extensions"]) and not skipped(str(f.relative_to(base)))
    found = {f for x in picked for f in ([fm.resolve(x)] if fm.resolve(x).is_file() else fm.resolve(x).rglob("*")) if ok(f)}
    return {config.get("give", "files"): sorted(str(f.relative_to(base)) for f in found)}   # relative to the space: what file_read and file_write take

async def file_write(config: dict, inputs: dict, ctx) -> dict:
    """Proposes a file's new content; nothing in the space changes until someone with a say accepts it. config: {"space", "path" (template), "of": input name of the content, "give": name}. Gives {"path", "status"}."""
    entry = ctx.director.aviary.space(config["space"])[1].stage(P.render(config["path"], inputs), str(inputs[config["of"]]), author=f"""Aviary, for {ctx.run.meta.get("owner") or "nobody"}""", tag=ctx.run.id)
    return {config.get("give", "written"): {"path": entry["path"], "status": entry["status"]}}

async def fetch(config: dict, inputs: dict, ctx) -> dict:
    """A file the role's provider made (an image, a clip, a sound) brought into a space. config: {"role", "location" (where it is kept there, "outputs"), "path" (template), "space" (template), "into" (folder template), "give"}. Only a space that takes files directly ("direct": true) receives it: a made file is new, there is nothing of the person's to overwrite. Gives the path in the space."""
    _, p = _bound(ctx, config["role"])
    space = P.render(config["space"], inputs)
    sp = ctx.director.aviary.lib.get("space", space)
    if not sp.get("direct"): raise PermissionError(f"""space {space} takes proposals only: a fetched file needs a space that takes files directly ("direct": true)""")
    src = P.render(config["path"], inputs)
    rel = "/".join(x for x in (P.render(config.get("into", ""), inputs).strip("/"), src.rsplit("/", 1)[-1]) if x)
    dest = ctx.director.aviary.space(space)[0].resolve(rel)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(await providers.fetch(p, config.get("location", "outputs"), src))
    return {config.get("give", "file"): rel}

TYPES = {"fetch": fetch, "capability": capability, "describe": describe, "fill": fill, "digest": digest, "shape": shape, "expr": expr, "regex": regex, "chunk": chunk, "pipeline": pipeline, "python": python, "file_read": file_read, "file_list": file_list, "file_write": file_write}
