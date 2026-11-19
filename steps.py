"""steps.py - the step types Aviary ships. Each is one function: async fn(config, inputs, ctx) -> {name: value}; config is the step's own data, inputs are the keys it needs, by the names it uses.
None of them knows a machine, a model or a program: work on a provider goes through the role the step names (ctx.run.meta["bound"][role], decided by roles.place)."""
import mimetypes, re, sys, json, math, time, base64, random, asyncio, fnmatch, hashlib, statistics
from types import SimpleNamespace
from . import plan as P, providers, text

def _dig(d, path: str):
    for k in path.split("."): d = d.get(k) if isinstance(d, dict) else None
    return d

def _bound(ctx, role: str) -> tuple:
    b = ctx.run.meta["bound"][role]   # placed by the director before the step started (Aviary.bind)
    return b, ctx.director.aviary.lib.get("provider", b["provider"])

def _empty(ctx, config: dict) -> bool:
    """An optional role nobody filled: the step gives nothing (None under each name) instead of failing the run."""
    return ctx.run.meta.get("bound", {}).get(config["role"], ...) is None

def _body(b: dict, config: dict, inputs: dict) -> dict:
    """The request a capability step sends to the provider its role was placed on: the binding's own body (its options), the step's body over it, and the placed asset as the model. "model": false - asked of the role's provider without naming its asset (a program that is not a model, like a prompt encoder beside an image model)"""
    mine = P.render(config.get("body") or {}, inputs)
    return {**b.get("body", {}), **mine, **({"options": {**b["body"]["options"], **mine["options"]}} if isinstance(b.get("body", {}).get("options"), dict) and isinstance(mine.get("options"), dict) else {}), **({"model": b["asset"]} if b.get("asset") and config.get("model", True) else {})}

async def capability(config: dict, inputs: dict, ctx) -> dict:
    """Any capability, through a role. config: {"role", "capability", "body": the request ({key} filled from inputs), "give": {name: part of the answer}}.
    Parts: text, thinking, tool_calls, result (and result.<field>), usage (and usage.<field>), seconds, job, provider and asset (where it was done). "all": true - where the role holds several assets (its binding's "also"), every one is asked and the answers are joined. While it works, what the provider reports is passed to whoever watches the run (ctx.run.emit). The binding's own body values come first, the step's over them ("options" joined key by key, so a binding's context size and a step's cache session both hold); the asset is the binding's.
    "detached": true - the work goes on at the provider even if Aviary stops, and a run picked up later joins it there instead of asking twice. For long work; a short answer someone is waiting on does not need it."""
    if _empty(ctx, config): return {name: None for name in config.get("give") or {"text": "text"}}
    first, lib = ctx.run.meta["bound"][config["role"]], ctx.director.aviary.lib
    if ctx.run.meta.get("warm"): return await _warm(config, inputs, ctx, first)
    async def one(b: dict, n: int) -> dict:
        p, t0 = lib.get("provider", b["provider"]), time.time()
        rid, body = f"""{ctx.run.id}.{ctx.step["id"]}""" + (f"+{n}" if n else ""), {**_body(b, config, inputs), **({"priority": ctx.run.meta["priority"]} if ctx.run.meta.get("priority") else {})}   # a run started with priority: the machine closes what stands in its way as soon as that has finished what it is doing
        out = await (providers.run_detached(p, config["capability"], body, rid) if config.get("detached") else providers.run(p, config["capability"], body, rid, on_event=lambda ev: ctx.run.emit(ctx.step, ev)))
        out.update(seconds=round(time.time() - t0, 3), provider=p["id"], asset=b.get("asset", ""))   # where it was done: a result kept on that machine (an encoding) names it
        ctx.run.note("work", step=ctx.step["id"], role=config["role"], capability=config["capability"], provider=p["id"], asset=b.get("asset", ""), usage=out.get("usage") or {})   # where the work was done and what the provider reported of it, kept with the run whatever the plan gives or keeps: the source of measure.py's rows
        return out
    give = config.get("give") or {"text": "text"}
    if not (config.get("all") and first.get("also")): return {name: _dig(out, part) for out in [await one(first, 0)] for name, part in give.items()}
    used = [first, *first["also"]]   # "all": true - the same asked of every asset the role holds, at once; texts are joined under each asset's name, lists are joined, anything else is given as a list in the role's order
    outs = await asyncio.gather(*(one(b, n) for n, b in enumerate(used)))
    def joined(part: str):
        vals = [_dig(o, part) for o in outs]
        if all(isinstance(v, str) or v is None for v in vals): return "\n\n".join(f"""[from {b["asset"]}]\n{v}""" for b, v in zip(used, vals) if v)
        return [x for v in vals for x in v] if all(isinstance(v, list) for v in vals) else vals
    return {name: joined(part) for name, part in give.items()}

async def _warm(config: dict, inputs: dict, ctx, b: dict) -> dict:
    """A warm run (Aviary.warm): what this step would use is brought up on the machine its role was placed on, and how that went is noted on the run, per role; the step gives nothing (None under each name), so what follows is walked through the same way."""
    if config["role"] in (ctx.run.meta.get("warm_roles") or [config["role"]]) and config["role"] not in ctx.run.meta.setdefault("warmed", {}):
        p, t0 = ctx.director.aviary.lib.get("provider", b["provider"]), time.time()
        try: said = await providers.warm(p, config["capability"], b.get("asset", ""), _body(b, config, inputs))
        except providers.ProviderError as e: said = {"error": str(e)[:400]}
        ctx.run.meta["warmed"][config["role"]] = {**said, "provider": p["id"], "asset": b.get("asset", ""), "capability": config["capability"], "seconds": round(time.time() - t0, 2)}
        ctx.run.note("warmed", step=ctx.step["id"], role=config["role"], **ctx.run.meta["warmed"][config["role"]])
    return {name: None for name in config.get("give") or {"text": "text"}}

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

# --- Pipelines as tools (for the step "agent", agent_loop.py) ---

_JSON_TYPES = {"integer": "integer", "number": "number", "boolean": "boolean", "list": "array", "items": "array", "object": "object", "options": "object"}

def _tool(eng, spec) -> dict:
    """One tool an agent may call: a pipeline of the library. spec: its id, or {"pipeline": id, "name": what the model calls it, "about": what it is told it does, "fixed": {input: value the plan sets and the model does not see}, "give": [which of its results go back]}. What goes back by default is what the pipeline's last step gives."""
    spec = {"pipeline": spec} if isinstance(spec, str) else spec
    plan, fixed = eng.lib.get("pipeline", spec["pipeline"]), spec.get("fixed") or {}
    props = {k: {"type": _JSON_TYPES.get(p.get("type"), "string"), "description": " ".join(x for x in (p.get("label", ""), p.get("hint", "")) if x)} for k, p in (plan.get("params") or {}).items() if k not in fixed}
    return {"name": spec.get("name") or spec["pipeline"], "plan": plan, "fixed": fixed, "give": spec.get("give") or plan["steps"][-1].get("gives", []), "schema": {"type": "function", "function": {"name": spec.get("name") or spec["pipeline"], "description": spec.get("about") or plan.get("objective") or plan.get("name", ""), "parameters": {"type": "object", "properties": props, "required": [k for k, p in (plan.get("params") or {}).items() if k not in fixed and "default" not in p]}}}}

async def texts(config: dict, inputs: dict, ctx) -> dict:
    """Texts kept in the library (prompts, contexts, skills: any kind whose objects have a text), joined into one. config: {"kind", "ids": an input's template or a list (one id, a list of ids, or nothing), "field": which part ("text"), "join": what goes between them (a blank line), "before"/"after": templates put around the whole, "give": name}.
    Nothing chosen gives "before" and "after" alone. An id the library does not hold stops the run: a text that was asked for and is missing is never left out silently."""
    lib, ids = ctx.director.aviary.lib, P.render(config.get("ids"), inputs)
    ids = [i for i in (ids if isinstance(ids, list) else [ids]) if i]
    missing = [i for i in ids if not lib.has(config["kind"], i)]
    if missing: raise ValueError(f"""the library holds no {config["kind"]} named {", ".join(missing)}""")
    parts = [str(P.render(config.get("before", ""), inputs) or ""), *(str(lib.get(config["kind"], i).get(config.get("field", "text"), "")) for i in ids), str(P.render(config.get("after", ""), inputs) or "")]
    return {config.get("give", "text"): config.get("join", "\n\n").join(p for p in parts if p.strip())}

def kept_encoding(record: dict, text, max_len, default_len: int = 512) -> dict:
    """The one rule for whether a saved prompt already has an encoding to paint from: its newest noted encoding ("encodings": [{"embed", "at", "text", "max_len", ...}]) made of this text (spaces and line ends aside) at this length; {} when it has none. Everything that asks - the step below, a prompt editor showing "ready" - asks here, so they cannot disagree."""
    same = lambda a: " ".join(str(a or "").split())
    return next((e for e in reversed(record.get("encodings") or []) if e.get("embed") and same(e.get("text")) == same(text) and int(e.get("max_len") or default_len) == int(max_len or default_len)), {})

async def kept(config: dict, inputs: dict, ctx) -> dict:
    """Is the encoding this run needs already there? config: {"kind", "id": the saved prompt, "text", "max_len", "give": {name: "embed" | "at"}}. Gives the kept encoding's id and where it is kept, or "" for each when there is none (kept_encoding) - a run then encodes, and one that found it does not."""
    lib, c = ctx.director.aviary.lib, P.render(config, inputs)
    e = kept_encoding(lib.get(c["kind"], c["id"]) if c.get("id") and lib.has(c["kind"], c["id"]) else {}, c.get("text"), c.get("max_len"))
    return {name: e.get(field, "") for name, field in (config.get("give") or {"kept_embed": "embed", "kept_at": "at"}).items()}

async def lib_get(config: dict, inputs: dict, ctx) -> dict:
    """A value from a library object: config {"kind", "id": template, "path": dotted part ("" the whole object), "default": given when the object or the part is not there (without one, that stops the run), "give"}."""
    oid, lib = str(P.render(config["id"], inputs) or ""), ctx.director.aviary.lib
    have = lib.has(config["kind"], oid) if oid else False
    if not have and "default" not in config: raise ValueError(f"""the library holds no {config["kind"]} named {oid or "(none given)"}""")
    v = _dig(lib.get(config["kind"], oid), config["path"]) if have and config.get("path") else lib.get(config["kind"], oid) if have else None
    return {config.get("give", "value"): config.get("default") if v is None else v}

TEXT_KINDS = ("skill", "context", "prompt")   # library kinds a pipeline may write: text for models and people, never code, grants or machines

async def lib_put(config: dict, inputs: dict, ctx) -> dict:
    """Library objects saved from a run (skills, contexts, prompts brought in from elsewhere). config: {"kind": one of TEXT_KINDS, "of": input name holding an object or a list ({"id"?, "name", "text", ...}), "replace": false, "give"}.
    An object whose id is free is saved under it; one the library already holds the same is left ("same"); a different one is saved under a free name (id-2 ...) unless "replace" - and a replaced one is kept as an earlier version either way. Gives [{"id", "done": "saved" | "same" | "renamed" | "replaced"}]."""
    kind, lib, out = config["kind"], ctx.director.aviary.lib, []
    if kind not in TEXT_KINDS: raise ValueError(f"""a pipeline writes only {", ".join(TEXT_KINDS)} to the library, not {kind}""")
    items = inputs[config["of"]]
    for o in items if isinstance(items, list) else [items]:
        oid = re.sub(r"[^a-z0-9_-]+", "_", str(o.get("id") or o.get("name") or "").lower()).strip("_")[:60] or "item"
        o = {k: v for k, v in o.items() if k not in ("id", "modified", "by")}
        old = {k: v for k, v in lib.get(kind, oid).items() if k not in ("id", "modified", "by")} if lib.has(kind, oid) else None
        if old == o: out.append({"id": oid, "done": "same"})
        elif old is None or config.get("replace"):
            lib.save(kind, {"id": oid, **o}, by=f"""run {ctx.run.id}""")
            out.append({"id": oid, "done": "replaced" if old else "saved"})
        else:
            free = next(f"{oid}-{n}" for n in range(2, 1000) if not lib.has(kind, f"{oid}-{n}"))
            lib.save(kind, {"id": free, **o}, by=f"""run {ctx.run.id}""")
            out.append({"id": free, "done": "renamed"})
    return {config.get("give", "saved"): out}

async def lib_note(config: dict, inputs: dict, ctx) -> dict:
    """A record added to a list in a library object (a kept prompt's encodings): config {"kind", "id": template, "field", "record": {name: template}, "same": [names that make two records the same - the newer replaces the older], "when": a key's name (nothing is written unless its value is true), "give"}. The object's earlier version is kept by the library; an object that is not there is left alone."""
    oid, lib = str(P.render(config["id"], inputs) or ""), ctx.director.aviary.lib
    if (config.get("when") and not inputs.get(config["when"])) or not oid or not lib.has(config["kind"], oid): return {config.get("give", "noted"): False}
    o, rec = lib.get(config["kind"], oid), {**P.render(config["record"], inputs), "t": time.time()}
    same = lambda r: all(r.get(k) == rec.get(k) for k in config.get("same") or [])
    lib.save(config["kind"], {**o, config["field"]: [r for r in o.get(config["field"]) or [] if not (config.get("same") and same(r))] + [rec]}, by=ctx.run.meta.get("owner", ""))
    return {config.get("give", "noted"): True}

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
    "each": input name of a list - one run per item ({item} and {n} in "inputs"), their gives as lists. "carry": [keys] - the items run in turn, each given what the one before left under those keys, and the last values are given. "in_turn": true - one after the other without carrying anything. Otherwise the items run as the means allow.
    "failed": name - with "each": an item whose run fails is left out instead of stopping this run; it gives nothing under each name, and every such item is given under this name as {"item", "error"} (the error in full) and noted on the run ("left_out"). "failed_stop": how many failures in a row stop the run after all (5; 0 never) - a provider that is down fails every item, a file it cannot read fails one."""
    eng, me, give = ctx.director.aviary, ctx.run, config.get("give") or {}
    plan, items, carry = config.get("plan") or eng.lib.get("pipeline", config["pipeline"]), inputs[config["each"]] if config.get("each") else [None], {k: inputs[k] for k in config.get("carry", []) if k in inputs}
    name, failed, row = lambda n: f"""{me.id}~{ctx.step["id"]}""" + (f"~{n}" if config.get("each") else ""), [], [0]   # row: failures in a row
    async def one(n, item, carried):
        given = {k: v for k, v in P.render(config.get("inputs") or {}, {**inputs, **carried, "item": item, "n": n}).items() if not (isinstance(v, str) and re.fullmatch(r"\{[\w.@]+\}", v))}   # a value nothing has filled yet (a carried key before the first item) is left to the pipeline's own default
        if (state := await eng.sub(me, name(n), plan, given, config.get("roles"))) == "done":
            row[0] = 0
            return eng.director.runs[name(n)].data
        why = f"""{name(n)} is {state}: {eng.director.runs[name(n)].meta.get("error", "")}"""
        if not (config.get("failed") and config.get("each")) or state != "failed": raise RuntimeError(why)   # only a failure is left out: a stop or a held step is someone's decision
        failed.append({"item": item, "error": why})
        row[0] += 1
        me.note("left_out", step=ctx.step["id"], item=item, error=why)
        if 0 < int(config.get("failed_stop", 5)) <= row[0]: raise RuntimeError(f"""{row[0]} items in a row failed, the last: {why}""")
        return {}
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
    return {**({k: [o.get(v) for o in outs] for k, v in give.items()} if config.get("each") else {k: outs[0].get(v) for k, v in give.items()}), **carry, **({config["failed"]: failed} if config.get("failed") and config.get("each") else {})}

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

async def file_uri(config: dict, inputs: dict, ctx) -> dict:
    """The first of the picked files as a data URI (a picture sent to a program), or nothing when none was picked - for optional pictures (a clip's start and end frames). config: {"space", "paths": the picked list (template), "give"}."""
    fm, picked = ctx.director.aviary.space(P.render(config["space"], inputs))[0], P.render(config["paths"], inputs) or []
    first = next((fm.resolve(x) for x in (picked if isinstance(picked, list) else [picked]) if x and fm.resolve(x).is_file()), None)
    return {config.get("give", "uri"): f"""data:{mimetypes.guess_type(first.name)[0] or "application/octet-stream"};base64,{base64.b64encode(first.read_bytes()).decode()}""" if first else None}

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

async def carry(config: dict, inputs: dict, ctx) -> dict:
    """Files one role's machine made, taken to the machine of another role when the two differ: a prompt encoded on one node and painted on another, a recording heard on one and answered on another. config: {"from": role (or "from_at": a template naming the provider, where a kept result already is), "to": role, "location" (the same name on both, "vault"), "path" (template: a file or a folder), "give"}. Through the portal, so no machine is given another's key. A file already there is left as it is (what is carried this way is named by a hash of what made it). Gives the number of files carried; 0 when both roles are on one machine."""
    src = ctx.director.aviary.lib.get("provider", P.render(config["from_at"], inputs)) if config.get("from_at") else _bound(ctx, config["from"])[1]   # from_at: the machine named by a key (where a kept result is), instead of a role's
    dst = _bound(ctx, config["to"])[1]
    loc, path, give = config.get("location", "vault"), P.render(config["path"], inputs).strip("/"), config.get("give", "carried")
    if src["id"] == dst["id"]: return {give: 0}
    listing = lambda p: providers.machine(p, "GET", f"/files/{loc}?under={path}&recursive=1")
    try: have = {f["path"] for f in (await listing(dst))["files"]}
    except providers.ProviderError: have = set()   # nothing there yet under that name
    try: files = [f["path"] for f in (await listing(src))["files"] if not f["dir"]]
    except providers.ProviderError: files = [path]   # a single file has no listing of its own
    n = 0
    for f in [f for f in files if f not in have]:
        await providers.put(dst, loc, f, await providers.fetch(src, loc, f))
        n += 1
    ctx.run.note("carried", step=ctx.step["id"], src=src["id"], to=dst["id"], location=loc, path=path, files=n)
    return {give: n}

# --- Simulation: populations drawn to stated shares, and roles taking turns on one transcript ---

def json_in(text):
    """The first JSON object or list in a text (a model's answer around it, a fenced block), or the value itself when it is one already; ValueError when there is none."""
    if isinstance(text, (dict, list)): return text
    s, dec = str(text or ""), json.JSONDecoder()
    for i in [i for i, ch in enumerate(s) if ch in "{["]:
        try: return dec.raw_decode(s, i)[0]
        except ValueError: continue
    raise ValueError(f"no JSON object in: {s[:300]}")

async def json_read(config: dict, inputs: dict, ctx) -> dict:
    """A value read from JSON written in a text (json_in). config: {"of": input name, "default": given when there is none (without one, that stops the run), "give"}."""
    try: v = json_in(inputs[config["of"]])
    except ValueError:
        if "default" not in config: raise
        v = config["default"]
    return {config.get("give", "value"): v}

def _quantile(a: dict, q: float) -> float:
    """The value at quantile q (0..1) of one numeric attribute: "range": [lo, hi] with "dist" uniform (the default), "triangular" (its "mode") or "normal" ("mean", "sd"; cut to the range)."""
    lo, hi = (float(x) for x in a["range"])
    if a.get("dist") == "normal": return min(hi, max(lo, statistics.NormalDist(float(a.get("mean", (lo + hi) / 2)), float(a.get("sd", (hi - lo) / 6) or 1e-9)).inv_cdf(min(max(q, 1e-9), 1 - 1e-9))))
    if a.get("dist") == "triangular":
        m = min(hi, max(lo, float(a.get("mode", (lo + hi) / 2))))
        f = (m - lo) / (hi - lo) if hi > lo else 0.5
        return lo + math.sqrt(q * (hi - lo) * (m - lo)) if q < f else hi - math.sqrt((1 - q) * (hi - lo) * (hi - m))
    return lo + q * (hi - lo)

def _fits(person: dict, when: dict) -> bool:
    """Whether a drawn person meets an attribute's "when": {attribute: [lo, hi] for a number | [values] for a choice}."""
    return all((isinstance(c, list) and len(c) == 2 and all(isinstance(x, (int, float)) for x in c) and isinstance(person.get(k), (int, float)) and c[0] <= person[k] <= c[1]) or (isinstance(c, list) and person.get(k) in c) for k, c in (when or {}).items())

async def draw(config: dict, inputs: dict, ctx) -> dict:
    """People (or anything with attributes) drawn so the whole set holds the stated shares: no model involved, the same set for the same seed. config: {"spec": input name (an object, or a text holding JSON), "count", "seed", "give"}.
    The spec: {"attributes": {name: one of
      {"choices": {value: weight}} - dealt out in the weights' proportions (the largest remainders get the odd ones), then shuffled;
      {"range": [lo, hi], "dist": uniform | triangular | normal, "mode", "mean", "sd", "int": true} - one value per equal slice of the distribution, shuffled, so a small set still spans it;
      each with "when": {earlier attribute: [lo, hi] or [values]} - only those who meet it get one, the rest get "else" (null)}}.
    Attributes are drawn in the order given, so a "when" can only refer to one listed before it. Each drawn item has "n" (its place) as well."""
    spec, n = json_in(inputs[config["spec"]] if config["spec"] in inputs else P.render(config["spec"], inputs)), int(P.render(config["count"], inputs))
    rng, out = random.Random(str(P.render(config.get("seed", 0), inputs))), [{"n": i} for i in range(int(n))]
    for name, a in (spec.get("attributes") or spec).items():
        who = [p for p in out if _fits(p, a.get("when"))]
        for p in out: p[name] = a.get("else")
        if "choices" in a:
            w, total = {str(k): float(v) for k, v in a["choices"].items() if float(v) > 0}, sum(float(v) for v in a["choices"].values() if float(v) > 0) or 1
            share = {k: v / total * len(who) for k, v in w.items()}
            got = {k: int(v) for k, v in share.items()}
            for k in sorted(share, key=lambda k: share[k] - got[k], reverse=True)[:len(who) - sum(got.values())]: got[k] += 1
            vals = [k for k, c in got.items() for _ in range(c)]
        else:
            vals = [_quantile(a, (i + rng.random()) / len(who)) for i in range(len(who))]
            vals = [int(round(v)) for v in vals] if a.get("int") else [round(v, 3) for v in vals]
        rng.shuffle(vals)
        for p, v in zip(who, vals): p[name] = v
    return {config.get("give", "drawn"): out}

def _said(entries: list, me: str) -> list:
    """A transcript as one speaker sees it: its own turns as its answers, everyone else's as what it is told ("Name: text"); turns from the same side joined, so the roles alternate as chat models expect."""
    msgs = []
    for e in entries:
        role, text = ("assistant", e["text"]) if e["who"] == me else ("user", f"""{e["who"]}: {e["text"]}""")
        if msgs and msgs[-1]["role"] == role: msgs[-1]["content"] += "\n\n" + text
        else: msgs.append({"role": role, "content": text})
    return msgs

async def turns(config: dict, inputs: dict, ctx) -> dict:
    """Several roles taking turns on one transcript until a judge says it is over, or the rounds run out: a scene played by a person and a game master, a debate, an interview.
    config: {"speakers": [{"name", "role", "system"}] (in the order they speak each round), "judge": {"name", "role", "system", "done": a word in the judge's notes that ends it ("DONE")} (optional), "opening": what the scene starts from (said by "Scene"), "rounds", "think", "give": {"transcript", "notes", "ended"}}.
    Every speaker sees the whole transcript (its own turns as its own answers). After each round the judge reads all of it and writes notes (what happened, what does not fit); they are kept with the round. The transcript and notes are kept as they grow (Run.live), so whoever watches sees each turn and a run that stops goes on from the last whole turn. Gives the transcript [{"who", "text", "round"}], the notes [{"round", "text"}] and why it ended ("judge" | "rounds")."""
    c, give = P.render({k: v for k, v in config.items() if k != "give"}, inputs), {"transcript": "transcript", "notes": "notes", "ended": "ended", **(config.get("give") or {})}
    sid = ctx.step["id"]
    log, notes = ctx.run.live.get(f"{sid}:transcript") or [{"who": "Scene", "text": str(c.get("opening") or ""), "round": 0}], ctx.run.live.get(f"{sid}:notes") or []
    judge, rounds = c.get("judge") or {}, int(c.get("rounds") or 6)
    done = str(judge.get("done") or "DONE")
    async def say(who: dict, msgs: list, n: int) -> str:
        sub = SimpleNamespace(**{**vars(ctx), "step": {**ctx.step, "id": f"""{sid}.{n}"""}})   # each turn is its own request to the provider
        got = await capability({"role": who["role"], "capability": "chat", "body": {"messages": [{"role": "system", "content": str(who.get("system") or "")}, *msgs], "think": bool(c.get("think"))}, "give": {"text": "text"}}, {}, sub)
        return re.sub(rf"""^\s*\**{re.escape(str(who.get("name", "")))}\**\s*:\s*""", "", str(got["text"] or "")).strip()   # a model that writes its own name before its turn: the name is the transcript's to show
    over = lambda: bool(judge and notes and re.search(rf"\b{re.escape(done)}\b", notes[-1]["text"], re.I))
    whole = lambda r: r == 0 or (all(any(e["round"] == r and e["who"] == s["name"] for e in log) for s in c["speakers"]) and (not judge or any(x["round"] == r for x in notes)))
    r = max(e["round"] for e in log)   # a run picked up after a stop goes on in the round it was in
    while not over():
        if whole(r): r += 1
        if r > rounds: break
        for sp in [s for s in c["speakers"] if not any(e["round"] == r and e["who"] == s["name"] for e in log)]:
            log.append({"who": sp["name"], "text": await say(sp, [*_said(log, sp["name"]), *([{"role": "user", "content": "(go on)"}] if log[-1]["who"] == sp["name"] else [])], len(log)), "round": r})
            ctx.run.keep_live(f"{sid}:transcript", log)
            ctx.run.emit(ctx.step, {"turn": len(log) - 1, "who": sp["name"], "round": r})
        if judge and not any(x["round"] == r for x in notes):
            notes.append({"round": r, "text": await say(judge, [{"role": "user", "content": "\n\n".join(f"""{e["who"]}: {e["text"]}""" for e in log)}], 1000 + r)})
            ctx.run.keep_live(f"{sid}:notes", notes)
    return {give["transcript"]: log, give["notes"]: notes, give["ended"]: "judge" if over() else "rounds"}

TYPES = {"turns": turns, "draw": draw, "json_read": json_read, "carry": carry, "fetch": fetch, "capability": capability, "describe": describe, "fill": fill, "digest": digest, "shape": shape, "expr": expr, "regex": regex, "chunk": chunk, "pipeline": pipeline, "python": python, "file_read": file_read, "file_list": file_list, "file_write": file_write, "texts": texts, "lib_get": lib_get, "lib_put": lib_put, "kept_encoding": kept, "lib_note": lib_note, "file_uri": file_uri}   # "agent" is agent_loop.agent, added by Aviary (it imports this file)
for _f in (capability, describe, fill, digest, shape, expr, regex, chunk, file_read, file_list, texts, lib_get, kept, file_uri, json_read, draw): _f.warm = True   # run in a warm run as in any other (director.py): they only read, or warm instead of working (capability)
