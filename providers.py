"""providers.py - a provider is a place that offers capabilities and speaks the node contract: describe, offer, plan, run, state. A provider record is data: {"id", "name", "url", "key", "speaks": a translator, for one that does not speak the contract}.
The contract's source is the provider itself (its /interface): Aviary keeps no copy of any capability's shape. Anything that does not speak the contract is brought to it by a translator in front of it; no other way of talking to a machine exists in the engine."""
import json, asyncio, httpx

VERBS = {"describe": "/node", "interface": "/interface", "offer": "/node/capabilities", "assets": "/models", "state": "/services", "memory": "/budget", "modules": "/modules", "knobs": "/knobs"}   # the contract's read-only verbs
LONG = httpx.Timeout(connect=10.0, read=None, write=60.0, pool=10.0)   # work takes as long as it takes; only reaching the provider is timed

class ProviderError(RuntimeError): pass

# --- Translators: a provider that does not speak the contract ("speaks": <translator id> in its record) is asked through a translator, which is data (library kind "translator"): where each verb is asked, how a request becomes its body, which parts of its answer are the contract's. What such a provider cannot answer (memory, modules, plans) is said to be unknown, never guessed. ---

translator = None   # fn(id) -> the translator object; set by Aviary when it starts (the library is its)

def _tr(p: dict) -> dict: return translator(p["speaks"]) if p.get("speaks") else None

def _dig(d, path: str):
    for k in path.split("."): d = d.get(k) if isinstance(d, dict) else None
    return d

def _take(d, spec):
    """One part of a foreign answer: a dotted path, or [path, factor] for a number in other units."""
    return (_dig(d, spec[0]) or 0) * spec[1] if isinstance(spec, list) else _dig(d, spec)

def _fill(node, req: dict):
    """A foreign request body from a contract request: {name} filled from the request; a value whose {name} the request does not have is left out, so the foreign side uses its own default."""
    if isinstance(node, dict): return {k: v for k, v in ((k, _fill(v, req)) for k, v in node.items()) if v is not _GONE}
    if isinstance(node, str) and node.startswith("{") and node.endswith("}") and node[1:-1].isidentifier(): return req[node[1:-1]] if req.get(node[1:-1]) not in (None, "") else _GONE
    return node
_GONE = object()

async def _foreign(p: dict, method: str, path: str, body=None, timeout=60.0):
    try:
        async with httpx.AsyncClient(timeout=timeout, trust_env=False) as c: r = await c.request(method, p["url"] + path, headers=_head(p), json=body)
    except httpx.HTTPError as e: _fail(p, f"""cannot be reached ({p["url"]}): {type(e).__name__}: {e}""")
    if r.status_code != 200: _fail(p, f"{method} {path} answered {r.status_code}: {r.text[:300]}")
    return r.json()

async def _tr_assets(p: dict, tr: dict, capability: str) -> list:
    """Its assets in the contract's shape. Which capabilities an asset has is the translator's rule: "capabilities": {"default": [...], "match": {capability: regex over the name - such names have that capability only}}."""
    import re
    a, rule = tr["assets"], tr["assets"]["capabilities"]
    def caps(name): return next(([c] for c, rx in rule.get("match", {}).items() if re.search(rx, name, re.I)), rule["default"])
    rows = [{"name": _dig(x, a["name"]), "capabilities": caps(_dig(x, a["name"])), "size": _dig(x, a["size"]) if a.get("size") else 0, "kind": "model", "notes": "through " + p["speaks"]} for x in _dig(await _foreign(p, "GET", a["path"]), a["list"]) or []]
    return [r for r in rows if not capability or capability in r["capabilities"]]

async def fetch(p: dict, location: str, path: str) -> bytes:
    """A file from a provider's own locations (GET /files/<location>/<path>), whole."""
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(connect=10.0, read=600.0, write=30.0, pool=10.0), trust_env=False) as c: r = await c.get(f"""{p["url"]}/files/{location}/{path}""", headers=_head(p))
    except httpx.HTTPError as e: _fail(p, f"""lost fetching {location}/{path}: {type(e).__name__}: {e}""")
    if r.status_code != 200: _fail(p, f"""{location}/{path} answered {r.status_code}: {r.text[:300]}""")
    return r.content

def _head(p: dict) -> dict: return {"Authorization": f"""Bearer {p["key"]}"""} if p.get("key") else {}

async def ask(p: dict, verb: str, query: str = "", body: dict = None, timeout: float = 60.0):
    """One of the contract's verbs (VERBS) or "plan"; ProviderError says which provider and what it answered."""
    if tr := _tr(p):
        if verb == "plan": return {"fits": "yes", "will": f"""It speaks {p["speaks"]}, not the node contract: whether it fits there is not known here; the work is sent as it is."""}
        if verb == "describe": return {"node": p.get("name") or p["id"], "speaks": p["speaks"], **({k: v for k, v in (await _foreign(p, "GET", tr["describe"]["path"], timeout=timeout)).items() if isinstance(v, (str, int, float))} if tr.get("describe") else {})}
        if verb == "assets": return {"models": await _tr_assets(p, tr, query.partition("capability=")[2])}
        _fail(p, f"""it speaks {p["speaks"]}, which has no {verb}: that is not known for this provider""")
    try:
        async with httpx.AsyncClient(timeout=timeout, trust_env=False) as c: r = await (c.post(f"""{p["url"]}/plan""", headers=_head(p), json=body) if verb == "plan" else c.get(f"""{p["url"]}{VERBS[verb]}{query}""", headers=_head(p)))
    except httpx.HTTPError as e: raise ProviderError(f"""{p.get("name") or p["id"]} ({p["url"]}) cannot be reached: {type(e).__name__}: {e}""")
    if r.status_code != 200: raise ProviderError(f"""{p.get("name") or p["id"]} answered {verb} with {r.status_code}: {r.text[:300]}""")
    return r.json()

async def assets(p: dict, capability: str = "", timeout: float = 60.0) -> list: return (await ask(p, "assets", f"?capability={capability}" if capability else "", timeout=timeout))["models"]

async def manage(p: dict, method: str, kind: str = "", name: str = "", body: dict = None, query: str = ""):
    """The things a manager makes at a provider, by kind (a knowledge group, ...): GET with no kind = the kinds and what one of each is made of; GET kind = the ones there; PUT kind/name = make or change; DELETE kind/name (query "?confirm=1", "&data=1") = remove."""
    if p.get("speaks"): _fail(p, f"""it speaks {p["speaks"]}: nothing can be made there from here""")
    path = "/assets/" + (f"{kind}/{name}" if name else kind or "kinds") + query
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(connect=6.0, read=120.0, write=30.0, pool=6.0), trust_env=False) as c: r = await c.request(method, f"""{p["url"]}{path}""", headers=_head(p), json=body)
    except httpx.HTTPError as e: _fail(p, f"""cannot be reached ({p["url"]}): {type(e).__name__}: {e}""")
    if r.status_code != 200: _fail(p, f"""{method} {path} answered {r.status_code}: {r.text[:400]}""")
    return r.json()

async def machine(p: dict, method: str, path: str, body: dict = None, timeout: float = 60.0):
    """The provider's own management (its programs: /modules...), for a manager. Only a provider that speaks the contract has one."""
    if p.get("speaks"): _fail(p, f"""it speaks {p["speaks"]}: its programs are managed where it runs, not from here""")
    return await _foreign(p, method, path, body, timeout)

async def machine_lines(p: dict, path: str, body: dict):
    """A management call that answers line by line as it goes (installing a program): each line as the object it is."""
    try:
        async with httpx.AsyncClient(timeout=LONG, trust_env=False) as c:
            async with c.stream("POST", p["url"] + path, headers=_head(p), json=body) as r:
                if r.status_code != 200: _fail(p, f"""POST {path} answered {r.status_code}: {(await r.aread()).decode(errors="replace")[:400]}""")
                async for line in r.aiter_lines():
                    if line.strip(): yield json.loads(line)
    except httpx.HTTPError as e: _fail(p, f"""lost during {path} ({p["url"]}): {type(e).__name__}: {e}""")

def address(text: str) -> str:
    """A provider's address as it is stored: with its scheme (http:// when none was typed) and no trailing slash."""
    text = text.strip().rstrip("/")
    return text if "://" in text else f"http://{text}"

def _fail(p: dict, text: str): raise ProviderError(f"""{p.get("name") or p["id"]}: {text}""")

async def run(p: dict, capability: str, body: dict, rid: str = "", on_event=None) -> dict:
    """Does the work while the caller waits and gathers its events: {"text", "thinking", "tool_calls", "result", "usage", "reason", "job"}. on_event sees each event as it comes. If the caller goes away, the provider drops the work."""
    out = {"text": "", "thinking": "", "tool_calls": [], "result": None, "usage": {}, "reason": "", "job": ""}
    if tr := _tr(p): return await _tr_run(p, tr, capability, body, out, on_event)
    try:
        async with httpx.AsyncClient(timeout=LONG, trust_env=False) as c:
            async with c.stream("POST", f"""{p["url"]}/run/{capability}?shape=events""", headers={**_head(p), **({"X-Request-Id": rid} if rid else {})}, json=body) as r:
                if r.status_code != 200: _fail(p, f"""refused {capability} with {r.status_code}: {(await r.aread()).decode(errors="replace")[:400]}""")
                async for line in r.aiter_lines():
                    if not line.strip(): continue
                    ev = json.loads(line)
                    if on_event: on_event(ev)
                    t = ev.get("type")
                    if t in ("text", "thinking"): out[t] += ev["text"]
                    elif t == "tool_calls": out["tool_calls"] += ev["calls"]
                    elif t == "result": out["result"] = ev["data"]
                    elif t == "usage": out["usage"].update({k: v for k, v in ev.items() if k != "type"})
                    elif t == "job": out["job"] = ev.get("id", "")
                    elif t == "done": out["reason"] = ev.get("reason", "")
                    elif t == "error": _fail(p, f"""{capability}: {ev.get("message")}""")
    except httpx.HTTPError as e: _fail(p, f"""lost during {capability} ({p["url"]}): {type(e).__name__}: {e}""")
    return out

async def _tr_run(p: dict, tr: dict, capability: str, body: dict, out: dict, on_event) -> dict:
    """The work through a translator: {"path", "body", "stream": "ndjson" (a line per piece) | absent (one answer), "text", "thinking", "tool_calls", "usage": {contract name: part}, "result": part, "done": the part that is true on the last line}."""
    if capability not in tr["run"]: _fail(p, f"""it speaks {p["speaks"]}, which does not do {capability} here""")
    w, req = tr["run"][capability], body
    def piece(d):
        for k in ("text", "thinking"):
            if w.get(k) and (t := _take(d, w[k])):
                out[k] += t
                if on_event: on_event({"type": k, "text": t})
        if w.get("tool_calls") and (calls := _take(d, w["tool_calls"])): out["tool_calls"] += calls
        if w.get("usage") and (not w.get("done") or _take(d, w["done"])): out["usage"].update({k: _take(d, x) for k, x in w["usage"].items() if _take(d, x) is not None})
        if w.get("result"): out["result"] = _take(d, w["result"])
    try:
        async with httpx.AsyncClient(timeout=LONG, trust_env=False) as c:
            if not w.get("stream"):
                r = await c.post(p["url"] + w["path"], headers=_head(p), json=_fill(w["body"], req))
                if r.status_code != 200: _fail(p, f"""{capability}: {w["path"]} answered {r.status_code}: {r.text[:300]}""")
                piece(r.json())
            else:
                async with c.stream("POST", p["url"] + w["path"], headers=_head(p), json=_fill(w["body"], req)) as r:
                    if r.status_code != 200: _fail(p, f"""{capability}: {w["path"]} answered {r.status_code}: {(await r.aread()).decode(errors="replace")[:300]}""")
                    async for line in r.aiter_lines():
                        if line.strip(): piece(json.loads(line))
    except httpx.HTTPError as e: _fail(p, f"""lost during {capability} ({p["url"]}): {type(e).__name__}: {e}""")
    out["reason"] = "stop"
    return out

async def run_detached(p: dict, capability: str, body: dict, rid: str) -> dict:
    """The same work, but it goes on at the provider whatever happens to the caller. rid names it: asking again under the same rid joins the work already there (running or finished) instead of starting it twice, which is how a run that was cut finds its answer.
    The answer is taken from the provider once read. A caller that is stopped on purpose (cancelled) stops the work too; one that simply dies does not.
    A provider behind a translator cannot keep work for a caller that is gone: there it is asked attached, and a cut means asking again."""
    if p.get("speaks"): return await run(p, capability, body, rid)
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(connect=10.0, read=90.0, write=60.0, pool=10.0), trust_env=False) as c:
            r = await c.post(f"""{p["url"]}/run/{capability}?background=1""", headers={**_head(p), "X-Request-Id": rid}, json=body)
            if r.status_code != 200: _fail(p, f"refused {capability} with {r.status_code}: {r.text[:400]}")
            jid = r.json()["job"]
            try:
                while (d := (await c.get(f"""{p["url"]}/jobs/{jid}?wait=30""", headers=_head(p))).json()).get("phase") != "done":
                    if d.get("error"): _fail(p, f"""{capability}: job {jid} is gone: {d["error"]}""")
            except asyncio.CancelledError:
                await c.post(f"""{p["url"]}/jobs/{jid}/cancel""", headers=_head(p))
                raise
            await c.delete(f"""{p["url"]}/jobs/{jid}""", headers=_head(p))
    except httpx.HTTPError as e: _fail(p, f"""lost during {capability} ({p["url"]}): {type(e).__name__}: {e}. The work may still be going on there; going on with the run asks for it again by the same name.""")
    if d.get("error") or d.get("outcome") != "served": _fail(p, f"""{capability}: {(d.get("error") or {}).get("message") or d.get("reason") or d.get("outcome")}""")
    return {**{k: d.get(k) for k in ("text", "thinking", "tool_calls", "result", "usage", "reason")}, "job": jid}
