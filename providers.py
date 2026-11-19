"""providers.py - a provider is a place that offers capabilities and speaks the node contract: describe, offer, plan, run, state. A provider record is data: {"id", "name", "url", "key", "speaks": a translator, for one that does not speak the contract}.
The contract's source is the provider itself (its /interface): Aviary keeps no copy of any capability's shape. Anything that does not speak the contract is brought to it by a translator in front of it; no other way of talking to a machine exists in the engine."""
import json, time, asyncio, contextvars, httpx

VERBS = {"describe": "/node", "interface": "/interface", "offer": "/node/capabilities", "assets": "/models", "state": "/services", "memory": "/budget", "modules": "/modules", "knobs": "/knobs"}   # the contract's read-only verbs
CONNECT_S = 3.0   # a machine on the network takes milliseconds to accept a connection; one that has not within this is off, asleep or away (a sleeping machine never refuses, it just does not answer)

class ProviderError(RuntimeError): pass
class Unreachable(ProviderError): """Nothing answered at its address: off, asleep, or out of reach. Not a fault of the provider or of the request."""

DOWN: dict = {}   # provider id -> (time, why, wait s): ones that did not answer; asked again only after wait, which doubles while it stays silent (DOWN_S up to DOWN_MAX_S), so an asleep or far-away machine does not slow every page and placement
DOWN_S, DOWN_MAX_S = 20.0, 300.0

TRY_ANYWAY = contextvars.ContextVar("try_anyway", default=False)   # set by a question asked in the background (seen.py), which holds up nothing and so tries a machine marked down as well: that is how one back up is noticed

def _client(p: dict, read=60.0, write=60.0) -> httpx.AsyncClient:
    """A client for one provider: connecting may take CONNECT_S, the answer as long as the work needs (read None: no limit). One that did not connect is not tried again until its wait has passed (forget(p) to try now), except by a question asked in the background (TRY_ANYWAY); one that answers again is no longer marked down."""
    if (d := DOWN.get(p["id"])) and d[0] > time.monotonic() - d[2] and not TRY_ANYWAY.get(): raise Unreachable(d[1])
    return httpx.AsyncClient(timeout=httpx.Timeout(connect=CONNECT_S, read=read, write=write, pool=CONNECT_S), trust_env=False)

HEARD_S = 75.0   # a node says every 15 s that work is still going (a heartbeat, with its phase), however long the work is quiet - reading a long prompt, loading a model; nothing at all for this long means the connection is lost, never that the work is slow
SLOW_S = 2.0   # a machine slower than this to answer a short question is named in the server log, so a slow page can be traced to the machine and the question

def _slow(p: dict, what: str, t0: float):
    if (dt := time.monotonic() - t0) > SLOW_S: print(f"""[aviary] {p.get("name") or p["id"]} took {dt:.1f} s to answer {what}""", flush=True)

def forget(pid: str = ""):
    """Tries a provider (every one, with no id) again on the next call, whatever its wait: after it was woken, or when a person asks the machines again."""
    DOWN.pop(pid, None) if pid else DOWN.clear()

def _lost(p: dict, e: Exception, what: str):
    """A failed request said as it was: not answering at all (marked down for DOWN_S, Unreachable), or lost on the way (ProviderError)."""
    if isinstance(e, (httpx.ConnectError, httpx.ConnectTimeout)):
        DOWN[p["id"]] = (time.monotonic(), f"""{p.get("name") or p["id"]} ({p["url"]}) is not answering: {type(e).__name__}: {e}""", min(DOWN_MAX_S, DOWN[p["id"]][2] * 2) if p["id"] in DOWN else DOWN_S)
        raise Unreachable(DOWN[p["id"]][1])
    _fail(p, f"""{what} ({p["url"]}): {type(e).__name__}: {e}""")

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
        t0 = time.monotonic()
        async with _client(p, timeout, timeout) as c: r = await c.request(method, p["url"] + path, headers=_head(p), json=body)
        _slow(p, f"{method} {path}", t0)
        DOWN.pop(p["id"], None)
    except httpx.HTTPError as e: _lost(p, e, "cannot be reached")
    if r.status_code != 200: _fail(p, f"{method} {path} answered {r.status_code}: {r.text[:300]}")
    return r.json() if r.content.strip() else {}   # some servers answer an act with 200 and nothing (Ollama's delete)

async def server_do(p: dict, verb: str, model: str, timeout: float = 60.0) -> dict:
    """One act on a model that a translator describes for its server ("manage": {verb: {"method", "path", "body"}}, {model} filled in): "remove" a model there, "pull" one onto it. What the translator does not describe is done on the server itself, and the error says so."""
    spec = ((_tr(p) or {}).get("manage") or {}).get(verb)
    if not spec: _fail(p, f"""it speaks {p.get("speaks") or "the node contract"}, which describes no "{verb}" here: that is done on the server itself""")
    return await _foreign(p, spec.get("method", "POST"), spec["path"], {k: v.replace("{model}", model) if isinstance(v, str) else v for k, v in (spec.get("body") or {}).items()}, timeout)

async def _tr_assets(p: dict, tr: dict, capability: str) -> list:
    """Its assets in the contract's shape. Which capabilities an asset has is the translator's rule: "capabilities": {"default": [...], "match": {capability: regex over the name - such names have that capability only}, "add": {capability: regex - such names have it as well as the others}}."""
    import re
    a, rule = tr["assets"], tr["assets"]["capabilities"]
    def caps(name): return next(([c] for c, rx in rule.get("match", {}).items() if re.search(rx, name, re.I)), [*rule["default"], *(c for c, rx in rule.get("add", {}).items() if re.search(rx, name, re.I))])
    rows = [{"name": _dig(x, a["name"]), "capabilities": caps(_dig(x, a["name"])), "size": _dig(x, a["size"]) if a.get("size") else 0, "kind": "model", "notes": "through " + p["speaks"]} for x in _dig(await _foreign(p, "GET", a["path"]), a["list"]) or []]
    return [r for r in rows if not capability or capability in r["capabilities"]]

async def fetch(p: dict, location: str, path: str) -> bytes:
    """A file from a provider's own locations (GET /files/<location>/<path>), whole."""
    try:
        async with _client(p, 600.0, 30.0) as c: r = await c.get(f"""{p["url"]}/files/{location}/{path}""", headers=_head(p))
    except httpx.HTTPError as e: _lost(p, e, f"lost fetching {location}/{path}")
    if r.status_code != 200: _fail(p, f"""{location}/{path} answered {r.status_code}: {r.text[:300]}""")
    return r.content

async def put(p: dict, location: str, path: str, data: bytes, overwrite: bool = False) -> dict:
    """A file written into a provider's own locations (PUT /files/<location>/<path>). A name already taken is refused unless overwrite."""
    try:
        async with _client(p, 600.0, 600.0) as c: r = await c.put(f"""{p["url"]}/files/{location}/{path}""" + ("?overwrite=1" if overwrite else ""), content=data, headers=_head(p))
    except httpx.HTTPError as e: _lost(p, e, f"lost writing {location}/{path}")
    if r.status_code != 200: _fail(p, f"""writing {location}/{path} answered {r.status_code}: {r.text[:300]}""")
    return r.json()

def _head(p: dict) -> dict: return {"Authorization": f"""Bearer {p["key"]}"""} if p.get("key") else {}

async def ask(p: dict, verb: str, query: str = "", body: dict = None, timeout: float = 60.0):
    """One of the contract's verbs (VERBS) or "plan"; ProviderError says which provider and what it answered."""
    if tr := _tr(p):
        if verb == "plan": return {"fits": "yes", "will": f"""It speaks {p["speaks"]}, not the node contract: whether it fits there is not known here; the work is sent as it is."""}
        if verb == "describe": return {"node": p.get("name") or p["id"], "speaks": p["speaks"], **({k: v for k, v in (await _foreign(p, "GET", tr["describe"]["path"], timeout=timeout)).items() if isinstance(v, (str, int, float))} if tr.get("describe") else {})}
        if verb == "assets": return {"models": await _tr_assets(p, tr, query.partition("capability=")[2])}
        _fail(p, f"""it speaks {p["speaks"]}, which has no {verb}: that is not known for this provider""")
    try:
        t0 = time.monotonic()
        async with _client(p, timeout, timeout) as c: r = await (c.post(f"""{p["url"]}/plan""", headers=_head(p), json=body) if verb == "plan" else c.get(f"""{p["url"]}{VERBS[verb]}{query}""", headers=_head(p)))
        _slow(p, verb, t0)
    except httpx.HTTPError as e: _lost(p, e, "cannot be reached")
    DOWN.pop(p["id"], None)
    if r.status_code != 200: raise ProviderError(f"""{p.get("name") or p["id"]} answered {verb} with {r.status_code}: {r.text[:300]}""")
    return r.json()

async def assets(p: dict, capability: str = "", timeout: float = 60.0) -> list: return (await ask(p, "assets", f"?capability={capability}" if capability else "", timeout=timeout))["models"]

async def manage(p: dict, method: str, kind: str = "", name: str = "", body: dict = None, query: str = ""):
    """The things a manager makes at a provider, by kind (a knowledge group, ...): GET with no kind = the kinds and what one of each is made of; GET kind = the ones there; PUT kind/name = make or change; DELETE kind/name (query "?confirm=1", "&data=1") = remove."""
    if p.get("speaks"): _fail(p, f"""it speaks {p["speaks"]}: nothing can be made there from here""")
    path = "/assets/" + (f"{kind}/{name}" if name else kind or "kinds") + query
    try:
        async with _client(p, 120.0, 30.0) as c: r = await c.request(method, f"""{p["url"]}{path}""", headers=_head(p), json=body)
    except httpx.HTTPError as e: _lost(p, e, "cannot be reached")
    if r.status_code != 200: _fail(p, f"""{method} {path} answered {r.status_code}: {r.text[:400]}""")
    return r.json()

async def machine(p: dict, method: str, path: str, body: dict = None, timeout: float = 60.0):
    """The provider's own management (its programs: /modules...), for a manager. Only a provider that speaks the contract has one."""
    if p.get("speaks"): _fail(p, f"""it speaks {p["speaks"]}: its programs are managed where it runs, not from here""")
    return await _foreign(p, method, path, body, timeout)

async def machine_lines(p: dict, path: str, body: dict):
    """A management call that answers line by line as it goes (installing a program): each line as the object it is."""
    try:
        async with _client(p, None, 60.0) as c:
            async with c.stream("POST", p["url"] + path, headers=_head(p), json=body) as r:
                if r.status_code != 200: _fail(p, f"""POST {path} answered {r.status_code}: {(await r.aread()).decode(errors="replace")[:400]}""")
                async for line in r.aiter_lines():
                    if line.strip(): yield json.loads(line)
    except httpx.HTTPError as e: _lost(p, e, f"lost during {path}")

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
        async with _client(p, HEARD_S, 60.0) as c:
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
    except httpx.ReadTimeout: _fail(p, f"""lost during {capability}: no sign from it for {HEARD_S:.0f} s (it says every 15 s that work is going on); the machine or the way to it went away""")
    except httpx.HTTPError as e: _lost(p, e, f"lost during {capability}")
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
        async with _client(p, None, 60.0) as c:
            if not w.get("stream"):
                r = await c.post(p["url"] + w["path"], headers=_head(p), json=_fill(w["body"], req))
                if r.status_code != 200: _fail(p, f"""{capability}: {w["path"]} answered {r.status_code}: {r.text[:300]}""")
                piece(r.json())
            else:
                async with c.stream("POST", p["url"] + w["path"], headers=_head(p), json=_fill(w["body"], req)) as r:
                    if r.status_code != 200: _fail(p, f"""{capability}: {w["path"]} answered {r.status_code}: {(await r.aread()).decode(errors="replace")[:300]}""")
                    async for line in r.aiter_lines():
                        if line.strip(): piece(json.loads(line))
    except httpx.HTTPError as e: _lost(p, e, f"lost during {capability}")
    out["reason"] = "stop"
    return out

async def warm(p: dict, capability: str, model: str, body: dict) -> dict:
    """What a request would use, up ahead of it: {"seconds", "read_ahead"?, "why"?, "how"}. A node loads it through its scheduler (POST /warm). Where that is not offered (a provider behind a translator, an older node), a chat with messages is asked for one token instead - that loads the model and reads the standing start the same way - and anything else says it cannot be warmed from here."""
    t0 = time.monotonic()
    if not p.get("speaks"):
        try: return {**await _foreign(p, "POST", "/warm", {"capability": capability, "model": model, "body": body}, timeout=900.0), "how": "the node's own warm"}
        except ProviderError as e:
            if "answered 404" not in str(e) or "nothing active here" in str(e): raise
    if capability != "chat" or not body.get("messages"): _fail(p, f"""it offers no warm (an older Magpie, or not Magpie): {capability} loads with its first request""")
    await run(p, "chat", {**body, "options": {**(body.get("options") or {}), "num_predict": 1}, "max_tokens": 1})
    return {"seconds": round(time.monotonic() - t0, 2), "read_ahead": True, "how": "a one-token chat (no warm offered)"}

async def run_detached(p: dict, capability: str, body: dict, rid: str) -> dict:
    """The same work, but it goes on at the provider whatever happens to the caller. rid names it: asking again under the same rid joins the work already there (running or finished) instead of starting it twice, which is how a run that was cut finds its answer.
    The answer is taken from the provider once read. A caller that is stopped on purpose (cancelled) stops the work too; one that simply dies does not.
    A provider behind a translator cannot keep work for a caller that is gone: there it is asked attached, and a cut means asking again."""
    if p.get("speaks"): return await run(p, capability, body, rid)
    try:
        async with _client(p, 90.0, 60.0) as c:
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
