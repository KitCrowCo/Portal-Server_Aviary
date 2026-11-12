"""test_core.py - the plan / director core against stand-in step types. Run: python3 test_core.py"""
import asyncio, tempfile, time
from pathlib import Path
from .director import Director

LOG = []
async def work(config, inputs, ctx):
    LOG.append(("start", ctx.run.id, ctx.step["id"], time.time()))
    await asyncio.sleep(config.get("s", 0.05))
    if config.get("fail"): raise RuntimeError("as told")
    LOG.append(("end", ctx.run.id, ctx.step["id"], time.time()))
    return {k: f"""{ctx.step["id"]}({",".join(sorted(inputs))})""" for k in config.get("give", ctx.step.get("gives", []))}
TYPES = {"work": work}
def step(i, needs=(), gives=(), **kw): return {"id": i, "type": "work", "needs": list(needs), "gives": list(gives), **{k: v for k, v in kw.items() if k in ("keep", "once")}, "config": {k: v for k, v in kw.items() if k not in ("keep", "once")}}
def ran(rid=None): return [s for w, r, s, _ in LOG if w == "start" and (rid is None or r == rid)]
async def until(d, rid, states=("done", "failed", "held", "stopped"), t=5):
    for _ in range(int(t / 0.01)):
        if d.runs[rid].meta["state"] in states: return d.runs[rid].meta["state"]
        await asyncio.sleep(0.01)
    raise TimeoutError(d.runs[rid].view())
CHAIN = {"name": "chain", "params": {"q": {}}, "steps": [step("a", ["q"], ["x"]), step("b", ["x"], ["y"], keep=True), step("c", ["y"], ["z"]), step("d", ["x", "z"], ["out"], s=0.3)]}

async def main():
    root = Path(tempfile.mkdtemp())
    # 1. a plan runs to its end; the same plan as a graph where it allows
    d = Director(root / "1", TYPES); r = d.create(CHAIN, {"q": 1}); d.go(r)
    assert await until(d, r) == "done" and ran(r) == ["a", "b", "c", "d"], ran(r)
    wide = {"steps": [step("p", [], ["x"], s=0.2), step("q", [], ["y"], s=0.2), step("j", ["x", "y"], ["z"])]}
    LOG.clear(); r = d.create(wide); t0 = time.time(); d.go(r); await until(d, r); together = time.time() - t0
    # 2. the same plan as a list: both first steps occupy one lane. The plan is unchanged; only the means differ
    d2 = Director(root / "2", TYPES, lane=lambda run, s: "one model" if s["id"] in ("p", "q") else None)
    LOG.clear(); r = d2.create(wide); t0 = time.time(); d2.go(r); await until(d2, r); in_turn = time.time() - t0
    assert together < 0.35 < in_turn, (together, in_turn)
    print(f"1-2 ok: graph {together:.2f}s, list {in_turn:.2f}s, same plan")
    # 3. everything stops while d runs (the director is simply dropped: nothing in memory survives). b was kept, the rest was not
    d3 = Director(root / "3", TYPES); r = d3.create(CHAIN, {"q": 1}); d3.go(r)
    while "d" not in d3.runs[r].busy: await asyncio.sleep(0.01)
    [t.cancel() for t in d3.runs[r].busy.values()]; d3.runs.clear()
    LOG.clear(); d3 = Director(root / "3", TYPES); assert d3.open() == [] and d3.runs[r].meta["state"] == "interrupted"
    assert Director(root / "3", TYPES, on_restart="continue").open() == []   # it is no longer under way; and below, the director's own setting applies when the plan is silent
    assert d3.runs[r].view()["course"] == [["a", "c"], ["d"]], d3.runs[r].view()
    d3.go(r); assert await until(d3, r) == "done" and sorted(ran(r)) == ["a", "c", "d"], ran(r)
    print("3 ok: picked up from the kept key; b not done again, a done again because d needs x")
    # 4. a plan that keeps nothing starts over; one that keeps everything and says continue goes on by itself from the step that was cut
    for keep, cont, expect in ((False, False, ["a", "b", "c", "d"]), (True, True, ["d"])):
        pl = {**CHAIN, "keep": keep, "steps": [{k: v for k, v in s.items() if k != "keep"} for s in CHAIN["steps"]]}
        d4 = Director(root / f"4{keep}", TYPES); r = d4.create(pl, {"q": 1}); d4.go(r)
        while "d" not in d4.runs[r].busy: await asyncio.sleep(0.01)
        [t.cancel() for t in d4.runs[r].busy.values()]; d4.runs.clear()
        LOG.clear(); d4 = Director(root / f"4{keep}", TYPES, on_restart="continue" if cont else "ask"); on = d4.open()
        d4.go(r)
        assert (on == [r]) == cont and await until(d4, r) == "done" and ran(r) == expect, (on, ran(r))
    print("4 ok: nothing kept -> from the start; all kept + continue -> only the cut step, without being told")
    # 5. a step that must not happen twice: started, end not kept -> held for a decision
    once = {"steps": [step("pay", [], ["receipt"], once=True, s=0.3), step("after", ["receipt"], [])]}
    d5 = Director(root / "5", TYPES); r = d5.create(once); d5.go(r); await asyncio.sleep(0.05)
    [t.cancel() for t in d5.runs[r].busy.values()]; d5.runs.clear()
    LOG.clear(); d5 = Director(root / "5", TYPES); d5.open(); d5.go(r)
    assert await until(d5, r) == "held" and ran(r) == [], (d5.runs[r].meta, ran(r))
    d5.decide(r, "pay", values={"receipt": "it did go through"}); d5.go(r)
    assert await until(d5, r) == "done" and ran(r) == ["after"] and d5.runs[r].data["receipt"] == "it did go through"
    print("5 ok: once-step held, decided as happened, the rest went on")
    # 6. two runs under one director sharing a lane: the more important one goes first; a failure stops its run only
    d6 = Director(root / "6", TYPES, lane=lambda run, s: "one model")
    LOG.clear(); low = d6.create({"steps": [step("l1", [], ["x"]), step("l2", ["x"], [])]}, priority=0); high = d6.create({"steps": [step("h1", [], ["x"]), step("h2", ["x"], [], fail=True)]}, priority=5)
    d6.runs[low].state("running"); d6.go(high)
    assert await until(d6, high) == "failed" and await until(d6, low) == "done" and ran() == ["h1", "h2", "l1", "l2"], ran()
    print("6 ok: two plans interleaved by priority on one lane; one failing leaves the other alone:", d6.runs[high].meta["error"])
    # 7. a branch: the step gives one of two keys; what waits on the other is unreached, and the run is done
    br = {"steps": [step("choose", [], ["yes", "no"], give=["yes"]), step("y", ["yes"], []), step("n", ["no"], [])]}
    d7 = Director(root / "7", TYPES); LOG.clear(); r = d7.create(br); d7.go(r)
    assert await until(d7, r) == "done" and ran(r) == ["choose", "y"] and d7.runs[r].meta["unreached"] == ["n"], d7.runs[r].meta
    # 8. stop, then go
    d8 = Director(root / "8", TYPES); LOG.clear(); r = d8.create({**CHAIN, "keep": True}, {"q": 1}); d8.go(r); await asyncio.sleep(0.08); d8.stop(r)
    assert await until(d8, r) == "stopped"; n = len(ran(r)); d8.go(r)
    assert await until(d8, r) == "done" and len(ran(r)) == 4 and n < 4, ran(r)
    try: d8.create({"steps": [step("a", ["missing"], [])]})
    except ValueError as e: print("7-8 ok: branch, stop/go; a bad plan is refused:", e)
async def more():
    """The step types that need no provider, through Aviary itself: a branch by expression, regex, chunking, and a pipeline as a step - per item, in turn with a carried value, and picked up after a stop."""
    from .aviary import Aviary
    e = Aviary(Path(tempfile.mkdtemp()), [Path(__file__).parent / "library"]); e.types["work"] = work
    async def run(plan, inputs=None):
        r = await e.start(plan, inputs); assert await until(e.director, r) == "done", e.view(r); return e.director.runs[r]
    r = await run({"params": {"n": {}}, "steps": [{"id": "size", "type": "expr", "needs": ["n"], "gives": ["small", "large"], "config": {"expr": "'s' if n < 10 else 'l'", "routes": {"s": "small", "l": "large"}}}, step("on_small", ["small"], ["a"]), step("on_large", ["large"], ["b"])]}, {"n": 3})
    assert "a" in r.data and "b" not in r.data and r.meta["unreached"] == ["on_large"], r.meta
    r = await run({"params": {"t": {}}, "steps": [{"id": "find", "type": "regex", "needs": ["t"], "gives": ["m"], "config": {"of": "t", "pattern": r"(\w+)@(\w+)", "give": "m"}}, {"id": "swap", "type": "regex", "needs": ["t"], "gives": ["out"], "config": {"of": "t", "pattern": r"(\w+)@(\w+)", "replace": "{1} at {0}", "give": "out"}}, {"id": "cut", "type": "chunk", "needs": ["t"], "gives": ["parts"], "config": {"of": "t", "chars": 12, "give": "parts"}}]}, {"t": "a@b and c@d.\n\nSecond paragraph here."})
    assert [m["groups"] for m in r.data["m"]] == [["a", "b"], ["c", "d"]] and r.data["out"].startswith("b at a and d at c") and len(r.data["parts"]) >= 2, r.data
    child = {"params": {"item": {}, "sum": {"default": ""}}, "keep": True, "steps": [{"id": "add", "type": "expr", "needs": ["item", "sum"], "gives": ["sum"], "config": {"expr": "sum + str(item)", "give": "total"}}]}
    child["steps"][0].update(gives=["total"])
    outer = {"params": {"items": {}}, "steps": [{"id": "fold", "type": "pipeline", "needs": ["items"], "gives": ["total"], "config": {"plan": child, "each": "items", "carry": ["total"], "inputs": {"item": "{item}", "sum": "{total}"}}}, {"id": "all", "type": "pipeline", "needs": ["items"], "gives": ["each_total"], "config": {"plan": child, "each": "items", "inputs": {"item": "{item}"}, "give": {"each_total": "total"}}}]}
    r = await run(outer, {"items": ["a", "b", "c"]})
    assert r.data["total"] == "abc" and r.data["each_total"] == ["a", "b", "c"] and f"{r.id}~fold~2" in e.director.runs and e.director.runs[f"{r.id}~fold~2"].meta["parent"] == r.id, r.data
    r = await run({"params": {"note": {"default": "a"}}, "steps": [{"id": "renew", "type": "expr", "needs": ["note"], "gives": ["note"], "config": {"expr": "note + '!'", "give": "note"}}]})
    assert r.data["note"] == "a!", r.data   # a step that renews its own input runs, once
    code = {"params": {"x": {}}, "steps": [{"id": "py", "type": "python", "needs": ["x"], "gives": ["double", "kept"], "config": {"script": "import sys, json\nd = json.load(sys.stdin)\nprint('noise')\nprint(json.dumps({'double': d['x'] * 2, 'kept': '{x}'}))"}}]}
    r = e.director.runs[await e.start(code, {"x": 4})]; assert await until(e.director, r.id) == "failed" and "may not run the Python code written into" in r.meta["error"], r.meta   # not allowed unless whoever starts it was
    r = e.director.runs[await e.start(code, {"x": 4}, may=["python"])]; assert await until(e.director, r.id) == "failed", r.meta   # code of one's own choosing is another decision than code written into the pipeline
    r = e.director.runs[await e.start(code, {"x": 4}, may=["python_fixed"])]; assert await until(e.director, r.id) == "done" and r.data["double"] == 8 and r.data["kept"] == "{x}", r.data   # and its own braces are left alone
    assert e.allows("python", "ann", True) and not e.allows("python", "ann", False)
    slow = {"params": {"item": {}}, "keep": True, "steps": [step("w", ["item"], ["out"], s=0.25)]}
    rid = await e.start({"params": {"items": {}}, "steps": [{"id": "each", "type": "pipeline", "needs": ["items"], "gives": ["outs", "last"], "config": {"plan": slow, "each": "items", "carry": ["last"], "inputs": {"item": "{item}"}, "give": {"outs": "out"}}}]}, {"items": [1, 2, 3]})
    await asyncio.sleep(0.35); e.director.stop(rid, now=True); assert await until(e.director, rid) == "stopped"
    LOG.clear(); await e.go(rid); assert await until(e.director, rid) == "done" and len(e.director.runs[rid].data["outs"]) == 3
    print("9 ok: branch by expression, regex, chunks, Python only where allowed, a pipeline per item / in turn with a carried value; after a stop the finished items were not run again:", sorted({r_ for w, r_, s_, _ in LOG if w == "start"}))
    assert not any(r_.endswith("~0") for w, r_, s_, _ in LOG if w == "start")

asyncio.run(main())
asyncio.run(more())
