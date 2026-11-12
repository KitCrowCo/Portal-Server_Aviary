"""test_node.py - Aviary against a real node (AVIARY_TEST_NODE=url, AVIARY_TEST_KEY, AVIARY_TEST_MODELS="a,b": two chat models it offers). Run from the portal root: python3 -m tools.aviary.test_node"""
import os, sys, json, asyncio, tempfile
from pathlib import Path
from .aviary import Aviary

URL, KEY, (M1, M2) = os.environ["AVIARY_TEST_NODE"], os.getenv("AVIARY_TEST_KEY", ""), os.environ["AVIARY_TEST_MODELS"].split(",")
SHIPPED = [Path(__file__).parent / "library"]
async def until(e, rid, states=("done", "failed", "held", "stopped", "unplaced"), t=120):
    for _ in range(int(t / 0.02)):
        if e.director.runs[rid].meta["state"] in states: return e.director.runs[rid].meta["state"]
        await asyncio.sleep(0.02)
    raise TimeoutError(e.view(rid))
def starts(e, rid): return [json.loads(l)["step"] for l in (e.director.runs[rid].dir / "journal.jsonl").read_text().splitlines() if json.loads(l)["what"] == "start"]

async def main():
    data = Path(tempfile.mkdtemp()); e = Aviary(data, SHIPPED)
    e.lib.save("provider", {"id": "n1", "name": "Test node", "url": URL, "key": KEY})
    hello = {"messages": [{"role": "user", "content": "hello"}]}
    # a role nobody bound, with more than one possible model: refused with the choices, nothing guessed
    r = await e.start("chat_turn", hello); assert await until(e, r) == "unplaced", e.view(r); print("unbound:", e.view(r)["error"][:150])
    # bound by a profile; then the model changed by editing the profile and nothing else
    e.lib.save("bindings", {"id": "base", "roles": {"main": {"asset": M1}, "subject": {"asset": M1}}})
    r = await e.start("chat_turn", hello, profiles=["base"]); assert await until(e, r) == "done"; a1 = e.director.runs[r].data["text"]
    e.lib.save("bindings", {"id": "base", "roles": {"main": {"asset": M2}, "subject": {"asset": M1}}})
    r = await e.start("chat_turn", hello, profiles=["base"]); assert await until(e, r) == "done"; a2 = e.director.runs[r].data["text"]
    print("binding edit:", repr(a1), "->", repr(a2), "| versions kept:", e.lib.versions("bindings", "base"), "| bound:", {k: e.director.runs[r].meta["bound"]["main"][k] for k in ("provider", "asset", "lane", "fits")})
    # a profile that inherits, and a pin for one run over both
    # candidates by case: a long prompt takes the second model, a short one the first; a profile that inherits changes one role only; a pin stands above both
    cases = dict(e.lib.get("pipeline", "chat_turn")); cases["roles"] = {"main": {"needs": ["chat"], "facts": {"chars": "len(str(messages))"}}}
    e.lib.save("bindings", {"id": "mine", "inherits": "base", "roles": {"main": [{"asset": M2, "when": "chars > 200", "note": "long prompts"}, {"asset": "not-there", "when": "images"}, {"asset": M1}]}})
    for text, want in (("hello", M1), ("hello " * 60, M2)):
        r = await e.start(cases, {"messages": [{"role": "user", "content": text}]}, profiles=["mine"]); assert await until(e, r) == "done"; b = e.director.runs[r].meta["bound"]["main"]; assert b["asset"] == want and not b["pinned"], b
    print("candidates by case: short ->", M1, "| long ->", M2, "| passed over for the short one:", e.director.runs[r].meta["bound"]["main"]["passed"] or "none", "/", [x["why"] for x in e.director.runs[await e.start(cases, hello, profiles=["mine"])].meta["bound"]["main"]["passed"]])
    r = await e.start(cases, {"messages": [{"role": "user", "content": "hello " * 60}]}, profiles=["mine"], pins={"main": {"asset": M1}}); await until(e, r); b = e.director.runs[r].meta["bound"]["main"]; assert b["asset"] == M1 and b["pinned"], b
    r = await e.start(cases, hello, profiles=["mine"], pins={"main": {"asset": "not-there"}}); assert await until(e, r) == "unplaced" and "pinned" in e.view(r)["error"], e.view(r)
    # the measurement set, from its file; everything stops in the middle; another Aviary opens the same data and goes on
    m = await e.start("measure_text", {"lengths": [128, 2048], "note": "test"}, profiles=["base"])
    while len(e.director.runs[m].view()["done"]) < 6: await asyncio.sleep(0.02)
    c = await e.start("chat_turn", hello, profiles=["base"], pins={"main": {"asset": M1}}, priority=5)   # more important, same model: it gets the next turn on the lane
    assert await until(e, c) == "done"; mid = len(e.director.runs[m].view()["done"]); assert mid < 20, mid
    [t.cancel() for t in e.director.runs[m].busy.values()]; before = starts(e, m); e.director.runs.clear()   # the old Aviary is gone: nothing it held in memory acts again
    e = Aviary(data, SHIPPED); await e.open(); assert e.director.runs[m].meta["state"] == "interrupted", e.view(m)
    left = sum(map(len, e.view(m)["course"])); await e.go(m); assert await until(e, m) == "done", e.view(m)
    after, d = starts(e, m)[len(before):], e.director.runs[m].data
    print(f"measurement: {len(e.director.runs[m].plan['steps'])} steps; stopped with {len(before)} started, {left} left; after pickup ran {len(after)}; done again: {sorted(set(after) & set(before))}")
    print("chat turn got its turn in the middle:", mid, "steps of the measurement were done when it finished")
    print("set hash:", d["set_hash"][:16], "| speed.0:", {k: d["speed.0"].get(k) for k in ("prompt_n", "ttft_ms", "tps", "gen_n")}, "| card:", sorted(d["card"]))
    assert len(set(after) & set(before)) <= 2 and all(k in d for k in ("speed.3", "head.3", "hashes")) and len(d["hashes"]) == 12
DETACHED = {"name": "long answer", "roles": {"main": {"needs": ["chat"]}}, "params": {"messages": {}}, "steps": [{"id": "answer", "type": "capability", "keep": True, "needs": ["messages"], "gives": ["text", "job"], "config": {"role": "main", "capability": "chat", "detached": True, "body": {"messages": "{messages}"}, "give": {"text": "text", "job": "job"}}}]}

async def cut(data: str):
    """Started as its own process by detached(): begins the run, then dies without a word, as in a power cut."""
    e = Aviary(Path(data), SHIPPED)
    await e.start(DETACHED, {"messages": [{"role": "user", "content": "long"}]}, pins={"main": {"asset": M1}})
    await asyncio.sleep(0.15)
    os._exit(1)

async def detached():
    data = Path(tempfile.mkdtemp()); e = Aviary(data, SHIPPED)
    e.lib.save("provider", {"id": "n1", "name": "Test node", "url": URL, "key": KEY})
    await (await asyncio.create_subprocess_exec(sys.executable, "-m", "tools.aviary.test_node", "cut", str(data))).wait()
    e = Aviary(data, SHIPPED); await e.open(); (r,) = e.director.runs; assert e.director.runs[r].meta["state"] == "interrupted"
    await e.go(r); assert await until(e, r) == "done", e.view(r)
    d = e.director.runs[r].data; print("detached: Aviary died after asking; the next one joined the same work:", repr(d["text"]), "| job", d["job"], "| asked once more:", starts(e, r))
    assert d["job"] == f"{r}.answer"

if sys.argv[1:2] == ["cut"]: asyncio.run(cut(sys.argv[2]))
else:
    asyncio.run(main())
    asyncio.run(detached())
