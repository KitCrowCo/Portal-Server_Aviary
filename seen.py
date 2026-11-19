"""seen.py - what each machine last said, kept: the answer to every question asked of a machine for showing (what it offers, its programs, what is running, its code, a program's own status), with when it last answered and when it was last asked, kept on disk across restarts.
A view draws at once from what is known - however old, and saying how old - and never waits on a machine. Each entry also keeps how many checks in a row were missed and since when (health): answering, answered before but missed its last check (shown with caution), and never reached are different states.
An entry older than the freshness its view asks for is asked again in the background, each (machine, question) on its own, so a machine that is slow or away holds up only its own entries: they keep their last answer, with its age and why the last ask failed.
Work and changes (a run, an install, a stop) never go through here: they ask the machine itself, and answer from it."""
import json, time, asyncio
from pathlib import Path
from .providers import TRY_ANYWAY

SAVE_S = 2.0   # answers are written to disk at most this often

class Seen:
    def __init__(self, path: Path):
        self.path, self.tasks, self.saved, self.later = path, {}, 0.0, None
        self.rows = json.loads(path.read_text()) if path.exists() else {}   # key -> {"value", "t" (answered), "asked", "error", "away"}

    def get(self, key: str, ask, fresh_s: float = 15.0) -> dict:
        """What is known under key now: {"value" (the last answer, None if it never answered), "t" (when it answered, 0 never), "asked", "error" (why the last ask failed, "" when it answered), "away" (the last ask did not reach it), "asking"}. ask: a function giving the coroutine that asks again; it is started in the background when the entry is older than fresh_s and not already being asked."""
        r = self.rows.get(key) or {"value": None, "t": 0.0, "asked": 0.0, "error": "", "away": False}
        if time.time() - r["asked"] > fresh_s and not self.asking(key): self.tasks[key] = asyncio.ensure_future(self._ask(key, ask))
        return {**r, "asking": self.asking(key)}

    def asking(self, key: str) -> bool: return key in self.tasks and not self.tasks[key].done()

    async def fresh(self, key: str, ask, fresh_s: float = 15.0, wait_s: float = 0.0) -> dict:
        """get, then waiting up to wait_s for an ask under way - for a caller that would rather have the answer now if it comes quickly (never longer than wait_s)."""
        r = self.get(key, ask, fresh_s)
        if r["asking"] and wait_s: await asyncio.wait([self.tasks[key]], timeout=wait_s)
        return self.get(key, ask, fresh_s) if wait_s else r

    async def _ask(self, key: str, ask):
        r = self.rows.setdefault(key, {"value": None, "t": 0.0, "asked": 0.0, "error": "", "away": False})
        r["asked"] = time.time()
        TRY_ANYWAY.set(True)   # in this background task only: a machine marked down is tried too, so one back up is seen
        try: r.update(value=await ask(), t=time.time(), error="", away=False, misses=0, missed_since=0.0)
        except Exception as e: r.update(error=str(e)[:2000], away=type(e).__name__ == "Unreachable", misses=r.get("misses", 0) + 1, missed_since=r.get("missed_since") or time.time())   # the last answer stays, marked by its age and why it is not newer
        self._save()

    def _save(self, later: bool = False):
        """At most once every SAVE_S; one held back is written when that time has passed."""
        if later: self.later = None
        elif time.time() - self.saved < SAVE_S:
            if not self.later: self.later = asyncio.get_running_loop().call_later(SAVE_S, self._save, True)
            return
        self.saved = time.time()
        self.path.with_suffix(".tmp").write_text(json.dumps(self.rows))
        self.path.with_suffix(".tmp").replace(self.path)

    def forget(self, prefix: str = ""):
        """Asks again on the next look: every entry, or those under prefix (one machine: "<id>:")."""
        for k in [k for k in self.rows if k.startswith(prefix)]: self.rows[k]["asked"] = 0.0

    def drop(self, prefix: str):
        """A machine taken out of the library: what it said goes with it."""
        for k in [k for k in self.rows if k.startswith(prefix)]: del self.rows[k]
        self.saved = 0.0
        self._save()

def age(r: dict) -> str:
    """How old an entry's answer is, in words: "now", "40 s ago", "3 min ago", "2 h ago", "never answered"."""
    if not r.get("t"): return "never answered"
    s = time.time() - r["t"]
    return "now" if s < 10 else f"{s:.0f} s ago" if s < 120 else f"{s / 60:.0f} min ago" if s < 7200 else f"{s / 3600:.0f} h ago" if s < 172800 else f"{s / 86400:.0f} days ago"

def health(r: dict) -> tuple:
    """(state, words) of a kept entry - four different things a single "reached: yes/no" would blur:
    "ok": it answered its last check; "missed": it answered before, but not its last check(s) - it may be down, and what is kept is as it was then, which may or may not still hold; "never": it has not answered yet; "asking": its first check is under way.
    An answer that is a refusal of the question itself ("answered 404": an older program without it) is still an answer: the machine is there."""
    if r.get("error") and "answered " in r["error"]: return "ok", "answering (it does not have this list: an older program, or not a node)"
    if not r.get("error"): return ("ok", f"answering ({age(r)})") if r.get("t") else ("asking", "being asked ...")
    n, since = r.get("misses") or 1, age({"t": r.get("missed_since")}) if r.get("missed_since") else ""
    if not r.get("t"): return "never", f"""never reached ({n} check{"s" if n > 1 else ""} missed)"""
    return "missed", f"""missed its last {n} check{"s" if n > 1 else ""}{", the first " + since if since else ""}; shown as it was {age(r)}"""
