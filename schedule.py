"""schedule.py - saved set-ups started by the clock.
A schedule is a library object (kind "schedule"): {"name", "setup": a saved set-up's id (which names the pipeline, its inputs and who fills its roles), "every_minutes": a number, and/or "at": ["HH:MM", ...] local time, "days": [0-6, Monday first] (none: every day), "on": true, "owner", "may": what the owner's policy allowed the person who saved it}.
What it did last is kept beside the library, not in it (<data>/schedule_state.json), so firing never makes a new version of the object.
Rules: a run it starts belongs to the schedule's owner, with the limits that person had when saving it. While the run it started last is still going, it does not start another. A time that passed while the portal was off is made up once, the same day; an interval starts counting when the schedule is saved."""
import json, time, asyncio
from pathlib import Path

TICK = 30   # seconds between looks at the clock
LIVE = ("queued", "running", "waiting", "stopping")

class Schedule:
    def __init__(self, aviary, data: Path):
        self.av, self.file = aviary, Path(data) / "schedule_state.json"

    def state(self) -> dict: return json.loads(self.file.read_text()) if self.file.exists() else {}
    def _note(self, sid: str, **kw): self.file.write_text(json.dumps({**(st := self.state()), sid: {**st.get(sid, {}), **kw}}, indent=1))

    def due(self, s: dict, now: float, last: float) -> bool:
        """Whether it is time: the interval has passed since it last started (or was saved), or one of today's times lies between the last start and now, on a day it runs."""
        if not s.get("on") or not s.get("setup"): return False
        lt = time.localtime(now)
        if s.get("days") and lt.tm_wday not in [int(d) for d in s["days"]]: return False
        if s.get("every_minutes") and now - last >= float(s["every_minutes"]) * 60: return True
        midnight = time.mktime((lt.tm_year, lt.tm_mon, lt.tm_mday, 0, 0, 0, 0, 0, -1))
        return any(last < midnight + int(h) * 3600 + int(m) * 60 <= now for h, m in (str(t).split(":")[:2] for t in s.get("at") or []))

    def status(self, sid: str) -> str:
        """One line on what a schedule did last, for whoever lists it."""
        st = self.state().get(sid, {})
        run = self.av.director.runs.get(st.get("run", ""))
        return " | ".join(x for x in (f"""last started {time.strftime("%Y-%m-%d %H:%M", time.localtime(st["last"]))}""" if st.get("last") else "not started yet", f"""{st["run"]}: {run.meta["state"]}""" if run else "", st.get("error", "")) if x)

    async def look(self, now: float = None) -> list:
        """Starts what is due. Answers the ids started."""
        now, st, started = now or time.time(), self.state(), []
        for s in self.av.lib.list("schedule"):
            mine = st.get(s["id"], {})
            if not self.due(s, now, max(mine.get("last", 0), s.get("modified", 0))): continue
            if (run := self.av.director.runs.get(mine.get("run", ""))) and run.meta["state"] in LIVE: continue   # the one it started last is still going
            try:
                setup = self.av.lib.get("setup", s["setup"])
                rid = await self.av.start(setup["pipeline"], setup.get("inputs") or {}, owner=s.get("owner", ""), profiles=s.get("profiles", []), pins=setup.get("pins") or {}, may=s.get("may", []))
                self._note(s["id"], last=now, run=rid, error="")
                started.append(s["id"])
            except Exception as e: self._note(s["id"], last=now, error=f"not started: {type(e).__name__}: {e}")   # a schedule that cannot start says so where it is listed and is tried again at its next time; the others go on
        return started

    async def loop(self):
        while True:
            await asyncio.sleep(TICK)
            await self.look()
