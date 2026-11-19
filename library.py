"""library.py - every collection Aviary keeps is the same thing: JSON objects of a kind (pipeline, bindings, provider, prompt, ...), each one file named by its id.
Two places per kind: shipped (comes with a tool or module, never written here) and mine (the data folder). An object in mine with a shipped id replaces it; removing mine brings the shipped one back.
Saving keeps the version it replaces under history/, numbered. Nothing here knows what any kind means."""
import json, time, shutil
from pathlib import Path

class Library:
    def __init__(self, mine: Path, shipped: list = ()):
        self.mine, self.shipped = Path(mine), [Path(p) for p in shipped]   # shipped: folders that each hold <kind>/<id>.json; a later folder wins over an earlier one

    def retired(self, kind: str) -> set: return {i for root in self.shipped if (root / "retired.json").exists() for i in json.loads((root / "retired.json").read_text()).get(kind, [])}   # a shipped folder's retired.json {kind: [ids]}: shipped objects it no longer has, ignored if an older copy of the folder still holds them
    def _files(self, kind: str) -> dict:
        old = self.retired(kind)
        return {f.stem: f for root in (*self.shipped, self.mine) for f in sorted((root / kind).glob("*.json")) if root == self.mine or f.stem not in old}
    def stale(self) -> list: return [f for root in self.shipped for k in self.kinds() for f in (root / k).glob("*.json") if f.stem in self.retired(k)]   # retired files still on disk, to be deleted by hand
    def kinds(self) -> list: return sorted({d.name for root in (*self.shipped, self.mine) if root.exists() for d in root.iterdir() if d.is_dir() and d.name != "history"})
    def origin(self, kind: str, oid: str) -> str: return "mine" if (self.mine / kind / f"{oid}.json").exists() else "shipped"
    def has(self, kind: str, oid: str) -> bool: return oid in self._files(kind)
    def get(self, kind: str, oid: str) -> dict: return {"id": oid, **json.loads(self._files(kind)[oid].read_text())}
    def list(self, kind: str, tag: str = "") -> list: return [o for o in (self.get(kind, i) for i in self._files(kind)) if not tag or tag in o.get("tags", [])]

    def versions(self, kind: str, oid: str) -> list: return sorted(int(f.stem) for f in (self.mine / "history" / kind / oid).glob("*.json"))
    def version(self, kind: str, oid: str, n: int) -> dict: return json.loads((self.mine / "history" / kind / oid / f"{n}.json").read_text())

    def save(self, kind: str, obj: dict, by: str = "") -> dict:
        """Writes the object to mine. What it replaces there is kept as the next numbered version first."""
        f, hist = self.mine / kind / f"""{obj["id"]}.json""", self.mine / "history" / kind / obj["id"]
        f.parent.mkdir(parents=True, exist_ok=True)
        if f.exists():
            hist.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, hist / f"""{max(self.versions(kind, obj["id"]), default=0) + 1}.json""")
        f.write_text(json.dumps({**obj, "modified": time.time(), "by": by}, indent=1, ensure_ascii=False))
        return self.get(kind, obj["id"])

    def remove(self, kind: str, oid: str):
        """Removes mine (kept as a version); a shipped object of that id is in use again."""
        f, hist = self.mine / kind / f"{oid}.json", self.mine / "history" / kind / oid
        hist.mkdir(parents=True, exist_ok=True)
        f.replace(hist / f"{max(self.versions(kind, oid), default=0) + 1}.json")
