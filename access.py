"""access.py - named sets of what the providers hold (pools), and who may use what.
A pool (library kind "pool"): {"name", "about", "members": ["<provider id or *>|<asset name or glob>", ...], "users": [portal user names], "prefer": "running" | "listed"}.
As a set: a role's candidate may name a pool instead of one asset ({"pool": id}). The role is then filled by a member that offers what the role needs and fits now, so one role draws on several machines and models without naming any (roles.py).
As a limit: a pool that lists users restricts its members. An asset held by at least one such pool is used only by the people one of those pools lists, and by work that runs as them. An asset no restricting pool holds is open to everyone, as it was before any pool existed.
Nobody stands above a limit, an admin included: an admin changes a pool (a library change, kept with its earlier versions, so it shows) and then uses its members like anyone it lists. Work with no owner uses only what is open to everyone."""
import fnmatch

def holds(pool: dict, pid: str, asset: str) -> bool: return any(m.split("|", 1)[0] in ("*", pid) and fnmatch.fnmatchcase(asset, m.split("|", 1)[-1] or "*") for m in pool.get("members") or [])
def rank(pool: dict, pid: str, asset: str) -> int: return next(i for i, m in enumerate(pool["members"]) if m.split("|", 1)[0] in ("*", pid) and fnmatch.fnmatchcase(asset, m.split("|", 1)[-1] or "*"))   # where in the pool's own order a member stands
def limits(lib, pid: str, asset: str) -> list: return [p for p in lib.list("pool") if p.get("users") and holds(p, pid, asset)]   # the pools that restrict this asset

def allowed(lib, user: str, pid: str, asset: str) -> bool:
    """Whether work running as user may use this asset."""
    over = limits(lib, pid, asset)
    return not over or any(user in p["users"] for p in over)

def why_not(lib, user: str, pid: str, asset: str) -> str: return f"""{asset} is kept for the people of {", ".join(p.get("name") or p["id"] for p in limits(lib, pid, asset))}; {user or "work without an owner"} is not among them"""
