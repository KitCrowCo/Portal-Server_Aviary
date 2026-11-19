"""text.py - cutting text into pieces a model can take, with no model involved. Taken as it was from AI Manager, where it was worked out against real notes and source files.
paragraphs(): paragraph boundaries first, then sentences, and a marked hard cut only when a run-on leaves no other way. code(): whole definitions packed in order, never across a section header, each piece headed with file, section, names defined and lines."""
import re, ast

_SENTENCE_SPLIT_RE = re.compile(r'(?<=[.!?])\s+')
_FORCED_CUT_MARK = "[...chunk split - no sentence boundary found, cut forced here...]"

def _pack(units: list, target: int, overlap: int = 0) -> list:
    chunks, cur = [], ""
    for u in units:
        if cur and len(cur) + len(u) + 1 > target:
            chunks.append(cur)
            cur = (cur[-overlap:] + " " + u) if overlap else u
        else:
            cur = f"{cur} {u}" if cur else u
    if cur: chunks.append(cur)
    return chunks

def _hard_split(text: str, target: int) -> list:
    """Last-resort character split for a run-on with no usable punctuation at all - marks every forced cut so a downstream reader (human or LLM) can see a guess was forced, rather than trusting what looks like a clean break but isn't one."""
    out = []
    for i in range(0, len(text), target):
        piece = text[i:i+target]
        out.append(piece + (f" {_FORCED_CUT_MARK}" if i + target < len(text) else ""))
    return out

def _chunk_paragraphs(text: str, target_chars: int, overlap_chars: int = 0, hard_split_ratio: float = 2.0) -> list:
    """Tries paragraph boundaries first, falls back to sentence boundaries for an oversized paragraph, and only hard-splits (marked, not silent) when a single run-on still exceeds target_chars * hard_split_ratio with no punctuation to split on at all.
    This is deliberately NOT a rule that guarantees clean breaks - raw notes have too many edge cases for that to be worth hand-coding; the goal is graceful degradation with a visible flag, not silent failure."""
    segments = []
    for p in (p for p in text.split("\n\n") if p.strip()):
        if len(p) <= target_chars:
            segments.append(p); continue
        for seg in _pack(_SENTENCE_SPLIT_RE.split(p), target_chars):
            segments.extend(_hard_split(seg, target_chars) if len(seg) > target_chars * hard_split_ratio else [seg])
    return _pack(segments, target_chars, overlap_chars)

_SECTION_RE = re.compile(r"^\s*(?:#|//)\s*-{3}\s*(.+?)\s*-{3}\s*$")   # `# --- Title ---` / `// --- Title ---` section headers
_CODE_START_RE = re.compile(r"""^(?:export\s+)?(?:async\s+)?(?:def|class|function)\b|^(?:export\s+)?(?:const|let|var)\s+[\w$]+\s*=\s*(?:async\s*)?(?:function\b|\([^)]*\)\s*=>|[\w$]+\s*=>)|^[A-Z_][A-Z0-9_]*\s*=|^#{1,6}\s""")
_SYMBOL_RE = re.compile(r"""^(?:export\s+)?(?:async\s+)?(?:def|class|function)\s+([\w$]+)|^(?:export\s+)?(?:const|let|var)\s+([\w$]+)|^([A-Z_][A-Z0-9_]*)\s*=|^#{1,6}\s+(.+)""")
_INNER_BREAK_RE = re.compile(r"^\s+(?:async\s+)?(?:def|function)\s|^\s*$")   # where an oversized definition may be cut: an inner def or a blank line

def _py_spans(lines: list):
    """Top-level statements of a Python file as (first, last) 1-based line spans covering every line: the blank lines, comments and section headers before a statement belong to it. None when the file does not parse (the generic splitter is used instead)."""
    try: tree = ast.parse("\n".join(lines))
    except SyntaxError: return None
    spans, prev = [], 0
    for node in tree.body: spans.append((prev + 1, node.end_lineno)); prev = node.end_lineno
    if prev < len(lines): spans = spans[:-1] + [(spans[-1][0], len(lines))] if spans else [(1, len(lines))]
    return spans

def _generic_spans(lines: list) -> list:
    """Spans between lines that start a definition, a constant, a markdown heading or a section header, for files without a parser here."""
    starts = sorted({0} | {i for i, l in enumerate(lines) if _CODE_START_RE.match(l) or _SECTION_RE.match(l)})
    return [(a + 1, b) for a, b in zip(starts, starts[1:] + [len(lines)]) if b > a]

def _split_span(lines: list, a: int, b: int, target: int) -> list:
    """An oversized span cut into pieces of at most target chars, each ending before the last inner def or blank line that fits (at a plain line only when none does). A single line longer than target is a piece by itself."""
    out, start, size, brk = [], a, 0, None
    for i in range(a, b + 1):
        n = len(lines[i - 1]) + 1
        if i > start and size + n > target:
            cut = brk or i   # the piece ends just before line cut
            out.append((start, cut - 1)); start, brk = cut, None; size = sum(len(lines[k - 1]) + 1 for k in range(start, i))
        if i > start and _INNER_BREAK_RE.match(lines[i - 1]): brk = i
        size += n
    return out + [(start, b)]

def _code_chunk(lines: list, path: str, a: int, b: int, section: str, part: str = "") -> dict:
    symbols = [next(g for g in m.groups() if g) for m in (_SYMBOL_RE.match(l) for l in lines[a - 1:b] if not _SECTION_RE.match(l)) if m and not (m.group(4) and path.endswith(".py"))]   # in Python a leading # is a comment, not a markdown heading
    return {"text": f"""File: {path or "(unnamed)"} | Section: {section or "-"} | Defines: {", ".join(symbols[:12]) or "-"} | Lines {a}-{b}{part}\n{chr(10).join(lines[a - 1:b])}""", "path": path, "section": section, "symbols": symbols, "start": a, "end": b}

def _chunk_code(text: str, path: str, target: int, ratio: float = 2.0) -> list:
    """Definition-aware chunks of a source file (Python by syntax tree; other files by lines that start a definition, constant or markdown heading).
    Whole definitions are packed in order up to target chars and never across a section header; one definition over ratio x target is split inside itself, and those pieces say so in their header."""
    lines = text.splitlines()
    if not lines: return []
    spans = (_py_spans(lines) if path.endswith(".py") else None) or _generic_spans(lines)
    sections, cur_sec = [], ""
    for l in lines: cur_sec = (_SECTION_RE.match(l).group(1) if _SECTION_RE.match(l) else cur_sec); sections.append(cur_sec)
    size = lambda a, b: sum(len(lines[i]) + 1 for i in range(a - 1, b))
    chunks, cur = [], []
    for a, b in spans:
        if size(a, b) > target * ratio:
            if cur: chunks.append(_code_chunk(lines, path, cur[0][0], cur[-1][1], sections[cur[-1][1] - 1])); cur = []
            pieces = _split_span(lines, a, b, target)
            chunks += [_code_chunk(lines, path, x, y, sections[y - 1], f" | Part {k + 1}/{len(pieces)} of one definition") for k, (x, y) in enumerate(pieces)]
            continue
        if cur and (sections[b - 1] != sections[cur[-1][1] - 1] or size(cur[0][0], b) > target): chunks.append(_code_chunk(lines, path, cur[0][0], cur[-1][1], sections[cur[-1][1] - 1])); cur = []
        cur.append((a, b))
    if cur: chunks.append(_code_chunk(lines, path, cur[0][0], cur[-1][1], sections[cur[-1][1] - 1]))
    return chunks

def paragraphs(text: str, chars: int = 4000, overlap: int = 0, hard_split_ratio: float = 2.0) -> list: return _chunk_paragraphs(text, chars, overlap, hard_split_ratio)
def code(text: str, path: str = "", chars: int = 4000, hard_split_ratio: float = 2.0) -> list: return [c["text"] for c in _chunk_code(text, path, chars, max(1.0, hard_split_ratio))]
