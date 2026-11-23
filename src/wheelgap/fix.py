"""--fix: for each gap cell (a package on a target, `source` or `missing`, not --ignore'd), the way out, in order of
preference:

1. an upgrade to an official release: the one --suggest names for that target (suggest.suggest: the lowest newer
   release with a live, compatible wheel whose Requires-Python admits the target's Python, no pre-release unless the
   pin is one, and that keeps a wheel on every target the package has one on now);
2. an unofficial gap wheel of the pinned version from wheelgap's gap wheel index (gapwheels.fillable), the fallback;
3. none, with the reason, and the package's issue tracker where wheelgap has it.

The releases come from the same lookups as --suggest (cli.release_lookups), so only names origin.known_pypi accepts
are looked up. Like the rest of --fix, nothing here changes a status, the summary or the exit code.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from packaging.specifiers import InvalidSpecifier, SpecifierSet
from packaging.utils import canonicalize_name
from packaging.version import InvalidVersion, Version

from . import gapwheels
from .check import ABI3_ON_FREETHREADED, Row, Status, rejection
from .sources import Source
from .suggest import Suggestion
from .targets import Target

UPGRADE, GAP_WHEEL, NONE = "upgrade", "gap_wheel", "none"
# Why a cell has no fix (`reason` in --json).
NO_NEWER = "no_newer_release"  # looked up: no newer release has a wheel for the target (and no gap wheel fits)
LOSES = "loses_wheel"  # every newer release with a wheel for the target drops one the package has elsewhere
NOT_LOOKED_UP = "not_looked_up"  # the releases weren't looked up (see the why line), and no gap wheel fits
NO_BINARY = "no_binary"  # --no-binary: pip uses no wheel for it, so neither an upgrade nor a gap wheel helps

RESOLVE_NOTE = "wheelgap doesn't resolve dependencies: the rest of them must allow these releases."
# Inputs whose ranged requirement (Package.spec) is the user's own constraint, written in that input.
SPEC_KINDS = frozenset({"requirements", "pyproject", "script", "conda-env", None})


@dataclass(frozen=True)
class Cell:
    row: Row
    target: Target
    source: Source
    kind: str  # UPGRADE, GAP_WHEEL or NONE
    version: str | None = None  # UPGRADE: the release to move to
    fill: gapwheels.Fill | None = None  # an unofficial gap wheel of the pinned version (for UPGRADE, the fallback)
    reason: str | None = None  # NONE: one of the reasons above
    abi3_only: bool = False  # NONE, on a free-threaded target: the release has abi3 wheels for its platform only


@dataclass(frozen=True)
class Constraint:
    """The user's own requirement for an upgrade cell's package, as wheelgap read it, and whether the release fits."""

    file: str
    requirements: tuple[str, ...]  # NAME + specifier; when `allows` is False, only the ones that exclude the release
    allows: bool


def gap_targets(row: Row, targets: list[Target]) -> list[Target]:
    return [t for t in targets if row.results[t] in (Status.SOURCE, Status.MISSING) and t not in row.ignored]


def abi3_only(row: Row, target: Target) -> bool:
    """Whether a free-threaded target's gap is down to abi3 wheels, by --explain's reason (check.rejection)."""
    pkg = row.package
    return target.freethreaded and any(
        f.endswith(".whl") and f not in pkg.yanked and rejection(f, target) == ABI3_ON_FREETHREADED
        for f in pkg.files or []
    )


def plan(
    gaps: list[Row],
    targets: list[Target],
    source_of: dict[int, Source],
    suggestions: dict[int, Suggestion],
    fills: list[gapwheels.Fill],
    not_looked_up: dict[int, str],
) -> list[Cell]:
    """One Cell per gap cell, in row and target order. `suggestions` and `not_looked_up` (why a package's releases
    weren't looked up) are by id() of the package; `fills` are gapwheels.fillable's."""
    filled = {(id(f.row), f.target): f for f in fills}
    cells = []
    for r in gaps:
        source = source_of[id(r.package)]
        found = suggestions.get(id(r.package))
        for t in gap_targets(r, targets):
            fill = filled.get((id(r), t))
            if found is not None and t in found.found:
                cells.append(Cell(r, t, source, UPGRADE, version=found.found[t], fill=fill))
                continue
            if fill is not None:
                cells.append(Cell(r, t, source, GAP_WHEEL, fill=fill))
                continue
            if r.package.no_binary:
                reason = NO_BINARY
            elif id(r.package) in not_looked_up or found is None:
                reason = NOT_LOOKED_UP
            else:
                reason = LOSES if t in found.loses else NO_NEWER
            cells.append(Cell(r, t, source, NONE, reason=reason, abi3_only=abi3_only(r, t)))
    return cells


def _uv_specifiers(source: Source, key: str) -> list[str] | None:
    """The specifiers a uv.lock's own projects (workspace members: editable or virtual) write for `key` in their
    dependencies and dependency groups, plus the lock's [manifest] constraints; None when there are none to go by, or
    when an override replaces them (uv then ignores them)."""
    manifest = source.uv_manifest if isinstance(source.uv_manifest, dict) else {}
    if any(isinstance(o, dict) and _key(o) == key for o in manifest.get("overrides", [])):
        return None
    entries: list[dict] = []
    for p in source.uv_packages or []:
        src, meta = p.get("source"), p.get("metadata")
        if not (isinstance(src, dict) and ("editable" in src or "virtual" in src) and isinstance(meta, dict)):
            continue
        entries += [d for d in meta.get("requires-dist", []) if isinstance(d, dict)]
        groups = meta.get("requires-dev", {})
        for group in groups.values() if isinstance(groups, dict) else []:
            entries += [d for d in group if isinstance(d, dict)] if isinstance(group, list) else []
    entries += [c for c in manifest.get("constraints", []) if isinstance(c, dict)]
    found = [e for e in entries if _key(e) == key]
    if not found or any(not isinstance(e.get("specifier", ""), str) for e in found):
        return None
    return [e.get("specifier", "") for e in found]


def _key(entry: dict) -> str | None:
    name = entry.get("name")
    return canonicalize_name(name) if isinstance(name, str) else None


def constraint(cell: Cell) -> Constraint | None:
    """The user's own constraint on an upgrade cell's package, only where wheelgap already reads one: a uv.lock's
    project requirements, or a ranged requirement in a requirements file, pyproject.toml, script or the command line.
    None otherwise (a transitive dependency, a Poetry or PDM lock, ...): then nothing is said."""
    pkg, source = cell.row.package, cell.source
    if cell.version is None:
        return None
    if source.kind == "uv" and source.uv_script is None:
        specs = _uv_specifiers(source, pkg.key)
        file = source.uv_pyproject.name if source.uv_pyproject is not None else "pyproject.toml"
    elif pkg.spec and source.kind in SPEC_KINDS:
        specs, file = [pkg.spec], source.label
    else:
        return None
    if not specs:
        return None
    try:
        version = Version(cell.version)
        sets = [SpecifierSet(s) for s in specs]
    except (InvalidVersion, InvalidSpecifier):
        return None
    shown = [f"{pkg.name}{s}" for s, ss in zip(specs, sets, strict=True) if not ss.contains(version, prereleases=True)]
    if shown:
        return Constraint(file, tuple(dict.fromkeys(shown)), allows=False)
    return Constraint(file, tuple(dict.fromkeys(f"{pkg.name}{s}" for s in specs)), allows=True)


def counts(cells: list[Cell]) -> dict[str, int]:
    """--json's fix_summary."""
    return {
        "gaps": len(cells),
        UPGRADE: sum(c.kind == UPGRADE for c in cells),
        GAP_WHEEL: sum(c.kind == GAP_WHEEL for c in cells),
        NONE: sum(c.kind == NONE for c in cells),
    }


def summary_line(cells: list[Cell]) -> str:
    """Only upgrades count as fixed: a gap wheel is unofficial, so it's counted on its own."""
    n = counts(cells)
    gaps = f"{n['gaps']} gap{'s' * (n['gaps'] != 1)} (a package on a target)"
    return (
        f"--fix: {gaps}: {n[UPGRADE]} fixed by upgrading to an official release; "
        f"{n[GAP_WHEEL]} {'has' if n[GAP_WHEEL] == 1 else 'have'} only an unofficial gap wheel; "
        f"{n[NONE]} {'has' if n[NONE] == 1 else 'have'} no fix yet."
    )


def _plain(text: str) -> str:
    return text


def _code(text: str) -> str:
    return f"`{text}`"  # names, versions, specifiers, file labels and commands made of them: no backticks in them


def _by_package(cells: list[Cell]) -> dict[tuple[str, str, str | None], list[Cell]]:
    """Cells grouped by (name as the input writes it, pinned version, upgrade version), in order."""
    grouped: dict[tuple[str, str, str | None], list[Cell]] = {}
    for c in cells:
        grouped.setdefault((c.row.package.name, c.row.package.version or "?", c.version), []).append(c)
    return grouped


def _labels(cells: list[Cell], multi_py: bool) -> str:
    return ", ".join(dict.fromkeys(c.target.label(multi_py) for c in cells))


def _pins(cells: list[Cell]) -> list[str]:
    """NAME==VERSION per upgrade, each once, in order."""
    return list(dict.fromkeys(f"{c.row.package.name}=={c.version}" for c in cells))


def _hashed(cells: list[Cell]) -> bool:
    return any(c.row.package.hashes for c in cells)


def how(source: Source, cells: list[Cell], code: Callable[[str], str] = _plain) -> list[str]:
    """What to run or change to move `source`'s packages to the releases of these upgrade cells, one line each."""
    pins = _pins(cells)
    options = " ".join(f"--upgrade-package {p}" for p in pins)
    label = code(source.label)
    kind = source.kind
    if kind == "uv" and source.uv_script is not None:
        return [
            f"For {code(Path(source.uv_script).name)}'s lock, run "
            f"{code(f'uv lock --script {source.uv_script} {options}')}."
        ]
    if kind == "uv":
        where = code(str(source.uv_pyproject.parent)) if source.uv_pyproject is not None else "the project"
        return [f"In {where}, run {code(f'uv lock {options}')}, then {code('uv sync')}."]
    if kind in ("poetry", "pdm"):
        tool = "poetry" if kind == "poetry" else "pdm"
        update = f"{tool} update {' '.join(dict.fromkeys(c.row.package.name for c in cells))}"
        return [
            f"Run {code(update)} for {label}: it stays within {code('pyproject.toml')}'s constraints and picks the "
            f"newest release they allow, so check that it's at least {', '.join(map(code, pins))} "
            "(widen the constraint if not), then run wheelgap again."
        ]
    if kind == "requirements":
        out = [f"In {label}, change the pins to {', '.join(map(code, pins))}."]
        if _hashed(cells):
            out.append(
                f"{label} pins hashes, so regenerate it instead of editing it: {code(f'pip-compile {options}')}, or "
                f"{code(f'uv pip compile {options}')} with the options you made it with."
            )
        return out
    if kind == "conda-env":
        return [f"In {label}'s {code('pip:')} list, change the pins to {', '.join(map(code, pins))}."]
    if kind == "script":
        return [f"In {label}'s {code('# /// script')} block, change the requirements to {', '.join(map(code, pins))}."]
    if kind is None:  # package names on the command line
        return [f"Use {', '.join(map(code, pins))} instead."]
    if kind == "pyproject":
        return [f"In {label}, change the requirements so they allow {', '.join(map(code, pins))}."]
    # pylock.toml, Pipfile.lock, pixi.lock, a pip report: made by a tool wheelgap doesn't drive.
    return [f"Upgrade to {', '.join(map(code, pins))} with the tool that made {label}, then lock again."]


def constraint_text(c: Constraint, code: Callable[[str], str] = _plain) -> str:
    reqs = ", ".join(map(code, c.requirements))
    if c.allows:
        return f"within your {code(c.file)} constraint {reqs}"
    return f"needs {code(c.file)}'s {reqs} relaxed"


def _why_not_looked_up(cells: list[Cell], not_looked_up: dict[int, str]) -> list[tuple[str, list[str]]]:
    """(why, the names) for packages whose releases weren't looked up, in order; reasons come from
    cli.release_lookups."""
    by_why: dict[str, list[str]] = {}
    for c in cells:
        if c.kind != UPGRADE and (why := not_looked_up.get(id(c.row.package))):
            names = by_why.setdefault(why, [])
            if c.row.package.name not in names:
                names.append(c.row.package.name)
    return list(by_why.items())


REASON_TEXT = {
    NO_NEWER: "no newer release has a wheel",
    LOSES: "every newer release with a wheel drops one it has elsewhere",
    NOT_LOOKED_UP: "newer releases weren't looked up",
    NO_BINARY: "--no-binary, so pip uses no wheel for it",
}


def _none_line(name: str, pinned: str, cs: list[Cell], multi_py: bool, ask: str | None, code: Callable) -> str:
    reasons = "; ".join(
        f"{REASON_TEXT[reason]} for {_labels([c for c in cs if c.reason == reason], multi_py)}"
        for reason in dict.fromkeys(c.reason for c in cs if c.reason)
    )
    line = f"{code(f'{name} {pinned}')}: {reasons}"
    if abi3 := [c for c in cs if c.abi3_only]:
        tags = " or ".join(dict.fromkeys(f"cp{c.target.major}{c.target.minor}t" for c in abi3))
        line += (
            f"; only abi3 wheels for {_labels(abi3, multi_py)}, which free-threaded Python can't use: "
            f"ask {code(name)} for a {tags} wheel"
        )
        return line + (f" at {ask}" if ask else "")
    return line + (f"; ask for a wheel: {ask}" if ask else "")


def _lines(
    cells: list[Cell],
    multi_py: bool,
    not_looked_up: dict[int, str],
    issues_urls: dict[int, str | None],
    code: Callable[[str], str],
    item: str,
) -> tuple[list[str], list[str], list[str], list[str]]:
    """(the upgrade lines, the how lines, the no-fix lines, the not-looked-up lines)."""
    upgrades = [c for c in cells if c.kind == UPGRADE]
    up = []
    for (name, pinned, version), cs in _by_package(upgrades).items():
        line = f"{item}{code(f'{name} {pinned} -> {version}')}: {_labels(cs, multi_py)}"
        if (own := constraint(cs[0])) is not None:
            line += f" ({constraint_text(own, code)})"
        up.append(line)
    versions: dict[int, set[str | None]] = {}
    for c in upgrades:
        versions.setdefault(id(c.row), set()).add(c.version)
    # A row whose gap targets got different releases: no one release has wheels for all of them.
    split = list(dict.fromkeys(c.row.package.name for c in upgrades if len(versions[id(c.row)]) > 1))
    ways: list[str] = []
    for cs in _by_source(upgrades).values():
        ways += how(cs[0].source, cs, code)
    if split:
        ways.append(
            f"No one release of {', '.join(map(code, split))} has wheels for all of these targets: pick the one for "
            "the targets you need."
        )
    if upgrades:
        ways.append(RESOLVE_NOTE)
    none = [
        item + _none_line(name, pinned, cs, multi_py, issues_urls.get(id(cs[0].row.package)), code)
        for (name, pinned, _), cs in _by_package([c for c in cells if c.kind == NONE]).items()
    ]
    unlooked = [
        f"Newer releases of {', '.join(map(code, names))} weren't looked up: {why}."
        for why, names in _why_not_looked_up(cells, not_looked_up)
    ]
    return up, ways, none, unlooked


def _by_source(cells: list[Cell]) -> dict[int, list[Cell]]:
    out: dict[int, list[Cell]] = {}
    for c in cells:
        out.setdefault(id(c.source), []).append(c)
    return out


def alternatives(cells: list[Cell]) -> set[tuple[int, Target]]:
    """(id(row), target) of the gap wheels that are only the fallback: an upgrade fixes that cell too."""
    return {(id(c.row), c.target) for c in cells if c.kind == UPGRADE and c.fill is not None}


UPGRADES_HEAD = "Upgrade: a newer official release has the wheel:"


def text_lines(
    cells: list[Cell], multi_py: bool, not_looked_up: dict[int, str], issues_urls: dict[int, str | None]
) -> list[str]:
    """--fix's summary, upgrades and gaps with no fix, before the gap wheel section (empty when there's no gap)."""
    if not cells:
        return []
    up, ways, none, unlooked = _lines(cells, multi_py, not_looked_up, issues_urls, _plain, "  ")
    out = ["", summary_line(cells)]
    if up:
        out += ["", UPGRADES_HEAD, *up, *ways]
    if none:
        out += ["", "No fix yet:", *none]
    if unlooked:
        out += ["", *unlooked]
    return out


def markdown_lines(
    cells: list[Cell], multi_py: bool, not_looked_up: dict[int, str], issues_urls: dict[int, str | None]
) -> list[str]:
    """--fix's summary, upgrades and gaps with no fix in the Markdown report (empty when there's no gap)."""
    if not cells:
        return []
    up, ways, none, unlooked = _lines(cells, multi_py, not_looked_up, issues_urls, _code, "- ")
    out = ["#### Fixes", "", summary_line(cells), ""]
    if up:
        out += [UPGRADES_HEAD, "", *up, "", *ways, ""]
    if none:
        out += ["No fix yet:", "", *none, ""]
    if unlooked:
        out += [*unlooked, ""]
    return out


def _gap_wheel_json(fill: gapwheels.Fill) -> dict[str, object]:
    return {"wheel": fill.wheel.wheel, "url": fill.wheel.url, "sha256": fill.wheel.sha256, "unofficial": True}


def to_json(cell: Cell) -> dict[str, object]:
    """--json --fix: one gap cell's `fix`."""
    if cell.kind == UPGRADE:
        c = constraint(cell)
        return {
            "kind": UPGRADE,
            "version": cell.version,
            "how": " ".join(how(cell.source, [cell])),
            "constraint": None
            if c is None
            else {"file": c.file, "requirements": list(c.requirements), "allows": c.allows},
            "gap_wheel": None if cell.fill is None else _gap_wheel_json(cell.fill),
        }
    if cell.kind == GAP_WHEEL and cell.fill is not None:
        return {"kind": GAP_WHEEL, **_gap_wheel_json(cell.fill)}
    return {"kind": NONE, "reason": cell.reason, "abi3_only": cell.abi3_only}
