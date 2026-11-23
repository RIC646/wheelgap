"""End-to-end check of `wheelgap --fix` on a clean machine (the Fix end-to-end workflow runs it).

    python scripts/fix_e2e.py plain -t linux-x86_64 --expect lz4,zstandard --pypi-only build
    python scripts/fix_e2e.py plain -t win-amd64 --expect lz4,zstandard --pypi-only only-binary
    python scripts/fix_e2e.py hashed -t linux-x86_64,win-amd64 --expect lz4,zstandard

It runs on the Python under test and uses only the standard library, so it needs nothing but that Python and the
network. Every install goes into a fresh venv; wheelgap is installed from this checkout into a venv of its own.

`plain` writes a requirements file with orjson, zstandard and lz4 pinned, and:
1. shows that installing it from PyPI alone fails: with `--pypi-only build`, a plain `pip install -r` must fail while
   building from source (a machine with no compiler); with `--pypi-only only-binary` (a machine with a compiler,
   such as GitHub's Windows runners), `pip install --only-binary=:all: -r` must find no wheel, which is what PyPI
   lacks, and the --fix install below must then take our wheels rather than build;
2. runs `wheelgap requirements.txt -p <this Python> -t <target> [--glibc <this glibc>] --fix`, checks that it names
   exactly the `--expect` packages, then appends what `--fix --print-gap-wheel-snippet` prints to the file, as the
   docs say (and checks that its one non-comment line is the line the plain --fix output says to add);
3. installs the file with `pip install --report` (which only records what pip did) in a fresh venv: it must succeed,
   each `--expect` package must come from the gap wheel index's release files and every other package from PyPI;
4. imports orjson, zstandard and lz4 and round-trips data through each.

`hashed` does the same from a fully hash-pinned file made by `uv pip compile --generate-hashes`, with --fix run for
every `-t` target (a cross-platform file needs a gap wheel's hash for each target it fills). Under "If the
requirements pin hashes, add each package's hashes to its existing line, keeping the hashes it has ...", --fix prints
an "add to the NAME==VERSION line:" label per package with its gap wheels' `--hash` options alone below it, and
--print-gap-wheel-snippet says the same in a comment per package; the two must agree. The script puts each of those
`--hash` options inside the package's own continued requirement block, keeping PyPI's hashes, as both say, and
`pip install --require-hashes -r` must succeed and take the gap wheels.
(Not as a line of their own: pip allows only the hashes all of a package's lines share, so it would build the sdist.)

Exit code 0 when every check passes, 1 when one fails (with a ::error:: annotation saying which), and each step's
evidence goes to the job summary when GITHUB_STEP_SUMMARY is set.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
INDEX = "https://wheelgap.github.io/wheels/simple/"
GAP_FILES = "https://github.com/wheelgap/wheels/releases/download/"
PYPI_FILES = "https://files.pythonhosted.org/"
PINS = ["orjson==3.13.0", "zstandard==0.25.0", "lz4==4.4.5"]
UV = "uv==0.13.9"  # only to make the hash-pinned file, the way a project would
COMPILERS = ["cc", "gcc", "c++", "g++", "clang", "cargo", "rustc"]
FILLS_HEADER = "wheelgap's gap wheel index has wheels PyPI lacks for:"
ADD_LINE_HEADER = "To install them with pip, add this line to your requirements file:"
HASHES_HEADER = (
    "If the requirements pin hashes, add each package's hashes to its existing line, keeping the hashes it has "
    "(on a second line for the same package, pip won't use them):"
)
# Under it, --fix's text output gives each package a label line, then its --hash options alone on the next line.
HASH_LABEL = re.compile(r"add to the (\S+==\S+) line:")
HASH_OPTS = re.compile(r"(?:--hash=sha256:[0-9a-f]{64} ?)+")
# --print-gap-wheel-snippet's comment for one package of a hash-pinned file: its NAME==VERSION, then its options.
SNIPPET_HASHES = re.compile(
    r"# If the requirements pin hashes, add these to the existing (\S+) line, keeping the hashes it has "
    r"\(not as a line of their own, which pip won't use\): ((?:--hash=sha256:[0-9a-f]{64} ?)+)"
)
ROUND_TRIP = """
import lz4.frame, orjson, zstandard
from importlib.metadata import version
doc = {"name": "wheelgap", "python": "3.15", "nested": [1, 2.5, None, True, "\\u00fc"]}
assert orjson.loads(orjson.dumps(doc)) == doc
data = b"wheelgap gap wheel " * 4096
packed = zstandard.ZstdCompressor(level=10).compress(data)
assert zstandard.ZstdDecompressor().decompress(packed) == data and len(packed) < len(data)
framed = lz4.frame.compress(data)
assert lz4.frame.decompress(framed) == data and len(framed) < len(data)
for name in ("orjson", "zstandard", "lz4"):
    print(name, version(name), "round-trip ok")
"""


class CheckFailed(Exception):
    pass


def summary(*lines: str) -> None:
    path = os.environ.get("GITHUB_STEP_SUMMARY")
    if path:
        with open(path, "a", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")


def summary_block(title: str, text: str) -> None:
    summary(f"**{title}**", "", "```text", text.rstrip("\n"), "```", "")


def run(cmd: list[str | Path], *, title: str, cwd: Path | None = None) -> tuple[int, str]:
    """Run cmd, print its output in a log group and return (exit code, stdout+stderr)."""
    args = [str(c) for c in cmd]
    print(f"::group::{title}", flush=True)
    print("$ " + " ".join(args), flush=True)
    proc = subprocess.run(args, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    print(proc.stdout, end="" if proc.stdout.endswith("\n") else "\n")
    print(f"(exit code {proc.returncode})")
    print("::endgroup::", flush=True)
    return proc.returncode, proc.stdout


def venv(path: Path) -> Path:
    """A fresh venv at path; returns its python."""
    if path.exists():
        shutil.rmtree(path)
    subprocess.run([sys.executable, "-m", "venv", str(path)], check=True)
    py = path / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    assert py.exists(), py
    return py


def this_python() -> str:
    return f"{sys.version_info.major}.{sys.version_info.minor}"


def glibc_args(targets: list[str]) -> list[str]:
    """--glibc for the glibc this machine has, when a Linux target is checked here."""
    libc, ver = platform.libc_ver()
    if sys.platform.startswith("linux") and any(t.startswith("linux-") for t in targets):
        if libc != "glibc":
            raise CheckFailed(f"expected a glibc Linux, found {libc or 'no libc'} {ver}")
        return ["--glibc", ".".join(ver.split(".")[:2])]
    return []


def check_no_compiler(require_none: bool) -> None:
    found = {c: shutil.which(c) for c in COMPILERS}
    present = {c: p for c, p in found.items() if p}
    line = "compilers on PATH: " + (", ".join(f"{c} ({p})" for c, p in present.items()) or "none")
    print(line)
    summary(f"- {line} (looked for {', '.join(COMPILERS)})")
    if require_none and present:
        raise CheckFailed(f"this machine should have no compiler, but has {', '.join(present)}")


def install_wheelgap(work: Path) -> Path:
    py = venv(work / "wheelgap-venv")
    code, _ = run([py, "-m", "pip", "install", REPO], title="Install wheelgap from this checkout")
    if code:
        raise CheckFailed("could not install wheelgap from the checkout")
    exe = py.parent / ("wheelgap.exe" if os.name == "nt" else "wheelgap")
    run([exe, "--version"], title="wheelgap --version")
    return exe


def pypi_only(work: Path, reqs: Path, mode: str) -> None:
    """Installing reqs from PyPI alone must fail, for the reason the mode names."""
    py = venv(work / "pypi-only-venv")
    if mode == "build":
        code, out = run([py, "-m", "pip", "install", "-r", reqs], title="PyPI alone: pip install -r requirements.txt")
        # pip names the sdist it failed to build, and the build's own error must be the missing compiler (zstandard's
        # setup.py runs gcc to preprocess a header; a setuptools compile says "error: command 'gcc' failed").
        failed = r"Failed to build|Failed building wheel"
        no_cc = r"No such file or directory: '(cc|gcc|c\+\+|g\+\+|cargo|rustc)'|error: command '[^']*' failed"
        why = [ln.strip() for ln in out.splitlines() if re.search(f"{failed}|{no_cc}", ln)]
        expected = code != 0 and any(re.search(failed, ln) for ln in why) and any(re.search(no_cc, ln) for ln in why)
    else:
        cmd: list[str | Path] = [py, "-m", "pip", "install", "--only-binary=:all:", "-r", reqs]
        code, out = run(cmd, title="PyPI alone, wheels only: pip install --only-binary=:all: -r requirements.txt")
        why = [ln for ln in out.splitlines() if "No matching distribution found" in ln]
        expected = code != 0 and bool(why)
    summary_block(f"PyPI alone ({mode}): exit code {code}", "\n".join(why) or out[-3000:])
    if not expected:
        raise CheckFailed(f"installing from PyPI alone should fail ({mode}), but exit code {code} and no such error")


def run_fix(wheelgap: Path, reqs: Path, targets: list[str], extra: list[str] | None = None) -> str:
    cmd: list[str | Path] = [wheelgap, reqs.name, "-p", this_python(), "-t", ",".join(targets)]
    cmd += [*glibc_args(targets), "--fix", *(extra or [])]
    code, out = run(cmd, cwd=reqs.parent, title=" ".join(["wheelgap", *map(str, cmd[1:])]))
    summary_block(" ".join(["wheelgap", *map(str, cmd[1:])]) + f" (exit code {code})", out)
    # The check still reports PyPI's gaps, so 1; 0 would mean PyPI has the wheels now (the test needs new pins),
    # and 2 is an error.
    if code != 1:
        raise CheckFailed(f"wheelgap --fix exited {code}, expected 1 (gaps found)")
    return out


def section(out: str, header: str) -> list[str]:
    """The indented lines right under header in --fix's text output."""
    lines = out.splitlines()
    if header not in lines:
        raise CheckFailed(f"--fix printed no {header!r} line")
    got = []
    for ln in lines[lines.index(header) + 1 :]:
        if not ln.startswith("  "):
            break
        got.append(ln.strip())
    return got


def check_fills(out: str, expect: set[str], targets: list[str]) -> None:
    named = {}
    for ln in section(out, FILLS_HEADER):
        m = re.fullmatch(r"(\S+) (\S+): (.+)", ln)
        if not m:
            raise CheckFailed(f"unexpected line under {FILLS_HEADER!r}: {ln!r}")
        named[m[1]] = m[3].split(", ")
    if set(named) != expect:
        raise CheckFailed(f"--fix names {sorted(named)}, expected exactly {sorted(expect)}")
    for name, filled in named.items():
        if sorted(filled) != sorted(targets):
            raise CheckFailed(f"--fix says {name} fills {filled}, expected {targets}")


def check_report(report: Path, expect: set[str]) -> None:
    """pip's --report: the --expect packages came from the gap wheel index's files, the rest from PyPI."""
    items = json.loads(report.read_text(encoding="utf-8"))["install"]
    rows, bad = [], []
    for item in items:
        name, url = item["metadata"]["name"].lower(), item["download_info"]["url"]
        rows.append(f"{name} {item['metadata']['version']}  {url}")
        want = GAP_FILES if name in expect else PYPI_FILES
        if not url.startswith(want) or not url.endswith(".whl"):
            bad.append(f"{name} came from {url}, expected a wheel under {want}")
    summary_block("What pip installed (from its --report)", "\n".join(rows))
    print("\n".join(rows))
    if bad or {i["metadata"]["name"].lower() for i in items} < expect:
        raise CheckFailed("; ".join(bad) or f"pip's report lacks one of {sorted(expect)}")


def install_and_use(work: Path, reqs: Path, expect: set[str], require_hashes: bool) -> None:
    py = venv(work / "fixed-venv")
    report = work / "report.json"
    cmd: list[str | Path] = [py, "-m", "pip", "install", "--report", report, "-r", reqs]
    if require_hashes:
        cmd[4:4] = ["--require-hashes"]
    code, out = run(cmd, title="With --fix's lines: " + " ".join(map(str, cmd[3:])))
    tail = [ln for ln in out.splitlines() if ln.startswith(("Successfully installed", "ERROR"))]
    summary_block(f"pip install with --fix's lines (exit code {code})", "\n".join(tail) or out[-4000:])
    if code:
        raise CheckFailed(f"pip install with --fix's lines failed (exit code {code}): {' / '.join(tail)[:500]}")
    check_report(report, expect)
    code, out = run([py, "-c", ROUND_TRIP], title="Import and round-trip orjson, zstandard and lz4")
    summary_block(f"Import and round-trip (exit code {code})", out)
    if code:
        raise CheckFailed("importing or round-tripping the installed packages failed")


def plain(args: argparse.Namespace, work: Path) -> None:
    reqs = work / "requirements.txt"
    reqs.write_text("\n".join(PINS) + "\n", encoding="utf-8")
    check_no_compiler(require_none=args.pypi_only == "build")
    pypi_only(work, reqs, args.pypi_only)
    wheelgap = install_wheelgap(work)
    out = run_fix(wheelgap, reqs, args.target)
    check_fills(out, args.expect, args.target)
    add = section(out, ADD_LINE_HEADER)
    snippet = run_fix(wheelgap, reqs, args.target, ["--print-gap-wheel-snippet"])
    config = [ln for ln in snippet.splitlines() if ln.strip() and not ln.startswith("#")]
    if config != add or config != [f"--extra-index-url {INDEX}"]:
        raise CheckFailed(f"the snippet's lines {config} differ from the line --fix says to add {add}")
    with reqs.open("a", encoding="utf-8") as f:
        f.write(snippet)
    summary_block("requirements.txt with the snippet appended", reqs.read_text(encoding="utf-8"))
    install_and_use(work, reqs, args.expect, require_hashes=False)


def merge_hashes(text: str, printed: dict[str, list[str]]) -> str:
    """Add each package's printed `--hash=sha256:H` options to its existing (multi-line) entry, keeping its own."""
    lines = text.splitlines()
    for req, opts in printed.items():
        starts = [i for i, x in enumerate(lines) if x.split(" ", 1)[0] == req]
        if len(starts) != 1:
            raise CheckFailed(f"{req} is on {len(starts)} lines of the compiled file, expected 1")
        for opt in opts:
            line = lines[starts[0]]
            lines[starts[0]] = line if line.endswith("\\") else line + " \\"
            lines.insert(starts[0] + 1, f"    {opt}" + (" \\" if line.endswith("\\") else ""))
    return "\n".join(lines) + "\n"


def hash_lines(out: str, expect: set[str], targets: list[str]) -> dict[str, list[str]]:
    """{NAME==VERSION: its --hash options} from --fix's text output: per package a label, then a hash per target."""
    printed: dict[str, list[str]] = {}
    got = section(out, HASHES_HEADER)
    if len(got) % 2:
        raise CheckFailed(f"expected label and options lines in pairs under the hashes header, got {got}")
    for label, opts in zip(got[::2], got[1::2], strict=True):
        m = HASH_LABEL.fullmatch(label)
        if not m or m[1] in printed or not HASH_OPTS.fullmatch(opts):
            raise CheckFailed(f"unexpected lines under the hashes header: {label!r}, {opts!r}")
        printed[m[1]] = opts.split()
    names = {req.split("==", 1)[0] for req in printed}
    if names != expect or any(len(opts) != len(targets) for opts in printed.values()):
        raise CheckFailed(f"expected one label per package of {sorted(expect)}, a hash per target, got {printed}")
    return printed


def hashed(args: argparse.Namespace, work: Path) -> None:
    check_no_compiler(require_none=sys.platform.startswith("linux"))
    tools = venv(work / "uv-venv")
    if run([tools, "-m", "pip", "install", UV], title=f"Install {UV}")[0]:
        raise CheckFailed(f"could not install {UV}")
    (work / "requirements.in").write_text("\n".join(PINS) + "\n", encoding="utf-8")
    uv = tools.parent / ("uv.exe" if os.name == "nt" else "uv")
    cmd: list[str | Path] = [uv, "pip", "compile", "--generate-hashes", "--python-version", this_python()]
    cmd += ["requirements.in", "-o", "requirements.txt"]
    if run(cmd, cwd=work, title="uv pip compile --generate-hashes")[0]:
        raise CheckFailed("uv pip compile --generate-hashes failed")
    reqs = work / "requirements.txt"
    pypi_only(work, reqs, "build" if sys.platform.startswith("linux") else "only-binary")
    wheelgap = install_wheelgap(work)
    out = run_fix(wheelgap, reqs, args.target)
    check_fills(out, args.expect, args.target)
    add, printed = section(out, ADD_LINE_HEADER), hash_lines(out, args.expect, args.target)
    snippet = run_fix(wheelgap, reqs, args.target, ["--print-gap-wheel-snippet"])
    in_snippet = {m[1]: m[2].split() for m in map(SNIPPET_HASHES.fullmatch, snippet.splitlines()) if m}
    if in_snippet != printed:
        raise CheckFailed(f"the snippet's hash comments {in_snippet} differ from --fix's hash lines {printed}")
    text = "\n".join(add) + "\n" + merge_hashes(reqs.read_text(encoding="utf-8"), printed)
    reqs.write_text(text, encoding="utf-8")
    ours = [h for opts in printed.values() for h in opts]
    shown = [ln for ln in text.splitlines() if not ln.startswith("    --hash=") or any(h in ln for h in ours)]
    summary_block(
        "requirements.txt with --fix's hashes on each package's line (PyPI's own hashes left out here)",
        "\n".join(shown),
    )
    install_and_use(work, reqs, args.expect, require_hashes=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    sub = parser.add_subparsers(dest="mode", required=True)
    for name in ("plain", "hashed"):
        p = sub.add_parser(name)
        p.add_argument("-t", "--target", required=True, type=lambda s: s.split(","), help="wheelgap target labels")
        p.add_argument("--expect", required=True, type=lambda s: set(s.split(",")), help="packages --fix must name")
        if name == "plain":
            p.add_argument("--pypi-only", choices=["build", "only-binary"], required=True)
    args = parser.parse_args()
    summary(f"### `{args.mode}` on Python {platform.python_version()}, {platform.platform()}", "")
    work = Path(tempfile.mkdtemp(prefix="fix-e2e-"))
    try:
        plain(args, work) if args.mode == "plain" else hashed(args, work)
    except CheckFailed as e:
        print(f"::error::{e}")
        summary(f"**Failed:** {e}")
        return 1
    summary("**Passed.**")
    return 0


if __name__ == "__main__":
    sys.exit(main())
