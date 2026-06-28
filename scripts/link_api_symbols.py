"""Link API symbols to mkdocstrings autorefs in narrative docs.

Scans docs/api/**/*.md for `::: gradiend...` stubs to build a symbol map, then
updates narrative markdown (outside code fences) to use autoref links.

Usage:
    python scripts/link_api_symbols.py          # update files in place
    python scripts/link_api_symbols.py --check  # exit 1 if changes would be made
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
API_STUB_PATTERN = re.compile(r"^::: (gradiend\.\S+)", re.MULTILINE)
API_INDEX_LINK = re.compile(
    r"\*\*\[(?P<sym>[A-Za-z_][A-Za-z0-9_]*)\]\((?P<sym2>[A-Za-z_][A-Za-z0-9_]*)\.md\)\*\*"
)
ALREADY_AUTOREF = re.compile(r"\[`[^`]+`\]\[gradiend\.")

# Instance-style entry points documented on these classes (not separate API stubs).
METHOD_OWNERS: dict[str, frozenset[str]] = {
    "trainer": frozenset({
        "plot_training_convergence",
        "plot_encoder_distributions",
        "plot_encoder_scatter",
        "plot_encoder_by_target",
        "plot_probability_shifts",
        "evaluate_encoder",
        "evaluate_decoder",
        "evaluate",
    }),
    "suite": frozenset({
        "plot_similarity_heatmap",
        "plot_cross_encoding_heatmap",
        "plot_topk_overlap_heatmap",
        "plot_anchor_aligned_encoding_heatmap",
        "evaluate_encoder",
        "evaluate_decoder",
        "evaluate",
    }),
    "view": frozenset({
        "plot_encoder_by_target",
        "plot_encoder_distributions",
        "plot_encoder_scatter",
        "plot_training_convergence",
        "plot_probability_shifts",
        "evaluate_encoder",
    }),
    "SymmetricTrainerSuite": frozenset({
        "plot_cross_encoding_heatmap",
    }),
}
METHOD_OWNER_FQN: dict[str, str] = {
    "trainer": "gradiend.trainer.trainer.Trainer",
    "suite": "gradiend.trainer.suite.base.TrainerSuite",
    "view": "gradiend.trainer.core.multi_seed.MultiSeedTrainerView",
    "SymmetricTrainerSuite": "gradiend.trainer.suite.symmetric.SymmetricTrainerSuite",
}
# Bare calls in backticks that refer to a class method (no `trainer.` prefix).
BARE_METHOD_ALIASES: dict[str, str] = {
    "evaluate_encoder": "gradiend.trainer.trainer.Trainer.evaluate_encoder",
}


def load_symbol_map() -> dict[str, str]:
    mapping: dict[str, str] = {}
    for path in sorted((DOCS / "api").rglob("*.md")):
        match = API_STUB_PATTERN.search(path.read_text(encoding="utf-8"))
        if not match:
            continue
        fqn = match.group(1)
        short = fqn.rsplit(".", 1)[-1]
        mapping[short] = fqn
    return mapping


def split_code_fences(text: str) -> list[tuple[str, bool]]:
    """Split text into (segment, is_code_fence) pairs."""
    parts: list[tuple[str, bool]] = []
    in_fence = False
    current: list[str] = []
    for line in text.splitlines(keepends=True):
        if line.startswith("```"):
            if current:
                parts.append(("".join(current), in_fence))
                current = []
            parts.append((line, True))
            in_fence = not in_fence
        else:
            current.append(line)
    if current:
        parts.append(("".join(current), in_fence))
    return parts


def _symbol_pattern(symbols: dict[str, str]) -> re.Pattern[str]:
    ordered = sorted(symbols, key=len, reverse=True)
    names = "|".join(re.escape(s) for s in ordered)
    return re.compile(names)


def link_method_calls(segment: str) -> str:
    if not segment:
        return segment

    owners = "|".join(re.escape(k) for k in METHOD_OWNER_FQN)
    pattern = re.compile(
        rf"(?<!\[)`(?P<owner>{owners})\."
        rf"(?P<method>[a-z_][a-z0-9_]*)(?P<args>\([^`]*\))?`(?!\])"
    )

    def repl(match: re.Match[str]) -> str:
        owner = match.group("owner")
        method = match.group("method")
        if method not in METHOD_OWNERS[owner]:
            return match.group(0)
        args = match.group("args") or ""
        display = f"{owner}.{method}{args}"
        fqn = f"{METHOD_OWNER_FQN[owner]}.{method}"
        return f"[`{display}`][{fqn}]"

    return pattern.sub(repl, segment)


def link_standalone_calls(segment: str, symbols: dict[str, str]) -> str:
    if not segment:
        return segment

    all_names = dict(symbols)
    all_names.update(BARE_METHOD_ALIASES)
    names = _symbol_pattern(all_names)
    pattern = re.compile(
        rf"(?<!\[)`(?P<name>{names.pattern})(?P<args>\([^`]*\))`(?!\])"
    )

    def repl(match: re.Match[str]) -> str:
        name = match.group("name")
        args = match.group("args")
        fqn = all_names[name]
        return f"[`{name}{args}`][{fqn}]"

    return pattern.sub(repl, segment)


def link_backtick_symbols(segment: str, symbols: dict[str, str]) -> str:
    if not segment or not symbols:
        return segment

    segment = link_method_calls(segment)
    segment = link_standalone_calls(segment, symbols)

    names = _symbol_pattern(symbols)
    # `Class.method()` or `Class.attr`
    qualified = re.compile(
        rf"(?<!\[)`({names.pattern})(\.(?:[A-Za-z_][A-Za-z0-9_]*)(?:\(\))?)`(?!\])"
    )
    simple = re.compile(rf"(?<!\[)`({names.pattern})`(?!\])")

    def qual_repl(match: re.Match[str]) -> str:
        sym, suffix = match.group(1), match.group(2)
        return f"[`{sym}`][{symbols[sym]}]{suffix}"

    def simple_repl(match: re.Match[str]) -> str:
        sym = match.group(1)
        return f"[`{sym}`][{symbols[sym]}]"

    segment = qualified.sub(qual_repl, segment)
    return simple.sub(simple_repl, segment)


def link_bold_symbols(segment: str, symbols: dict[str, str]) -> str:
    if not segment or not symbols:
        return segment

    names = _symbol_pattern(symbols)
    bold = re.compile(rf"(?<!\[)\*\*({names.pattern})\*\*(?!\[)")

    def repl(match: re.Match[str]) -> str:
        sym = match.group(1)
        return f"**[`{sym}`][{symbols[sym]}]**"

    return bold.sub(repl, segment)


def link_api_index_pages(text: str, symbols: dict[str, str]) -> str:
    def repl(match: re.Match[str]) -> str:
        sym = match.group("sym")
        if sym != match.group("sym2") or sym not in symbols:
            return match.group(0)
        return f"**[`{sym}`][{symbols[sym]}]**"

    return API_INDEX_LINK.sub(repl, text)


def link_file(path: Path, symbols: dict[str, str]) -> str:
    text = path.read_text(encoding="utf-8")
    rel = path.relative_to(DOCS)

    if rel.parts[0] == "api" and rel.name == "index.md":
        return link_api_index_pages(text, symbols)

    chunks = split_code_fences(text)
    out: list[str] = []
    for chunk, in_fence in chunks:
        if in_fence:
            out.append(chunk)
        else:
            linked = link_backtick_symbols(chunk, symbols)
            linked = link_bold_symbols(linked, symbols)
            out.append(linked)
    return "".join(out)


def should_process(path: Path) -> bool:
    rel = path.relative_to(DOCS)
    if rel.parts[0] != "api":
        return True
    return rel.name == "index.md"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Do not write files; exit 1 if any file would change.",
    )
    args = parser.parse_args()

    symbols = load_symbol_map()
    if not symbols:
        print("No API symbols found under docs/api/", file=sys.stderr)
        return 1

    changed_files: list[Path] = []
    for path in sorted(DOCS.rglob("*.md")):
        if not should_process(path):
            continue
        new_text = link_file(path, symbols)
        if new_text != path.read_text(encoding="utf-8"):
            changed_files.append(path)
            if not args.check:
                path.write_text(new_text, encoding="utf-8")

    if changed_files:
        for path in changed_files:
            print(path.relative_to(ROOT))
        if args.check:
            print(f"\n{len(changed_files)} file(s) need API autoref links.", file=sys.stderr)
            return 1
        print(f"\nUpdated {len(changed_files)} file(s).")
    else:
        print("All docs already use API autoref links.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
