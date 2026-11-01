"""Verify that the documented notebooks preserve the original executable code.

Usage:
    python tools/verify_code_integrity.py <original_dir> <documented_dir>

Both directories must contain notebooks with the same filenames. For each pair
the script checks that:

  1. The documented notebook contains exactly the same code cells, in the same
     order, as the original (added cells may only be markdown cells).
  2. Every code cell differs from the original ONLY by added full-line
     comments: removing the added lines yields the original source exactly
     (character for character).
  3. The Python token stream of each code cell, ignoring comments and blank
     lines, is identical to the original.
  4. Outputs, execution counts and cell metadata of every code cell, the
     original markdown cells and the notebook-level metadata are unchanged.

Exit code is 0 if every check passes, 1 otherwise.
"""
import io
import json
import os
import sys
import tokenize

IGNORED_TOKENS = {tokenize.COMMENT, tokenize.NL, tokenize.ENCODING}


def src(cell):
    s = cell["source"]
    return "".join(s) if isinstance(s, list) else s


def code_tokens(text):
    toks = []
    for tok in tokenize.generate_tokens(io.StringIO(text).readline):
        if tok.type in IGNORED_TOKENS:
            continue
        toks.append((tok.type, tok.string))
    return toks


def only_added_comment_lines(orig, new):
    """True if `new` equals `orig` plus inserted lines that are full-line comments."""
    o = orig.splitlines(keepends=True)
    n = new.splitlines(keepends=True)
    i = 0
    for line in n:
        if i < len(o) and line == o[i]:
            i += 1
        elif line.lstrip().startswith("#"):
            continue  # an added comment line
        else:
            return False
    return i == len(o)


def check(orig_path, new_path):
    errors = []
    a = json.load(open(orig_path, encoding="utf-8"))
    b = json.load(open(new_path, encoding="utf-8"))
    if a["metadata"] != b["metadata"]:
        errors.append("notebook metadata changed")

    # Layout check: each cell source must be a string or a list of complete
    # lines (only the last element may lack a trailing newline). A list of
    # fragments is valid JSON but renders badly in some viewers.
    for k, cell in enumerate(b["cells"]):
        s = cell["source"]
        if isinstance(s, list) and any(not part.endswith("\n") for part in s[:-1]):
            errors.append(f"documented cell {k}: source is not stored as whole lines")

    # Walk through the documented notebook, matching original cells in order.
    oc = a["cells"]
    j = 0
    added_md = 0
    stats = {"code": 0, "markdown": 0}
    for cell in b["cells"]:
        if j < len(oc) and cell["cell_type"] == oc[j]["cell_type"]:
            o = oc[j]
            if cell["cell_type"] == "code":
                if not only_added_comment_lines(src(o), src(cell)):
                    errors.append(f"original code cell {j}: non-comment change")
                elif code_tokens(src(o)) != code_tokens(src(cell)):
                    errors.append(f"original code cell {j}: token stream differs")
                for key in ("outputs", "execution_count", "metadata"):
                    if o.get(key) != cell.get(key):
                        errors.append(f"original code cell {j}: '{key}' changed")
                stats["code"] += 1
                j += 1
                continue
            if cell["cell_type"] == "markdown" and src(cell) == src(o) and cell.get("metadata") == o.get("metadata"):
                stats["markdown"] += 1
                j += 1
                continue
        if cell["cell_type"] == "markdown":
            added_md += 1
            continue
        errors.append(f"unexpected {cell['cell_type']} cell (documented index) before original cell {j}")
    if j != len(oc):
        errors.append(f"only {j}/{len(oc)} original cells found in order")
    n_orig_code = sum(c["cell_type"] == "code" for c in oc)
    n_new_code = sum(c["cell_type"] == "code" for c in b["cells"])
    if n_orig_code != n_new_code:
        errors.append(f"code cell count {n_orig_code} -> {n_new_code}")
    return errors, stats, added_md, n_orig_code


def main():
    if len(sys.argv) != 3:
        print(__doc__)
        return 2
    orig_dir, new_dir = sys.argv[1:]
    ok = True
    for name in sorted(os.listdir(orig_dir)):
        if not name.endswith(".ipynb"):
            continue
        errors, stats, added_md, n_code = check(os.path.join(orig_dir, name), os.path.join(new_dir, name))
        status = "PASS" if not errors else "FAIL"
        print(
            f"[{status}] {name}: {stats['code']}/{n_code} original code cells preserved, "
            f"{stats['markdown']} original markdown cells preserved, {added_md} documentation cells added"
        )
        for e in errors:
            print("    -", e)
        ok &= not errors
    print("ALL CHECKS PASSED" if ok else "VERIFICATION FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
