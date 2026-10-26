"""Fill every number on the site from the ledger (paper/numbers.json) or from web/tools/facts.json.

HTML carries placeholders:  <span data-num="p2.H1" data-f="pp">x</span>   <span data-fact="dataset.pairs">x</span>
Run:  python web/tools/fill_site.py          (rewrites the pages in web/public in place, writes web/public/data/site-data.js)
      python web/tools/fill_site.py --check  (changes nothing; exits 1 if any page is out of date or an id is unknown)
No number is typed by hand: ledger -> text, as in the paper. Facts that are not in the ledger carry a source in facts.json.
"""
import json, re, sys
from pathlib import Path

TOOLS = Path(__file__).resolve().parent
WEB = TOOLS.parent
PUBLIC = WEB / "public"  # the folder that is deployed (GitHub Pages root); nothing outside it is published
ROOT = WEB.parent
rows = {r["id"]: r for r in json.load(open(ROOT / "paper/numbers.json"))["rows"]}
facts = json.load(open(TOOLS / "facts.json"))
MINUS = "−"


def sgn(x, s):
    return s.replace("-", MINUS)


def one(row, f, x):
    """format one number x (value or CI end) of ledger row `row` in format f"""
    if f == "pct1":  return f"{100 * x:.1f}%"
    if f == "pct0":  return f"{100 * x:.0f}%"
    if f == "pp":    return sgn(x, f"{100 * x:+.1f} pp")      # unit diff, signed
    if f == "ppn":   return sgn(x, f"{100 * x:+.1f}")          # same, no unit (CI ends)
    if f == "ppabs": return f"{abs(100 * x):.1f} pp"
    if f == "ppraw": return f"{x:.2f} pp"                       # unit pp (already points)
    if f == "auc":   return f"{x:.3f}"
    if f == "int":   return f"{int(round(x))}"
    if f == "p":     return f"{x:.4f}"
    raise SystemExit(f"unknown format {f}")


def fill_one(m):
    attrs = m.group(1)
    a = dict(re.findall(r'(data-[a-z]+)="([^"]*)"', attrs))
    if "data-num" in a:
        i = a["data-num"]
        if i not in rows:
            raise SystemExit(f"unknown ledger id: {i}")
        r, f = rows[i], a.get("data-f")
        if not f:
            raise SystemExit(f"{i}: data-f missing")
        part = a.get("data-part", "value")
        if part == "value":
            txt = one(r, f, r["value"])
        else:  # ci: [lo, hi] with the unit-free form for differences
            if not r["ci"]:
                raise SystemExit(f"{i}: no CI in the ledger")
            g = "ppn" if f in ("pp",) else f
            txt = f"[{one(r, g, r['ci'][0])}, {one(r, g, r['ci'][1])}]"
    elif "data-fact" in a:
        k = a["data-fact"]
        if k not in facts:
            raise SystemExit(f"unknown fact: {k}")
        v, fmt = facts[k]["value"], facts[k]["fmt"]
        txt = {"int": lambda: f"{v:,}", "auc": lambda: f"{v:.3f}", "pct1": lambda: f"{100 * v:.1f}%", "pct2": lambda: f"{100 * v:.2f}%"}.get(fmt, lambda: str(v))()
    else:
        return m.group(0)
    return f"<span {attrs}>{txt}</span>"


SPAN = re.compile(r"<span\s+([^>]*?(?:data-num|data-fact)=[^>]*?)>((?s:.*?))</span>")


def process(text):
    return SPAN.sub(fill_one, text)


def main(check):
    bad = 0
    for p in sorted(PUBLIC.glob("*.html")):
        old = p.read_text()
        new = process(old)
        if new != old:
            if check:
                print("out of date:", p.name); bad += 1
            else:
                p.write_text(new)
    # chart data: the whole ledger as id -> [value, lo, hi], plus the facts
    EXCLUDE = {"p5.1a.p_perm_primary_approx"}  # superseded by the post hoc value p5.1.posthoc.p; no page uses it
    L = {i: [r["value"]] + (r["ci"] or []) for i, r in rows.items() if i not in EXCLUDE}
    F = {k: v["value"] for k, v in facts.items() if not k.startswith("_")}
    js = "/* generated file, do not edit */\nwindow.SITE=" + json.dumps({"L": L, "F": F}, separators=(",", ":")) + ";\n"
    target = PUBLIC / "data/site-data.js"
    if check:
        if not target.exists() or target.read_text() != js:
            print("out of date: data/site-data.js"); bad += 1
    else:
        target.write_text(js)
    n = sum(len(re.findall(r'data-(?:num|fact)="', p.read_text())) for p in PUBLIC.glob("*.html"))
    print(f"{'checked' if check else 'filled'} {n} numbers in {len(list(PUBLIC.glob('*.html')))} pages; {len(L)} ledger rows exported")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main("--check" in sys.argv)
