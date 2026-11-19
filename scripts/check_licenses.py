"""Fail when an installed distribution carries a licence outside the allowed set.

Stdlib only, so it runs with the interpreter it checks (also piped into a container: python - < this file).
`--no-exceptions` turns off the named exceptions; the CPU install must pass without any.
"""

import re
import sys
from importlib.metadata import distributions

ALLOWED_IDS = {
    "MIT", "MIT-0", "BSD-2-Clause", "BSD-3-Clause", "0BSD", "Apache-2.0", "ISC", "PSF-2.0", "Python-2.0",
    "MPL-2.0", "LGPL-2.1-or-later", "LGPL-3.0-or-later", "LGPL-3.0-only", "Unlicense", "Zlib", "CC0-1.0",
}
ALLOWED_CLASSIFIERS = (
    "MIT License", "BSD License", "Apache Software License", "ISC License", "Python Software Foundation License",
    "Mozilla Public License 2.0", "GNU Lesser General Public License", "zlib/libpng License", "The Unlicense",
)
SHORT_NAMES = {"mit", "bsd", "apache", "apache 2.0", "isc", "mpl-2.0", "psf"}
MAX_SHORT_LICENSE = 200  # longer is pasted licence text, not a name

# name pattern -> reason. Applied only to packages that fail the rules above.
EXCEPTIONS = [
    (re.compile(r"^(nvidia-|cuda-(bindings|python|toolkit)$)"), "optional GPU runtime; the CPU path installs none of them"),
]


def _expression_ok(expression: str) -> bool:
    expression = re.sub(r"\s+WITH\s+[\w.\-+]+", "", expression)  # the exception after WITH is ignored
    if "(" in expression:
        # shortcut: grouped expressions need every id allowed (may fail a valid one, never passes a bad one);
        # parse the grouping if a real package trips it
        return all(i in ALLOWED_IDS for i in re.findall(r"[\w.\-+]+", expression) if i not in ("AND", "OR"))
    # AND binds tighter than OR, so split on OR first
    return any(
        all(i in ALLOWED_IDS for i in re.split(r"\s+AND\s+", alt.strip()))
        for alt in re.split(r"\s+OR\s+", expression)
    )


def verdict(expression: str | None, license_text: str | None, classifiers: list[str]) -> bool:
    if expression:
        return _expression_ok(expression)
    # the bare "License :: OSI Approved" names no licence, so it falls through to the License field
    licence_classifiers = [c for c in classifiers if c.startswith("License ::") and c != "License :: OSI Approved"]
    if licence_classifiers:
        return all(any(a in c for a in ALLOWED_CLASSIFIERS) for c in licence_classifiers)
    text = (license_text or "").strip()
    return bool(text) and len(text) <= MAX_SHORT_LICENSE and (text.lower() in SHORT_NAMES or text in ALLOWED_IDS)


def main(argv: list[str]) -> int:
    use_exceptions = "--no-exceptions" not in argv
    failures = checked = 0
    for dist in sorted(distributions(), key=lambda d: (d.metadata["Name"] or "").lower()):
        name, version = dist.metadata["Name"] or "?", dist.version
        expression = dist.metadata.get("License-Expression")
        text = dist.metadata.get("License")
        if text and len(text) > MAX_SHORT_LICENSE:
            text = None
        classifiers = dist.metadata.get_all("Classifier") or []
        checked += 1
        if verdict(expression, text, classifiers):
            continue
        normalized = re.sub(r"[-_.]+", "-", name.lower())  # PEP 503: metadata may spell cuda_bindings
        reason = next((r for p, r in EXCEPTIONS if p.match(normalized)), None) if use_exceptions else None
        if reason:
            print(f"EXCEPTION {name} {version}: {reason}")
            continue
        found = expression or "; ".join(c for c in classifiers if c.startswith("License ::")) or text or "no licence metadata"
        print(f"FAIL {name} {version}: {found}")
        failures += 1
    print(f"{checked} packages checked, {failures} failing")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
