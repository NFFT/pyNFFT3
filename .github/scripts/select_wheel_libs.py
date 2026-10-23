#!/usr/bin/env python3
"""
Keep only a chosen subset of src/pyNFFT3/lib/, so `python -m build --wheel`
produces a wheel for one (OS, arch, glibc floor) combination instead of
bundling every platform's binaries into one oversized wheel (that hit
PyPI's 100 MB per-file limit). Used by the build-wheels matrix in
.github/workflows/release.yml, once per matrix entry, in a fresh checkout.

Each positional argument is either:
  - "ARCH/GLIBCVER" - keep that whole directory (Linux .so's, which live
    under a glibc-version subdirectory per CPU arch/SIMD level).
  - "ARCH:EXT"      - keep only files with that extension directly under
    ARCH (Windows .dll's / macOS .dylib's, which sit next to each other -
    and next to the Linux glibc2.*/ subdirectories - under the same ARCH
    directory, so an extension filter is needed to tell them apart).
"""

import sys
from pathlib import Path

LIB_DIR = Path("src/pyNFFT3/lib")


def main(specs):
    keep_dirs = set()
    keep_files = set()

    for spec in specs:
        if "/" in spec:
            arch, glibcver = spec.split("/", 1)
            keep_dirs.add(Path(arch) / glibcver)
        else:
            arch, ext = spec.split(":", 1)
            for f in (LIB_DIR / arch).glob(f"*.{ext}"):
                keep_files.add(f.relative_to(LIB_DIR))

    for path in LIB_DIR.rglob("*"):
        if path.is_dir():
            continue
        rel = path.relative_to(LIB_DIR)
        if rel in keep_files or any(str(rel).startswith(f"{d}/") for d in keep_dirs):
            continue
        path.unlink()

    # Drop now-empty directories (deepest first).
    for d in sorted((p for p in LIB_DIR.rglob("*") if p.is_dir()), reverse=True):
        if not any(d.iterdir()):
            d.rmdir()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("usage: select_wheel_libs.py SPEC [SPEC ...]")
    main(sys.argv[1:])
