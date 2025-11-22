#!/usr/bin/env python3
from pathlib import Path
import json, re

ROOT   = Path.home()/ "project/results/mcmc"
PERBIN = ROOT / "per_bin"
JOINT  = ROOT / "joint" / "y3_joint_bins1-4.npz"

bins = []
pat = re.compile(r"y3_bin(\d+)_z([0-9.]+)-([0-9.]+)\.npz$")
for p in sorted(PERBIN.glob("y3_bin*_z*.npz")):
    m = pat.match(p.name)
    if not m: 
        continue
    idx, zmin, zmax = int(m.group(1)), float(m.group(2)), float(m.group(3))
    zc = round(0.5*(zmin+zmax), 3)
    bins.append({"idx": idx, "file": str(p), "zmin": zmin, "zmax": zmax, "z_center": zc})

manifest = {"per_bin": bins, "joint": str(JOINT)}
with open(ROOT / "manifest.json", "w") as f:
    json.dump(manifest, f, indent=2)
print("Wrote", ROOT / "manifest.json")
