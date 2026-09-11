"""The aggregate paired-episode tally the abstract quotes, computed, not hand-assembled
(dual-track audit finding 4.3). Sweeps every result file that carries paired per-episode
outcomes for a frozen-faulted and an adaptive arm, assigns it to a category by path, applies
the EXCLUSION rules below, and writes a manifest naming every included and excluded file with
its counts. Nothing here is hand-typed except the rules.

Rules (the only editorial content):
  * excluded directories: diagnostics and other authors' studies that the paper does not sum
    (ace*, oracle, gate04, observers, saturation, sweep, joint_followup, composite_*, descriptor,
    mimo, norm_channels, joint_map, jointmap, hardware, suites/*_n20 superseded reruns);
  * excluded files: integral/matched-baseline sweeps (arm tag 'integral' or args.baseline != none),
    estimate-only probes, aborted/partial files (name contains 'aborted' or 'partial'), the
    unpaired GR1 cohorts (files without per_ep in both arms are skipped anyway);
  * a healthy control is counted (it is a paired comparison the paper reports);
  * a corrected-protocol rerun extended to n=40 supersedes its own n=20 version (jf_rerun/X.json
    is dropped when jf_rerun/X_n40.json exists: same scenarios, counted once).
Usage: python aggregate_tally.py [--root results] [--out results/aggregate_manifest.json]
"""
import argparse, glob, json, os, pathlib, re

EXCL_DIRS = ("ace", "oracle", "gate04", "observers", "saturation", "sweep", "joint_followup",
             "composite_", "descriptor", "mimo", "norm_channels", "joint_map", "jointmap", "hardware")
CATS = [("suites", "headline four suites"), ("phase05/jf_", "faults below the controller"),
        ("phase05/abl_", "constants ablation"), ("phase05/map_", "map cells n=40"),
        ("phase05", "LIBERO pi0.5: fault families, severities, profiles, controls"),
        ("oft", "OpenVLA-OFT"), ("groot", "GR00T N1.7"), ("aloha", "ALOHA"), ("gr1", "GR1 humanoid"),
        ("widowx", "WidowX / SimplerEnv"), ("gate", "healthy-phantom channel gate"),
        ("heldout", "held-out calibration"), ("jf_rerun", "joint-level reruns, corrected fault protocol"),
        ("google", "SimplerEnv Google robot")]

def category(rel):
    for pre, name in CATS:
        if rel.startswith(pre):
            return name
    return "other"

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--root", default="results")
    ap.add_argument("--out", default="results/aggregate_manifest.json"); a = ap.parse_args()
    inc, exc, totals = [], [], {}
    for f in sorted(glob.glob(os.path.join(a.root, "**", "*.json"), recursive=True)):
        rel = os.path.relpath(f, a.root)
        if any(part.startswith(EXCL_DIRS) for part in rel.split(os.sep)[:-1]):
            exc.append(dict(file=rel, why="excluded directory")); continue
        if re.search(r"aborted|partial|probe|estimate", os.path.basename(rel)):
            exc.append(dict(file=rel, why="probe / aborted / partial")); continue
        try:
            d = json.load(open(f))
        except Exception:
            continue
        arms = d.get("arms") if isinstance(d, dict) else None
        if not arms or "frozen_faulted" not in arms or "adaptive" not in arms:
            continue
        pa, pb = arms["frozen_faulted"].get("per_ep"), arms["adaptive"].get("per_ep")
        if not pa or not pb:
            exc.append(dict(file=rel, why="no per-episode outcomes (unpaired)")); continue
        args = d.get("args") or {}
        if args.get("baseline") not in (None, "none") or args.get("estimate_only"):
            exc.append(dict(file=rel, why=f"baseline={args.get('baseline')} estimate_only={args.get('estimate_only')}")); continue
        ka = {(e["task"], e["init"]): e["ok"] for e in pa}; kb = {(e["task"], e["init"]): e["ok"] for e in pb}
        if len(ka) != len(pa) or len(kb) != len(pb):
            exc.append(dict(file=rel, why="duplicate scenario keys")); continue
        keys = sorted(set(ka) & set(kb))
        fixed = sum(kb[k] and not ka[k] for k in keys); broken = sum(ka[k] and not kb[k] for k in keys)
        cat = category(rel)
        inc.append(dict(file=rel, category=cat, n=len(keys), frozen=sum(ka[k] for k in keys),
                        corrected=sum(kb[k] for k in keys), fixed=fixed, broken=broken))
        t = totals.setdefault(cat, dict(files=0, n=0, fixed=0, broken=0))
        t["files"] += 1; t["n"] += len(keys); t["fixed"] += fixed; t["broken"] += broken
    # superseded n=20 headline reruns: drop when an n=40 file for the same suite exists
    n40 = {i["file"] for i in inc if i["file"].startswith("suites") and "_n40" in i["file"]}
    for i in list(inc):
        if i["file"].startswith("suites") and "_n40" not in i["file"]:
            suite = re.sub(r"_rotonly.*", "", os.path.basename(i["file"]))
            if any(suite in x for x in n40):
                inc.remove(i); exc.append(dict(file=i["file"], why="superseded by the n=40 rerun"))
                t = totals[i["category"]]; t["files"] -= 1; t["n"] -= i["n"]; t["fixed"] -= i["fixed"]; t["broken"] -= i["broken"]
    names = {i["file"] for i in inc}
    for i in list(inc):
        if i["file"].startswith("jf_rerun") and not i["file"].endswith("_n40.json") \
                and i["file"].replace(".json", "_n40.json") in names:
            inc.remove(i); exc.append(dict(file=i["file"], why="superseded by its n=40 extension"))
            t = totals[i["category"]]; t["files"] -= 1; t["n"] -= i["n"]; t["fixed"] -= i["fixed"]; t["broken"] -= i["broken"]
    grand = dict(files=len(inc), n=sum(i["n"] for i in inc), fixed=sum(i["fixed"] for i in inc), broken=sum(i["broken"] for i in inc),
                 frozen_successes=sum(i["frozen"] for i in inc))
    # audit finding 1.4: a regression can only occur on an episode the frozen policy was winning
    grand["regression_rate_all"] = grand["broken"] / max(grand["n"], 1)
    grand["regression_rate_vs_frozen_successes"] = grand["broken"] / max(grand["frozen_successes"], 1)
    pathlib.Path(a.out).write_text(json.dumps(dict(rules=__doc__, included=inc, excluded=exc, totals=totals, grand=grand), indent=1))
    print(f"{'category':60s} files      n  fixed broken")
    for c, t in sorted(totals.items(), key=lambda kv: -kv[1]['n']):
        print(f"{c:60s} {t['files']:5d} {t['n']:6d} {t['fixed']:6d} {t['broken']:6d}")
    print(f"{'ALL':60s} {grand['files']:5d} {grand['n']:6d} {grand['fixed']:6d} {grand['broken']:6d}")
    print(f"regression rate: {grand['broken']}/{grand['n']} = {100*grand['regression_rate_all']:.1f}% of all paired episodes; "
          f"{grand['broken']}/{grand['frozen_successes']} = {100*grand['regression_rate_vs_frozen_successes']:.1f}% of episodes the frozen policy was winning")
    print(f"excluded {len(exc)} files; manifest {a.out}")

if __name__ == "__main__":
    main()
