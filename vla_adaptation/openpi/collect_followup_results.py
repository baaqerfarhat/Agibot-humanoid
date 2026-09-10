"""Compact follow-up outcomes without downloading trajectories or telemetry.

Run this on the filesystem containing the original study (including remotely via
SSH). No network requests, simulations, or statistical tests are launched here.

  python collect_followup_results.py collect RUN/study.json --out-dir COMPACT \
      --hash-telemetry
  python collect_followup_results.py manifest --preset aloha_joint \
      --summaries COMPACT/*/summary.json --registration 'lock 3 at REV' --out-dir MANIFESTS

Manifest presets require complete registered allocations. An explicit --plan
supports other predeclared Panda cells/families; see --example-plan. Incomplete
collection is diagnostic-only, requires an explicit flag, and emits no scoreable
views or manifest. Original bytes and full-state hashes remain traceable.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import sys


ALOHA_TASK = "gym_aloha/AlohaTransferCube-v0"
COMPOSITE_ARMS = ["off", "legacy_all", "legacy_corrected", "dob", "kalman",
                  "composite_001", "composite_005", "composite_010", "composite_025"]
PHYSICAL_ARMS = ["off", "legacy_allchannels", "legacy_correctedchannels", "kalman"]
PANDA_ARMS = ["off", "legacy_translation", "legacy_full", "legacy_rotation",
              "dob_translation", "legacy_half", "legacy_hold30"]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def safe_name(value):
    require(isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", value),
            "unsafe or empty output identifier: %r" % value)
    return value


def write_new(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def read_snapshot(path):
    """Hash and parse the same open file, even if an atomic writer replaces its path."""
    path = Path(path).resolve()
    with path.open("rb") as stream:
        before = os.fstat(stream.fileno())
        digest = hashlib.sha256()
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
        stream.seek(0)
        value = json.load(stream)
        after = os.fstat(stream.fileno())
    require((before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns),
            "source was modified in place during collection")
    return value, dict(path=str(path), sha256=digest.hexdigest(), bytes=before.st_size)


def telemetry_source(path, hash_contents=False, allow_changing=False):
    record = dict(path=str(Path(path).resolve()), hashed=False)
    if not hash_contents:
        return record
    with Path(path).open("rb") as stream:
        before = os.fstat(stream.fileno())
        remaining = before.st_size
        digest = hashlib.sha256()
        while remaining:
            chunk = stream.read(min(1024 * 1024, remaining))
            require(chunk, "telemetry shrank while being hashed")
            digest.update(chunk)
            remaining -= len(chunk)
        after = os.fstat(stream.fileno())
    unchanged = (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns)
    require(unchanged or allow_changing, "completed telemetry changed while being hashed")
    record.update(hashed=True, sha256=digest.hexdigest(), bytes=before.st_size,
                  hash_scope="whole_file" if unchanged else "prefix_at_collection_start",
                  unchanged_during_hash=unchanged)
    return record


def trim_state(value):
    """Preserve state fingerprints/checks; arrays are recoverable via source-study SHA."""
    if isinstance(value, dict):
        # Robot metadata also uses qpos for the scalar model address. It is not
        # a captured state array and must remain available for joint mapping.
        removed = [key for key in ("qpos", "qvel") if isinstance(value.get(key), list)]
        out = {key: trim_state(item) for key, item in value.items()
               if key not in removed}
        if removed:
            require(isinstance(value.get("sha256"), str)
                    and re.fullmatch(r"[0-9a-f]{64}", value["sha256"]),
                    "cannot remove initial-state arrays without their recorded SHA-256")
            out["omitted_arrays"] = {key: len(value[key]) for key in removed}
        return out
    if isinstance(value, list):
        return [trim_state(item) for item in value]
    return value


def layouts(study):
    if "conditions" in study:
        return "aloha_composite", study["conditions"]
    layout = "aloha_joint" if "robot_joints" in study else "panda"
    condition = study["args"].get("condition", "default")
    return layout, {condition: dict(arms=study["arms"],
                                   shared_control_id=study["shared_control_id"])}


def validate(study, complete):
    require(isinstance(study.get("study_id"), str), "study_id required")
    require(isinstance(study.get("args"), dict), "recorded args required")
    require(isinstance(study.get("arm_configs"), dict), "recorded arm_configs required")
    require(isinstance(study.get("pairing"), dict), "recorded pairing required")
    layout, cells = layouts(study)
    count = study["args"].get("episodes")
    require(type(count) is int and count > 0, "positive planned episode count required")
    require("off" in study["arm_configs"], "shared off configuration missing")
    checks = 0
    for condition, cell in cells.items():
        safe_name(condition)
        require(set(cell["arms"]) == set(study["arm_configs"]), "arm allocation differs from recorded configuration")
        require(isinstance(cell.get("shared_control_id"), str), "shared control ID missing")
        expected = {(row["task"], row["init"]) for row in study["schedule"]
                    if layout != "aloha_composite" or row["condition"] == condition}
        require(len(expected) == count, "schedule disagrees with planned scenario count")
        for name, arm in cell["arms"].items():
            safe_name(name)
            rows = arm.get("per_ep")
            require(isinstance(rows, list), "per-episode outcomes required")
            keys = []
            wins = 0
            for row in rows:
                require(type(row.get("task")) is int and type(row.get("init")) is int,
                        "integer task/init identifiers required")
                key = (row["task"], row["init"])
                require(key in expected, "unexpected scenario in outcomes")
                keys.append(key)
                require(type(row.get("ok")) in (bool, int) and row["ok"] in (0, 1),
                        "boolean episode outcome required")
                wins += int(row["ok"])
                state = row.get("initial_state", {})
                require(isinstance(state.get("sha256"), str)
                        and re.fullmatch(r"[0-9a-f]{64}", state["sha256"]),
                        "recorded initial-state hash required")
                if complete:
                    require(state.get("pairing_valid") is True, "episode failed the physical pairing check")
                if layout.startswith("aloha"):
                    seed = study["args"].get("seed")
                    require(type(seed) is int and row["task"] == 0 and 0 <= row["init"] < count,
                            "ALOHA requires a base seed and local episode indices")
                    if "actual_seed" in row:
                        require(row["actual_seed"] == seed + row["init"], "recorded actual seed disagrees")
            require(len(keys) == len(set(keys)), "duplicate scenario in one arm")
            require(type(arm.get("n")) is int and arm["n"] == len(rows)
                    and type(arm.get("successes")) is int and arm["successes"] == wins,
                    "arm counts disagree with per-episode outcomes")
            if complete:
                require(set(keys) == expected, "complete study has missing arm/scenario outcomes")
            checks += len(rows)
    if complete:
        require(study["pairing"].get("valid_so_far") is True, "study failed physical pairing")
        require(study["pairing"].get("checked_states") == checks, "pairing check count disagrees with outcomes")
    return layout, cells


def view_args(study, layout, condition, cell, name):
    args = dict(study["args"], **study["arm_configs"][name])
    args.update(followup_arm=name, static_corr=None, freeze_after=args.get("freeze_after"))
    if layout.startswith("aloha"):
        args.update(condition=condition, corr_joints=",".join(map(str, args["corr"])),
                    warm_start=False, identify_episodes=None)
        if layout == "aloha_composite":
            args.update(fault_vec=",".join(str(float(x)) for x in cell["fault_vec"]), gain=None)
    else:
        args.update(corr_dims=",".join(map(str, args["corr_dims"])), estimate_only=False)
    return args


def collect(path, out_dir, *, telemetry=None, hash_telemetry=False, allow_incomplete=False):
    study, source = read_snapshot(path)
    complete = study.get("status") == "complete"
    require(complete or allow_incomplete, "study is incomplete; use --allow-incomplete only for diagnostics")
    layout, cells = validate(study, complete)
    telemetry_path = telemetry or study["args"].get("telemetry")
    require(telemetry_path or not hash_telemetry, "telemetry path required to hash telemetry")
    tel = (telemetry_source(telemetry_path, hash_telemetry, allow_incomplete and not complete)
           if telemetry_path else None)
    base = {key: trim_state(value) for key, value in study.items()
            if key not in ("arms", "conditions", "schedule")}
    base.update(schema_version=1, compact_schema_version=1, layout=layout, source_study=source,
                source_telemetry=tel, scoreable=complete, cells={},
                collector_source=dict(path=str(Path(__file__).resolve()),
                    sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),
                omission="Estimate/trajectory arrays and full qpos/qvel omitted; retain source study/telemetry by hash.")
    views = {}
    for condition, cell in cells.items():
        compact = {name: {key: trim_state(arm[key]) for key in ("n", "successes", "per_ep", "diagnostics")
                          if key in arm} for name, arm in cell["arms"].items()}
        entry = dict(shared_control_id=cell["shared_control_id"], arms=compact, views={})
        if "fault_vec" in cell:
            entry["fault_vec"] = cell["fault_vec"]
        for name in cell["arms"]:
            if name == "off" or not complete:
                continue
            filename = (f"{condition}_{name}.json" if layout == "aloha_composite" else f"{name}.json")
            entry["views"][name] = filename
            view = {key: value for key, value in base.items() if key not in ("cells", "arm_configs")}
            view.update(args=view_args(study, layout, condition, cell, name),
                        shared_control_id=cell["shared_control_id"],
                        shared_control_source=f"{source['path']}:" +
                            (f"conditions.{condition}.arms.off" if layout == "aloha_composite" else "arms.off"),
                        arms=dict(frozen_faulted=compact["off"], adaptive=compact[name]))
            views[filename] = view
        base["cells"][condition] = entry
    out_dir = Path(out_dir)
    require(not out_dir.exists(), "output directory exists; choose a fresh compact output path")
    out_dir.mkdir(parents=True, exist_ok=False)
    for name, view in views.items():
        write_new(out_dir / name, view)
    # Publish summary last. Manifests consume only this completed collection index.
    write_new(out_dir / "summary.json", base)
    return base


def _load_summary(path):
    summary, source = read_snapshot(path)
    require(summary.get("compact_schema_version") == 1, "collector summary required")
    require(summary.get("scoreable") is True and summary.get("status") == "complete",
            "no scoring manifest for an incomplete/diagnostic collection")
    require(summary.get("pairing", {}).get("valid_so_far") is True, "pairing check failed")
    return summary, source


def example_plan():
    return dict(schema_version=1, study="declared_panda_screen", stage="development",
        suite="libero_spatial", pairing="libero_set_init_state", registration="lock at REV",
        expected_keys=[[t, 30] for t in range(10)], forbidden_keys=[[0, 25]],
        cells=[dict(id="joint5", summary="compact/summary.json", condition="default",
                    expected_args=dict(suite="libero_spatial", joint_fault="torque:5:5.0",
                                       episodes=10, gamma=.08, dead=.008, norm_r=.15, clip=.3),
                    expected_arms=PANDA_ARMS)],
        families=[dict(id="candidate_vs_off", comparisons=[dict(id=f"joint5_{name}_vs_off",
                          cell="joint5", left="off", right=name) for name in PANDA_ARMS[1:]]),
                  dict(id="alternative_vs_legacy", comparisons=[dict(id=f"joint5_{name}_vs_legacy",
                          cell="joint5", left="legacy_translation", right=name) for name in PANDA_ARMS[2:]])])


def preset_plan(preset, summary_paths, registration):
    """Fixed allocation and constants, independent of successes or observed winners."""
    require(registration, "registration and revision are required")
    summaries = [(Path(path).resolve(), _load_summary(path)[0]) for path in summary_paths]
    require(len({str(path) for path, _ in summaries}) == len(summaries), "duplicate summary input")
    common = dict(episodes=10, gamma=.08, dead=.002, norm_r=.4, clip=.08,
                  law="legacy", deadzone_mode="zero", damping=0., profile="step", onset=0)
    if preset == "aloha_composite":
        require(len(summaries) == 1, "composite allocation requires one complete two-condition study")
        path, summary = summaries[0]
        require(summary["layout"] == "aloha_composite" and set(summary["cells"]) == {"healthy", "offset"},
                "both registered composite conditions are required")
        seed, arms, baseline = 2700, COMPOSITE_ARMS, "legacy_all"
        sources = [(condition, path, summary, condition) for condition in ("healthy", "offset")]
        common.update(seed=seed, corr=list(range(6)))
        arm_args = {
            "legacy_all": dict(baseline="none", norm_channels="all", tracking_rate=0., tracking_strength=0.),
            "legacy_corrected": dict(baseline="none", norm_channels="corrected", tracking_rate=0., tracking_strength=0.),
            "dob": dict(baseline="dob", norm_channels="all", tracking_rate=0., tracking_strength=0.),
            "kalman": dict(baseline="kalman", norm_channels="all", tracking_rate=0., tracking_strength=0.),
        }
        for name, strength, rate in (("composite_001", .01, 686923.6043281124),
                ("composite_005", .05, 3434618.021640562),
                ("composite_010", .1, 6869236.043281124),
                ("composite_025", .25, 17173090.10820281)):
            arm_args[name] = dict(baseline="composite", norm_channels="all",
                                  tracking_strength=strength, tracking_rate=rate)
    elif preset == "aloha_joint":
        expected = {"healthy"} | {f"joint{i}" for i in range(12)}
        require(len(summaries) == 13, "physical ALOHA allocation requires all thirteen complete cells")
        sources = []
        for path, summary in summaries:
            require(summary["layout"] == "aloha_joint" and len(summary["cells"]) == 1,
                    "physical ALOHA cell summary required")
            condition = next(iter(summary["cells"]))
            sources.append((condition, path, summary, condition))
        require({row[0] for row in sources} == expected, "physical ALOHA cells are missing or duplicated")
        seed, arms, baseline = 2900, PHYSICAL_ARMS, "legacy_allchannels"
        common.update(seed=seed, corr=list(range(6)) + list(range(7, 13)), tracking_rate=0.)
        arm_args = {
            "legacy_allchannels": dict(baseline="none", norm_channels="all"),
            "legacy_correctedchannels": dict(baseline="none", norm_channels="corrected"),
            "kalman": dict(baseline="kalman", norm_channels="all"),
        }
    else:
        raise ValueError("unknown manifest preset")
    plan = dict(schema_version=1, study=preset + "_development", stage="development", suite=ALOHA_TASK,
                pairing="aloha_seeded_reset", registration=registration,
                expected_keys=[[0, seed+i] for i in range(10)],
                forbidden_keys=[[0, i] for i in list(range(2600, 2610)) + [2500, 2501, 2690]],
                cells=[], families=[dict(id="candidate_vs_off", comparisons=[]),
                                    dict(id="alternative_vs_legacy", comparisons=[])])
    if preset == "aloha_composite":
        plan["families"].append(dict(id="composite_vs_kalman", comparisons=[]))
    for cid, path, summary, condition in sorted(sources):
        require(summary.get("stage") in ("dev", "development"), "development preset cannot relabel another stage")
        require(set(summary["cells"][condition]["arms"]) == set(arms), "registered arm allocation differs")
        expected_args = dict(common, condition=condition)
        if preset == "aloha_joint":
            torque = None if condition == "healthy" else [16.,32.,16.,.2,1.,.4][int(condition[5:]) % 6]
            expected_args["joint_fault"] = None if torque is None else f"torque:{int(condition[5:])}:{torque}"
        else:
            expected_args["fault_vec"] = ",".join(str(x) for x in
                ([0.]*14 if condition == "healthy" else [0.02]*6+[0.]*8))
        plan["cells"].append(dict(id=cid, summary=str(path), condition=condition,
                                  expected_args=expected_args, expected_arms=arms,
                                  expected_arm_args=arm_args))
        for name in arms[1:]:
            plan["families"][0]["comparisons"].append(dict(id=f"{cid}_{name}_vs_off", cell=cid, left="off", right=name))
            if name != baseline:
                plan["families"][1]["comparisons"].append(dict(id=f"{cid}_{name}_vs_legacy", cell=cid, left=baseline, right=name))
            if name.startswith("composite_"):
                plan["families"][2]["comparisons"].append(dict(id=f"{cid}_{name}_vs_kalman", cell=cid, left="kalman", right=name))
    return plan


def generate_manifests(plan, out_dir, *, plan_directory=Path(".")):
    """Resolve every declared cell and comparison before writing any manifest."""
    require(plan.get("schema_version") == 1 and plan.get("registration"), "explicit versioned registered plan required")
    require(plan.get("stage") in ("development", "confirmation"), "declared analysis stage required")
    if plan["stage"] == "confirmation":
        require(plan.get("locked_before_outcomes") is True and plan.get("selection_record")
                and plan.get("forbidden_keys"), "confirmation requires a prior lock, selection record, and forbidden keys")
    require(plan.get("cells") and plan.get("families"), "explicit cells and families required")
    expected_keys = {tuple(key) for key in plan["expected_keys"]}
    require(len(expected_keys) == len(plan["expected_keys"]) and expected_keys,
            "nonempty distinct planned scenarios required")
    require(not expected_keys.intersection(tuple(key) for key in plan.get("forbidden_keys", [])),
            "planned scenarios overlap forbidden calibration/development scenarios")
    out_dir = Path(out_dir).resolve()
    cells = {}
    for spec in plan["cells"]:
        cid = safe_name(spec["id"])
        require(cid not in cells, "duplicate declared cell")
        path = (Path(plan_directory) / spec["summary"]).resolve()
        summary, source = _load_summary(path)
        recorded_stage = {"dev": "development", "confirm": "confirmation"}.get(
            summary.get("stage"), summary.get("stage"))
        require(recorded_stage == plan["stage"], "manifest stage differs from the recorded study stage")
        require(summary["pairing"]["mechanism"] == plan["pairing"], "declared pairing mechanism differs")
        condition = spec["condition"]
        require(condition in summary["cells"], "declared condition missing from collection")
        cell = summary["cells"][condition]
        require(set(spec["expected_arms"]) == set(cell["arms"]), "expected arm allocation differs")
        require(spec.get("expected_args"), "cell configuration must be declared explicitly")
        records = {}
        for name, relative in cell["views"].items():
            view_path = (path.parent / relative).resolve()
            record, view_source = read_snapshot(view_path)
            require(record.get("source_study") == summary["source_study"], "view belongs to another source study")
            require(record.get("shared_control_id") == cell["shared_control_id"], "view uses another control cohort")
            require(record.get("status") == "complete" and record.get("scoreable") is True,
                    "view is not complete and scoreable")
            require(record["arms"]["frozen_faulted"] == cell["arms"]["off"]
                    and record["arms"]["adaptive"] == cell["arms"][name], "view outcome differs from authoritative collection")
            for key, value in spec["expected_args"].items():
                require(record["args"].get(key) == value, f"{cid}/{name}: expected args.{key}={value!r}")
            for key, value in spec.get("expected_arm_args", {}).get(name, {}).items():
                require(record["args"].get(key) == value, f"{cid}/{name}: expected arm args.{key}={value!r}")
            for arm in record["arms"].values():
                observed_keys = {(row["task"], row["init"] + record["args"]["seed"]
                                  if plan["pairing"] == "aloha_seeded_reset" else row["init"])
                                 for row in arm["per_ep"]}
                require(observed_keys == expected_keys, "outcomes differ from explicitly planned scenarios")
            records[name] = (view_path, record, view_source)
        require(set(records) == set(spec["expected_arms"]) - {"off"}, "one view per declared candidate required")
        cells[cid] = dict(spec=spec, summary=summary, source=source, records=records)
    outputs = {}
    family_names = set()
    for family in plan["families"]:
        fid = safe_name(family["id"])
        require(fid not in family_names, "duplicate family")
        family_names.add(fid)
        manifest = {key: copy.deepcopy(plan[key]) for key in ("schema_version", "study", "stage", "suite",
                    "pairing", "registration", "expected_keys", "forbidden_keys", "selection_record",
                    "locked_before_outcomes") if key in plan}
        manifest.update(runs=[], comparisons=[], collection_sources=[cell["source"] for cell in cells.values()])
        runs, pairs = set(), set()
        for contrast in family["comparisons"]:
            cid = contrast["cell"]
            require(cid in cells, "contrast names an undeclared cell")
            cell = cells[cid]
            left, right = contrast["left"], contrast["right"]
            require(left != right and left in cell["spec"]["expected_arms"]
                    and right in cell["spec"]["expected_arms"], "invalid contrast arms")
            pair = (cid, tuple(sorted((left, right))))
            require(pair not in pairs, "duplicate or reversed comparison within family")
            pairs.add(pair)
            refs = []
            for name in (left, right):
                owner = (right if left == "off" else left) if name == "off" else name
                rid = f"{cid}_{owner}"
                path, record, source = cell["records"][owner]
                if rid not in runs:
                    runs.add(rid)
                    # Bind the effective recorded arm configuration as well as the
                    # independently declared common cell settings.
                    expected = dict(cell["spec"]["expected_args"])
                    expected.update({key: record["args"][key] for key in
                        ("baseline", "norm_channels", "followup_arm", "corr_dims", "corr_joints",
                         "tracking_rate", "tracking_strength", "correction_scale", "freeze_after")
                        if key in record["args"]})
                    manifest["runs"].append(dict(id=rid, cell=cid,
                        path=os.path.relpath(path, out_dir), expected_args=expected,
                        collected_view_sha256=source["sha256"], source_study=record["source_study"],
                        shared_control_id=record["shared_control_id"]))
                refs.append(dict(run=rid, arm="frozen_faulted" if name == "off" else "adaptive"))
            manifest["comparisons"].append(dict(id=contrast["id"], family=fid, left=refs[0], right=refs[1]))
        require(manifest["comparisons"], "empty comparison family")
        outputs[fid + ".json"] = manifest
    require(not out_dir.exists(), "manifest output directory already exists")
    out_dir.mkdir(parents=True, exist_ok=False)
    for name, manifest in outputs.items():
        write_new(out_dir / name, manifest)
    return outputs


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--example-plan", action="store_true")
    sub = parser.add_subparsers(dest="command")
    cp = sub.add_parser("collect")
    cp.add_argument("study", type=Path)
    cp.add_argument("--out-dir", type=Path, required=True)
    cp.add_argument("--telemetry", type=Path)
    cp.add_argument("--hash-telemetry", action="store_true")
    cp.add_argument("--allow-incomplete", action="store_true")
    mp = sub.add_parser("manifest")
    selection = mp.add_mutually_exclusive_group(required=True)
    selection.add_argument("--preset", choices=("aloha_composite", "aloha_joint"))
    selection.add_argument("--plan", type=Path)
    mp.add_argument("--summaries", type=Path, nargs="+")
    mp.add_argument("--registration")
    mp.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.example_plan:
        print(json.dumps(example_plan(), indent=2))
        return 0
    try:
        if args.command == "collect":
            summary = collect(args.study, args.out_dir, telemetry=args.telemetry,
                              hash_telemetry=args.hash_telemetry, allow_incomplete=args.allow_incomplete)
            print(json.dumps(dict(status=summary["status"], scoreable=summary["scoreable"],
                source_sha256=summary["source_study"]["sha256"], cells={cid: {name:
                    dict(n=arm["n"], successes=arm["successes"]) for name, arm in cell["arms"].items()}
                    for cid, cell in summary["cells"].items()})))
        elif args.command == "manifest":
            if args.plan:
                plan = json.loads(args.plan.read_text())
                directory = args.plan.resolve().parent
            else:
                require(args.summaries, "preset requires explicit summary paths")
                plan = preset_plan(args.preset, args.summaries, args.registration)
                directory = Path(".")
            manifests = generate_manifests(plan, args.out_dir, plan_directory=directory)
            print(json.dumps({name: len(value["comparisons"]) for name, value in manifests.items()}))
        else:
            parser.error("collect or manifest subcommand required")
    except (ValueError, OSError, KeyError) as error:
        print("No usable collection/manifest produced: " + str(error), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
