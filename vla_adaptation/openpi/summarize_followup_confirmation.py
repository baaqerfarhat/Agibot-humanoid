"""Validate complete confirmation allocations, summarize outcomes and export figures.

Panda supports exactly the registered three-suite, all-seven-joint confirmation.
Only its prespecified joint5 candidate-versus-legacy contrast receives a new pooled
test. The joint5 candidate-versus-off aggregate is descriptive. Other fault cells
never become additional independent scenarios in a pooled test. ALOHA supports
complete healthy-plus-twelve-joint per-cell tables without pooled testing.

No simulation, policy query, directory scan, outcome-dependent selection or partial
efficacy report is performed. Original raw studies/telemetry remain traceable by
their collector-recorded hashes; this tool does not re-download those raw files.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import pathlib
import re

try:
    from . import score_joint_followup as scoring
except ImportError:
    import score_joint_followup as scoring

PANDA_SUITES = ("libero_spatial", "libero_object", "libero_goal")
PANDA_ARMS = ("off", "legacy_translation", "weighted_full")
PANDA_FAMILIES = {"legacy_vs_off": 24, "candidate_vs_off": 24, "candidate_vs_legacy": 24}
PRIMARY = "primary_joint5_candidate_vs_legacy"
SECONDARY = "secondary_joint5_candidate_vs_off"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def valid_hash(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def condition(args, robot):
    value = args.get("joint_fault")
    if robot == "aloha" and value is None:
        return "healthy", None, 0.
    require(isinstance(value, str), "explicit physical torque condition required")
    fields = value.split(":")
    require(len(fields) == 3 and fields[0] == "torque", "only declared physical torque cells supported")
    try:
        joint, torque = int(fields[1]), float(fields[2])
    except ValueError as error:
        raise ValueError("invalid joint torque condition") from error
    require(0 <= joint < (7 if robot == "panda" else 12), "joint outside the robot's physical arm indices")
    require(torque == torque and abs(torque) != float("inf"), "finite torque required")
    if robot == "panda":
        require(torque in (0., 5.), "Panda confirmation only permits healthy or registered +5Nm faults")
        require(torque != 0. or joint == 0, "healthy Panda cell must use the registered joint0 zero torque")
    return ("healthy", None, 0.) if torque == 0. else (f"joint{joint}", joint, torque)


def validate_allocation(manifest):
    """Reject changed/missing allocations before reading any outcome file."""
    require(manifest.get("stage") == "confirmation" and manifest.get("locked_before_outcomes") is True,
            "a locked confirmation manifest is required")
    require(manifest.get("selection_record") and manifest.get("registration"), "selection and registration records required")
    robot = "aloha" if manifest.get("pairing") == "aloha_seeded_reset" else "panda"
    runs = manifest.get("runs", [])
    require(runs and manifest.get("comparisons"), "complete declared runs and contrasts required")
    specs = {}
    cells = defaultdict(dict)
    for spec in runs:
        rid = spec.get("id")
        require(isinstance(rid, str) and rid not in specs, "distinct declared run IDs required")
        args = spec.get("expected_args", {})
        name = args.get("followup_arm")
        require(isinstance(name, str) and name != "off", "each compact view must name its candidate arm")
        suite = spec.get("suite", manifest.get("suite"))
        label, joint, torque = condition(args, robot)
        key = (suite, label)
        require(name not in cells[key], "duplicate arm view for one suite/condition")
        require(spec.get("expected_source_hashes") or valid_hash(spec.get("collected_view_sha256")),
                "each view needs declared source hashes or a declared compact-view hash")
        if spec.get("expected_source_hashes"):
            require(all(valid_hash(v) for v in spec["expected_source_hashes"].values()), "invalid expected source hash")
        item = dict(spec=spec, suite=suite, condition=label, joint=joint, torque=torque, arm=name)
        cells[key][name] = item
        specs[rid] = item
    if robot == "panda":
        require(manifest.get("pairing") == "libero_set_init_state", "Panda physical pairing declaration required")
        require(set(manifest.get("suites", [])) == set(PANDA_SUITES), "all three registered Panda suites required")
        expected_cells = {(suite, label) for suite in PANDA_SUITES
                          for label in ["healthy"]+[f"joint{j}" for j in range(7)]}
        require(set(cells) == expected_cells and len(runs) == 48, "all24 Panda cells and48 compact views required")
        require(set(map(tuple, manifest.get("expected_keys", []))) == {(t, i) for t in range(10) for i in (35, 36)},
                "Panda confirmation requires all tasks0..9 at untouched inits35/36")
        require(manifest.get("family_sizes") == PANDA_FAMILIES, "three global comparison families of24 required")
        require(len(manifest["comparisons"]) == 72, "all72 registered Panda cell contrasts required")
        for arms in cells.values():
            require(set(arms) == set(PANDA_ARMS[1:]), "each Panda cell requires legacy_translation and weighted_full")
            for name, item in arms.items():
                args = item["spec"]["expected_args"]
                expected_baseline = "weighted_dob" if name == "weighted_full" else "none"
                expected_dims = "0,1,2,3,4,5" if name == "weighted_full" else "0,1,2"
                require(args.get("baseline") == expected_baseline and args.get("corr_dims") == expected_dims,
                        "Panda comparison arms differ from the selected masks/observers")
        declarations = manifest.get("planned_aggregates", [])
        require(len(declarations) == 2 and {d.get("id") for d in declarations} == {PRIMARY, SECONDARY},
                "only the two registered joint5 pooled summaries are permitted")
    else:
        require(not manifest.get("planned_aggregates"), "ALOHA mode produces per-cell summaries only; no pooled tests")
        require(len(cells) == 13 and {key[1] for key in cells} == {"healthy"}|{f"joint{j}" for j in range(12)},
                "all twelve ALOHA joints and the healthy cell are required")
        require(len({tuple(sorted(views)) for views in cells.values()}) == 1,
                "ALOHA arm allocation must be consistent across all declared cells")
    by_id = {}
    observed = Counter()
    for contrast in manifest["comparisons"]:
        require(contrast.get("id") not in by_id, "duplicate contrast identifier")
        by_id[contrast["id"]] = contrast
        left, right = contrast["left"], contrast["right"]
        require(left.get("run") in specs and right.get("run") in specs, "unknown contrast view")
        lhs, rhs = specs[left["run"]], specs[right["run"]]
        require((lhs["suite"], lhs["condition"]) == (rhs["suite"], rhs["condition"]), "cross-condition contrast forbidden")
        require(left.get("arm") in ("adaptive", "frozen_faulted") and right.get("arm") in ("adaptive", "frozen_faulted"),
                "unsupported compact-view arm")
        left_name = lhs["arm"] if left["arm"] == "adaptive" else "off"
        right_name = rhs["arm"] if right["arm"] == "adaptive" else "off"
        if robot == "panda":
            allowed = {"legacy_vs_off": ("off", "legacy_translation"),
                       "candidate_vs_off": ("off", "weighted_full"),
                       "candidate_vs_legacy": ("legacy_translation", "weighted_full")}
            require(allowed.get(contrast.get("family")) == (left_name, right_name), "changed registered contrast direction or arms")
        observed[(lhs["suite"], lhs["condition"], contrast["family"])] += 1
    if robot == "panda":
        require(set(observed) == {(s, c, f) for s, c in cells for f in PANDA_FAMILIES}
                and all(n == 1 for n in observed.values()), "every cell must appear once in each global family")
        for aggregate in manifest["planned_aggregates"]:
            primary = aggregate["id"] == PRIMARY
            require(aggregate.get("role") == ("prespecified_primary" if primary else "descriptive_secondary"),
                    "aggregate role differs from registration")
            require(aggregate.get("expected_scenarios") == 60 and aggregate.get("scenario_identity") == ["suite", "task", "init"],
                    "aggregate must preserve60 unique suite/task/init scenarios")
            ids = aggregate.get("comparison_ids", [])
            require(len(ids) == 3 and len(set(ids)) == 3 and all(i in by_id for i in ids), "three distinct declared aggregate contrasts required")
            selected = [by_id[i] for i in ids]
            family = "candidate_vs_legacy" if primary else "candidate_vs_off"
            require(all(c["family"] == family for c in selected), "wrong aggregate comparison family")
            identities = {(specs[c["right"]["run"]]["suite"], specs[c["right"]["run"]]["condition"]) for c in selected}
            require(identities == {(s, "joint5") for s in PANDA_SUITES}, "only disjoint suites of the joint5 condition may be pooled")
            if primary:
                require(aggregate.get("family_tests") == 1 and aggregate.get("family") == PRIMARY,
                        "one prespecified pooled primary test is permitted")
    return robot, specs, cells


def validate_compact(record, spec, raw):
    require(record.get("compact_schema_version") == 1 and record.get("scoreable") is True
            and record.get("status") == "complete", "complete scoreable collector views required")
    require(record.get("stage") in ("confirm", "confirmation"), "compact view was not collected from confirmation")
    for name in ("source_study", "collector_source"):
        source = record.get(name, {})
        require(source.get("path") and valid_hash(source.get("sha256")), f"{name} provenance missing")
    telemetry = record.get("source_telemetry", {})
    require(telemetry.get("hashed") is True and telemetry.get("hash_scope") == "whole_file"
            and telemetry.get("unchanged_during_hash") is True and valid_hash(telemetry.get("sha256")),
            "complete unchanged telemetry SHA256 required")
    if "collected_view_sha256" in spec:
        require(digest(raw) == spec["collected_view_sha256"], "collected compact-view hash mismatch")
    pairing = record.get("pairing", {})
    require(pairing.get("valid_so_far") is True, "recorded physical pairing failed")
    require(pairing.get("mechanism") in ("libero_set_init_state", "aloha_seeded_reset"), "physical pairing mechanism missing")
    tolerance = pairing.get("tolerance", record["args"].get("state_tolerance", 0.))
    for arm in record["arms"].values():
        for row in arm["per_ep"]:
            state = row.get("initial_state", {})
            require(valid_hash(state.get("sha256")) and state.get("pairing_valid") is True,
                    "episode physical-state hash or pairing check missing")
            for field in ("max_abs_qpos", "max_abs_qvel", "nonstate_physics_mismatch"):
                if field in state:
                    require(0 <= state[field] <= (0 if field == "nonstate_physics_mismatch" else tolerance),
                            "episode physical-state difference exceeds recorded tolerance")


def descriptive(left, right):
    require(left and set(left) == set(right), "matched nonempty descriptive scenario sets required")
    lost = sum(left[k] and not right[k] for k in left)
    gained = sum(right[k] and not left[k] for k in left)
    base, success, n = sum(left.values()), sum(right.values()), len(left)
    return dict(n=n, left_successes=base, right_successes=success,
                right_only=gained, left_only=lost, gain_points=100*(success-base)/n,
                regression_opportunities=base, regressions_conditional=lost/base if base else None,
                note="Descriptive pooled counts only; no pooled secondary p-value or significance claim.")


def summarize(manifest, directory=pathlib.Path("."), *, reader=None, scored_report=None):
    robot, specs, cells = validate_allocation(manifest)
    reader = reader or (lambda path: path.read_bytes())
    cache = {}
    def cached_read(path):
        path = pathlib.Path(path).resolve()
        if path not in cache:
            cache[path] = reader(path)
        return cache[path]
    # The unchanged core scorer validates every declared view, scenario, source hash,
    # pairing flag and comparison family before any summary is returned.
    score = scoring.score(manifest, pathlib.Path(directory), reader=cached_read)
    if scored_report is not None:
        def comparable(value):
            # Downloaded compact files may have moved from Lambda to the local
            # workspace. IDs, content hashes and all numerical results still match.
            result = dict(value)
            result["artifacts"] = [{k: v for k, v in artifact.items() if k != "path"}
                                   for artifact in value.get("artifacts", [])]
            return result
        require(comparable(scored_report) == comparable(score),
                "supplied score report does not match recomputed validated scores")
    records = {}
    for rid, item in specs.items():
        spec = item["spec"]
        raw = cached_read((pathlib.Path(directory)/spec["path"]).resolve())
        record = json.loads(raw)
        validate_compact(record, spec, raw)
        require(record["args"]["followup_arm"] == item["arm"], "compact arm identity mismatch")
        records[rid] = record
    table = []
    arm_outcomes = {}
    source_cells = []
    for (suite, label), views in sorted(cells.items(), key=lambda item:
            (item[0][0], -1 if item[0][1] == "healthy" else int(item[0][1][5:]))):
        first = records[next(iter(views.values()))["spec"]["id"]]
        off = first["arms"]["frozen_faulted"]
        n = off["n"]
        names = ["off"]+sorted(views)
        outcomes = {"off": off}
        for name, item in views.items():
            record = records[item["spec"]["id"]]
            require((record.get("study_id"), record.get("shared_control_id"), record["source_study"], record["source_telemetry"])
                    == (first.get("study_id"), first.get("shared_control_id"), first["source_study"], first["source_telemetry"]),
                    "candidate views do not share one authoritative study/control source")
            require(record["arms"]["frozen_faulted"] == off, "shared off copies disagree, including physical-state evidence")
            require(record["pairing"].get("checked_states") == n*len(names), "pairing count differs from complete cell allocation")
            outcomes[name] = record["arms"]["adaptive"]
        item = next(iter(views.values()))
        row = dict(suite=suite, condition=label, joint=item["joint"], torque_Nm=item["torque"], n=n,
                   successes={name: arm["successes"] for name, arm in outcomes.items()}, comparisons=[])
        for name, arm in outcomes.items():
            offset = first["args"]["seed"] if robot == "aloha" else 0
            arm_outcomes[(suite, label, name)] = {(suite, x["task"], x["init"]+offset): bool(x["ok"]) for x in arm["per_ep"]}
        for contrast, scored in zip(manifest["comparisons"], score["comparisons"]):
            ref = specs[contrast["right"]["run"]]
            if (ref["suite"], ref["condition"]) != (suite, label):
                continue
            def name(side):
                entry = contrast[side]
                return specs[entry["run"]]["arm"] if entry["arm"] == "adaptive" else "off"
            row["comparisons"].append(dict(left_arm=name("left"), right_arm=name("right"), **scored))
        table.append(row)
        source_cells.append(dict(suite=suite, condition=label, study_id=first["study_id"],
            source_study=first["source_study"], source_telemetry=first["source_telemetry"], shared_control_id=first["shared_control_id"]))
    aggregates = []
    if robot == "panda":
        for declaration in manifest["planned_aggregates"]:
            primary = declaration["id"] == PRIMARY
            left_name = "legacy_translation" if primary else "off"
            left, right = {}, {}
            for suite in PANDA_SUITES:
                for target, name in ((left, left_name), (right, "weighted_full")):
                    values = arm_outcomes[(suite, "joint5", name)]
                    require(not set(values).intersection(target), "pooled scenarios are not disjoint")
                    target.update(values)
            require(len(left) == len(right) == 60, "pooled joint5 comparison must contain60 unique suite/task/init keys")
            stats = scoring.compare(left, right) if primary else descriptive(left, right)
            if primary:
                stats.update(family_tests=1, p_bonferroni=stats["p"],
                    inference=("candidate_higher" if stats["gain_points"] > 0 else "legacy_higher")
                    if stats["p"] < .05 else "not_resolved")
            aggregates.append(dict(id=declaration["id"], role=declaration["role"], left_arm=left_name,
                right_arm="weighted_full", condition="joint5", torque_Nm=5., scenario_identity=["suite", "task", "init"],
                scenarios=[list(key) for key in sorted(left)], **stats))
    return dict(schema_version=1, status="complete", study=manifest["study"], robot=robot,
        registration=manifest["registration"], selection_record=manifest["selection_record"],
        cells=len(table), unique_rollout_assignments=sum(row["n"]*len(row["successes"]) for row in table),
        family_sizes=dict(Counter(row["family"] for row in score["comparisons"])),
        validation=dict(complete_allocation=True, all_declared_sources_and_views_checked=True,
            complete_telemetry_hashes=True, shared_controls_and_episode_pairing_checked=True,
            supplied_score_report_matched=scored_report is not None),
        source_cells=source_cells, artifacts=score["artifacts"], per_cell=table, aggregates=aggregates,
        caveats=["The same task/init scenarios recur across different fault cells; those cells are not pooled as independent trials.",
                 *(["Only the registered joint5 primary contrast receives a new pooled exact McNemar test.",
                    "A positive primary result does not establish improvement versus off, healthy safety, or universal repair."]
                   if robot == "panda" else ["No pooled ALOHA test was registered; all inference remains per cell."]),
                 "All cell comparisons retain their original complete global multiplicity families.",
                 "Unresolved significance is not equivalence or a no-harm certificate; policy sampling is not pinned."],
        severity_context=("Panda +5Nm is6.25% of the installed motor limits on joints0–4 and41.7% on joints5–6; "
                          "the tested absolute torques are not equal fractional actuator severity."
                          if robot == "panda" else "Each ALOHA cell's physical torque is shown explicitly in Nm; joint indices0–11 exclude grippers."),
        validated_score_report=score)


def markdown(report):
    lines = [f"{report['study']}: complete confirmation; {report['cells']} cells, "
             f"{report['unique_rollout_assignments']} unique rollout assignments.", "",
             "Cell successes use the explicit denominator shown. Fixed/broken counts compare the named arms.", "",
             "| Suite | Condition; torque | Successes | Comparison | Fixed | Broken / baseline successes | Regression upper95% | Adjusted p |",
             "|---|---|---|---|---:|---:|---:|---:|"]
    for cell in report["per_cell"]:
        counts = "; ".join(f"{k} {v}/{cell['n']}" for k, v in cell["successes"].items())
        for i, comparison in enumerate(cell["comparisons"]):
            bound = comparison["regression_upper95_one_sided"]
            upper = "undefined" if bound is None else f"{100*bound:.1f}%"
            lines.append(f"| {cell['suite'] if i == 0 else ''} | "
                f"{cell['condition']+'; '+str(cell['torque_Nm'])+' Nm' if i == 0 else ''} | {counts if i == 0 else ''} | "
                f"{comparison['left_arm']} → {comparison['right_arm']} | {comparison['right_only']} | "
                f"{comparison['left_only']}/{comparison['regression_opportunities']} | {upper} | {comparison['p_bonferroni']:.6g} |")
    for aggregate in report["aggregates"]:
        lines.extend(["", f"{aggregate['role']}: {aggregate['left_arm']} → {aggregate['right_arm']}, joint5 +5 Nm, "
            f"{aggregate['left_successes']}/{aggregate['n']} → {aggregate['right_successes']}/{aggregate['n']}; "
            f"fixed {aggregate['right_only']}, broken {aggregate['left_only']}/{aggregate['regression_opportunities']} baseline successes."])
        if aggregate["role"] == "prespecified_primary":
            lines.append(f"Exact two-sided McNemar p={aggregate['p']:.6g}; {aggregate['inference']}.")
        else:
            lines.append("Descriptive pooling only; no pooled secondary significance test.")
    lines.extend(["", "Regression bounds are one-sided95% Clopper–Pearson bounds conditional on baseline successes.",
                  report["severity_context"]]+report["caveats"])
    return "\n".join(lines)+"\n"


def plot(report, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    outputs = []
    suites = sorted({row["suite"] for row in report["per_cell"]})
    for suite in suites:
        cells = sorted([row for row in report["per_cell"] if row["suite"] == suite],
                       key=lambda row: -1 if row["joint"] is None else row["joint"])
        arms = list(cells[0]["successes"])
        require(all(set(row["successes"]) == set(arms) for row in cells), "plot arm allocations must agree")
        fig, axes = plt.subplots(2, 1, figsize=(max(11, 1.25*len(cells)), 8), sharex=True)
        x = np.arange(len(cells)); width = .8/len(arms)
        for i, arm in enumerate(arms):
            rates = [100*row["successes"][arm]/row["n"] for row in cells]
            bars = axes[0].bar(x+(i-(len(arms)-1)/2)*width, rates, width, label=arm)
            for bar, row in zip(bars, cells):
                axes[0].text(bar.get_x()+bar.get_width()/2, bar.get_height()+1,
                             f"{row['successes'][arm]}/{row['n']}", ha="center", va="bottom", fontsize=7)
        axes[0].set_ylabel("Task success (%)"); axes[0].set_ylim(0, 117)
        axes[0].legend(loc="upper center", bbox_to_anchor=(.5, 1.18), ncol=min(4, len(arms)), frameon=False)
        contrasts = [(c["left_arm"], c["right_arm"]) for c in cells[0]["comparisons"]]
        width = .8/len(contrasts)
        for i, pair in enumerate(contrasts):
            rows = [next(c for c in cell["comparisons"] if (c["left_arm"], c["right_arm"]) == pair) for cell in cells]
            positions = x+(i-(len(contrasts)-1)/2)*width
            bars = axes[1].bar(positions, [r["right_only"] for r in rows], width, label=" → ".join(pair))
            color = bars[0].get_facecolor()
            axes[1].bar(positions, [-r["left_only"] for r in rows], width, color=color, hatch="///", alpha=.7)
        axes[1].axhline(0, color="black", linewidth=.7)
        axes[1].set_ylabel("Fixed (+) / broken (−) paired scenarios")
        axes[1].legend(loc="upper center", bbox_to_anchor=(.5, 1.21), ncol=min(3, len(contrasts)), fontsize=8, frameon=False)
        axes[1].set_xticks(x)
        axes[1].set_xticklabels([f"{row['condition']}\n{row['torque_Nm']:g} Nm\nn={row['n']}" for row in cells])
        fig.suptitle(f"{suite}: complete confirmation — all declared cells", y=.99)
        fig.text(.5, .01, "Repeated task/init identities across faults are not independent pooled trials. "
                 "Hatched bars show regressions; complete multiplicity families remain in the JSON/Markdown.",
                 ha="center", fontsize=8)
        fig.tight_layout(rect=(0, .045, 1, .95))
        stem = re.sub(r"[^A-Za-z0-9_.-]", "_", suite)+"_confirmation"
        for extension in ("svg", "png"):
            output = pathlib.Path(out_dir)/(stem+"."+extension)
            fig.savefig(output, dpi=180)
            outputs.append(output)
        plt.close(fig)
    return outputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=pathlib.Path)
    parser.add_argument("--score-report", type=pathlib.Path)
    parser.add_argument("--out-dir", type=pathlib.Path, required=True)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    require(not args.out_dir.exists(), "output directory exists; preserve prior summaries")
    raw = args.manifest.read_bytes()
    manifest = json.loads(raw)
    supplied_raw = args.score_report.read_bytes() if args.score_report else None
    supplied = json.loads(supplied_raw) if supplied_raw is not None else None
    report = summarize(manifest, args.manifest.resolve().parent, scored_report=supplied)
    report["provenance"] = dict(manifest=dict(path=str(args.manifest.resolve()), sha256=digest(raw)),
        helper=dict(path=str(pathlib.Path(__file__).resolve()), sha256=digest(pathlib.Path(__file__).read_bytes())),
        scorer=dict(path=str(pathlib.Path(scoring.__file__).resolve()), sha256=digest(pathlib.Path(scoring.__file__).read_bytes())))
    if args.score_report:
        report["provenance"]["supplied_score_report"] = dict(path=str(args.score_report.resolve()), sha256=digest(supplied_raw))
    args.out_dir.mkdir(parents=True)
    (args.out_dir/"summary.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    (args.out_dir/"summary.md").write_text(markdown(report))
    if not args.no_plots:
        plot(report, args.out_dir)
    print(json.dumps(dict(status="complete", cells=report["cells"],
        rollout_assignments=report["unique_rollout_assignments"], pooled_analyses=[x["id"] for x in report["aggregates"]],
        output=str(args.out_dir))))


if __name__ == "__main__":
    main()
