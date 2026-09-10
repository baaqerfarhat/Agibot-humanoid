"""Read completed ALOHA observer telemetry; emit bounded forensic diagnostics.

Truth is used only to describe executed offsets and estimate errors. Reference
and gain decompositions are identities under the recorded fitted model/on the
recorded trajectory, not new rollouts, causal comparisons, or task-error bounds.
All declared arms/episodes are required. Raw telemetry stays on its filesystem.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np


LEFT = np.arange(6)
RIGHT = np.arange(7,13)
GRIP = np.array([6,13])
INACTIVE = np.arange(6,14)


def array(value):
    return np.asarray(value, dtype=float)


def fingerprint(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b""):
            digest.update(chunk)
    return dict(path=str(Path(path).resolve()), sha256=digest.hexdigest(), bytes=Path(path).stat().st_size)


def json_default(value):
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, np.generic): return value.item()
    raise TypeError(type(value).__name__)


def stats(values):
    data = array(values)
    if data.ndim == 1: data = data[:,None]
    return dict(samples=len(data), mean=data.mean(0), rms=np.sqrt(np.mean(data**2,0)),
                rms_norm=float(np.sqrt(np.mean(np.sum(data**2,1)))),
                max_norm=float(np.max(np.linalg.norm(data,axis=1))))


def summarize_rows(rows):
    keys = sorted(set().union(*(row.keys() for row in rows)) - {"t"})
    result = {key:stats([row[key] for row in rows if key in row]) for key in keys}
    pred = array([row["prediction_increment_left"] for row in rows])
    tracking = array([row["tracking_increment_left"] for row in rows])
    denominator = float(np.sum(pred**2))
    result["tracking_to_prediction_increment_rms_ratio"] = float(np.sqrt(np.sum(tracking**2)/denominator)) if denominator else None
    magnitude = np.linalg.norm(pred,axis=1)*np.linalg.norm(tracking,axis=1)
    nonzero = magnitude>1e-20
    result["prediction_tracking_cosine_mean"] = float(np.mean(np.sum(pred[nonzero]*tracking[nonzero],axis=1)/magnitude[nonzero])) if nonzero.any() else None
    return result


def training_distribution(path):
    episodes = json.loads(Path(path).read_text())
    features = np.concatenate([np.c_[array(ep["q_before"])[:,:6],array(ep["u"])[:,:6]] for ep in episodes[:6]])
    center = features.mean(0)
    covariance = np.cov(features,rowvar=False)
    inverse = np.linalg.pinv(covariance)
    mahal = np.einsum('ni,ij,nj->n',features-center,inverse,features-center)
    return dict(center=center,inverse=inverse,minimum=features.min(0),maximum=features.max(0),
                mahalanobis2_q99=float(np.quantile(mahal,.99)),samples=len(features),fit_episodes=list(range(6)))


def extrapolation(feature, training):
    centered = feature-training["center"]
    mahal = float(centered@training["inverse"]@centered)
    outside = (feature<training["minimum"]) | (feature>training["maximum"])
    return mahal, float(mahal>training["mahalanobis2_q99"]), outside.astype(float)


def analyze(telemetry, summary_path, calibration_log):
    summary = json.loads(Path(summary_path).read_text())
    if summary.get("status") != "complete" or not summary.get("scoreable"):
        raise ValueError("a completed, scoreable compact summary is required")
    training = training_distribution(calibration_log)
    known_outcomes = {(condition,name,row["init"]):row["ok"]
        for condition,cell in summary["cells"].items() for name,arm in cell["arms"].items() for row in arm["per_ep"]}
    pooled = defaultdict(lambda:defaultdict(list))
    episodes = []
    seen = set()
    integrity = dict(executed_command_max_error=0., estimate_update_max_error=0.,
                     recorded_reference_max_error=0., reference_decomposition_max_error=0.)
    current, rows = None, []
    digest = hashlib.sha256()
    header = None
    def finish(key, episode_rows):
        if key is None: return
        if key not in known_outcomes or key in seen: raise ValueError("unexpected/repeated telemetry episode")
        seen.add(key)
        windows = {"steps_0_20":[row for row in episode_rows if row["t"]<20],
                   "steps_20_100":[row for row in episode_rows if 20<=row["t"]<100],
                   "last_100":episode_rows[-100:]}
        record = dict(condition=key[0],arm=key[1],episode=key[2],success=known_outcomes[key],
                      steps=len(episode_rows),windows={})
        for name, selected in windows.items():
            if not selected: continue
            pooled[key[:2]][name].extend(selected)
            metrics = summarize_rows(selected)
            # Full component statistics are pooled below; retain compact per-episode
            # endpoint diagnostics so unequal episode lengths remain transparent.
            record["windows"][name] = {metric:metrics[metric] for metric in
                ("net_offset_left","estimate_error_left","reference_error_left",
                 "reference_error_from_net_offset","reference_error_from_model_mismatch",
                 "one_step_model_error_left","estimate_at_clip_left","estimate_at_clip_gripper",
                 "tracking_to_prediction_increment_rms_ratio")}
        episodes.append(record)
    with Path(telemetry).open("rb") as stream:
        for raw in stream:
            digest.update(raw)
            row=json.loads(raw)
            if row.get("type")=="header":
                if header is not None: raise ValueError("multiple telemetry headers")
                header=row; config=header["config"]
                M=array(config["observer"]["M"]); R=array(config["observer"]["R"]); inverse=np.linalg.pinv(M)
                model=config["reference_model"]; A=array(model["A"]); B=array(model["B"]); offset=array(model["offset"])
                if model["state_indices"]!=list(range(6)) or model["command_indices"]!=list(range(6)):
                    raise ValueError("this declared forensic analysis is for the left-six reference")
                common=config["common"]; clip=common["clip"]; gamma=common["gamma"]
                continue
            if row.get("type")!="step": continue
            if header is None: raise ValueError("telemetry header missing")
            condition,name=row["arm"].split("/",1); key=(condition,name,row["episode"])
            qbefore=array(row["joint_before"]); q=array(row["measured"])
            if key!=current:
                finish(current,rows); current=key; rows=[]
                reference=qbefore[:6].copy(); error_net=np.zeros(6); error_model=np.zeros(6)
                previous_unprojected=np.zeros(14)
            if row["t"]!=len(rows): raise ValueError("missing or reordered episode steps")
            u=array(row["raw_action"]); command=array(row["command"]); correction=array(row["correction"])
            truth=array(row["f_true"]); theta=array(row["f_hat"]); before=array(row["f_hat_before"]); residual=array(row["r"])
            net=correction+truth
            integrity["executed_command_max_error"]=max(integrity["executed_command_max_error"],float(np.max(np.abs(command-u-net))))
            refbefore=reference.copy(); reference=A@reference+B@u[:6]+offset
            model_error=q[:6]-(A@qbefore[:6]+B@command[:6]+offset)
            error_net=A@error_net+B@net[:6]; error_model=A@error_model+model_error
            referror=q[:6]-reference
            integrity["reference_decomposition_max_error"]=max(integrity["reference_decomposition_max_error"],float(np.max(np.abs(referror-error_net-error_model))))
            if row.get("reference_position") is not None:
                integrity["recorded_reference_max_error"]=max(integrity["recorded_reference_max_error"],float(np.max(np.abs(reference-array(row["reference_position"])))))
            record=dict(t=row["t"],net_offset_left=net[:6],estimate_error_left=theta[:6]-truth[:6],
                estimate_error_right=theta[RIGHT]-truth[RIGHT],estimate_error_gripper=theta[GRIP]-truth[GRIP],
                residual_left=residual[:6],residual_right=residual[RIGHT],residual_gripper=residual[GRIP],
                reference_error_left=referror,reference_error_from_net_offset=error_net.copy(),
                reference_error_from_model_mismatch=error_model.copy(),one_step_model_error_left=model_error,
                estimate_at_clip_left=(np.abs(theta[:6])>=clip-1e-10).astype(float),
                estimate_at_clip_right=(np.abs(theta[RIGHT])>=clip-1e-10).astype(float),
                estimate_at_clip_gripper=(np.abs(theta[GRIP])>=clip-1e-10).astype(float),
                any_left_clip=float(np.any(np.abs(theta[:6])>=clip-1e-10)),
                any_inactive_clip=float(np.any(np.abs(theta[INACTIVE])>=clip-1e-10)))
            squared=float(residual@residual)
            record["gripper_share_residual_squared_norm"]=float(residual[GRIP]@residual[GRIP]/squared) if squared else 0.
            record["attenuation_if_all_norm"]=1./(1.+(np.linalg.norm(residual)/common["norm_r"])**2)
            record["attenuation_if_left_norm"]=1./(1.+(np.linalg.norm(residual[:6])/common["norm_r"])**2)
            tracking=np.zeros(14); prediction=np.zeros(14)
            arm=config["arm_configs"][name]
            if arm["adapt"]:
                if row.get("gain") is not None:
                    gain=array(row["gain"]); innovation=residual-M@before
                    prediction=gain@innovation
                    if row.get("tracking_increment") is not None: tracking=array(row["tracking_increment"])
                    for label,indices in (("left",LEFT),("right",RIGHT),("gripper",GRIP)):
                        record[f"prediction_from_{label}_observations_left"]=gain[:6,indices]@innovation[indices]
                    record["prediction_from_inactive_estimates_left"]=-gain[:6]@M[:,INACTIVE]@before[INACTIVE]
                    # This is the algebraic difference caused by retaining the
                    # previous inactive projection, with all current observations
                    # and gains frozen. It is not a recursively rerun observer.
                    record["prediction_difference_from_prior_inactive_clipping_left"]=gain[:6]@M[:,INACTIVE]@(previous_unprojected[INACTIVE]-before[INACTIVE])
                elif arm["baseline"]=="dob": prediction=gamma*(inverse@residual-before)
                elif arm["baseline"]=="none":
                    estimate=inverse@residual
                    nr=np.linalg.norm(residual[:6] if arm["norm_channels"]=="corrected" else residual)
                    if nr<common["dead"]: estimate=np.zeros(14)
                    prediction=gamma*(estimate/(1.+(nr/common["norm_r"])**2)-before)
                else: raise ValueError("unhandled observer arm")
            unprojected=before+prediction+tracking
            expected=np.clip(unprojected,-clip,clip) if arm["adapt"] else before
            integrity["estimate_update_max_error"]=max(integrity["estimate_update_max_error"],float(np.max(np.abs(expected-theta))))
            previous_unprojected=unprojected
            record.update(prediction_increment_left=prediction[:6],tracking_increment_left=tracking[:6],
                          post_projection_increment_left=theta[:6]-before[:6])
            for label,feature in (("actual",np.r_[qbefore[:6],command[:6]]),
                                  ("reference",np.r_[refbefore,u[:6]])):
                mahal,outside,coordinate_outside=extrapolation(feature,training)
                record[label+"_mahalanobis2"]=mahal
                record[label+"_above_training_q99"]=outside
                record[label+"_outside_training_coordinate_range"]=coordinate_outside
            rows.append(record)
    finish(current,rows)
    if seen!=set(known_outcomes): raise ValueError("telemetry lacks declared episodes")
    expected_sha=summary["source_telemetry"].get("sha256")
    if expected_sha and digest.hexdigest()!=expected_sha: raise ValueError("telemetry differs from completed collection hash")
    if max(integrity.values())>1e-10: raise ValueError("recorded arithmetic failed replay/decomposition checks: "+str(integrity))
    std=np.sqrt(np.diag(R)); correlation=R/(std[:,None]*std[None,:])
    matrix=dict(M=M,R=R,M_inverse=inverse,R_eigenvalues=np.linalg.eigvalsh(R),R_condition=float(np.linalg.cond(R)),blocks={})
    for label,indices in (("left",LEFT),("right",RIGHT),("gripper",GRIP)):
        matrix["blocks"][label]=dict(M_into_left_fro=float(np.linalg.norm(M[:6,indices])),
            inverse_into_left_fro=float(np.linalg.norm(inverse[:6,indices])),
            R_correlation_with_left_max=float(np.max(np.abs(correlation[:6,indices]))))
    return dict(schema_version=1,study_id=summary["study_id"],status="complete_diagnostics",args=header["args"],
        source_study=summary["source_study"],source_summary=fingerprint(summary_path),
        source_telemetry=dict(path=str(Path(telemetry).resolve()),sha256=digest.hexdigest(),bytes=Path(telemetry).stat().st_size),
        source_calibration_log=fingerprint(calibration_log),source_script=fingerprint(__file__),
        windows=dict(steps_0_20="0 <= t < 20",steps_20_100="20 <= t < 100",last_100="last min(100,episode_length) samples; may overlap early windows"),
        aggregation="Pooled step-weighted moments per arm/window; per-episode summaries retained separately",
        calibration_distribution={key:value for key,value in training.items() if key!="inverse"},
        matrix_diagnostics=matrix,arithmetic_integrity=integrity,
        arms={condition:{name:dict(successes=summary["cells"][condition]["arms"][name]["successes"],
            episodes=summary["cells"][condition]["arms"][name]["n"],
            windows={window:summarize_rows(values) for window,values in pooled[(condition,name)].items()})
            for name in summary["cells"][condition]["arms"]} for condition in summary["cells"]},
        per_episode=episodes,
        limits=["No new rollouts or outcomes; all declared arms and episodes are included.",
            "Fault truth is used only for offline errors and executed net-offset analysis.",
            "Independent policy sampling and diverging commands prevent causal across-arm trajectory subtraction.",
            "Reference error splits are additive identities under the fitted A/B, not identified physical causes.",
            "Actual one-step model errors include hidden velocity, contacts, omitted channels, and extrapolation.",
            "Gain contributions and inactive-clipping differences hold recorded observations/gains fixed; they are arithmetic diagnostics, not new controller evaluations.",
            "Covariance coupling can create cancelling contributions, so component RMS ratios need not sum to one.",
            "Training-domain thresholds use fitting episodes0–5 only; they are descriptive, not a new acceptance gate."])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--telemetry',type=Path,required=True)
    parser.add_argument('--summary',type=Path,required=True)
    parser.add_argument('--calibration-log',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    if args.out.exists(): raise ValueError('refusing to overwrite existing diagnostics')
    report=analyze(args.telemetry,args.summary,args.calibration_log)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    with args.out.open('x') as stream: json.dump(report,stream,default=json_default,allow_nan=False)
    print(json.dumps(dict(study_id=report['study_id'],integrity=report['arithmetic_integrity'],
                         episodes=len(report['per_episode']),out=str(args.out))))


if __name__=='__main__': main()
