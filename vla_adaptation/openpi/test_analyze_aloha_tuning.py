"""Scientific-integrity regressions for selection and held-out tuning analysis."""
import copy
import hashlib
import itertools
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

import analyze_aloha_tuning as analysis
from run_aloha_tuning import normalize_plan, select_candidates


def write(path, value):
    path.write_text(json.dumps(value))


def fixture(root, *, phase="search", conditions=None, seeds=None, singleton=False, endpoint=False):
    conditions = conditions or ["healthy", "offset"]
    seeds = seeds or [3200, 3201]
    count = 1 if singleton else 2
    candidates = {"off": dict(family="off"), "oracle": dict(family="oracle")}
    families = {}
    for family in analysis.FAMILIES:
        families[family] = []
        for i in range(count):
            name = f"{family}_{i}"
            families[family].append(name)
            candidates[name] = dict(family=family, gamma=.08 + .01*i)
            if family in ("kalman", "composite"):
                candidates[name].update(Q=(np.eye(14)*.001).tolist(), R=np.eye(14).tolist(),
                    P0=np.eye(14).tolist(), clip=.08+.01*i,
                    tracking_rate=.01 if family == "composite" else 0.0)
    families["composite"] += families["kalman"] if not singleton else []
    bank = {"candidates": [dict(name=n, parameters=p, qualification=dict(allowed=True))
                           for n,p in candidates.items()]}
    bank_path = root/"bank.json"
    write(bank_path, bank)
    source_path = root/"source.py"
    source_path.write_text("# frozen source\n")
    selection = None
    if phase != "search":
        selection_path = root/"prior_selection.json"
        write(selection_path, dict(kind="candidate_selection", candidate_bank_sha256=analysis.digest(bank_path),
                                  selected=families,allocation=dict(expected_seeds=[3200,3201])))
        selection = dict(path=str(selection_path), sha256=analysis.digest(selection_path))
    raw_plan = dict(schema_version=1, bank_sha256=analysis.digest(bank_path),
        stage="confirmation" if phase == "confirmation" else "tuning", seed=seeds[0], episodes=len(seeds),
        max_steps=3, correction_indices=list(range(6)), candidate_names=list(candidates),
        conditions=[dict(name=c, kind="offset", fault_vec=([0.0]*14 if c=="healthy" else [.02]*6+[0.0]*8))
                    for c in conditions])
    if phase == "confirmation":
        raw_plan["selection_record"] = selection
    registry_info = None
    if endpoint:
        registry_path = root / "registry.json"
        selected_endpoint = dict(id="fixture_8004", host="127.0.0.1", port=8004)
        write(registry_path, dict(schema_version=1, endpoints=[selected_endpoint]))
        registry_info = dict(path=str(registry_path),sha256=analysis.digest(registry_path))
        raw_plan["metadata"] = dict(policy_endpoint=selected_endpoint,
            policy_registry_sha256=registry_info["sha256"])
    plan_path = root/"plan.json"
    write(plan_path, raw_plan)
    plan = normalize_plan(raw_plan)
    sources = {str(source_path):analysis.digest(source_path), str(plan_path):analysis.digest(plan_path),
               str(bank_path):analysis.digest(bank_path)}
    if phase == "confirmation":
        sources[selection["path"]] = selection["sha256"]
    configs = select_candidates(plan, dict(candidates=candidates))
    study_path, telemetry_path = root/"study.json", root/"telemetry.jsonl"
    study = dict(schema_version=1, study_id="test_study", stage=plan["stage"], status="complete",
        completed_source_recheck=True, args=dict(plan=str(plan_path),candidate_bank=str(bank_path),
            telemetry=str(telemetry_path),seed=seeds[0],episodes=len(seeds),max_steps=3,corr=list(range(6)),
            reset_estimate=True,reset_covariance=True,reset_reference=True), plan=plan, source_hashes=sources, arm_configs=configs,
        robot_joints=[], pairing=dict(valid_so_far=True, checked_states=len(candidates)*len(conditions)*len(seeds),
            fields=["full_qpos","full_qvel"], tolerance=plan["state_tolerance"]), conditions={}, schedule=[])
    for condition in conditions:
        definition = next(c for c in plan["conditions"] if c["name"]==condition)
        cohort = dict(definition=definition, shared_control_id=f"test_study:{condition}:off", arms={})
        study["conditions"][condition] = cohort
        for name in candidates:
            arm = dict(n=len(seeds), successes=len(seeds), per_ep=[], traj=[], f_hat=[], f_true=[], diagnostics=[])
            cohort["arms"][name] = arm
            for i, seed in enumerate(seeds):
                qpos, qvel = np.zeros(16,dtype="<f8"), np.zeros(15,dtype="<f8")
                qpos[-1] = i*.1
                h=hashlib.sha256(json.dumps([list(qpos.shape),list(qvel.shape)]).encode()+qpos.tobytes()+qvel.tobytes()).hexdigest()
                arm["per_ep"].append(dict(task=0,init=i,episode=i,actual_seed=seed,ok=True,
                    initial_state=dict(qpos=qpos.tolist(),qvel=qvel.tolist(),sha256=h,pairing_valid=True)))
                arm["traj"].append(np.zeros((3,14)).tolist())
                arm["f_hat"].append([0.0]*14)
                arm["f_true"].append(np.zeros((3,14)).tolist())
                energy = 3*sum(x*x for x in definition["fault_vec"]) if name=="oracle" else 0.0
                arm["diagnostics"].append(dict(physical_steps=3,applied_correction_steps=3,
                    applied_correction_energy=energy))
        for i,seed in enumerate(seeds):
            study["schedule"].append(dict(condition=condition,seed=seed,arms=list(candidates)))
    write(study_path,study)
    if endpoint:
        write(root/"run_config.json",dict(args=dict(study["args"],host="127.0.0.1",port=8004),
            plan=plan,source_hashes=sources))
    header=dict(study_id=study["study_id"],source_hashes=sources,plan=plan,config=dict(arm_configs=configs))
    telemetry_path.write_text(json.dumps(header)+"\n")
    manifest=dict(schema_version=1,phase=phase,conditions=conditions,expected_seeds=seeds,
        expected_candidates=list(candidates),families=families,
        sources={str(source_path):analysis.digest(source_path)},
        candidate_bank=dict(path=str(bank_path),sha256=analysis.digest(bank_path)),
        studies=[dict(path=str(study_path),plan_path=str(plan_path),plan_sha256=analysis.digest(plan_path))])
    if selection:
        manifest["selection_record"] = selection
    if registry_info:
        manifest["policy_server_registry"] = registry_info
    manifest_path=root/"manifest.json"
    write(manifest_path,manifest)
    return manifest_path,study_path


class IntegrityTests(unittest.TestCase):
    def setUp(self):
        self.temporary=tempfile.TemporaryDirectory()
        self.root=Path(self.temporary.name)
        self.manifest,self.study=fixture(self.root)

    def tearDown(self):
        self.temporary.cleanup()

    def mutate(self, function):
        study=analysis.read(self.study)
        function(study)
        write(self.study,study)
        with self.assertRaises(ValueError):
            analysis.load_allocation(self.manifest)

    def test_complete_frozen_study_selects_and_reuses_kalman_rows(self):
        data=analysis.load_allocation(self.manifest)
        report=analysis.selection_report(data,2)
        self.assertEqual(report["selected"]["kalman"],["kalman_0","kalman_1"])
        self.assertEqual(len(data["outcomes"]),56)
        self.assertEqual(report["ranked"]["composite"][2]["eta_zero_alias"],True)

    def test_incomplete_status_and_missing_arm_are_rejected(self):
        self.mutate(lambda s:s.update(status="running"))
        self.manifest,self.study=fixture(self.root)
        self.mutate(lambda s:s["conditions"]["offset"]["arms"].pop("dob_0"))

    def test_duplicate_seed_and_success_total_mismatch_are_rejected(self):
        self.mutate(lambda s:s["conditions"]["healthy"]["arms"]["off"]["per_ep"][1].update(actual_seed=3200))
        self.manifest,self.study=fixture(self.root)
        self.mutate(lambda s:s["conditions"]["healthy"]["arms"]["off"].update(successes=0))

    def test_stale_source_bank_and_plan_are_rejected(self):
        (self.root/"source.py").write_text("# changed\n")
        with self.assertRaisesRegex(ValueError,"source hash"):
            analysis.load_allocation(self.manifest)
        self.manifest,self.study=fixture(self.root)
        bank=analysis.read(self.root/"bank.json");bank["extra"]=1;write(self.root/"bank.json",bank)
        with self.assertRaisesRegex(ValueError,"bank hash"):
            analysis.load_allocation(self.manifest)
        self.manifest,self.study=fixture(self.root)
        plan=analysis.read(self.root/"plan.json");plan["episodes"]=3;write(self.root/"plan.json",plan)
        with self.assertRaisesRegex(ValueError,"plan hash"):
            analysis.load_allocation(self.manifest)

    def test_runtime_parameter_edit_is_rejected(self):
        self.mutate(lambda s:s["arm_configs"]["kalman_0"].update(clip=.9))

    def test_full_cube_state_mismatch_even_with_updated_hash_is_rejected(self):
        def corrupt(study):
            state=study["conditions"]["offset"]["arms"]["dob_0"]["per_ep"][0]["initial_state"]
            state["qpos"][-1]=.1
            qpos,qvel=(np.asarray(state[k],dtype="<f8") for k in ("qpos","qvel"))
            state["sha256"]=hashlib.sha256(json.dumps([list(qpos.shape),list(qvel.shape)]).encode()+qpos.tobytes()+qvel.tobytes()).hexdigest()
        self.mutate(corrupt)

    def test_duplicate_study_and_missing_global_condition_are_rejected(self):
        manifest=analysis.read(self.manifest);manifest["studies"]*=2;write(self.manifest,manifest)
        with self.assertRaisesRegex(ValueError,"duplicate study"):
            analysis.load_allocation(self.manifest)
        self.manifest,self.study=fixture(self.root)
        manifest=analysis.read(self.manifest);manifest["conditions"].append("joint0");write(self.manifest,manifest)
        with self.assertRaisesRegex(ValueError,"global allocation incomplete"):
            analysis.load_allocation(self.manifest)

    def test_actual_energy_uses_previous_estimate_and_mask(self):
        data=analysis.load_allocation(self.manifest)
        study=analysis.read(self.study)
        arm=study["conditions"]["healthy"]["arms"]["dob_0"]
        arm["traj"][0][0][0]=.02
        arm["traj"][0][0][13]=9.0
        arm["diagnostics"][0]["applied_correction_energy"]=.02**2
        write(self.study,study)
        data=analysis.load_allocation(self.manifest)
        self.assertEqual(data["energies"]["healthy","dob_0",3200],(.02**2,3))
        self.mutate(lambda s:s["conditions"]["healthy"]["arms"]["dob_0"]["diagnostics"][0].update(applied_correction_energy=0))

    def test_selection_ties_use_healthy_harm_then_episode_mean_energy(self):
        data=analysis.load_allocation(self.manifest)
        # Both have score .5, but dob_0 breaks a healthy success; dob_1 does not.
        for c,s in itertools.product(data["manifest"]["conditions"],[3200,3201]):
            data["outcomes"][c,"dob_0",s]=int(c=="offset")
            data["outcomes"][c,"dob_1",s]=int(c=="healthy")
        self.assertEqual(analysis.selection_report(data,2)["selected"]["dob"][0],"dob_1")
        # Episode means differ from pooling: (10/10+0/1)/2=.5, not10/11.
        for c,s in itertools.product(data["manifest"]["conditions"],[3200,3201]):
            data["energies"][c,"off",s]=(10.,10) if s==3200 else (0.,1)
        self.assertEqual(analysis.candidate_table(data,"off")["per_step_applied_correction_energy"],.5)

    def test_matching_dob_kalman_are_annotated_without_pooling_outcomes(self):
        data=analysis.load_allocation(self.manifest)
        data["recipes"]["kalman_0"]=dict(q_recipe="matched",q_scale=1.,r_scale=1.,p0="gamma_c")
        # Fixture's parameters omit clip on DOB; real candidate bank expands it.
        data["candidates"]["dob_0"]["clip"]=.08
        data["candidates"]["dob_1"]["clip"]=.08
        pairs=analysis.deterministic_equivalences(data)
        self.assertEqual([(p["kalman"],p["dob"]) for p in pairs],[("kalman_0","dob_0")])
        self.assertEqual(len(data["outcomes"]),56)
        data["recipes"]["kalman_0"]["p0"]="diffuse_identity"
        self.assertEqual(analysis.deterministic_equivalences(data),[])

    def test_prior_selection_must_match_confirmation_candidate(self):
        manifest,study=fixture(self.root,phase="confirmation",conditions=["healthy"]+[f"joint{i}" for i in range(7)],
                              seeds=list(range(3400,3430)),singleton=True)
        data=analysis.load_allocation(manifest)
        self.assertEqual(len(data["outcomes"]),1920)
        prior=analysis.read(self.root/"prior_selection.json");prior["selected"]["dob"]=["missing"]
        write(self.root/"prior_selection.json",prior)
        with self.assertRaisesRegex(ValueError,"selection record hash"):
            analysis.load_allocation(manifest)

    def test_declared_endpoint_binds_actual_run_configuration(self):
        manifest,_=fixture(self.root,phase="validation",seeds=[3300,3301],endpoint=True)
        data=analysis.load_allocation(manifest)
        receipt=data["provenance"][0]["run_config"]
        self.assertEqual(receipt["sha256"],analysis.digest(self.root/"run_config.json"))
        self.assertEqual(receipt["policy_endpoint"]["port"],8004)

    def test_endpoint_host_or_port_tampering_is_rejected(self):
        for key,value in (("host","other-host"),("port",8002)):
            manifest,_=fixture(self.root,phase="validation",seeds=[3300,3301],endpoint=True)
            config=analysis.read(self.root/"run_config.json");config["args"][key]=value
            write(self.root/"run_config.json",config)
            with self.assertRaisesRegex(ValueError,"actual CLI endpoint"):
                analysis.load_allocation(manifest)

    def test_run_configuration_plan_source_and_hash_tampering_is_rejected(self):
        for tamper in (lambda c:c["plan"].update(seed=9999),
                       lambda c:c["source_hashes"].clear()):
            manifest,_=fixture(self.root,phase="validation",seeds=[3300,3301],endpoint=True)
            config=analysis.read(self.root/"run_config.json");tamper(config)
            write(self.root/"run_config.json",config)
            with self.assertRaisesRegex(ValueError,"run configuration"):
                analysis.load_allocation(manifest)
        manifest,_=fixture(self.root,phase="validation",seeds=[3300,3301],endpoint=True)
        config=analysis.read(manifest);config["studies"][0]["run_config_sha256"]="0"*64;write(manifest,config)
        with self.assertRaisesRegex(ValueError,"run configuration artifact hash"):
            analysis.load_allocation(manifest)

    def test_endpoint_requires_configuration_and_hash_bound_registry(self):
        manifest,_=fixture(self.root,phase="validation",seeds=[3300,3301],endpoint=True)
        (self.root/"run_config.json").unlink()
        with self.assertRaisesRegex(ValueError,"missing run configuration"):
            analysis.load_allocation(manifest)
        manifest,_=fixture(self.root,phase="validation",seeds=[3300,3301],endpoint=True)
        registry=analysis.read(self.root/"registry.json");registry["endpoints"][0]["port"]=8002
        write(self.root/"registry.json",registry)
        with self.assertRaisesRegex(ValueError,"registry hash"):
            analysis.load_allocation(manifest)


class StatisticsTests(unittest.TestCase):
    def test_seed_bootstrap_is_deterministic_and_matches_direct_resampling(self):
        differences=np.array([-.5,0,.25,.5])
        result=analysis.cluster_bootstrap_interval(differences)
        self.assertEqual(result,analysis.cluster_bootstrap_interval(differences))
        rng=np.random.default_rng(20260908)
        means=differences[rng.integers(0,4,size=(50000,4))].mean(axis=1)
        lower,upper=np.quantile(means,[.025,.975])
        self.assertEqual((result["lower"],result["upper"]),(lower,upper))
        self.assertEqual(result["seed_clusters"],4)

    def test_degenerate_bootstrap_retains_equivalence_caution(self):
        result=analysis.cluster_bootstrap_interval(np.zeros(30))
        self.assertEqual((result["lower"],result["upper"]),(0.,0.))
        self.assertIn("does not establish equivalence",result["limitation"])
        with self.assertRaises(ValueError):
            analysis.cluster_bootstrap_interval([0.,1.],resamples=10)

    def test_exact_cluster_dp_matches_independent_assignment_enumeration(self):
        for values in ([0,0,0], [14,14,14,14], [-7,1,2,4,0], [3,-2,3,-2]):
            observed=abs(sum(values))
            count=sum(abs(sum(s*v for s,v in zip(signs,values)))>=observed
                for signs in itertools.product((-1,1),repeat=len(values)))
            result=analysis.cluster_test(np.asarray(values)/14,14)
            self.assertTrue(result["exact"])
            self.assertEqual(result["p_two_sided"],count/2**len(values))
        strongest=analysis.cluster_test(np.ones(30),14)
        self.assertEqual(strongest["p_two_sided"],2/2**30)
        self.assertEqual(strongest["assignments"],2**30)
        self.assertLessEqual(strongest["distinct_signed_sums"],841)

    def test_noninteger_cluster_values_use_declared_monte_carlo_fallback(self):
        result=analysis.cluster_test([.12345,.67891],14)
        self.assertEqual(result["draws"],200000)

    def test_sign_flip_is_deterministic_and_matches_small_exact_distribution(self):
        differences=np.array([1.,1.,1.,1.])
        exact=sum(abs(sum(signs))>=4 for signs in itertools.product((-1,1),repeat=4))/16
        first=analysis.sign_flip_test(differences)
        self.assertEqual(first,analysis.sign_flip_test(differences))
        self.assertAlmostEqual(first["p_two_sided"],exact,delta=.003)
        self.assertEqual(analysis.sign_flip_test(np.zeros(30))["p_two_sided"],1)
        with self.assertRaises(ValueError):
            analysis.sign_flip_test(differences,100)

    def test_holm_monotonicity_and_exact_mcnemar(self):
        self.assertEqual(analysis.holm([.04,.001,.02,.8]),[.08,.004,.06,.8])
        self.assertEqual(analysis._mcnemar(0,9),2/512)
        self.assertEqual(analysis._mcnemar(0,0),1)

    def test_confirmation_cluster_unit_and_bonferroni8(self):
        with tempfile.TemporaryDirectory() as directory:
            path,_=fixture(Path(directory),phase="confirmation",
                conditions=["healthy"]+[f"joint{i}" for i in range(7)],seeds=list(range(3400,3430)),singleton=True)
            data=analysis.load_allocation(path)
            for c,s in itertools.product(data["manifest"]["conditions"],range(3400,3430)):
                data["outcomes"][c,"kalman_0",s]=int(s>=3409)
            report=analysis.confirmation_report(data)
            self.assertEqual(report["primary"]["seed_clusters"],30)
            self.assertEqual(report["primary"]["confidence_interval"]["seed_clusters"],30)
            self.assertAlmostEqual(report["primary"]["effect"],.3)
            self.assertEqual(report["primary"]["per_cell"][0]["p_bonferroni8"],8*2/512)
            self.assertEqual(len(report["secondary"]),4)
            self.assertNotIn("oracle",[r["family"] for r in report["numerical_order"]])


if __name__ == "__main__":
    unittest.main()
