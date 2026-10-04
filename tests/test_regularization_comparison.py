"""No fits, NumPy, Torch, network, or real datasets are needed for these checks."""
from contextlib import contextmanager
import copy
import csv
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "compare_regularization", Path(__file__).parents[1]/"scripts/compare_regularization.py")
comparison = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(comparison)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


def inventory(root):
    files = {}
    for path in root.rglob("*"):
        if path.is_file() and path.name != "artifact_inventory.json":
            files[path.relative_to(root).as_posix()] = dict(bytes=path.stat().st_size,
                                                          sha256=comparison.file_hash(path))
    write_json(root/"artifact_inventory.json",dict(kind="merged",files=files,
               bytes=sum(row["bytes"] for row in files.values())))


def make_fixture(root, new):
    root.mkdir()
    methods = list(comparison.ARMS if new else comparison.ARMS+comparison.BASELINES)
    config = dict(run_kind="FORMAL",B_requested=200,n_values=[40],methods=methods,
                  dgp={"p":4,"q":3},mechanism_seed=17,seed_base=99,device="cpu",
                  training={name:dict(epochs=300,weight_decay=.01 if new else 0.0)
                            for name in ("nn_cfg","ae_cfg","me_cfg")})
    truth = dict(value=2.0,converged=True)
    source_hashes = {name:hashlib.sha256(name.encode()).hexdigest() for name in comparison.SOURCES}
    if new:
        source_hashes["simulation/runner.py"] = hashlib.sha256(b"new-runner").hexdigest()
    manifest = dict(config=config,code_hashes=source_hashes,mechanism_hash="same-mechanism",
                    environment=dict(packages={k:"same-version" for k in ("numpy","scipy","scikit-learn","torch")},
                                     platform="new-kernel" if new else "old-kernel"))
    manifest["run_hash"] = comparison.digest(manifest)
    manifest["truth"] = truth
    write_json(root/"manifest.json",manifest)
    write_json(root/"mechanism.json",dict(config=config["dgp"],parameter_seed=17,
               mechanism_hash="same-mechanism",truth=truth))
    (root/"mechanism.npz").write_bytes(b"same validated saved mechanism")
    (root/"scores").mkdir()
    # The upstream merge validates NPZ numerics. Here only its byte checksums
    # are consumed, so a tiny opaque score file suffices without importing NumPy.
    score = root/"scores"/"fixture.npz"
    score.write_bytes(b"previously validated synthetic score fixture")
    for method in methods:
        for rep in range(100 if new else 50):
            # New first50 error=1, fresh50 error=3; old error=2.
            # Thus all100 RMSE=sqrt(5), not mean(1,3)=2.
            error = (1.0 if rep < 50 else 3.0) if new else 2.0
            if method == "mediencoder_l3zero":
                error *= .5
            theta,se = 2+error,.5
            key = f"n00040_r{rep:04d}_{method}"
            lower,upper = theta-comparison.Z*se,theta+comparison.Z*se
            row = dict(task_id=key,n=40,method=method,rep=rep,status="complete",
                       data_seed=1000+rep,training_seed=2000+rep,
                       observed_data_sha256=hashlib.sha256(str(rep).encode()).hexdigest(),
                       run_hash=manifest["run_hash"],mechanism_hash=manifest["mechanism_hash"],
                       theta_population=2.0,theta_hat=theta,error=error,se_IF=se,
                       ci_lower=lower,ci_upper=upper,ci_length=upper-lower,covered=lower<=2<=upper,
                       estimator_metadata=dict(resolved_config={
                           role:dict(weight_decay=.01 if new else 0.0)
                           for role in ("nuisance","representation_A","representation_B")}),
                       score_artifact="scores/fixture.npz",score_artifact_sha256=comparison.file_hash(score))
            write_json(root/"tasks"/(key+".json"),row)
    inventory(root)


@pytest.fixture(scope="module")
def artifacts(tmp_path_factory):
    base = tmp_path_factory.mktemp("regularization-inputs")
    old,new = base/"old",base/"new"
    make_fixture(old,False)
    make_fixture(new,True)
    return old,new


@contextmanager
def changed_record(root, name, change=None, remove=False):
    path = root/"tasks"/name
    original = path.read_bytes()
    try:
        if remove:
            path.unlink()
        else:
            row = json.loads(original)
            change(row)
            write_json(path,row)
        inventory(root)
        yield
    finally:
        path.write_bytes(original)
        inventory(root)


def test_complete_comparison_keeps_cohorts_pairing_and_baselines_separate(artifacts,tmp_path):
    old,new = artifacts
    assert comparison.main([str(old),str(new),"--output-dir",str(tmp_path/"report")]) == 0
    output = tmp_path/"report"
    audit = comparison.read_json(output/"comparison_audit.json")
    assert audit["complete"] and audit["old_run_hash"] != audit["new_run_hash"]
    assert [r["complete_pairs"] for r in audit["paired_audits"]] == [100,100]
    assert audit["environment_difference"]  # Kernel text may change without confounding numerics.
    with (output/"summaries.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 11
    lookup = {(r["cohort"],r["method"]):r for r in rows}
    assert float(lookup["new_wd.01_all100","mediencoder"]["RMSE"]) == pytest.approx(5**.5)
    assert float(lookup["new_wd.01_fresh50_99","mediencoder"]["Bias"]) == 3
    assert float(lookup["new_wd.01_first50","mediencoder"]["RMSE"]) == 1
    assert all(r["B_requested"] == "50" for r in rows if r["method"] in comparison.BASELINES)
    with (output/"paired_changes.csv").open() as stream:
        pairs = list(csv.DictReader(stream))
    assert float(pairs[0]["RMSE_change"]) == -1
    assert float(pairs[0]["RMSE_change_MCSE"]) == 0
    assert pairs[-1]["comparison"] == "new_wd.01_zero_minus_positive_all100"
    assert float(pairs[-1]["RMSE_change"]) == pytest.approx(-(5**.5)/2)
    assert "not a new all-method B=100" in (output/"comparison.md").read_text()


@pytest.mark.parametrize("field,value",[("data_seed",999),("training_seed",999),
                                      ("observed_data_sha256","a"*64)])
def test_rejects_unpaired_overlapping_data_or_training(artifacts,tmp_path,field,value):
    old,new = artifacts
    with changed_record(new,"n00040_r0002_mediencoder.json",lambda r:r.update({field:value})):
        with pytest.raises(ValueError,match="Mismatched paired "+field):
            comparison.compare(old,new,tmp_path/"report")


def test_rejects_fresh_alignment_pair_mismatch(artifacts,tmp_path):
    old,new = artifacts
    with changed_record(new,"n00040_r0075_mediencoder_l3zero.json",lambda r:r.update(training_seed=17)):
        with pytest.raises(ValueError,match="Mismatched paired training_seed"):
            comparison.compare(old,new,tmp_path/"report")


def test_rejects_worker_using_old_decay_despite_new_manifest(artifacts,tmp_path):
    old,new = artifacts
    def drift(row):
        row["estimator_metadata"]["resolved_config"]["nuisance"]["weight_decay"] = 0
    with changed_record(new,"n00040_r0002_mediencoder.json",drift):
        with pytest.raises(ValueError,match="Resolved nuisance weight decay"):
            comparison.compare(old,new,tmp_path/"report")


@pytest.mark.parametrize("failed",[False,True])
def test_incomplete_exit_has_honest_counts_and_no_success_only_metrics(artifacts,tmp_path,failed):
    old,new = artifacts
    change = lambda r:r.update(status="failed",error_type="ArithmeticError",error_message="test failure")
    with changed_record(new,"n00040_r0075_mediencoder.json",change,remove=not failed):
        assert comparison.main([str(old),str(new),"--output-dir",str(tmp_path/"report")]) == 2
        audit = comparison.read_json(tmp_path/"report"/"comparison_audit.json")
        assert not audit["complete"]
        assert len(audit["new_failed_tasks" if failed else "new_missing_tasks"]) == 1
        with (tmp_path/"report"/"summaries.csv").open() as stream:
            rows = list(csv.DictReader(stream))
        full = next(r for r in rows if r["cohort"]=="new_wd.01_all100" and r["method"]=="mediencoder")
        assert full["B_completed"] == "99" and full["RMSE"] == full["Coverage"] == ""
        assert full["B_failed" if failed else "B_missing"] == "1"
        first = next(r for r in rows if r["cohort"]=="new_wd.01_first50" and r["method"]=="mediencoder")
        assert first["status"] == "complete" and first["RMSE"] == "1.0"


@pytest.mark.parametrize("change,match",[
    (lambda m:m["config"]["dgp"].update(p=99),"Scientific settings changed"),
    (lambda m:m["config"].update(seed_base=1),"Scientific settings changed"),
    (lambda m:m["config"]["training"]["nn_cfg"].update(weight_decay=0),"shared weight_decay"),
    (lambda m:m["code_hashes"].update({"estimation.py":"different"}),"Scientific source changed"),
    (lambda m:m["environment"]["packages"].update(numpy="different"),"Numerical runtime changed"),
    (lambda m:m["truth"].update(value=3),"Mechanism or population truth changed"),
])
def test_rejects_scientific_confounding(artifacts,change,match):
    old,new = artifacts
    # Unit-test all provenance gates without repeatedly reading hundreds of tasks.
    left = dict(root=old,manifest=comparison.read_json(old/"manifest.json"))
    right = dict(root=new,manifest=copy.deepcopy(comparison.read_json(new/"manifest.json")))
    change(right["manifest"])
    with pytest.raises(ValueError,match=match):
        comparison.validate_comparability(left,right)


@pytest.mark.parametrize("field,value",[("error",float("nan")),("covered",True),("ci_upper",900)])
def test_rejects_invalid_numerics_instead_of_dropping_rows(artifacts,tmp_path,field,value):
    old,new = artifacts
    with changed_record(new,"n00040_r0002_mediencoder.json",lambda r:r.update({field:value})):
        assert comparison.main([str(old),str(new),"--output-dir",str(tmp_path/"report")]) == 1
        assert not (tmp_path/"report"/"summaries.csv").exists()


def test_rejects_tampered_score_and_input_output_overlap(artifacts,tmp_path):
    old,new = artifacts
    score = new/"scores"/"fixture.npz"
    original = score.read_bytes()
    try:
        score.write_bytes(b"tampered")
        with pytest.raises(ValueError,match="checksum mismatch"):
            comparison.compare(old,new,tmp_path/"report")
    finally:
        score.write_bytes(original)
    with pytest.raises(ValueError,match="separate from both input"):
        comparison.compare(old,new,old/"reports")


def test_paired_mcse_uses_covariance_and_zero_rmse_is_explicit():
    def record(error):
        return dict(status="complete",error=error,covered=True,se_IF=1,ci_length=2*comparison.Z)
    left = {i:record(float(i)) for i in range(1,51)}
    right = {i:record(float(i+2)) for i in range(1,51)}
    pairs = [(i,i) for i in left]
    result = comparison.paired_changes(left,right,pairs,"fixture",40,"mediencoder")
    assert result["Bias_change"] == 2 and result["Bias_change_MCSE"] == 0
    assert result["MSE_change_MCSE"] > 0
    zero = {i:record(0) for i in left}
    result = comparison.paired_changes(zero,right,pairs,"fixture",40,"mediencoder")
    assert result["RMSE_change"] > 0 and result["RMSE_change_MCSE"] is None
