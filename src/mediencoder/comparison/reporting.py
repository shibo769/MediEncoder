"""Inference summaries from each dataset's interval, with explicit denominators."""
import math
import numpy as np

EFFECTS = ("NIE", "NDE", "TE")


def validate_observed(X, M, A, Y):
    X, M, A, Y = np.asarray(X), np.asarray(M), np.asarray(A), np.asarray(Y)
    n = len(Y)
    if X.ndim != 2 or M.ndim != 2 or A.shape != (n,) or Y.shape != (n,) or len(X) != n or len(M) != n:
        raise ValueError("Expected matching X/M matrices and A/Y vectors")
    if n < 12 or not all(np.isfinite(v).all() for v in (X, M, A, Y)):
        raise ValueError("At least 12 finite observations are required")
    if set(np.unique(A)) != {0, 1}:
        raise ValueError("A must contain both binary treatment arms")
    return X, M, A, Y


def validate_result(result):
    """Reject an entire fit with any invalid effect/advertised interval."""
    for effect in EFFECTS:
        if not np.isfinite(result["effects"][effect]):
            raise ValueError(f"Nonfinite {effect} estimate")
        interval = result.get("effect_ci", {}).get(effect)
        se = result.get("effect_se", {}).get(effect)
        if result.get("inference_available", False) and interval is None:
            raise ValueError(f"Method advertises inference but supplied no {effect} interval")
        if interval is not None:
            if len(interval) != 2 or not np.isfinite(interval).all() or interval[0] > interval[1]:
                raise ValueError(f"Invalid {effect} interval")
        if se is not None and (not np.isfinite(se) or se < 0):
            raise ValueError(f"Invalid {effect} standard error")
    return result


def summarize(rows, requested_replications=None):
    """Rows contain one method/replicate and all three effects.

    Bias/SD/RMSE use successful fits; interval summaries use valid reported
    intervals. Failures and absent inference are counted explicitly. Coverage
    over all attempted runs is also reported, treating a failed attempt to
    produce an interval as a failure to cover. It is unavailable for methods
    that do not implement inference at all.
    """
    identities = [(r["mechanism_hash"], r["n"], r["replicate"], r["method"]) for r in rows]
    if len(set(identities)) != len(identities):
        raise ValueError("Duplicate comparison method/replication records")
    groups = {}
    for row in rows:
        groups.setdefault((row["mechanism_hash"], row["n"], row["method"]), []).append(row)
    output = []
    for (mechanism_hash, n, method), group in sorted(groups.items()):
        plans = {r.get("B_requested") for r in group if r.get("B_requested") is not None}
        if len(plans) > 1:
            raise ValueError("Mixed requested replication counts within one experiment")
        planned = requested_replications if requested_replications is not None else next(iter(plans), None)
        if planned is not None and (planned < len(group) or planned < 1):
            raise ValueError("Requested replication count is smaller than recorded attempts")
        for effect in EFFECTS:
            successful, intervals = [], []
            for r in group:
                if r["status"] != "ok":
                    continue
                value, truth = r["effects"][effect], r["truth"][effect]
                if not np.isfinite([value, truth]).all():
                    raise ValueError("A successful row contains a nonfinite estimate/truth")
                successful.append(value - truth)
                interval = r.get("effect_ci", {}).get(effect)
                if interval is not None:
                    if len(interval) != 2 or not np.isfinite(interval).all() or interval[0] > interval[1]:
                        raise ValueError("Invalid interval in successful row")
                    intervals.append((float(interval[1] - interval[0]), bool(interval[0] <= truth <= interval[1])))
            d = np.asarray(successful)
            covered = sum(c for _, c in intervals)
            advertised = any(r.get("inference_available", False) for r in group)
            coverage = covered / len(intervals) if intervals else None
            output.append({"mechanism_hash": mechanism_hash, "n": n, "method": method, "effect": effect,
                           "B_requested": planned, "B_attempted": len(group),
                           "B_pending": planned - len(group) if planned is not None else None,
                           "B_success": len(d), "B_failed": len(group) - len(d),
                           "B_intervals": len(intervals), "B_success_without_interval": len(d) - len(intervals),
                           "Bias": float(d.mean()) if len(d) else None,
                           "SD_error": float(d.std(ddof=1)) if len(d) > 1 else None,
                           "RMSE": float(np.sqrt(np.mean(d ** 2))) if len(d) else None,
                           "CI_Length": float(np.mean([x for x, _ in intervals])) if intervals else None,
                           "Coverage": coverage,
                           "Coverage_MCSE": math.sqrt(coverage * (1 - coverage) / len(intervals)) if intervals else None,
                           "Coverage_all_attempts": covered / len(group) if advertised else None})
    return output


def markdown_table(summary):
    columns = ("n", "method", "effect", "B_requested", "B_attempted", "B_pending", "B_failed", "B_intervals", "SD_error", "RMSE", "CI_Length", "Coverage")
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for row in summary:
        cells = []
        for key in columns:
            value = row[key]
            cells.append("Unavailable" if value is None else (f"{value:.3f}" if isinstance(value, float) else str(value)))
        lines.append("| " + " | ".join(cells) + " |")
    lines += ["", "SD_error is the Monte Carlo SD of estimation errors. CI length and coverage use each dataset's own reported interval.",
              "Point-estimate-only methods have unavailable CI columns. Failed fits remain in the attempted denominator; see summary.json for all counts."]
    return "\n".join(lines) + "\n"
