"""
Run every experiment in the report and write results/ (metrics JSON, CSVs, figures).

    python src/run.py                       # synthetic WESAD-format data
    python src/run.py --source wesad --wesad-root data/WESAD

Experiments
  E1  Baseline vs proposed (+ two ablations), LOSO, 15 subjects
  E2  Error analysis: which situations cause false alarms / misses
  E3  Confidence & abstention: coverage vs. quality
  E4  Calibration (reliability diagram, ECE)
  E5  Personalisation: how many minutes of rest calibration are needed
  E6  Minimum-data ablation: which sensors are really necessary
  E7  Fairness: error rates by sex and age group
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

sys.path.insert(0, os.path.dirname(__file__))
from data import LABEL_NAMES, load                      # noqa: E402
from evaluate import bootstrap_ci, loso_predict, metrics, per_subject  # noqa: E402
from features import ALL_FEATURES, FEATURE_GROUPS, extract_all        # noqa: E402
from models import (abstain, calibration_mask, make_baseline, make_lr_norm,  # noqa: E402
                    make_proposed, make_rf_cal)

CALIB_MIN = 10
plt.rcParams.update({"figure.dpi": 130, "font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False})
C_BASE, C_PROP, C_GREY = "#8a8f98", "#2f6fdb", "#c9ccd1"


def save(fig, out, name):
    fig.tight_layout()
    fig.savefig(os.path.join(out, "figures", name), bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["synthetic", "wesad"], default="synthetic")
    ap.add_argument("--wesad-root", default=None)
    ap.add_argument("--out", default="results")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    os.makedirs(os.path.join(args.out, "figures"), exist_ok=True)
    t0 = time.time()
    R = {"source": args.source, "calib_min": CALIB_MIN}

    # ------------------------------------------------------------------ data
    subjects = load(args.source, args.wesad_root, seed=args.seed)
    df = extract_all(subjects)
    df.to_csv(os.path.join(args.out, "features.csv"), index=False)
    R["data"] = {
        "n_subjects": int(df["subject"].nunique()),
        "n_windows": int(len(df)),
        "windows_per_label": {LABEL_NAMES[k]: int(v) for k, v in df["label"].value_counts().sort_index().items()},
        "stress_prevalence": float(df["stress"].mean()),
        "sex_counts": df.groupby("subject")["sex"].first().value_counts().to_dict(),
    }
    print("data:", R["data"])
    feats = ALL_FEATURES

    # ------------------------------------------------------------------ E1
    configs = {
        "Baseline: LogReg, raw": (make_baseline, 0),
        "LogReg + personal norm": (make_lr_norm, CALIB_MIN),
        "RandomForest, raw": (make_rf_cal, 0),
        "RandomForest + personal norm": (make_rf_cal, CALIB_MIN),
        "Proposed: personal norm + selected, calibrated model": (make_proposed, CALIB_MIN),
    }
    preds, persub = {}, {}
    R["E1"] = {}
    for name, (fn, cm) in configs.items():
        # every config is scored on the same windows (calibration minutes excluded)
        p = loso_predict(df, feats, fn, calib_min=cm)
        if cm == 0:
            p = p.loc[p.index.intersection(df.index[~calibration_mask(df, CALIB_MIN)])]
        preds[name], persub[name] = p, per_subject(p)
        res = {"pooled": metrics(p["stress"].values, p["p_stress"].values)}
        for col in ["macro_f1", "balanced_acc", "auroc", "ece"]:
            m, lo, hi = bootstrap_ci(persub[name], col)
            res[f"subject_mean_{col}"] = {"mean": m, "ci95": [lo, hi]}
        R["E1"][name] = res
        print(f"{name:45s} macroF1={res['pooled']['macro_f1']:.3f} "
              f"AUROC={res['pooled']['auroc']:.3f} ECE={res['pooled']['ece']:.3f}")
    BASE, PROP = "Baseline: LogReg, raw", "Proposed: personal norm + selected, calibrated model"
    a, b = persub[BASE]["macro_f1"].values, persub[PROP]["macro_f1"].values
    stat = wilcoxon(b, a, alternative="greater")
    R["E1_test"] = {"wilcoxon_p_proposed_gt_baseline_macro_f1": float(stat.pvalue),
                    "subjects_improved": int((b > a).sum()), "n": int(len(a))}
    pd.concat([d.assign(model=k) for k, d in persub.items()]).to_csv(
        os.path.join(args.out, "per_subject_metrics.csv"), index=False)

    R["E1_selected_classifier_per_fold"] = (preds[PROP].groupby("subject")["chosen_model"]
                                            .first().value_counts().to_dict())
    fig, ax = plt.subplots(figsize=(8, 3.2))
    names = list(configs)
    short = ["Baseline\nLogReg raw", "LogReg\n+ norm", "RF\nraw", "RF\n+ norm", "Proposed\nnorm + select + cal"]
    for i, n in enumerate(names):
        r = R["E1"][n]["subject_mean_macro_f1"]
        col = C_PROP if n == PROP else (C_BASE if n == BASE else C_GREY)
        ax.bar(i, r["mean"], color=col, width=0.6)
        ax.errorbar(i, r["mean"], yerr=[[r["mean"] - r["ci95"][0]], [r["ci95"][1] - r["mean"]]],
                    color="k", capsize=4, lw=1)
        ax.text(i, r["ci95"][1] + 0.015, f"{r['mean']:.2f}", ha="center")
    ax.set_xticks(range(len(short)), short)
    ax.set_ylabel("Macro-F1 (mean over subjects)")
    ax.set_ylim(0, 1.05)
    ax.set_title("E1. Unseen-person stress detection (LOSO, 95% bootstrap CI)")
    save(fig, args.out, "e1_model_comparison.png")

    fig, ax = plt.subplots(figsize=(7.5, 3.2))
    order = persub[PROP].sort_values("macro_f1")["subject"].values
    pb = persub[BASE].set_index("subject").loc[order, "macro_f1"]
    pp = persub[PROP].set_index("subject").loc[order, "macro_f1"]
    x = np.arange(len(order))
    ax.vlines(x, pb, pp, color=C_GREY, lw=2)
    ax.scatter(x, pb, color=C_BASE, label="Baseline", zorder=3)
    ax.scatter(x, pp, color=C_PROP, label="Proposed", zorder=3)
    ax.set_xticks(x, order, rotation=0, fontsize=8)
    ax.set_ylabel("Macro-F1")
    ax.set_title("Per-subject macro-F1 (sorted by proposed model)")
    ax.legend(frameon=False, loc="lower right")
    save(fig, args.out, "e1_per_subject.png")

    # ------------------------------------------------------------------ E2
    P = preds[PROP].copy()
    B = preds[BASE].copy()
    P["yhat"] = (P["p_stress"] >= 0.5).astype(int)
    B["yhat"] = (B["p_stress"] >= 0.5).astype(int)
    fa = {}
    for lab in (1, 3, 4):
        fa[LABEL_NAMES[lab]] = {
            "baseline": float(B.loc[B["label"] == lab, "yhat"].mean()),
            "proposed": float(P.loc[P["label"] == lab, "yhat"].mean()),
        }
    # misses by minutes since stress onset
    st = P[P["label"] == 2].copy()
    st["min_since_onset"] = st.groupby("subject")["t_start"].transform(lambda s: (s - s.min()) / 60)
    st["bucket"] = pd.cut(st["min_since_onset"], [-0.1, 1, 3, 6, 20], labels=["0-1", "1-3", "3-6", "6+"])
    miss = (1 - st.groupby("bucket", observed=True)["yhat"].mean()).to_dict()
    worst = persub[PROP].nsmallest(3, "macro_f1")[["subject", "macro_f1"]].to_dict("records")
    meta = {s["id"]: s["meta"] for s in subjects}
    for w in worst:
        w["responder_strength"] = meta[w["subject"]].get("responder")
    R["E2"] = {"false_alarm_rate_by_true_condition": fa,
               "proposed_miss_rate_by_min_since_stress_onset": {str(k): float(v) for k, v in miss.items()},
               "worst_subjects_proposed": worst}

    fig, axes = plt.subplots(1, 2, figsize=(8, 3))
    labs = list(fa)
    x = np.arange(len(labs))
    axes[0].bar(x - 0.18, [fa[l]["baseline"] for l in labs], 0.36, color=C_BASE, label="Baseline")
    axes[0].bar(x + 0.18, [fa[l]["proposed"] for l in labs], 0.36, color=C_PROP, label="Proposed")
    axes[0].set_xticks(x, labs)
    axes[0].set_ylabel("Share wrongly flagged as stress")
    axes[0].set_title("False alarms by true condition")
    axes[0].legend(frameon=False)
    axes[1].bar(list(map(str, miss)), list(miss.values()), color=C_PROP)
    axes[1].set_xlabel("Minutes since stress task started")
    axes[1].set_ylabel("Miss rate")
    axes[1].set_title("Misses by time since onset")
    save(fig, args.out, "e2_error_analysis.png")

    # timeline demo figure for one median subject
    med_sid = persub[PROP].sort_values("macro_f1").iloc[len(order) // 2]["subject"]
    g = P[P["subject"] == med_sid].sort_values("t_start")
    fig, ax = plt.subplots(figsize=(8.5, 2.8))
    for lab, col in [(2, "#f4c7c3"), (3, "#fde7b0"), (4, "#cfe8d6")]:
        for t in g.loc[g["label"] == lab, "t_start"]:
            ax.axvspan(t / 60, (t + 30) / 60, color=col, lw=0)
    ax.plot(g["t_start"] / 60, g["p_stress"], color=C_PROP, lw=1.6)
    ax.axhspan(0.35, 0.65, color="#eeeeee", zorder=0)
    ax.text(g["t_start"].max() / 60, 0.5, "abstain zone ", va="center", ha="right", fontsize=8, color="#666")
    ax.set_ylim(0, 1)
    ax.set_xlabel("Minutes into session")
    ax.set_ylabel("P(stress)")
    ax.set_title(f"Subject {med_sid}: red = true stress, yellow = amusement, green = meditation")
    save(fig, args.out, "demo_timeline.png")

    # ------------------------------------------------------------------ E3
    y, p = P["stress"].values, P["p_stress"].values
    curve = []
    for half in np.linspace(0, 0.45, 10):
        ab = abstain(p, 0.5 - half, 0.5 + half) if half > 0 else np.zeros(len(p), bool)
        keep = ~ab
        m = metrics(y[keep], p[keep]) if keep.sum() > 10 and len(np.unique(y[keep])) > 1 else None
        if m:
            curve.append({"band": [round(0.5 - half, 3), round(0.5 + half, 3)],
                          "coverage": float(keep.mean()), "macro_f1": m["macro_f1"],
                          "false_alarm_rate": m["false_alarm_rate"]})
    ab = abstain(p)
    R["E3"] = {"curve": curve,
               "chosen_band": [0.35, 0.65],
               "chosen_coverage": float((~ab).mean()),
               "chosen_macro_f1": metrics(y[~ab], p[~ab])["macro_f1"],
               "chosen_false_alarm_rate": metrics(y[~ab], p[~ab])["false_alarm_rate"],
               "error_rate_in_abstained": float(((p[ab] >= 0.5) != y[ab]).mean()) if ab.any() else None,
               "error_rate_in_kept": float(((p[~ab] >= 0.5) != y[~ab]).mean())}
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    ax.plot([c["coverage"] for c in curve], [c["macro_f1"] for c in curve], "o-", color=C_PROP)
    ax.scatter([R["E3"]["chosen_coverage"]], [R["E3"]["chosen_macro_f1"]], s=90,
               facecolor="none", edgecolor="k", zorder=4, label="chosen rule (0.35-0.65)")
    ax.set_xlabel("Coverage (share of windows answered)")
    ax.set_ylabel("Macro-F1 on answered windows")
    ax.set_title("E3. Abstaining trades coverage for quality")
    ax.invert_xaxis()
    ax.legend(frameon=False, fontsize=8)
    save(fig, args.out, "e3_abstention.png")

    # ------------------------------------------------------------------ E4
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    bins = np.linspace(0, 1, 11)
    for name, col, lab in [(BASE, C_BASE, "Baseline"), (PROP, C_PROP, "Proposed")]:
        d = preds[name]
        idx = np.clip(np.digitize(d["p_stress"], bins) - 1, 0, 9)
        xs, ys = [], []
        for k in range(10):
            mk = idx == k
            if mk.sum() >= 5:
                xs.append(d["p_stress"].values[mk].mean()); ys.append(d["stress"].values[mk].mean())
        ax.plot(xs, ys, "o-", color=col,
                label=f"{lab} (ECE {R['E1'][name]['pooled']['ece']:.3f})")
    ax.plot([0, 1], [0, 1], "--", color="#999", lw=1)
    ax.set_xlabel("Predicted P(stress)")
    ax.set_ylabel("Observed stress frequency")
    ax.set_title("E4. Reliability diagram")
    ax.legend(frameon=False, fontsize=8)
    save(fig, args.out, "e4_calibration.png")

    # ------------------------------------------------------------------ E5
    R["E5"] = {}
    for cm in [0, 2, 5, 10, 15]:
        p5 = loso_predict(df, feats, make_proposed, calib_min=cm)
        # score on the windows left after the largest calibration so all points are comparable
        p5 = p5.loc[p5.index.intersection(df.index[~calibration_mask(df, 15)])]
        ps = per_subject(p5)
        R["E5"][cm] = {"macro_f1": bootstrap_ci(ps, "macro_f1"),
                       "pooled_macro_f1": metrics(p5["stress"].values, p5["p_stress"].values)["macro_f1"]}
        print("calib", cm, R["E5"][cm]["macro_f1"])
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    ks = list(R["E5"])
    mu = [R["E5"][k]["macro_f1"][0] for k in ks]
    lo = [R["E5"][k]["macro_f1"][1] for k in ks]
    hi = [R["E5"][k]["macro_f1"][2] for k in ks]
    ax.fill_between(ks, lo, hi, color=C_PROP, alpha=0.15)
    ax.plot(ks, mu, "o-", color=C_PROP)
    ax.set_xlabel("Minutes of personal rest calibration")
    ax.set_ylabel("Macro-F1 (mean over subjects)")
    ax.set_title("E5. Personalisation")
    save(fig, args.out, "e5_personalisation.png")

    # ------------------------------------------------------------------ E6
    sets = {
        "EDA only": FEATURE_GROUPS["EDA"],
        "BVP (heart) only": FEATURE_GROUPS["BVP"],
        "EDA + BVP": FEATURE_GROUPS["EDA"] + FEATURE_GROUPS["BVP"],
        "EDA + BVP + TEMP": FEATURE_GROUPS["EDA"] + FEATURE_GROUPS["BVP"] + FEATURE_GROUPS["TEMP"],
        "All four (with ACC)": ALL_FEATURES,
    }
    R["E6"] = {}
    for k, fs in sets.items():
        p6 = loso_predict(df, fs, make_proposed, calib_min=CALIB_MIN)
        R["E6"][k] = {"macro_f1": bootstrap_ci(per_subject(p6), "macro_f1"),
                      "n_features": len(fs)}
        print("E6", k, R["E6"][k]["macro_f1"])
    fig, ax = plt.subplots(figsize=(6, 2.9))
    ks = list(R["E6"])
    v = [R["E6"][k]["macro_f1"] for k in ks]
    ax.barh(ks, [x[0] for x in v], color=[C_PROP if k == "All four (with ACC)" else C_GREY for k in ks],
            xerr=[[x[0] - x[1] for x in v], [x[2] - x[0] for x in v]], capsize=3)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("Macro-F1 (mean over subjects)")
    ax.set_title("E6. Which sensors are necessary?")
    save(fig, args.out, "e6_min_data.png")

    # ------------------------------------------------------------------ E7
    R["E7"] = {}
    ages = df.groupby("subject")["age"].first()
    age_cut = float(ages.median())
    for name in (BASE, PROP):
        d = preds[name]
        groups = {f"sex={s}": d[d["sex"] == s] for s in sorted(d["sex"].unique())}
        groups[f"age<={age_cut:.0f}"] = d[d["age"] <= age_cut]
        groups[f"age>{age_cut:.0f}"] = d[d["age"] > age_cut]
        out = {}
        for gname, gd in groups.items():
            m = metrics(gd["stress"].values, gd["p_stress"].values)
            out[gname] = {k: m[k] for k in ("macro_f1", "false_alarm_rate", "miss_rate", "ece")}
            out[gname]["n_subjects"] = int(gd["subject"].nunique())
        sexes = [g for g in out if g.startswith("sex=")]
        out["sex_gap_macro_f1"] = float(abs(out[sexes[0]]["macro_f1"] - out[sexes[-1]]["macro_f1"]))
        out["sex_gap_false_alarm"] = float(abs(out[sexes[0]]["false_alarm_rate"] - out[sexes[-1]]["false_alarm_rate"]))
        R["E7"][name] = out

    # feature importance (proposed classifier fitted on all normalised data)
    from models import personal_normalise
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.inspection import permutation_importance
    dn = personal_normalise(df, feats, CALIB_MIN)
    rf = RandomForestClassifier(n_estimators=300, min_samples_leaf=3, random_state=42,
                                class_weight="balanced_subsample", n_jobs=-1)
    rf.fit(dn[feats], dn["stress"])
    pi = permutation_importance(rf, dn[feats], dn["stress"], n_repeats=5, random_state=0,
                                scoring="f1_macro", n_jobs=-1)
    imp = pd.Series(pi.importances_mean, index=feats).sort_values(ascending=False)
    R["feature_importance_top8"] = {k: float(v) for k, v in imp.head(8).items()}
    fig, ax = plt.subplots(figsize=(5, 3))
    top = imp.head(8)[::-1]
    ax.barh(top.index, top.values, color=C_PROP)
    ax.set_xlabel("Drop in macro-F1 when shuffled")
    ax.set_title("Most influential features")
    save(fig, args.out, "feature_importance.png")

    R["runtime_s"] = round(time.time() - t0, 1)
    with open(os.path.join(args.out, "results.json"), "w") as f:
        json.dump(R, f, indent=2, default=str)
    print(f"done in {R['runtime_s']} s -> {args.out}/results.json")


if __name__ == "__main__":
    main()

