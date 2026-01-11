# -*- coding: utf-8 -*-
"""
Nature Energy submission app (Streamlit)
Integrated Q-methodology (4 factors) + TPPP mapping + ABM/SD-informed simulations and sensitivity analyses.

Author:
  Prof. Dr. Songhee Kang (Tech University of Korea)
Run:
  streamlit run ne-tppp-q-abm-analysis_streamlit.py
"""

import io
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

# =========================================================
# 0) Page config
# =========================================================
st.set_page_config(page_title="NE Q-TPPP + ABM Pre-evaluation", layout="wide")
RNG_SEED = 42
rng = np.random.default_rng(RNG_SEED)

# =========================================================
# 1) Constants
# =========================================================
STATEMENTS = [
    "데이터센터는 재생에너지를 사용할 때 환경 책임성을 갖춘 시설로 평가받을 수 있다.",
    "디젤이나 가스 발전기를 백업 전력으로 사용할 경우 환경적 우려가 제기될 수 있다.",
    "물 절약이나 친환경 냉각 기술의 도입은 시민 신뢰에 긍정적 영향을 줄 수 있다.",
    "기술이 최신이더라도 안전성 확보가 부족하면 시민 불안을 유발할 수 있다.",
    "데이터센터 기술은 비용 효율성보다는 사회적 책임을 우선시해야 한다는 견해가 있다.",
    "기술이 낯설거나 복잡하게 인식되면 시민과의 거리감이 커질 수 있다.",
    "데이터센터 건설 과정에 시민 의견이 반영되지 않으면 반발 가능성이 높아질 수 있다.",
    "지역 사회와 장기적 관계를 맺어온 기업은 더 높은 신뢰를 받을 수 있다.",
    "설명회가 형식적으로 보일 경우, 시민 불신을 유발할 수 있다.",
    "정보 접근성이 낮을수록 시민의 불안과 의심이 증가할 수 있다.",
    "갈등 상황에서는 중립적 제3자의 개입이 조정에 도움이 될 수 있다.",
    "동일한 설명이라도 정부가 전달할 경우 기업보다 더 신뢰받을 가능성이 있다.",
    "기존 공장이나 발전소 부지를 재활용한 데이터센터는 수용성이 높아질 수 있다.",
    "지역 정체성과 조화를 이루지 못하는 입지는 거부감을 유발할 수 있다.",
    "자연경관 훼손이 발생하는 경우, 기술 우수성만으로 수용성 확보는 어려울 수 있다.",
    "수도권과 지방은 데이터센터 입지에 대해 서로 다른 기준을 가질 수 있다.",
    "외부 자본 주도의 일방적인 입지 결정은 지역사회의 신뢰를 저해할 수 있다.",
    "지역에 실질적인 혜택이 제공되면 시민 수용성이 높아질 수 있다.",
    "초기 단계에서 정보가 투명하게 공개되면 시민 신뢰가 높아질 수 있다.",
    "환경영향평가 결과는 시민들의 수용 여부에 중요한 판단 기준이 될 수 있다.",
    "기업과 지자체가 공동으로 결정한 프로젝트는 더 높은 신뢰를 얻을 수 있다.",
    "법적 요건을 충족하더라도 시민 신뢰를 확보하려면 추가적인 설명이 필요할 수 있다.",
    "지역 언론이 신속하고 정확하게 정보를 전달하면 신뢰성 제고에 기여할 수 있다.",
    "데이터센터 완공 이후에도 모니터링과 피드백 체계가 지속되면 신뢰 유지에 도움이 될 수 있다."
]
Q_MAP = {f"Q{i+1:02d}": txt for i, txt in enumerate(STATEMENTS)}

TPPP_CATEGORIES = {
    "Technology": ["Q01", "Q02", "Q03", "Q04", "Q05", "Q06"],
    "People": ["Q08", "Q09", "Q10", "Q12", "Q22", "Q23"],
    "Place": ["Q13", "Q14", "Q15", "Q16", "Q17", "Q18"],
    "Process": ["Q07", "Q11", "Q19", "Q20", "Q21", "Q24"],
}
Q_TO_TPPP = {q: cat for cat, qs in TPPP_CATEGORIES.items() for q in qs}
FACTOR_NAMES = ["F1", "F2", "F3", "F4"]

# =========================================================
# 2) Q-methodology core
# =========================================================
def _standardize_rows(X: np.ndarray) -> np.ndarray:
    mean = np.nanmean(X, axis=1, keepdims=True)
    std = np.nanstd(X, axis=1, ddof=1, keepdims=True)
    std[std == 0] = 1.0
    return (X - mean) / std

class QEngine:
    def __init__(self, data_df: pd.DataFrame, n_factors: int = 4):
        self.q_df = data_df.select_dtypes(include=[np.number])
        temp = self.q_df.values.astype(float)
        row_means = np.nanmean(temp, axis=1)
        inds = np.where(np.isnan(temp))
        temp[inds] = np.take(row_means, inds[0])
        self.data = np.nan_to_num(temp, nan=0.0)
        self.n_factors = n_factors
        self.calculated_weights: Dict[str, float] = {}

    def fit(self) -> "QEngine":
        R, _ = spearmanr(self.data, axis=1)
        self.R = np.nan_to_num(R, nan=0.0)
        eigvals, eigvecs = np.linalg.eigh(self.R)
        idx = eigvals.argsort()[::-1]
        self.eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]

        k = self.n_factors
        valid_eigs = np.maximum(self.eigvals[:k], 0)
        L = eigvecs[:, :k] * np.sqrt(valid_eigs)
        self.loadings = self._varimax(L)

        # population weights by max abs loading
        max_idx = np.argmax(np.abs(self.loadings), axis=1)
        counts = {f"F{i+1}": 0 for i in range(k)}
        for i in max_idx:
            counts[f"F{i+1}"] += 1
        total = len(max_idx)
        self.calculated_weights = {k: v / total for k, v in counts.items()}

        z_data = _standardize_rows(self.data)
        self.factor_arrays = self._calculate_factor_arrays(self.loadings, z_data)
        return self

    def _varimax(self, Phi: np.ndarray, gamma: float = 1.0, q: int = 20, tol: float = 1e-6) -> np.ndarray:
        p, k = Phi.shape
        R = np.eye(k)
        d = 0.0
        for _ in range(q):
            d_old = d
            Lambda = np.dot(Phi, R)
            u, s, vh = np.linalg.svd(
                np.dot(Phi.T, (Lambda**3 - (gamma/p) * np.dot(Lambda, np.diag(np.diag(np.dot(Lambda.T, Lambda))))))
            )
            R = np.dot(u, vh)
            d = np.sum(s)
            if d_old != 0 and d/d_old < 1 + tol:
                break
        return np.dot(Phi, R)

    def _calculate_factor_arrays(self, loadings: np.ndarray, z_data: np.ndarray) -> np.ndarray:
        n_items = z_data.shape[1]
        arrays = np.zeros((n_items, self.n_factors))
        for f in range(self.n_factors):
            l_vec = loadings[:, f]
            l_clean = np.clip(l_vec, -0.95, 0.95)
            w = l_clean / (1 - l_clean**2)
            if np.sum(np.abs(w)) < 1e-6:
                continue
            weighted_sum = np.dot(w, z_data)
            mu = np.mean(weighted_sum)
            sd = np.std(weighted_sum, ddof=1)
            if sd == 0:
                sd = 1.0
            arrays[:, f] = (weighted_sum - mu) / sd
        return arrays

def calculate_tppp_scores(df_q: pd.DataFrame) -> pd.DataFrame:
    scores = pd.DataFrame(index=df_q.index)
    for cat, items in TPPP_CATEGORIES.items():
        scores[cat] = df_q[items].mean(axis=1)
    return scores

def calculate_type_tppp_profile(factor_arrays: np.ndarray, q_labels: List[str]) -> pd.DataFrame:
    df_arrays = pd.DataFrame(factor_arrays, index=q_labels, columns=FACTOR_NAMES)
    out = {}
    for f in FACTOR_NAMES:
        out[f] = {cat: float(df_arrays.loc[qs, f].mean()) for cat, qs in TPPP_CATEGORIES.items()}
    return pd.DataFrame(out)

# =========================================================
# 3) System network (TPPP links)
# =========================================================
def create_system_network(corr_matrix: pd.DataFrame, threshold: float = 0.3) -> go.Figure:
    nodes = list(corr_matrix.columns)
    pos = {nodes[0]: (0, 1), nodes[1]: (1, 0), nodes[2]: (0, -1), nodes[3]: (-1, 0)}
    fig = go.Figure()
    for i in range(len(nodes)):
        for j in range(i+1, len(nodes)):
            r = float(corr_matrix.iloc[i, j])
            if abs(r) >= threshold:
                n1, n2 = nodes[i], nodes[j]
                x0, y0 = pos[n1]
                x1, y1 = pos[n2]
                fig.add_trace(go.Scatter(
                    x=[x0, x1, None],
                    y=[y0, y1, None],
                    mode="lines",
                    line=dict(width=abs(r)*10, color=("#E63946" if r < 0 else "#457B9D")),
                    hoverinfo="text",
                    text=f"{n1} ↔ {n2}<br>Spearman ρ: {r:.2f}",
                    showlegend=False
                ))
    fig.add_trace(go.Scatter(
        x=[pos[n][0] for n in nodes],
        y=[pos[n][1] for n in nodes],
        mode="markers+text",
        text=nodes,
        textposition=["top center", "middle right", "bottom center", "middle left"],
        textfont=dict(size=15, color="black"),
        marker=dict(size=45, color="white", line=dict(width=3, color="#333")),
        hoverinfo="none",
        showlegend=False
    ))
    fig.update_layout(
        title="TPPP System Network (Spearman links)",
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-1.5, 1.5]),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-1.5, 1.5]),
        margin=dict(l=20, r=20, t=50, b=20),
        plot_bgcolor="white",
        height=560
    )
    return fig

# =========================================================
# 4) Simulation model (integrated from sensitivity notebook)
# =========================================================
def calculate_agent_profiles_from_factor_arrays(fa_only: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    profiles = {}
    for f in FACTOR_NAMES:
        profiles[f] = {cat: float(fa_only.loc[qs, f].median()) for cat, qs in TPPP_CATEGORIES.items()}
    return profiles

def calculate_metrics(series: np.ndarray) -> Dict[str, float]:
    series = np.asarray(series, dtype=float)
    n = len(series)
    k = max(1, n // 3)
    return {
        "mean": float(series.mean()),
        "early_mean": float(series[:k].mean()),
        "late_mean": float(series[-k:].mean()),
        "final": float(series[-1]),
    }

def _interp_curve(points: List[Tuple[float, float]], steps: int, gamma: float = 1.0) -> np.ndarray:
    t = np.linspace(0, 1, steps)
    tw = np.clip(t**gamma, 0.0, 1.0)
    xp, fp = zip(*points)
    return np.interp(tw, np.asarray(xp, float), np.asarray(fp, float))

def _shift_points(points: List[Tuple[float, float]], shift: float) -> List[Tuple[float, float]]:
    shifted = []
    for x, y in points:
        if x <= 0.0:
            shifted.append((0.0, y))
        elif x >= 1.0:
            shifted.append((1.0, y))
        else:
            shifted.append((float(np.clip(x - shift, 0.0, 1.0)), y))
    shifted.sort(key=lambda t: t[0])
    out = [shifted[0]]
    eps = 1e-6
    for x, y in shifted[1:]:
        if x <= out[-1][0]:
            x = min(1.0, out[-1][0] + eps)
        out.append((x, y))
    out[0] = (0.0, out[0][1])
    out[-1] = (1.0, out[-1][1])
    return out

@dataclass
class SimParams:
    tech_max: float = 1.2
    process_max: float = 1.0
    place_max: float = 0.9
    people_max: float = 1.0
    penalty_delta: float = 1.5
    tau_synergy: float = 0.6
    synergy_gain: float = 0.5
    interaction_low: float = 0.25
    tech_high: float = 0.8
    proc_shift: float = 0.0
    proc_gamma: float = 1.0

def simulate_series(profiles: Dict[str, Dict[str, float]], weights: Dict[str, float], steps: int, scenario: str, p: SimParams):
    scenario = scenario.strip().upper()
    if scenario not in ("BAU", "SITE"):
        raise ValueError("scenario must be BAU or SITE")

    # normalize weights
    s = sum(weights.get(a, 0.0) for a in profiles)
    if s <= 0:
        w = {a: 1.0/len(profiles) for a in profiles}
    else:
        w = {a: float(weights.get(a, 0.0)/s) for a in profiles}

    if scenario == "BAU":
        # BAU: technocratic push with declining People, but allow Process effort to vary via process_max
        tech_in = _interp_curve([(0, 0.6), (1, p.tech_max)], steps, gamma=1.0)
        place_in = np.full(steps, 0.2, dtype=float)

        # Process effort can be increased (e.g., administrative effort) while People decays
        process_pts = _shift_points([(0, 0.4), (0.4, 0.6), (1, float(np.clip(p.process_max, 0.0, 1.5)))], p.proc_shift)
        people_pts  = _shift_points([(0, 0.4), (0.4, 0.2),  (1, 0.1)], p.proc_shift)

        process_in = _interp_curve(process_pts, steps, gamma=p.proc_gamma)
        people_in  = _interp_curve(people_pts,  steps, gamma=p.proc_gamma)
    else:
        tech_in = _interp_curve([(0, 0.45), (1, p.tech_max)], steps, gamma=1.0)
        place_mid = max(0.5, p.place_max * 0.65)
        proc_start = float(np.clip(0.70 * p.process_max, 0.0, 1.5))
        proc_mid   = float(np.clip(0.80 * p.process_max, 0.0, 1.5))
        proc_end   = float(np.clip(0.75 * p.process_max, 0.0, 1.5))
        process_pts = _shift_points([(0, proc_start), (0.4, proc_mid), (1, proc_end)], p.proc_shift)
        place_pts   = _shift_points([(0, 0.3), (0.6, place_mid), (1, p.place_max)], p.proc_shift)
        people_pts  = _shift_points([(0, 0.30), (0.6, 0.35), (0.8, 0.6), (1, p.people_max)], p.proc_shift)
        process_in = _interp_curve(process_pts, steps, gamma=p.proc_gamma)
        place_in   = _interp_curve(place_pts,   steps, gamma=p.proc_gamma)
        people_in  = _interp_curve(people_pts,  steps, gamma=p.proc_gamma)

    A = np.zeros(steps, dtype=float)
    rows = []
    for t in range(steps):
        tech_val = float(tech_in[t])
        proc_val = float(process_in[t])
        ppl_val  = float(people_in[t])
        plc_val  = float(place_in[t])

        interaction = proc_val * ppl_val
        synergy_bonus = p.synergy_gain if proc_val > p.tau_synergy else 0.0

        total = 0.0
        row = {"t": t, "A_total": None, "Technology_in": tech_val, "Process_in": proc_val, "People_in": ppl_val, "Place_in": plc_val, "Interaction": interaction}

        for agent, sens in profiles.items():
            tech_eff = tech_val * sens.get("Technology", 0.0) * max(interaction, 0.15)
            plc_eff  = plc_val  * sens.get("Place", 0.0)
            proc_eff = proc_val * sens.get("Process", 0.0) * (1.0 + synergy_bonus)
            ppl_eff  = ppl_val  * sens.get("People", 0.0) * (1.0 + synergy_bonus)

            if (interaction < p.interaction_low) and (tech_val > p.tech_high):
                if tech_eff > 0:
                    tech_eff = tech_eff / p.penalty_delta
                elif tech_eff < 0:
                    tech_eff = tech_eff * p.penalty_delta

            raw = tech_eff + plc_eff + proc_eff + ppl_eff
            acc = np.tanh(raw) * 100.0
            row[agent] = acc
            total += acc * w.get(agent, 0.0)

        row["A_total"] = total
        A[t] = total
        rows.append(row)

    return pd.DataFrame(rows), A

# =========================================================
# 5) Cached sensitivity functions
# =========================================================
@st.cache_data(show_spinner=False)
def run_process_tech_grid(profiles, weights, steps, penalty_delta, tau_synergy, synergy_gain, tech_vals, process_vals):
    rows = []
    for scen in ("BAU", "SITE"):
        for tech in tech_vals:
            for proc in process_vals:
                p = SimParams(
                    tech_max=float(tech),
                    process_max=float(proc),
                    penalty_delta=float(penalty_delta),
                    tau_synergy=float(tau_synergy),
                    synergy_gain=float(synergy_gain),
                )
                _, A = simulate_series(profiles, weights, steps, scen, p)
                met = calculate_metrics(A)
                rows.append({"scenario": scen, "tech_max": float(tech), "process_max": float(proc), **met})
    return pd.DataFrame(rows)

def boundary_curve_from_grid(df_grid, scenario, metric="mean"):
    out = []
    sub = df_grid[df_grid["scenario"] == scenario].copy()
    for t in sorted(sub["tech_max"].unique()):
        s = sub[sub["tech_max"] == t].sort_values("process_max")
        ok = s[s[metric] > 0]
        out.append((float(t), np.nan if ok.empty else float(ok["process_max"].iloc[0])))
    return pd.DataFrame(out, columns=["tech_max", f"min_process_for_{metric}_gt0"])


# =========================================================
# 5b) Supplementary plot helpers (S9, S10a, S10b)
# =========================================================
def _find_zero_cross_piecewise(x: np.ndarray, y: np.ndarray, target: float = 0.0):
    \"\"\"Piecewise-linear zero-cross within observed grid (no extrapolation).\"\"\"
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    for i in range(1, len(x)):
        if (y[i-1] < target and y[i] >= target) or (y[i-1] > target and y[i] <= target):
            if y[i] == y[i-1]:
                return float(x[i])
            return float(x[i-1] + (target - y[i-1]) * (x[i] - x[i-1]) / (y[i] - y[i-1]))
    return None

def make_s10a_slice_plot(df_grid: pd.DataFrame, tech_slice: float, tau_synergy: float):
    \"\"\"S10a: Representative slice (tech_max fixed). Mean acceptance vs Process (BAU vs SITE).\"\"\"
    bau = df_grid[(df_grid["scenario"]=="BAU") & (np.isclose(df_grid["tech_max"], tech_slice))].sort_values("process_max")
    site = df_grid[(df_grid["scenario"]=="SITE") & (np.isclose(df_grid["tech_max"], tech_slice))].sort_values("process_max")

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(bau["process_max"], bau["mean"], marker="o", linewidth=2.5, color="darkred", label="BAU (mean)")
    ax.plot(site["process_max"], site["mean"], marker="o", linewidth=2.5, color="darkblue", label="SITE (mean)")

    ax.axhline(0, linestyle="--", linewidth=1)
    ax.axvline(tau_synergy, linestyle="--", linewidth=1)
    ax.text(tau_synergy + 0.01, ax.get_ylim()[1]*0.92, f"Synergy trigger (Process={tau_synergy:.2f})", va="top")

    # BAU zero-cross (if present)
    bau_cross = _find_zero_cross_piecewise(bau["process_max"].to_numpy(), bau["mean"].to_numpy(), 0.0)
    if bau_cross is not None:
        ax.scatter([bau_cross], [0], s=260, color="darkred", edgecolor="white", linewidth=0.8, zorder=5)
        ax.annotate(
            f"BAU mean zero-cross ≈ {bau_cross:.2f}",
            xy=(bau_cross, 0),
            xytext=(bau_cross + 0.06, 0.12*(ax.get_ylim()[1]-ax.get_ylim()[0])),
            ha="left", va="bottom",
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="gray", alpha=0.9),
            arrowprops=dict(arrowstyle="->", lw=0.8, color="gray")
        )

    # SITE note if always positive across observed grid
    if len(site) > 0 and float(site["mean"].min()) > 0:
        ax.annotate(
            "SITE mean > 0 across observed Process grid\n(boundary lies below min grid)",
            xy=(float(site["process_max"].iloc[0]), float(site["mean"].iloc[0])),
            xytext=(float(site["process_max"].iloc[0]) + 0.08, float(site["mean"].iloc[0]) + 0.15*(ax.get_ylim()[1]-ax.get_ylim()[0])),
            ha="left", va="bottom",
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="gray", alpha=0.9),
            arrowprops=dict(arrowstyle="->", lw=0.8, color="gray")
        )

    ax.set_title(f"Representative slice (tech_max={tech_slice}). Mean acceptance vs Process")
    ax.set_xlabel("Process input (normalized; observed grid range)")
    ax.set_ylabel("Mean Total Acceptance Index")
    ax.legend()
    fig.tight_layout()
    return fig

def make_s10b_boundary_curve_plot(df_grid: pd.DataFrame):
    \"\"\"S10b: Boundary curves across tech_max (mean acceptance criterion).\"\"\"
    bau_bc = boundary_curve_from_grid(df_grid, "BAU", metric="mean")
    site_bc = boundary_curve_from_grid(df_grid, "SITE", metric="mean")

    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.plot(bau_bc["tech_max"], bau_bc["min_process_for_mean_gt0"], marker="o", linewidth=2.5, color="darkred", label="BAU boundary (mean>0)")
    ax.plot(site_bc["tech_max"], site_bc["min_process_for_mean_gt0"], marker="o", linewidth=2.5, color="darkblue", label="SITE boundary (mean>0)")

    ax.set_title("Viability boundary curves across technology intensity (mean acceptance criterion)")
    ax.set_xlabel("tech_max (normalized)")
    ax.set_ylabel("Min Process for mean(A_total)>0")
    ax.legend()
    fig.tight_layout()
    return fig

@st.cache_data(show_spinner=False)
def run_s9_robustness_screening(
    profiles: Dict[str, Dict[str, float]],
    base_weights: Dict[str, float],
    tech_fixed: float,
    deltas: Tuple[float, ...],
    n_samples: int,
    process_vals: Tuple[float, ...],
    steps: int,
    tau_synergy: float,
    synergy_gain: float,
):
    \"\"\"S9: robustness screening over ω_i (Dirichlet around base_weights) and δ at fixed tech_max.\"\"\"
    base = np.array([base_weights.get("F1", 0.25), base_weights.get("F2", 0.25), base_weights.get("F3", 0.25), base_weights.get("F4", 0.25)], float)
    base = base / base.sum()
    alpha = base * 30.0  # concentration around base
    local_rng = np.random.default_rng(RNG_SEED)

    rows = []
    for d in deltas:
        for _ in range(n_samples):
            wv = local_rng.dirichlet(alpha)
            w = {"F1": float(wv[0]), "F2": float(wv[1]), "F3": float(wv[2]), "F4": float(wv[3])}

            # BAU boundary (mean>0) as function of process_max
            bau_means = []
            bau_finals = []
            for proc in process_vals:
                pB = SimParams(
                    tech_max=float(tech_fixed),
                    process_max=float(proc),
                    penalty_delta=float(d),
                    tau_synergy=float(tau_synergy),
                    synergy_gain=float(synergy_gain),
                )
                _, A = simulate_series(profiles, w, steps, "BAU", pB)
                met = calculate_metrics(A)
                bau_means.append(met["mean"])
                bau_finals.append(met["final"])

            bau_means = np.asarray(bau_means, float)
            bau_finals = np.asarray(bau_finals, float)
            idx = np.where(bau_means > 0)[0]
            bau_b = np.nan if len(idx) == 0 else float(process_vals[idx[0]])
            deadlock_like = bool((bau_means[0] <= 0) and (bau_finals[-1] <= 0))

            # SITE boundary (mean>0)
            site_means = []
            for proc in process_vals:
                pS = SimParams(
                    tech_max=float(tech_fixed),
                    process_max=float(proc),
                    penalty_delta=float(d),
                    tau_synergy=float(tau_synergy),
                    synergy_gain=float(synergy_gain),
                )
                _, A = simulate_series(profiles, w, steps, "SITE", pS)
                site_means.append(calculate_metrics(A)["mean"])

            site_means = np.asarray(site_means, float)
            idx2 = np.where(site_means > 0)[0]
            site_b = np.nan if len(idx2) == 0 else float(process_vals[idx2[0]])

            rows.append({
                "delta": float(d),
                "bau_mean_boundary": float(bau_b) if not np.isnan(bau_b) else np.nan,
                "site_mean_boundary": float(site_b) if not np.isnan(site_b) else np.nan,
                "improvement_bau_minus_site": float(bau_b - site_b) if (not np.isnan(bau_b) and not np.isnan(site_b)) else np.nan,
                "deadlock_like": deadlock_like,
            })
    return pd.DataFrame(rows)

def make_s9_plot(df_s9: pd.DataFrame, tech_fixed: float):
    \"\"\"S9: clarified two-panel plot (boxplots + deadlock-like rate).\"\"\"
    deltas = np.array(sorted(df_s9["delta"].unique()))
    data_bau = [df_s9[df_s9["delta"] == d]["bau_mean_boundary"].dropna().values for d in deltas]
    data_site = [df_s9[df_s9["delta"] == d]["site_mean_boundary"].dropna().values for d in deltas]
    deadlock_rate = [df_s9[df_s9["delta"] == d]["deadlock_like"].mean() for d in deltas]

    pos = np.arange(len(deltas))

    fig = plt.figure(figsize=(12, 7))
    gs = fig.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0.15)

    ax = fig.add_subplot(gs[0, 0])
    bp_bau = ax.boxplot(data_bau, positions=pos - 0.18, widths=0.30, patch_artist=True, showfliers=False)
    bp_site = ax.boxplot(data_site, positions=pos + 0.18, widths=0.30, patch_artist=True, showfliers=False)

    for b in bp_bau["boxes"]:
        b.set(facecolor="white", hatch="")
    for b in bp_site["boxes"]:
        b.set(facecolor="white", hatch="///")

    ax.set_xticks(pos)
    ax.set_xticklabels([f"{d:.1f}" for d in deltas])
    ax.set_ylabel(f"Min Process for mean(A_total)>0\n(coarse screening; tech_max={tech_fixed})")
    ax.set_title("Supplementary S9: Robustness to ω_i and δ (BAU vs SITE boundaries)")

    # manual legend proxies
    import matplotlib.patches as mpatches
    bau_patch = mpatches.Patch(facecolor="white", edgecolor="black", hatch="", label="BAU boundary (mean>0)")
    site_patch = mpatches.Patch(facecolor="white", edgecolor="black", hatch="///", label="SITE boundary (mean>0)")
    ax.legend(handles=[bau_patch, site_patch], loc="upper left", frameon=True)

    ax2 = fig.add_subplot(gs[1, 0], sharex=ax)
    ax2.plot(pos, deadlock_rate, marker="o")
    ax2.set_ylabel("Deadlock-like rate\n(BAU)")
    ax2.set_xlabel("Deadlock penalty δ")
    ax2.set_ylim(0, max(deadlock_rate) * 1.15 if len(deadlock_rate) else 1.0)
    ax2.text(
        0.01, 0.85,
        "deadlock-like = (BAU mean≤0 at lowest Process) AND (BAU final≤0 at highest Process)\nwithin the coarse grid used for screening",
        transform=ax2.transAxes, va="top"
    )

    fig.tight_layout()
    return fig

# =========================================================
# 6) UI
# =========================================================
st.title("Final Q-TPPP + ABM Pre-evaluation (Nature Energy)")
st.caption("Integrated Streamlit app for NE submission: Q-methodology (4 factors) → TPPP → SITE vs BAU simulation + sensitivity.")

uploaded_file = st.sidebar.file_uploader("Upload Final CSV (Q01–Q24 required)", type=["csv"])
if not uploaded_file:
    st.info("Upload a CSV to start.")
    st.stop()

df_raw = pd.read_csv(uploaded_file)
q_cols = [c for c in df_raw.columns if c.startswith("Q") and c[1:].isdigit() and int(c[1:]) <= 24]
q_cols = sorted(q_cols, key=lambda x: int(x[1:]))
if len(q_cols) < 24:
    st.error("Invalid data: requires columns Q01–Q24.")
    st.stop()

engine = QEngine(df_raw[q_cols], n_factors=4).fit()

fa_df = pd.DataFrame(engine.factor_arrays, index=q_cols, columns=FACTOR_NAMES)
fa_df_display = fa_df.copy()
fa_df_display.insert(0, "Category", [Q_TO_TPPP.get(q) for q in fa_df_display.index])
fa_df_display.insert(1, "Statement", [Q_MAP.get(q) for q in fa_df_display.index])

tppp_scores = calculate_tppp_scores(df_raw[q_cols])
corr_matrix = tppp_scores.corr(method="spearman")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "1) Typology (Factor Arrays)",
    "2) TPPP Profile (Radar)",
    "3) TPPP Links (Spearman)",
    "4) Respondent Loadings & Weights",
    "5) SITE vs BAU Simulation (Tab 5)",
])

with tab1:
    st.subheader("Factor Arrays (4 perspectives)")
    st.dataframe(fa_df_display, use_container_width=True)
    st.download_button("Download factor arrays CSV", fa_df_display.to_csv(index=True).encode("utf-8-sig"), "factor_arrays_final.csv", "text/csv")

with tab2:
    st.subheader("TPPP Structural Perception (Radar)")
    tppp_profile = calculate_type_tppp_profile(engine.factor_arrays, q_cols)
    fig = go.Figure()
    categories = list(tppp_profile.index)
    for col in tppp_profile.columns:
        fig.add_trace(go.Scatterpolar(r=tppp_profile[col], theta=categories, fill="toself", name=col))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[-1.5, 1.5])), title="TPPP Radar Chart")
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(tppp_profile, use_container_width=True)

with tab3:
    st.subheader("TPPP Association Structure (Spearman)")
    threshold = st.slider("Link threshold |ρ| >", 0.0, 0.8, 0.25, 0.05)
    c1, c2 = st.columns([1, 1.5])
    with c1:
        st.markdown("**Spearman correlation matrix**")
        st.dataframe(corr_matrix, use_container_width=True)
    with c2:
        st.markdown("**System network**")
        st.plotly_chart(create_system_network(corr_matrix, threshold), use_container_width=True)

with tab4:
    st.subheader("Respondent assignments & population weights")
    loadings_df = pd.DataFrame(engine.loadings, columns=FACTOR_NAMES)
    loadings_df["Assigned Type"] = loadings_df[FACTOR_NAMES].abs().idxmax(axis=1)
    summary = loadings_df["Assigned Type"].value_counts().sort_index()
    total = len(loadings_df)
    weights_est = (summary / total).to_dict()
    summary_df = pd.DataFrame({
        "Assigned Type": summary.index,
        "Count": summary.values,
        "Calculated Weight": [weights_est[k] for k in summary.index],
        "Percentage": [f"{weights_est[k]*100:.1f}%" for k in summary.index],
    })
    c1, c2 = st.columns([1, 2])
    with c1:
        st.dataframe(summary_df, use_container_width=True)
    with c2:
        st.dataframe(loadings_df, use_container_width=True)

with tab5:
    st.subheader("SITE vs BAU simulation + sensitivity (integrated)")
    st.caption("Tab 5 includes Supplementary-style figures: S9 (robustness), S10a (slice), S10b (boundary curves).")

    profiles = calculate_agent_profiles_from_factor_arrays(fa_df)
    st.markdown("**Agent sensitivities (median z by category)**")
    st.dataframe(pd.DataFrame(profiles).T, use_container_width=True)

    # --- weights and key parameters ---
    st.markdown("#### Controls")
    c1, c2, c3, c4 = st.columns(4)
    w1 = c1.number_input("F1 weight", 0.0, 1.0, float(weights_est.get("F1", 0.25)))
    w2 = c2.number_input("F2 weight", 0.0, 1.0, float(weights_est.get("F2", 0.25)))
    w3 = c3.number_input("F3 weight", 0.0, 1.0, float(weights_est.get("F3", 0.25)))
    w4 = c4.number_input("F4 weight", 0.0, 1.0, float(weights_est.get("F4", 0.25)))
    w_custom = {"F1": w1, "F2": w2, "F3": w3, "F4": w4}

    d1, d2, d3, d4 = st.columns(4)
    penalty_delta = d1.slider("Deadlock penalty δ", 1.0, 3.0, 1.5, 0.1)
    tau_synergy = d2.slider("Synergy threshold τ", 0.3, 0.9, 0.6, 0.05)
    synergy_gain = d3.slider("Synergy gain", 0.0, 1.0, 0.5, 0.05)
    steps = d4.slider("Simulation steps", 12, 60, 24)

    # --- illustrative trajectories ---
    st.markdown("#### Scenario trajectories (illustrative)")
    t1, t2, t3 = st.columns(3)
    tech_max = t1.slider("tech_max (illustrative)", 0.5, 1.5, 1.2, 0.05)
    process_max = t2.slider("process_max (controls Process effort)", 0.5, 1.5, 1.0, 0.05)
    proc_shift = t3.slider("proc_shift (timing lead/lag)", -0.2, 0.2, 0.0, 0.05)

    params = SimParams(
        tech_max=float(tech_max),
        process_max=float(process_max),
        penalty_delta=float(penalty_delta),
        tau_synergy=float(tau_synergy),
        synergy_gain=float(synergy_gain),
        proc_shift=float(proc_shift),
        proc_gamma=1.0
    )

    hist_bau, A_bau = simulate_series(profiles, w_custom, int(steps), "BAU", params)
    hist_site, A_site = simulate_series(profiles, w_custom, int(steps), "SITE", params)

    met_bau = calculate_metrics(A_bau)
    met_site = calculate_metrics(A_site)

    fig_ts = go.Figure()
    fig_ts.add_trace(go.Scatter(x=hist_bau["t"], y=hist_bau["A_total"], name="BAU A_total", line=dict(color="darkred", width=3)))
    fig_ts.add_trace(go.Scatter(x=hist_site["t"], y=hist_site["A_total"], name="SITE A_total", line=dict(color="darkblue", width=3)))
    fig_ts.add_hline(y=0, line_dash="dash", line_color="black")
    fig_ts.update_layout(title="A_total trajectories", xaxis_title="t", yaxis_title="A_total (±100)", template="plotly_white")
    st.plotly_chart(fig_ts, use_container_width=True)

    cA, cB = st.columns(2)
    with cA:
        st.markdown("**BAU metrics**")
        st.json(met_bau)
    with cB:
        st.markdown("**SITE metrics**")
        st.json(met_site)

    st.markdown("---")
    st.markdown("## Supplementary S10 (mean acceptance criterion)")
    st.caption("S10a: representative slice at tech_max=1.2. S10b: boundary curves across technology intensity.")

    # Keep grid in session_state so S10 plots can render after running.
    if "df_grid" not in st.session_state:
        st.session_state["df_grid"] = None

    tech_vals = tuple(np.linspace(0.5, 1.5, 20).round(6))
    process_vals = tuple(np.linspace(0.5, 1.5, 20).round(6))

    if st.button("Run S10 grid search (BAU & SITE)", type="primary"):
        with st.spinner("Running process×tech grid…"):
            st.session_state["df_grid"] = run_process_tech_grid(
                profiles=profiles,
                weights=w_custom,
                steps=int(steps),
                penalty_delta=float(penalty_delta),
                tau_synergy=float(tau_synergy),
                synergy_gain=float(synergy_gain),
                tech_vals=tech_vals,
                process_vals=process_vals,
            )
        st.success("S10 grid complete.")

    if st.session_state["df_grid"] is not None:
        df_grid = st.session_state["df_grid"]

        # S10a (slice at tech=1.2)
        fig_s10a = make_s10a_slice_plot(df_grid, tech_slice=1.2, tau_synergy=float(tau_synergy))
        st.pyplot(fig_s10a, use_container_width=True)

        # S10b (boundary curves)
        fig_s10b = make_s10b_boundary_curve_plot(df_grid)
        st.pyplot(fig_s10b, use_container_width=True)

        # downloads
        st.download_button("Download S10 grid CSV", df_grid.to_csv(index=False).encode("utf-8-sig"), "S10_process_tech_grid.csv", "text/csv")

    st.markdown("---")
    st.markdown("## Supplementary S9 (robustness screening: ω_i × δ)")
    st.caption("S9 screens robustness of the BAU vs SITE ordering by sampling ω_i around estimated weights and sweeping δ (coarse process grid).")

    if "df_s9" not in st.session_state:
        st.session_state["df_s9"] = None

    r1, r2, r3 = st.columns(3)
    tech_fixed = r1.slider("tech_max for S9", 0.5, 1.5, 1.2, 0.05)
    n_samples = r2.slider("ω samples per δ", 50, 300, 150, 50)
    deltas = r3.multiselect("δ values", [1.0, 1.2, 1.5, 1.8, 2.1, 2.5], default=[1.0, 1.5, 2.5])

    if st.button("Run S9 robustness screening"):
        if len(deltas) == 0:
            st.warning("Select at least one δ.")
        else:
            with st.spinner("Running S9 robustness screening…"):
                process_vals_r = tuple(np.linspace(0.5, 1.5, 12).round(6))  # coarse screening grid
                st.session_state["df_s9"] = run_s9_robustness_screening(
                    profiles=profiles,
                    base_weights=weights_est,
                    tech_fixed=float(tech_fixed),
                    deltas=tuple(float(d) for d in deltas),
                    n_samples=int(n_samples),
                    process_vals=process_vals_r,
                    steps=int(max(12, steps//2)),
                    tau_synergy=float(tau_synergy),
                    synergy_gain=float(synergy_gain),
                )
            st.success("S9 screening complete.")

    if st.session_state["df_s9"] is not None:
        df_s9 = st.session_state["df_s9"]
        fig_s9 = make_s9_plot(df_s9, tech_fixed=float(tech_fixed))
        st.pyplot(fig_s9, use_container_width=True)
        st.download_button("Download S9 CSV", df_s9.to_csv(index=False).encode("utf-8-sig"), "S9_robustness_runs.csv", "text/csv")

    st.markdown("---")
    st.markdown("**Concise claim for NE**")
    st.write(
        "These simulation-based evaluations provide preliminary, model-contingent evidence that the SITE protocol is more likely than BAU to achieve net-positive consensus across plausible configurations, particularly when procedural legitimacy is established early."
    )
