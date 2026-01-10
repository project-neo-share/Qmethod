# -*- coding: utf-8 -*-
"""
Final Q-Methodology Analysis (Fixed 4 Factors + System Dynamics)
- Purpose: Generate final report data for Nature Energy submission.
- Update: 
  - [NEW] Tab 4: Added Respondent Type Assignment & Weight Calculation Table.
"""

import io
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import spearmanr, norm as normal_dist
import itertools

# ==========================================
# 1. Configuration & Constants
# ==========================================
st.set_page_config(page_title="Final Q-Analysis (Nature Energy Ver.)", layout="wide")

RNG_SEED = 42
rng = np.random.default_rng(RNG_SEED)

# 24 Statements
STATEMENTS = [
    "데이터센터는 재생에너지를 사용할 때 환경 책임성을 갖춘 시설로 평가받을 수 있다.", # Q01
    "디젤이나 가스 발전기를 백업 전력으로 사용할 경우 환경적 우려가 제기될 수 있다.", # Q02
    "물 절약이나 친환경 냉각 기술의 도입은 시민 신뢰에 긍정적 영향을 줄 수 있다.", # Q03
    "기술이 최신이더라도 안전성 확보가 부족하면 시민 불안을 유발할 수 있다.", # Q04
    "데이터센터 기술은 비용 효율성보다는 사회적 책임을 우선시해야 한다는 견해가 있다.", # Q05
    "기술이 낯설거나 복잡하게 인식되면 시민과의 거리감이 커질 수 있다.", # Q06
    "데이터센터 건설 과정에 시민 의견이 반영되지 않으면 반발 가능성이 높아질 수 있다.", # Q07
    "지역 사회와 장기적 관계를 맺어온 기업은 더 높은 신뢰를 받을 수 있다.", # Q08
    "설명회가 형식적으로 보일 경우, 시민 불신을 유발할 수 있다.", # Q09
    "정보 접근성이 낮을수록 시민의 불안과 의심이 증가할 수 있다.", # Q10
    "갈등 상황에서는 중립적 제3자의 개입이 조정에 도움이 될 수 있다.", # Q11
    "동일한 설명이라도 정부가 전달할 경우 기업보다 더 신뢰받을 가능성이 있다.", # Q12
    "기존 공장이나 발전소 부지를 재활용한 데이터센터는 수용성이 높아질 수 있다.", # Q13
    "지역 정체성과 조화를 이루지 못하는 입지는 거부감을 유발할 수 있다.", # Q14
    "자연경관 훼손이 발생하는 경우, 기술 우수성만으로 수용성 확보는 어려울 수 있다.", # Q15
    "수도권과 지방은 데이터센터 입지에 대해 서로 다른 기준을 가질 수 있다.", # Q16
    "외부 자본 주도의 일방적인 입지 결정은 지역사회의 신뢰를 저해할 수 있다.", # Q17
    "지역에 실질적인 혜택이 제공되면 시민 수용성이 높아질 수 있다.", # Q18
    "초기 단계에서 정보가 투명하게 공개되면 시민 신뢰가 높아질 수 있다.", # Q19
    "환경영향평가 결과는 시민들의 수용 여부에 중요한 판단 기준이 될 수 있다.", # Q20
    "기업과 지자체가 공동으로 결정한 프로젝트는 더 높은 신뢰를 얻을 수 있다.", # Q21
    "법적 요건을 충족하더라도 시민 신뢰를 확보하려면 추가적인 설명이 필요할 수 있다.", # Q22
    "지역 언론이 신속하고 정확하게 정보를 전달하면 신뢰성 제고에 기여할 수 있다.", # Q23
    "데이터센터 완공 이후에도 모니터링과 피드백 체계가 지속되면 신뢰 유지에 도움이 될 수 있다." # Q24
]

Q_MAP = {f"Q{i+1:02d}": txt for i, txt in enumerate(STATEMENTS)}

# TPPP Mapping
TPPP_CATEGORIES = {
    "Technology": ["Q01", "Q02", "Q03", "Q04", "Q05", "Q06"],
    "People": ["Q08", "Q09", "Q10", "Q12", "Q22", "Q23"],
    "Place": ["Q13", "Q14", "Q15", "Q16", "Q17", "Q18"],
    "Process": ["Q07", "Q11", "Q19", "Q20", "Q21", "Q24"]
}
Q_TO_TPPP = {}
for cat, items in TPPP_CATEGORIES.items():
    for item in items: Q_TO_TPPP[item] = cat

# ==========================================
# 2. Q-Methodology Logic
# ==========================================

def standardize_rows(X):
    mean = np.nanmean(X, axis=1, keepdims=True)
    std = np.nanstd(X, axis=1, ddof=1, keepdims=True)
    std[std == 0] = 1.0 
    return (X - mean) / std

class QEngine:
    def __init__(self, data_df, n_factors=4):
        self.q_df = data_df.select_dtypes(include=[np.number])
        temp_data = self.q_df.values
        row_means = np.nanmean(temp_data, axis=1)
        inds = np.where(np.isnan(temp_data))
        temp_data[inds] = np.take(row_means, inds[0])
        self.data = np.nan_to_num(temp_data, nan=0.0)
        self.n_factors = n_factors
        self.calculated_weights = {} 

    def fit(self):
        R, _ = spearmanr(self.data, axis=1)
        self.R = np.nan_to_num(R, nan=0.0)
        eigvals, eigvecs = np.linalg.eigh(self.R)
        idx = eigvals.argsort()[::-1]
        self.eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]
        k = self.n_factors
        valid_eigvals = np.maximum(self.eigvals[:k], 0)
        L = eigvecs[:, :k] * np.sqrt(valid_eigvals)
        self.loadings = self._varimax(L)
        
        # Calculate Population Weights from Loadings
        max_idxs = np.argmax(np.abs(self.loadings), axis=1)
        counts = {f"F{i+1}": 0 for i in range(k)}
        for i in max_idxs: counts[f"F{i+1}"] += 1
        total = len(max_idxs)
        self.calculated_weights = {k: v/total for k, v in counts.items()}

        z_data = standardize_rows(self.data)
        self.factor_arrays = self._calculate_factor_arrays(self.loadings, z_data)
        return self

    def _varimax(self, Phi, gamma=1.0, q=20, tol=1e-6):
        p, k = Phi.shape
        R = np.eye(k)
        d = 0
        for i in range(q):
            d_old = d
            Lambda = np.dot(Phi, R)
            u, s, vh = np.linalg.svd(
                np.dot(Phi.T, (Lambda**3 - (gamma/p) * np.dot(Lambda, np.diag(np.diag(np.dot(Lambda.T, Lambda))))))
            )
            R = np.dot(u, vh)
            d = np.sum(s)
            if d_old != 0 and d/d_old < 1 + tol: break
        return np.dot(Phi, R)

    def _calculate_factor_arrays(self, loadings, z_data):
        n_items = z_data.shape[1]
        arrays = np.zeros((n_items, self.n_factors))
        for f in range(self.n_factors):
            l_vec = loadings[:, f]
            l_clean = np.clip(l_vec, -0.95, 0.95)
            weights = l_clean / (1 - l_clean**2)
            if np.sum(np.abs(weights)) < 1e-6: continue
            weighted_sum = np.dot(weights, z_data)
            arr_mean = np.mean(weighted_sum)
            arr_std = np.std(weighted_sum, ddof=1)
            if arr_std == 0: arr_std = 1.0
            arrays[:, f] = (weighted_sum - arr_mean) / arr_std
        return arrays

def calculate_type_tppp_profile(factor_arrays, q_labels, mapping):
    df_arrays = pd.DataFrame(factor_arrays, index=q_labels)
    profiles = {}
    for i in range(factor_arrays.shape[1]):
        f_name = f"F{i+1}"
        cat_scores = {}
        for cat, items in mapping.items():
            valid_items = [item for item in items if item in df_arrays.index]
            if valid_items:
                cat_scores[cat] = df_arrays.loc[valid_items, i].mean()
        profiles[f_name] = cat_scores
    return pd.DataFrame(profiles)

def calculate_tppp_scores(df_q, mapping):
    scores = pd.DataFrame(index=df_q.index)
    for cat, items in mapping.items():
        valid_items = [i for i in items if i in df_q.columns]
        if valid_items:
            scores[cat] = df_q[valid_items].mean(axis=1)
    return scores

def create_system_network(corr_matrix, threshold=0.3):
    nodes = list(corr_matrix.columns)
    pos = {
        nodes[0]: (0, 1),   # Tech
        nodes[1]: (1, 0),   # People
        nodes[2]: (0, -1),  # Place
        nodes[3]: (-1, 0)   # Process
    }
    
    fig = go.Figure()
    
    for i in range(len(nodes)):
        for j in range(i+1, len(nodes)):
            n1, n2 = nodes[i], nodes[j]
            corr_val = corr_matrix.iloc[i, j]
            
            if abs(corr_val) >= threshold:
                x0, y0 = pos[n1]
                x1, y1 = pos[n2]
                color = '#E63946' if corr_val < 0 else '#457B9D'
                width = abs(corr_val) * 10
                
                fig.add_trace(go.Scatter(
                    x=[x0, x1, None],
                    y=[y0, y1, None],
                    mode='lines',
                    line=dict(width=width, color=color),
                    hoverinfo='text',
                    text=f"{n1} ↔ {n2}<br>Corr: {corr_val:.2f}",
                    showlegend=False
                ))

    node_x = [pos[n][0] for n in nodes]
    node_y = [pos[n][1] for n in nodes]
    
    fig.add_trace(go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        text=nodes,
        textposition=["top center", "middle right", "bottom center", "middle left"],
        textfont=dict(size=15, color='black'),
        marker=dict(size=45, color='white', line=dict(width=3, color='#333')),
        hoverinfo='none',
        name='Factors'
    ))
    
    fig.update_layout(
        title="TPPP System Dynamics (Feedback Loops)",
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-1.5, 1.5]),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-1.5, 1.5]),
        margin=dict(l=20, r=20, t=50, b=20),
        plot_bgcolor='white',
        height=600
    )
    return fig

def calculate_agent_profiles(df):
    profiles = {}
    factors = [c for c in df.columns if c.startswith('F') and c[1:].isdigit()]
    
    if 'Q_ID' not in df.columns:
        if len(df) == 24:
            df['Q_ID'] = [f"Q{i+1:02d}" for i in range(24)]
        else:
            df['Q_ID'] = [f"Q{i+1:02d}" for i in range(len(df))]
    
    for f in factors:
        agent_props = {}
        for cat, q_ids in TPPP_CATEGORIES.items():
            mask = df['Q_ID'].isin(q_ids)
            if mask.sum() > 0:
                median_z = df.loc[mask, f].median()
                agent_props[cat] = median_z
            else:
                agent_props[cat] = 0.0
        profiles[f] = agent_props
    return profiles

def run_simulation(profiles, steps=24, scenario="BAU", weights=None, sensitivity_params=None):
    history = []

    # --- defaults ---
    if sensitivity_params is None:
        sensitivity_params = {
            "tech_max": 1.2,
            "place_max": 0.9,
            "process_max": 1.2,
            "people_max": 1.0,
            "penalty": 1.5,
            "synergy_coeff": 0.8,
        }
    else:
        if "synergy_coeff" not in sensitivity_params:
            if "synergy_bonus" in sensitivity_params:
                sensitivity_params["synergy_coeff"] = sensitivity_params["synergy_bonus"]
            elif "synergy" in sensitivity_params:
                sensitivity_params["synergy_coeff"] = sensitivity_params["synergy"]
            else:
                sensitivity_params["synergy_coeff"] = 0.8

    # --- weights default ---
    if weights is None:
        weights = {agent: 1.0 / len(profiles) for agent in profiles}
    else:
        s = sum(weights.get(a, 0.0) for a in profiles)
        if s > 0:
            weights = {a: weights.get(a, 0.0) / s for a in profiles}
        else:
            weights = {agent: 1.0 / len(profiles) for agent in profiles}

    scenario_norm = str(scenario).strip()
    scenario_alias = {
        "BAU": "BAU (Technocratic Push)",
        "bau": "BAU (Technocratic Push)",
        "BAU (Technocratic Push)": "BAU (Technocratic Push)",
        "SITE": "SITE Protocol (Socio-Technical)",
        "site": "SITE Protocol (Socio-Technical)",
        "SITE Protocol (Socio-Technical)": "SITE Protocol (Socio-Technical)",
        "Sensitivity Test (Custom)": "Sensitivity Test (Custom)",
    }
    scenario_norm = scenario_alias.get(scenario_norm, scenario_norm)

    def profile_curve(points):
        t = np.linspace(0, 1, steps)
        xp, fp = zip(*points)
        return np.interp(t, xp, fp)

    # --- Policy Inputs based on scenario ---
    if scenario_norm == "BAU (Technocratic Push)":
        tech_max = sensitivity_params.get("tech_max", 1.2)
        process_floor = sensitivity_params.get("process_floor", 0.1)
        people_floor = sensitivity_params.get("people_floor", 0.1)

        tech_in = profile_curve([(0, 0.6), (1, tech_max)])
        place_in = np.full(steps, 0.2)
        process_in = profile_curve([(0, 0.4), (0.4, 0.25), (1, process_floor)])
        people_in = profile_curve([(0, 0.4), (0.4, 0.2), (1, people_floor)])

    elif scenario_norm == "SITE Protocol (Socio-Technical)":
        tech_max = sensitivity_params.get("tech_max", 0.85)
        place_max = sensitivity_params.get("place_max", 0.9)
        place_mid = max(0.5, place_max * 0.65)
        process_start = sensitivity_params.get("process_start", 0.7)
        process_mid = sensitivity_params.get("process_mid", 0.8)
        process_end = sensitivity_params.get("process_end", 0.75)
        people_peak = sensitivity_params.get("people_max", 1.0)

        tech_in = profile_curve([(0, 0.45), (1, tech_max)])
        process_in = profile_curve([(0, process_start), (0.4, process_mid), (1, process_end)])
        place_in = profile_curve([(0, 0.3), (0.6, place_mid), (1, place_max)])
        people_in = profile_curve([(0, 0.30), (0.6, 0.35), (0.8, 0.6), (1, people_peak)])

    elif scenario_norm == "Sensitivity Test (Custom)":
        tech_in = np.linspace(0.4, sensitivity_params.get("tech_max", 0.8), steps)
        place_in = np.linspace(0.4, sensitivity_params.get("place_max", 0.9), steps)
        process_in = np.linspace(0.5, sensitivity_params.get("process_max", 1.2), steps)
        people_in = np.linspace(0.4, sensitivity_params.get("people_max", 1.0), steps)

    else:
        tech_in = np.linspace(0.4, sensitivity_params.get("tech_max", 0.8), steps)
        place_in = np.linspace(0.4, sensitivity_params.get("place_max", 0.9), steps)
        process_in = np.linspace(0.5, sensitivity_params.get("process_max", 1.2), steps)
        people_in = np.linspace(0.4, sensitivity_params.get("people_max", 1.0), steps)

    synergy_coeff = sensitivity_params.get("synergy_coeff", 0.8)
    penalty_factor = sensitivity_params.get("penalty", 1.5)

    for t in range(steps):
        row = {"Step": t}
        total_acc = 0.0

        tech_val = tech_in
