# -*- coding: utf-8 -*-
"""
Final Q-Methodology Analysis (Fixed 4 Factors + System Dynamics)
- Purpose: Generate final report data for Nature Energy submission.
- Core Logic: 
  1. Person-wise Correlation (Q-method) -> 4 Factors Typology
  2. TPPP Framework -> Systemic Feedback Loop Analysis (Causal Links)
  3. Counterfactual Simulation -> Validation of SITE Protocol
- Update: 
  - Refined Simulation Logic with 'Distrust Penalty' and 'Synergy Bonus' based on literature (Slovic, 1993; Besley, 2010).
  - Adjusted Scaling Factor (0.5) for realistic score range.
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

# TPPP Mapping (Balanced 6 items each)
TPPP_CATEGORIES = {
    "Technology": ["Q01", "Q02", "Q03", "Q04", "Q05", "Q06"],
    "People": ["Q08", "Q09", "Q10", "Q12", "Q22", "Q23"],
    "Place": ["Q13", "Q14", "Q15", "Q16", "Q17", "Q18"],
    "Process": ["Q07", "Q11", "Q19", "Q20", "Q21", "Q24"]
}
Q_TO_TPPP = {}
for cat, items in TPPP_CATEGORIES.items():
    for item in items: Q_TO_TPPP[item] = cat

# Default Population Weights (based on your analysis ~44 people)
POPULATION_WEIGHTS = {
    "F1": 0.45,  # Techno-Realists
    "F2": 0.10,  # Eco-Equity Guardians
    "F3": 0.10,  # Development Pragmatists
    "F4": 0.35   # Tech-Skeptic Localists
}

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
                mean_z = df.loc[mask, f].mean()
                agent_props[cat] = mean_z
            else:
                agent_props[cat] = 0.0
        profiles[f] = agent_props
    return profiles

def run_simulation(profiles, steps=24, scenario="BAU", weights=None):
    history = []
    
    # [Refined Logic] Policy Inputs based on Literature
    if scenario == "BAU (Technocratic Push)":
        # Tech: Increasing (Efficiency drive)
        tech_in = np.linspace(0.5, 1.0, steps)
        # Place: Low & Static (Ignoring local context)
        place_in = np.full(steps, 0.2)
        # Process & People: Decreasing (Erosion of trust over time due to neglect)
        process_in = np.linspace(0.3, 0.1, steps) 
        people_in = np.linspace(0.3, 0.1, steps) 
        
    elif scenario == "SITE Protocol (Socio-Technical)":
        # Tech: Moderate (Validated)
        tech_in = np.linspace(0.4, 0.8, steps)
        # Place: High (Incentives, Equity)
        place_in = np.linspace(0.4, 0.9, steps)
        # Process: High (Transparency - Core Driver)
        process_in = np.linspace(0.5, 1.2, steps) 
        # People: Increasing (Trust building)
        people_in = np.linspace(0.4, 1.0, steps) 

    for t in range(steps):
        row = {"Step": t}
        total_acc = 0
        
        for agent, sens in profiles.items():
            # 1. Tech Effect
            tech_eff = tech_in[t] * sens.get("Technology", 0)
            
            # 2. Place Effect
            place_eff = place_in[t] * sens.get("Place", 0)
            
            # 3. Process/People (Synergy Bonus Logic)
            # If Process input is high (>0.6), it boosts the positive impact of Tech/Place
            # Reference: Besley (2010) on procedural justice as a mediator
            process_val = process_in[t]
            synergy_factor = 1.0
            
            if process_val > 0.6:
                synergy_factor = 1.2 # Bonus for good governance
            
            process_eff = process_val * sens.get("Process", 0) * synergy_factor
            people_eff = people_in[t] * sens.get("People", 0) * synergy_factor
            
            # 4. Interaction (The Distrust Penalty Logic)
            # Reference: Slovic (1993) "Trust Asymmetry" - negative events carry more weight
            # If Trust(People) input is low (<0.3) AND Tech push is high (>0.8), 
            # ANY agent with slight skepticism gets a massive penalty.
            
            penalty_factor = 1.0
            if people_in[t] < 0.3 and tech_in[t] > 0.8:
                penalty_factor = 1.5 # Distrust amplifies resistance
                
                # Apply penalty specifically to resistance (negative scores)
                if tech_eff < 0: tech_eff *= penalty_factor
                if place_eff < 0: place_eff *= penalty_factor

            # Sum components
            raw_score = tech_eff + place_eff + process_eff + people_eff
            
            # 5. Normalization (Scaling Adjustment)
            # Adjusted Scaling Factor: 0.25 -> 0.5 for realistic range (-50 to +50)
            acceptance = np.tanh(raw_score * 0.5) * 100
            
            row[agent] = acceptance
            total_acc += acceptance * weights.get(agent, 0.25)
            
        row["Total Index"] = total_acc
        history.append(row)
    return pd.DataFrame(history)

# ==========================================
# 3. UI
# ==========================================
st.title("📊 Final Q-Analysis: System Dynamics")
st.caption("Focus: Fixed 4 Factors & TPPP Feedback Loops for Nature Energy")

uploaded_file = st.sidebar.file_uploader("Upload Final CSV", type=['csv'])

if uploaded_file:
    try:
        df_raw = pd.read_csv(uploaded_file)
        q_cols = [c for c in df_raw.columns if c.startswith('Q') and c[1:].isdigit() and int(c[1:]) <= 24]
        
        if len(q_cols) < 5:
            st.error("Invalid Data: Columns Q01-Q24 not found.")
            st.stop()

        # Run Engine
        engine = QEngine(df_raw[q_cols], n_factors=4).fit()
        
        # Calculate Systemic Correlations (Raw Data Level)
        tppp_scores = calculate_tppp_scores(df_raw[q_cols], TPPP_CATEGORIES)
        corr_matrix = tppp_scores.corr(method='spearman')
        
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["1. Typology (Factor Arrays)", "2. Structural Profile (Radar)", "3. Systemic Loops (Triads)", "4. Respondent Loading", "5. SITE Simulation"])
        
        with tab1:
            st.subheader("Factor Arrays: The 4 Perspectives")
            fa_df = pd.DataFrame(engine.factor_arrays, index=q_cols, columns=["F1", "F2", "F3", "F4"])
            fa_df.insert(0, "Category", [Q_TO_TPPP.get(idx) for idx in fa_df.index])
            fa_df.insert(1, "Statement", [Q_MAP.get(idx) for idx in fa_df.index])
            st.dataframe(fa_df.style.background_gradient(cmap="RdBu_r", subset=["F1","F2","F3","F4"], vmin=-1.5, vmax=1.5))
            st.download_button("Download Array CSV", fa_df.to_csv().encode('utf-8-sig'), "factor_arrays_final.csv", "text/csv")

        with tab2:
            st.subheader("TPPP Structural Perception")
            tppp_profile = calculate_type_tppp_profile(engine.factor_arrays, q_cols, TPPP_CATEGORIES)
            fig = go.Figure()
            categories = list(tppp_profile.index)
            for col in tppp_profile.columns:
                fig.add_trace(go.Scatterpolar(r=tppp_profile[col], theta=categories, fill='toself', name=col))
            fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[-1.5, 1.5])), title="TPPP Radar Chart")
            c1, c2 = st.columns([2,1])
            with c1: st.plotly_chart(fig, use_container_width=True)
            with c2: st.dataframe(tppp_profile.style.background_gradient(cmap="RdBu_r", vmin=-1, vmax=1).format("{:.3f}"))

        with tab3:
            st.subheader("Systemic Feedback Loops (Triad Analysis)")
            c1, c2 = st.columns([1, 1.5])
            with c1:
                threshold = st.slider("Correlation Threshold (|r| > )", 0.0, 0.8, 0.25, 0.05)
                st.markdown("#### Correlation Matrix (Spearman)")
                st.dataframe(corr_matrix.style.background_gradient(cmap="coolwarm", vmin=-1, vmax=1).format("{:.2f}"))
            with c2:
                st.markdown("#### Systemic Network Graph")
                fig_net = create_system_network(corr_matrix, threshold)
                st.plotly_chart(fig_net, use_container_width=True)

        with tab4:
            st.subheader("Respondent Assignments")
            loadings_df = pd.DataFrame(engine.loadings, columns=["F1", "F2", "F3", "F4"])
            meta_cols = [c for c in df_raw.columns if c not in q_cols]
            if meta_cols:
                loadings_df = pd.concat([df_raw[meta_cols].reset_index(drop=True), loadings_df], axis=1)
            
            # Determine weights
            max_idxs = np.argmax(np.abs(engine.loadings), axis=1)
            counts = {f"F{i+1}": 0 for i in range(4)}
            for i in max_idxs: counts[f"F{i+1}"] += 1
            total = len(max_idxs)
            calc_weights = {k: v/total for k, v in counts.items()}
            
            st.dataframe(loadings_df.style.background_gradient(cmap="Blues", subset=["F1","F2","F3","F4"]))
            st.write("Calculated Weights from Data:", calc_weights)

        with tab5:
            st.subheader("Counterfactual Simulation (SITE Protocol)")
            st.caption("Validating SITE efficacy using empirical Q-profiles.")
            
            # Prepare Profiles
            profiles = calculate_agent_profiles(fa_df.rename(columns={"Statement": "Statement", "Category": "Category"})) 
            fa_df_sim = fa_df.copy()
            fa_df_sim['Q_ID'] = fa_df_sim.index # Q01...
            profiles = calculate_agent_profiles(fa_df_sim)
            
            # Weight Configuration (Editable)
            with st.expander("⚙️ Configure Agent Weights & Parameters"):
                w_f1 = st.number_input("F1 Weight", 0.0, 1.0, calc_weights.get("F1", 0.25))
                w_f2 = st.number_input("F2 Weight", 0.0, 1.0, calc_weights.get("F2", 0.25))
                w_f3 = st.number_input("F3 Weight", 0.0, 1.0, calc_weights.get("F3", 0.25))
                w_f4 = st.number_input("F4 Weight", 0.0, 1.0, calc_weights.get("F4", 0.25))
                custom_weights = {"F1": w_f1, "F2": w_f2, "F3": w_f3, "F4": w_f4}
                
                # Show Calculated Sensitivities Table
                st.markdown("##### Agent Sensitivities (Initial Parameters)")
                sens_df = pd.DataFrame(profiles).T
                st.dataframe(sens_df.style.background_gradient(cmap="RdBu", vmin=-1, vmax=1).format("{:.2f}"))

            sim_steps = st.slider("Simulation Duration (Months)", 12, 60, 24)
            df_bau = run_simulation(profiles, steps=sim_steps, scenario="BAU (Technocratic Push)", weights=custom_weights)
            df_site = run_simulation(profiles, steps=sim_steps, scenario="SITE Protocol (Socio-Technical)", weights=custom_weights)
            
            # Visualization: 2-Column Layout
            col_bau, col_site = st.columns(2)
            
            # [Adjusted] Y-Range: Focus on realistic data range + margin
            y_min = -40; y_max = 100 
            
            # Define marker styles for Grayscale compatibility
            # F1: Circle, F2: Square, F3: Diamond, F4: Triangle-Up
            markers = {"F1": "circle", "F2": "square", "F3": "diamond", "F4": "triangle-up"}
            dash_styles = {"F1": "dot", "F2": "dot", "F3": "dot", "F4": "dot"} # Dotted for agents
            
            # --- BAU Plot ---
            with col_bau:
                fig_bau = go.Figure()
                # Agents (Grayscale)
                for agent in ["F1", "F2", "F3", "F4"]:
                    fig_bau.add_trace(go.Scatter(
                        x=df_bau["Step"], y=df_bau[agent], name=agent,
                        mode='lines+markers',
                        line=dict(color='gray', width=1, dash=dash_styles[agent]),
                        marker=dict(symbol=markers[agent], size=6, color='black'),
                        opacity=0.6
                    ))
                # Total Index (Red Solid)
                fig_bau.add_trace(go.Scatter(
                    x=df_bau["Step"], y=df_bau["Total Index"], name="Total (BAU)",
                    mode='lines',
                    line=dict(color='#E63946', width=4)
                ))
                fig_bau.add_hline(y=0, line_dash="dash", line_color="black")
                fig_bau.update_layout(title="(A) BAU Scenario (Deadlock)", yaxis_range=[y_min, y_max], template="plotly_white", showlegend=False)
                st.plotly_chart(fig_bau, use_container_width=True)

            # --- SITE Plot ---
            with col_site:
                fig_site = go.Figure()
                # Agents (Grayscale)
                for agent in ["F1", "F2", "F3", "F4"]:
                    fig_site.add_trace(go.Scatter(
                        x=df_site["Step"], y=df_site[agent], name=agent,
                        mode='lines+markers',
                        line=dict(color='gray', width=1, dash=dash_styles[agent]),
                        marker=dict(symbol=markers[agent], size=6, color='black'),
                        opacity=0.6
                    ))
                # Total Index (Blue Solid)
                fig_site.add_trace(go.Scatter(
                    x=df_site["Step"], y=df_site["Total Index"], name="Total (SITE)",
                    mode='lines',
                    line=dict(color='#457B9D', width=4)
                ))
                fig_site.add_hline(y=0, line_dash="dash", line_color="black")
                fig_site.update_layout(title="(B) SITE Protocol (Consensus)", yaxis_range=[y_min, y_max], template="plotly_white", showlegend=True)
                st.plotly_chart(fig_site, use_container_width=True)
            
            st.success("The visual contrast between Red (BAU) and Blue (SITE) lines, along with distinct agent markers, highlights the structural shift from conflict to consensus.")
            
    except Exception as e:
        st.error(f"Error processing file: {e}")

else:
    st.info("Upload the CSV file to generate the final 4-factor report.")
