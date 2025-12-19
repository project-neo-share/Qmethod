# -*- coding: utf-8 -*-
"""
General Q-Methodology Analysis (Single Dataset) - TPPP Framework & Network Analysis
- Purpose: Q-factor analysis with 'Technology, People, Place, Process' Framework.
- Input: CSV data (Q01~Q24 + Metadata)
- Features:
  1. Factor Extraction & Rotation (PCA/Varimax)
  2. Factor Arrays (Z-scores)
  3. Distinguishing Statements
  4. TPPP Framework Mapping & Analysis
     - Correlation Matrix (Feedback Loops)
     - Type-based Radar Charts (Structural Perception)
  5. Network Analysis (Visualizing Feedback Loops)
  6. Factor Optimization (Scree Plot & Kaiser Rule)
  7. [NEW] Enhanced P-Set Profiling (Demographics Integration)
"""

import io
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import spearmanr, pearsonr, norm as normal_dist
import itertools

# ==========================================
# 1. Configuration & Constants
# ==========================================
st.set_page_config(page_title="General Q-Analysis", layout="wide")

RNG_SEED = 42
rng = np.random.default_rng(RNG_SEED)

# 24 Statements provided by the researcher
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

# Map Q01 -> Statement[0]
Q_MAP = {f"Q{i+1:02d}": txt for i, txt in enumerate(STATEMENTS)}

# TPPP Mapping (Based on content)
TPPP_CATEGORIES = {
    "Technology": ["Q01", "Q02", "Q03", "Q04", "Q05", "Q06", "Q24"],
    "People (Trust)": ["Q08", "Q09", "Q10", "Q11", "Q12", "Q22", "Q23"],
    "Place (Location)": ["Q13", "Q14", "Q15", "Q16", "Q17", "Q18"],
    "Process (Governance)": ["Q07", "Q19", "Q20", "Q21"]
}

# Reverse mapping for easy lookup
Q_TO_TPPP = {}
for cat, items in TPPP_CATEGORIES.items():
    for item in items:
        Q_TO_TPPP[item] = cat

# ==========================================
# 2. Math & Q-Logic Core
# ==========================================

def standardize_rows(X):
    """Row-wise Z-score normalization"""
    mean = np.nanmean(X, axis=1, keepdims=True)
    std = np.nanstd(X, axis=1, ddof=1, keepdims=True)
    std[std == 0] = 1.0 
    return (X - mean) / std

class QEngine:
    def __init__(self, data_df, n_factors=3, rotation=True):
        self.raw_df = data_df
        self.n_factors = n_factors
        self.rotation = rotation
        
        # Data Cleaning
        self.q_df = data_df.select_dtypes(include=[np.number])
        temp_data = self.q_df.values
        
        # Row-wise Mean Imputation
        row_means = np.nanmean(temp_data, axis=1)
        inds = np.where(np.isnan(temp_data))
        temp_data[inds] = np.take(row_means, inds[0])
        self.data = np.nan_to_num(temp_data, nan=0.0)
        
        self.n_persons, self.n_items = self.data.shape
        self.loadings = None
        self.factor_arrays = None
        self.explained_variance = None
        self.eigenvalues = None
        
    def fit(self):
        # 1. Correlation (Spearman for Likert)
        R, _ = spearmanr(self.data, axis=1)
        z_data = standardize_rows(self.data)
        R = np.nan_to_num(R, nan=0.0)
        
        # 2. Eigen Decomposition
        eigvals, eigvecs = np.linalg.eigh(R)
        idx = eigvals.argsort()[::-1]
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]
        self.eigenvalues = eigvals
        
        # 3. Extract Factors
        k = self.n_factors
        valid_eigvals = np.maximum(eigvals[:k], 0)
        L = eigvecs[:, :k] * np.sqrt(valid_eigvals)
        
        # 4. Varimax Rotation
        if self.rotation and k > 1:
            L = self._varimax(L)
            
        self.loadings = L
        self.explained_variance = eigvals[:k]
        
        # 5. Factor Arrays (Z-scores)
        self.factor_arrays = self._calculate_factor_arrays(L, z_data)
        
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
            
            w_abs_sum = np.sum(np.abs(weights))
            if w_abs_sum < 1e-6:
                arrays[:, f] = 0
                continue
            
            weighted_sum = np.dot(weights, z_data)
            arr_mean = np.mean(weighted_sum)
            arr_std = np.std(weighted_sum, ddof=1)
            if arr_std == 0: arr_std = 1.0
            arrays[:, f] = (weighted_sum - arr_mean) / arr_std
        return arrays

def strict_respondent_assignment(loadings, threshold=0.4, min_gap=0.1):
    """Assigns respondents to factors strictly."""
    assignments = []
    abs_loads = np.abs(loadings)
    for i in range(len(loadings)):
        row = abs_loads[i]
        max_idx = np.argmax(row)
        max_val = row[max_idx]
        sorted_row = np.sort(row)
        second_max = sorted_row[-2] if len(row) > 1 else 0
        
        if max_val < threshold:
            assignments.append("None (Low)")
        elif (max_val - second_max) < min_gap:
            assignments.append("Confounded")
        else:
            assignments.append(f"Type {max_idx+1}")
    return assignments

def find_distinguishing_items(factor_arrays, n_factors, item_labels, q_map, alpha=0.01):
    """Identifies distinguishing items"""
    col_names = [f"F{i+1}" for i in range(n_factors)]
    df_arrays = pd.DataFrame(factor_arrays, columns=col_names, index=item_labels)
    crit_z = normal_dist.ppf(1 - alpha/2)
    se = 0.3 

    dist_dict = {}
    for i in range(n_factors):
        target_col = f"F{i+1}"
        other_cols = [c for c in df_arrays.columns if c != target_col]
        if not other_cols: continue
        
        min_diff_val = pd.Series(np.inf, index=df_arrays.index)
        is_significant = pd.Series(True, index=df_arrays.index)
        
        for other in other_cols:
            diff = df_arrays[target_col] - df_arrays[other]
            z_stat = diff / (np.sqrt(2) * se)
            sig_check = (np.abs(z_stat) > crit_z)
            is_significant &= sig_check
            update_mask = np.abs(diff) < np.abs(min_diff_val)
            min_diff_val[update_mask] = diff[update_mask]
            
        dist_items = df_arrays[is_significant].copy()
        if not dist_items.empty:
            dist_items['Min Difference'] = min_diff_val[is_significant]
            dist_items['Direction'] = np.where(dist_items['Min Difference'] > 0, 'Higher', 'Lower')
            dist_items['Statement'] = [q_map.get(idx, "") for idx in dist_items.index]
            dist_items = dist_items.sort_values('Min Difference', ascending=False, key=abs)
            
            # Add TPPP Category
            dist_items['Category'] = [Q_TO_TPPP.get(idx, "Unknown") for idx in dist_items.index]
            
            cols = ['Category', 'Statement', 'Min Difference', 'Direction'] + col_names
            dist_dict[target_col] = dist_items[cols]
    return dist_dict

def calculate_tppp_scores(df_q, mapping):
    """Calculates average scores for TPPP categories per respondent"""
    scores = pd.DataFrame(index=df_q.index)
    for cat, items in mapping.items():
        # Only use items present in df_q
        valid_items = [i for i in items if i in df_q.columns]
        if valid_items:
            scores[cat] = df_q[valid_items].mean(axis=1)
    return scores

def calculate_type_tppp_profile(factor_arrays, q_labels, mapping):
    """Calculates average Z-score for TPPP categories per Factor Type"""
    df_arrays = pd.DataFrame(factor_arrays, index=q_labels)
    n_factors = df_arrays.shape[1]
    
    profiles = {}
    for i in range(n_factors):
        f_name = f"F{i+1}"
        cat_scores = {}
        for cat, items in mapping.items():
            valid_items = [item for item in items if item in df_arrays.index]
            if valid_items:
                cat_scores[cat] = df_arrays.loc[valid_items, i].mean()
        profiles[f_name] = cat_scores
        
    return pd.DataFrame(profiles)

def create_network_graph(corr_matrix, threshold):
    """Creates a Network Graph for TPPP Feedback Loops using Plotly"""
    
    # 4 Nodes fixed position (Circular)
    nodes = list(corr_matrix.columns)
    # Positions: Tech(Top), People(Right), Place(Bottom), Process(Left)
    pos = {
        nodes[0]: (0, 1),   # Tech
        nodes[1]: (1, 0),   # People
        nodes[2]: (0, -1),  # Place
        nodes[3]: (-1, 0)   # Process
    }
    
    edge_x = []
    edge_y = []
    edge_text = []
    
    # Add Edges if corr > threshold
    for i in range(len(nodes)):
        for j in range(i+1, len(nodes)):
            n1, n2 = nodes[i], nodes[j]
            corr_val = corr_matrix.iloc[i, j]
            
            if abs(corr_val) >= threshold:
                x0, y0 = pos[n1]
                x1, y1 = pos[n2]
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])
                edge_text.append(f"{n1}-{n2}: {corr_val:.2f}")

    # Edge Trace
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=2, color='#888'),
        hoverinfo='text',
        text=str(edge_text),
        mode='lines')

    # Node Trace
    node_x = [pos[n][0] for n in nodes]
    node_y = [pos[n][1] for n in nodes]
    
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        text=nodes,
        textposition="top center",
        hoverinfo='text',
        marker=dict(
            showscale=False,
            color='#1f77b4',
            size=30,
            line_width=2))

    fig = go.Figure(data=[edge_trace, node_trace],
                 layout=go.Layout(
                    title=f'TPPP Feedback Loops (Correlation > {threshold})',
                    title_font_size=16,
                    showlegend=False,
                    hovermode='closest',
                    margin=dict(b=20,l=5,r=5,t=40),
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                )
    return fig

def detect_strongest_loops(corr_matrix):
    """Detects 3-node feedback loops based on correlation strength"""
    cols = corr_matrix.columns.tolist()
    triads = []
    
    # Iterate all combinations of 3
    for triad in itertools.combinations(cols, 3):
        a, b, c = triad
        # Sum of correlations (Strength of loop)
        score = abs(corr_matrix.loc[a, b]) + abs(corr_matrix.loc[b, c]) + abs(corr_matrix.loc[c, a])
        avg_score = score / 3
        triads.append({
            "Loop": f"{a} ↔ {b} ↔ {c}",
            "Strength (Avg Corr)": avg_score,
            "Links": [f"{corr_matrix.loc[a,b]:.2f}", f"{corr_matrix.loc[b,c]:.2f}", f"{corr_matrix.loc[c,a]:.2f}"]
        })
    
    return pd.DataFrame(triads).sort_values("Strength (Avg Corr)", ascending=False)

def plot_scree(eigenvalues):
    """Plots Scree Plot for Factor Selection"""
    x = range(1, len(eigenvalues) + 1)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(x), y=eigenvalues, mode='lines+markers', name='Eigenvalue'))
    fig.add_hline(y=1.0, line_dash="dash", line_color="red", annotation_text="Kaiser Criterion (1.0)")
    
    fig.update_layout(
        title="Scree Plot (Eigenvalues)",
        xaxis_title="Factor Number",
        yaxis_title="Eigenvalue",
        template="plotly_white"
    )
    return fig

# ==========================================
# 3. Main UI
# ==========================================

st.title("📊 General Q-Analysis: TPPP Framework")
st.markdown("### TPPP (Technology, People, Place, Process) 중심 분석")

# File Upload
uploaded_file = st.sidebar.file_uploader("Upload Responses CSV", type=['csv'])

if uploaded_file:
    try:
        df_raw = pd.read_csv(uploaded_file)
        q_cols = [c for c in df_raw.columns if c.startswith('Q') and c[1:].isdigit() and int(c[1:]) <= 24]
        
        if len(q_cols) < 5:
            st.error("데이터에서 Q01~Q24 컬럼을 찾을 수 없습니다.")
            st.stop()
            
        df_q = df_raw[q_cols]
        id_col = next((c for c in df_raw.columns if 'email' in c.lower()), None)
        ids = df_raw[id_col] if id_col else [f"P{i}" for i in range(len(df_raw))]
        meta_cols = [c for c in df_raw.columns if c not in q_cols and c != id_col]
        
        st.sidebar.success(f"Loaded {len(df_raw)} respondents.")
        
    except Exception as e:
        st.error(f"File Load Error: {e}")
        st.stop()

    with st.sidebar:
        st.header("Analysis Settings")
        n_factors = st.number_input("Number of Factors", 1, 10, 3)
        assign_thr = st.slider("Assignment Threshold (>)", 0.3, 0.7, 0.4, 0.05)
        assign_gap = st.slider("Confounded Gap (>)", 0.05, 0.3, 0.1, 0.05)

    # Run Engine
    engine = QEngine(df_q, n_factors=n_factors).fit()
    assignments = strict_respondent_assignment(engine.loadings, threshold=assign_thr, min_gap=assign_gap)
    
    # Calculate scores for TPPP analysis
    tppp_scores = calculate_tppp_scores(df_q, TPPP_CATEGORIES)
    corr_matrix = tppp_scores.corr(method='spearman')

    # Prepare Metadata for P-Set Analysis
    df_meta = df_raw[meta_cols].copy()
    df_meta['Assigned Type'] = assignments
    df_meta['ID'] = ids

    # Tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "1. Factor Structure", 
        "2. TPPP Profiles (Radar)", 
        "3. TPPP Network & Loops",
        "4. Distinguishing Statements",
        "5. Raw Data & Arrays",
        "6. P-Set Profiling (Demographics)"
    ])
    
    # --- Tab 1: Structure ---
    with tab1:
        st.header("1. Factor Optimization & Structure")
        
        # Factor Optimization Section
        col_opt1, col_opt2 = st.columns([2, 1])
        with col_opt1:
            st.subheader("Optimal Factors (Scree Plot & Kaiser)")
            st.plotly_chart(plot_scree(engine.eigenvalues[:10]), use_container_width=True)
        
        with col_opt2:
            st.markdown("<br><br>", unsafe_allow_html=True) # Spacer
            kaiser_k = sum(engine.eigenvalues > 1.0)
            st.metric("Kaiser Criterion (k)", f"{kaiser_k} Factors", "Eigenvalue > 1.0")
            st.info(f"데이터 통계상 **{kaiser_k}개 요인**이 권장됩니다. (그래프의 꺾임새를 확인하세요)")

        st.divider()

        st.subheader("Respondents by Type")
        counts = pd.Series(assignments).value_counts().sort_index()
        c1, c2 = st.columns([1, 2])
        with c1:
            st.dataframe(pd.DataFrame({"Count": counts, "Ratio": (counts/len(assignments)*100).apply(lambda x: f"{x:.1f}%")}))
        with c2:
            st.bar_chart(counts)
        
        st.subheader("Factor Interpretation (Top Items)")
        # Show top 3 agreement/disagreement items per factor
        fa_df = pd.DataFrame(engine.factor_arrays, index=q_cols, columns=[f"F{i+1}" for i in range(n_factors)])
        
        cols = st.columns(n_factors)
        for i, col in enumerate(cols):
            f_key = f"F{i+1}"
            with col:
                st.markdown(f"**{f_key} Top/Bottom**")
                sorted_f = fa_df[f_key].sort_values(ascending=False)
                top3 = sorted_f.head(3)
                bot3 = sorted_f.tail(3)
                
                st.markdown("👍 **Strong Agreement**")
                for idx, val in top3.items():
                    st.caption(f"**{idx}** ({Q_TO_TPPP.get(idx)}): {Q_MAP[idx][:30]}... (z={val:.2f})")
                
                st.markdown("👎 **Strong Disagreement**")
                for idx, val in bot3.items():
                    st.caption(f"**{idx}** ({Q_TO_TPPP.get(idx)}): {Q_MAP[idx][:30]}... (z={val:.2f})")

    # --- Tab 2: TPPP Profiles ---
    with tab2:
        st.header("TPPP Perception Profiles (Radar Chart)")
        type_profiles = calculate_type_tppp_profile(engine.factor_arrays, q_cols, TPPP_CATEGORIES)
        
        categories = list(type_profiles.index)
        fig = go.Figure()

        for col in type_profiles.columns:
            fig.add_trace(go.Scatterpolar(
                r=type_profiles[col],
                theta=categories,
                fill='toself',
                name=col
            ))

        fig.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[-1.5, 1.5]) 
            ),
            showlegend=True,
            title="Type-specific TPPP Weighting (Z-scores)"
        )
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(type_profiles.style.background_gradient(cmap="RdBu_r", vmin=-1, vmax=1).format("{:.3f}"))

    # --- Tab 3: TPPP Network & Loops (NEW) ---
    with tab3:
        st.header("TPPP Network Analysis & Feedback Loops")
        st.markdown("4개 차원(기술, 사람, 장소, 절차) 간의 상관관계를 통해 **자기강화 루프(Self-reinforcing Loop)**를 탐색합니다.")
        
        c1, c2 = st.columns([1, 2])
        with c1:
            st.subheader("Settings")
            net_threshold = st.slider("Correlation Threshold", 0.0, 0.8, 0.4, 0.05, help="이 값 이상의 상관관계만 연결선으로 표시합니다.")
            
            st.subheader("Correlation Matrix")
            st.dataframe(corr_matrix.style.background_gradient(cmap="coolwarm", vmin=-1, vmax=1).format("{:.3f}"))
            
            st.subheader("Strongest Loop Detection (Triads)")
            loops_df = detect_strongest_loops(corr_matrix)
            st.dataframe(loops_df.style.format({"Strength (Avg Corr)": "{:.3f}"}))
            st.caption("가장 강한 평균 상관관계를 가진 3자 순환 고리입니다.")

        with c2:
            st.subheader("Network Visualization")
            fig_net = create_network_graph(corr_matrix, net_threshold)
            st.plotly_chart(fig_net, use_container_width=True)
            st.info("""
            **해석 가이드:**
            * **연결선(Edge):** 두 요소 간의 강한 상호작용을 의미합니다.
            * **삼각형(Triangle):** 3개의 요소가 서로 강하게 연결된 경우, 한 요소의 변화가 루프를 타고 전체 시스템을 강화(Positive Feedback)하거나 약화시킬 수 있는 핵심 구조입니다.
            """)

    # --- Tab 4: Distinguishing ---
    with tab4:
        st.subheader("Distinguishing Statements per Type")
        dist_dict = find_distinguishing_items(engine.factor_arrays, n_factors, q_cols, Q_MAP, alpha=0.05)
        
        d_tabs = st.tabs([f"Factor {i+1}" for i in range(n_factors)])
        for i, tab in enumerate(d_tabs):
            with tab:
                f_key = f"F{i+1}"
                res = dist_dict.get(f_key)
                if res is not None:
                    st.dataframe(
                        res.style.background_gradient(cmap="coolwarm", subset=["Min Difference"], vmin=-2, vmax=2)
                        .format({"Min Difference": "{:.2f}"}),
                        use_container_width=True
                    )
                else:
                    st.info("이 요인을 구별하는 문항이 없습니다.")

    # --- Tab 5: Raw Data ---
    with tab5:
        st.subheader("Factor Arrays (All Items)")
        fa_df = pd.DataFrame(engine.factor_arrays, index=q_cols, columns=[f"F{i+1}" for i in range(n_factors)])
        fa_df.insert(0, "Category", [Q_TO_TPPP.get(idx) for idx in fa_df.index])
        fa_df.insert(1, "Statement", [Q_MAP.get(idx) for idx in fa_df.index])
        st.dataframe(fa_df.style.background_gradient(cmap="RdBu_r", subset=[f"F{i+1}" for i in range(n_factors)]))

    # --- Tab 6: P-Set Profiling (NEW) ---
    with tab6:
        st.header("P-Set Profiling (Demographics Integration)")
        st.markdown("각 유형(Type)에 속한 응답자들의 **인구통계학적 특성**을 교차 분석합니다.")
        
        if not meta_cols:
            st.warning("데이터셋에 메타데이터 컬럼이 없습니다.")
        else:
            selected_meta = st.selectbox("Select Demographic Variable:", meta_cols)
            
            # Check if numeric (like years) or categorical
            is_numeric_meta = pd.to_numeric(df_meta[selected_meta], errors='coerce').notna().all()
            
            if is_numeric_meta:
                # Group by mean
                st.subheader(f"Average {selected_meta} by Type")
                df_meta_num = df_meta.copy()
                df_meta_num[selected_meta] = pd.to_numeric(df_meta_num[selected_meta])
                
                avg_stats = df_meta_num.groupby('Assigned Type')[selected_meta].mean().sort_index()
                st.bar_chart(avg_stats)
                st.dataframe(avg_stats)
            else:
                # Cross-tabulation Heatmap
                st.subheader(f"Distribution of {selected_meta} by Type")
                ctab = pd.crosstab(df_meta['Assigned Type'], df_meta[selected_meta])
                
                # Plotly Heatmap
                fig_heat = px.imshow(ctab, text_auto=True, aspect="auto", 
                                   color_continuous_scale="Greens",
                                   title=f"Heatmap: Type vs {selected_meta}")
                st.plotly_chart(fig_heat, use_container_width=True)
                
                # Normalized (Row %)
                st.caption("Row Percentage (Type Composition)")
                ctab_norm = pd.crosstab(df_meta['Assigned Type'], df_meta[selected_meta], normalize='index') * 100
                st.dataframe(ctab_norm.style.format("{:.1f}%").background_gradient(cmap="Greens", axis=1))

else:
    st.info("좌측 사이드바에서 CSV 파일을 업로드해주세요.")
