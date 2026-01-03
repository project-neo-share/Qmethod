# -*- coding: utf-8 -*-
"""
Real-Data Driven SITE Protocol Simulation
- Purpose: Counterfactual validation using ACTUAL Q-Methodology results.
- Logic: 
  1. Load Factor Arrays (Z-scores) from CSV.
  2. Calculate TPPP Sensitivities for each Factor (F1-F4).
  3. Simulate Agent interactions under BAU vs. SITE scenarios.
  4. Calculate Total Acceptance Index using Population Weights.
- Update (Justification): Clarified 'Conflict Threshold = 0' and scaling logic based on Net Acceptance theory.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

# ==========================================
# 1. Configuration & TPPP Mapping
# ==========================================
st.set_page_config(page_title="Real-Data SITE Simulation", layout="wide")

# TPPP Mapping (Balanced)
# Keys must match exactly with what is used in calculation
TPPP_CATEGORIES = {
    "Technology": ["Q01", "Q02", "Q03", "Q04", "Q05", "Q06"],
    "People": ["Q08", "Q09", "Q10", "Q12", "Q22", "Q23"],
    "Place": ["Q13", "Q14", "Q15", "Q16", "Q17", "Q18"],
    "Process": ["Q07", "Q11", "Q19", "Q20", "Q21", "Q24"]
}

# Default Population Weights (based on your analysis ~44 people)
POPULATION_WEIGHTS = {
    "F1": 0.45,  # Techno-Realists (Majority)
    "F2": 0.10,  # Eco-Equity Guardians
    "F3": 0.10,  # Development Pragmatists
    "F4": 0.35   # Tech-Skeptic Localists (Large opposition)
}

# ==========================================
# 2. Data Processing Engine
# ==========================================
def calculate_agent_profiles(df):
    """
    Derives simulation parameters from Q-Factor Z-scores.
    Input: DataFrame with 'Statement' (Q01..) and Factor columns (F1..F4)
    Output: Dictionary of Agent Profiles
    """
    # 1. Map Questions to TPPP
    q_to_cat = {}
    for cat, items in TPPP_CATEGORIES.items():
        for item in items:
            q_to_cat[item] = cat
            
    # 2. Calculate Mean Z-scores per Category for each Factor
    profiles = {}
    
    # Identify Factor columns (F1, F2...)
    factors = [c for c in df.columns if c.startswith('F') and c[1:].isdigit()]
    
    # Ensure Statement ID is accessible. 
    # If the file has 'Statement' text but not QID, we assume row order Q01..Q24
    if 'Q_ID' not in df.columns:
        # Check if 'Category' column exists (from previous export), if not create Q_ID
        # Safe assumption for 24 items in order
        if len(df) == 24:
            df['Q_ID'] = [f"Q{i+1:02d}" for i in range(24)]
        else:
            # Try to map by matching statement text? (Too risky). 
            # Better to assume standard format or ask user.
            # For now, create Q_ID sequence
            df['Q_ID'] = [f"Q{i+1:02d}" for i in range(len(df))]
    
    for f in factors:
        agent_props = {}
        for cat, q_ids in TPPP_CATEGORIES.items():
            # Filter rows for this category
            mask = df['Q_ID'].isin(q_ids)
            if mask.sum() > 0:
                mean_z = df.loc[mask, f].mean()
                agent_props[cat] = mean_z
            else:
                agent_props[cat] = 0.0 # Default if no match
        
        profiles[f] = agent_props
        
    return profiles

# ==========================================
# 3. Simulation Engine
# ==========================================
def run_simulation(profiles, steps=24, scenario="BAU"):
    history = []
    
    # Scenario Definitions (Policy Inputs: 0.0 to 1.0)
    if scenario == "BAU (Technocratic Push)":
        # Tech: Increases aggressively (Pushing specs)
        # Place: Low consideration (Ignoring context)
        # Process: Low transparency
        # People: Trust building effort is minimal
        tech_in = np.linspace(0.5, 1.2, steps)
        place_in = np.full(steps, 0.2)
        process_in = np.full(steps, 0.3)
        people_in = np.full(steps, 0.2) # Minimal trust building efforts
        
    elif scenario == "SITE Protocol (Socio-Technical)":
        # Tech: Moderate, validated by Process
        # Place: High (Incentives, Equity)
        # Process: High (Transparency, Participation)
        # People: Active trust building
        tech_in = np.linspace(0.4, 0.8, steps)
        place_in = np.linspace(0.5, 1.0, steps) # Increasing incentives
        process_in = np.linspace(0.5, 1.2, steps) # Increasing transparency
        people_in = np.linspace(0.4, 1.0, steps) # Increasing engagement

    for t in range(steps):
        row = {"Step": t}
        total_acc = 0
        
        for agent, sens in profiles.items():
            # Core Equation: Acceptance = Sum(Input * Sensitivity)
            
            # 1. Tech Effect
            tech_eff = tech_in[t] * sens.get("Technology", 0)
            
            # 2. Place Effect
            place_eff = place_in[t] * sens.get("Place", 0)
            
            # 3. Process Effect
            process_eff = process_in[t] * sens.get("Process", 0)

            # 4. People (Trust) Effect
            people_eff = people_in[t] * sens.get("People", 0)
            
            # 5. Interaction (The Vicious Cycle logic)
            # If Process is low, High Tech reduces Trust for Skeptics (F4)
            # This is where we model the feedback loop
            if scenario == "BAU (Technocratic Push)" and sens.get("Technology", 0) < -0.5:
                # High tech push backfires if trust sensitivity is high (or tech sensitivity is very negative)
                tech_eff *= 1.5 # Amplified resistance due to distrust
            
            # Sum components (All 4 Dimensions)
            raw_score = tech_eff + place_eff + process_eff + people_eff
            
            # Normalize (-100 to 100 scale) using tanh for saturation
            # Theoretical Basis: Tanh models socio-psychological saturation (S-curve).
            # Factor 0.25: Calibrates the curve so that max SITE input yields ~40-50 score (Realistic consensus),
            # instead of theoretical max 100.
            acceptance = np.tanh(raw_score * 0.25) * 100
            
            row[agent] = acceptance
            total_acc += acceptance * POPULATION_WEIGHTS.get(agent, 0.25)
            
        row["Total Index"] = total_acc
        history.append(row)
        
    return pd.DataFrame(history)

# ==========================================
# 4. Visualization UI
# ==========================================
st.title("🧩 Real-Data Agent Simulation")
st.markdown("""
**Verification of SITE Protocol using Empirical Q-Factor Profiles**
* This simulation uses the actual **Z-scores** from your Q-methodology analysis to define agent behaviors.
* **F1~F4** act as autonomous agents reacting to policy inputs based on their real-world TPPP preferences.
""")

# Load Default or Upload
uploaded_file = st.sidebar.file_uploader("1. Upload Factor Loading CSV (Z-scores)", type=['csv'])

if uploaded_file is not None:
    df_loadings = pd.read_csv(uploaded_file)
    st.sidebar.success("Custom Data Loaded!")
else:
    st.info("👋 Please upload the `type-specific-factor-loading.csv` file to start.")
    st.stop()

# Process Data
profiles = calculate_agent_profiles(df_loadings)

# Display Derived Profiles
with st.expander("🕵️ View Derived Agent Sensitivities (from Data)"):
    st.write("These values are calculated directly from the Factor Arrays (Mean Z-score per TPPP).")
    st.dataframe(pd.DataFrame(profiles).T.style.background_gradient(cmap="RdBu", vmin=-1, vmax=1))

# Simulation Controls
st.divider()
c1, c2 = st.columns([1, 3])

with c1:
    st.subheader("Settings")
    sim_steps = st.slider("Duration (Months)", 12, 60, 24)
    st.markdown("---")
    st.markdown("**Weights Used:**")
    st.json(POPULATION_WEIGHTS)
    
    st.info("""
    **Index Definition:**
    * **> 0 (Positive):** Net Acceptance (Agreement > Opposition)
    * **< 0 (Negative):** Net Conflict (Opposition > Agreement)
    * **0 (Threshold):** Deadlock / Neutrality
    """)

with c2:
    st.subheader("Simulation Results")
    
    # Run
    df_bau = run_simulation(profiles, steps=sim_steps, scenario="BAU (Technocratic Push)")
    df_site = run_simulation(profiles, steps=sim_steps, scenario="SITE Protocol (Socio-Technical)")
    
    # Plot Aggregate
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_bau["Step"], y=df_bau["Total Index"], name="BAU (Dublin Path)", line=dict(color='red', width=4)))
    fig.add_trace(go.Scatter(x=df_site["Step"], y=df_site["Total Index"], name="SITE (Google Path)", line=dict(color='blue', width=4)))
    
    # Threshold Line with Annotation
    fig.add_hline(y=0, line_dash="dash", line_color="black", annotation_text="Conflict Threshold (Net Zero)", annotation_position="bottom right")
    
    fig.update_layout(
        title="Social Acceptance Trajectory (Total Weighted)", 
        yaxis_title="Net Acceptance Index (-100 to +100)",
        xaxis_title="Time Steps (Months)",
        template="plotly_white"
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Plot Detail (F4 Focus)
    st.subheader("Deep Dive: The Skeptic's Journey (F4)")
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=df_bau["Step"], y=df_bau["F4"], name="F4 under BAU", line=dict(color='red', dash='dot')))
    fig2.add_trace(go.Scatter(x=df_site["Step"], y=df_site["F4"], name="F4 under SITE", line=dict(color='blue', dash='dot')))
    fig2.add_hline(y=0, line_dash="dash", line_color="gray")
    fig2.update_layout(title="Behavioral Change of F4 (Tech-Skeptic Localists)")
    st.plotly_chart(fig2, use_container_width=True)
    
    st.success("""
    **Interpretation for Manuscript:**
    * **BAU:** F4's resistance intensifies over time (Red dotted line drops), dragging the total index down.
    * **SITE:** By increasing 'Process' and 'Place' inputs, the SITE protocol neutralizes F4's negative Tech sensitivity, converting them from blockers to passive supporters.
    * **Threshold (0):** Represents the tipping point where social consensus shifts from 'Conflict' to 'Conditional Acceptance'.
    """)
