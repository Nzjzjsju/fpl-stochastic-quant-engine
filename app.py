import streamlit as st
import requests
import pandas as pd
import numpy as np
from scipy import stats
from scipy.optimize import milp, LinearConstraint, Bounds

st.set_page_config(page_title="Premier League Stochastic Quant Optimizer", layout="wide")

st.title("⚽ Premier League Stochastic Portfolio & Sharpe Ratio Optimizer")
st.markdown("**Methodology:** 10,000-Iteration Poisson Monte Carlo Simulation, Markowitz Risk-Adjusted Sharpe Modeling, and Mixed-Integer Linear Programming (MILP).")

@st.cache_data(ttl=3600)
def load_and_simulate(min_minutes):
    np.random.seed(42)
    url = "https://fantasy.premierleague.com/api/bootstrap-static/"
    r = requests.get(url).json()
    elements = pd.DataFrame(r['elements'])
    teams = pd.DataFrame(r['teams'])[['id', 'name']].rename(columns={'id': 'team', 'name': 'Club'})
    positions = pd.DataFrame(r['element_types'])[['id', 'singular_name_short']].rename(columns={'id': 'element_type', 'singular_name_short': 'Pos'})
    df = elements.merge(teams, on='team').merge(positions, on='element_type')

    cols = ['now_cost', 'minutes', 'total_points', 'selected_by_percent', 'expected_goals', 'expected_assists', 'clean_sheets', 'bonus']
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)

    df = df[df['minutes'] >= min_minutes].copy().reset_index(drop=True)
    df['Price (£m)'] = df['now_cost'] / 10.0
    n90 = df['minutes'] / 90.0

    lam_g = (df['expected_goals'] / n90).clip(lower=0.01).values
    lam_a = (df['expected_assists'] / n90).clip(lower=0.01).values
    p_cs = (df['clean_sheets'] / n90).clip(upper=0.65, lower=0.02).values
    b_90 = (df['bonus'] / n90).clip(upper=1.5).values

    g_pts = df['Pos'].map({'GKP': 6, 'DEF': 6, 'MID': 5, 'FWD': 4}).values
    cs_pts = df['Pos'].map({'GKP': 4, 'DEF': 4, 'MID': 1, 'FWD': 0}).values

    sims = 10000
    pts = 2.0 + np.random.poisson(lam_g, (sims, len(df))) * g_pts + np.random.poisson(lam_a, (sims, len(df))) * 3.0 + np.random.binomial(1, p_cs, (sims, len(df))) * cs_pts + b_90

    df['xP / Match'] = pts.mean(axis=0).round(2)
    df['Volatility (σ)'] = pts.std(axis=0).round(2)
    df['5-GW Expected Pts'] = (df['xP / Match'] * 5.0).round(2)
    df['Sharpe Ratio'] = ((df['xP / Match'] - 2.0) / df['Volatility (σ)']).round(3)
    df['Haul Prob (≥10 pts) %'] = ((pts >= 10).mean(axis=0) * 100).round(1)
    df['xPPM Efficiency'] = (df['5-GW Expected Pts'] / df['Price (£m)']).round(2)
    return df

st.sidebar.header("Portfolio Constraints")
budget_cap = st.sidebar.slider("Total Squad Budget Cap (£m)", 95.0, 105.0, 100.0, 0.5)
risk_lambda = st.sidebar.slider("Risk Aversion Penalty (λ)", 0.0, 1.0, 0.25, 0.05)
min_mins = st.sidebar.slider("Minimum Season Minutes Played", 90, 900, 180, 90)

df = load_and_simulate(min_mins)
df['Risk_Adj_Utility'] = (df['5-GW Expected Pts'] - risk_lambda * df['Volatility (σ)']).round(2)

# Run MILP Solver
A, lb, ub = [df['Price (£m)'].values], [-np.inf], [budget_cap]
for pos, cnt in {'GKP': 2, 'DEF': 5, 'MID': 5, 'FWD': 3}.items():
    A.append((df['Pos'] == pos).astype(float).values)
    lb.append(cnt)
    ub.append(cnt)
for club in df['Club'].unique():
    A.append((df['Club'] == club).astype(float).values)
    lb.append(0)
    ub.append(3)

res = milp(c=-df['Risk_Adj_Utility'].values, integrality=np.ones(len(df)), bounds=Bounds(0, 1), constraints=LinearConstraint(np.array(A), lb, ub))
df['Optimal'] = np.round(res.x).astype(int)
opt = df[df['Optimal'] == 1].sort_values(by=['Pos', '5-GW Expected Pts'], ascending=[True, False])

col1, col2, col3 = st.columns(3)
col1.metric("Optimal 15-Man Squad Cost", f"£{opt['Price (£m)'].sum():.1f}m")
col2.metric("Projected 5-GW Squad xP", f"{opt['5-GW Expected Pts'].sum():.1f} pts")
col3.metric("Mean Portfolio Sharpe Ratio", f"{opt['Sharpe Ratio'].mean():.3f}")

st.subheader("🏆 MILP Mathematically Optimal 15-Player Squad")
display_cols = ['web_name', 'Club', 'Pos', 'Price (£m)', 'selected_by_percent', 'xP / Match', '5-GW Expected Pts', 'Volatility (σ)', 'Sharpe Ratio', 'Haul Prob (≥10 pts) %']
st.dataframe(opt[display_cols].rename(columns={'web_name': 'Player', 'selected_by_percent': 'Ownership %'}), use_container_width=True)

st.subheader("📊 Full League Asset Valuation & Sharpe Screener")
st.dataframe(df.sort_values(by='Sharpe Ratio', ascending=False)[display_cols].rename(columns={'web_name': 'Player', 'selected_by_percent': 'Ownership %'}), use_container_width=True)
