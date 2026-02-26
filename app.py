"""
CMVI Streamlit Dashboard
========================
Interactive visualization of the Critical Minerals Vulnerability Index.

Launch: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import yaml
import os
import datetime

# ============================================================================
# CONFIG & DATA LOADING
# ============================================================================

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / 'data'
RESULTS_DIR = BASE_DIR / 'results'


@st.cache_data
def load_config():
    cfg_path = BASE_DIR / 'config.yaml'
    if not cfg_path.exists():
        return {}
    try:
        with open(cfg_path) as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


def _file_mtime(path):
    """Return file modification time as cache key, so st.cache_data invalidates on file change."""
    import os
    try:
        return os.path.getmtime(path)
    except OSError:
        return 0


@st.cache_data
def load_panel(basis='weight', mtime=None):
    """Load CMVI scores for the given basis (weight or value)."""
    # Try basis-specific file first, then default
    path = RESULTS_DIR / f'cmvi_scores_{basis}.csv'
    if not path.exists():
        path = RESULTS_DIR / 'cmvi_scores.csv'
    if path.exists():
        return pd.read_csv(path)
    return None


@st.cache_data
def load_all_consumer_panels(basis='weight', mtime=None):
    """Load pre-computed consumer-specific CMVI panels for the given basis."""
    path = RESULTS_DIR / f'cmvi_all_consumers_{basis}.csv'
    if not path.exists():
        path = RESULTS_DIR / 'cmvi_all_consumers.csv'
    if path.exists():
        return pd.read_csv(path)
    return None


@st.cache_data
def load_csv(name, mtime=None):
    path = DATA_DIR / name
    if path.exists():
        return pd.read_csv(path)
    return None


@st.cache_data
def load_comtrade_shares(mtime=None, mtime_fallback=None):
    """Load Comtrade bilateral import shares (by stage: raw/intermediate)."""
    path = DATA_DIR / 'comtrade_import_shares_by_stage.csv'
    if path.exists():
        return pd.read_csv(path)
    # Fallback to aggregate file
    path2 = DATA_DIR / 'comtrade_import_shares.csv'
    if path2.exists():
        return pd.read_csv(path2)
    return None


@st.cache_data
def load_comtrade_indicators(mtime=None):
    """Load pre-computed consumer indicators (HHI, HHI-WGI)."""
    path = DATA_DIR / 'comtrade_consumer_indicators.csv'
    if path.exists():
        return pd.read_csv(path)
    return None


@st.cache_data
def load_sensitivity(mtime=None):
    mc = RESULTS_DIR / 'sensitivity_mc_weights.csv'
    dr = RESULTS_DIR / 'sensitivity_dropout.csv'
    ag = RESULTS_DIR / 'sensitivity_aggregation.csv'
    out = {}
    if mc.exists(): out['mc'] = pd.read_csv(mc)
    if dr.exists(): out['dropout'] = pd.read_csv(dr)
    if ag.exists(): out['aggregation'] = pd.read_csv(ag)
    return out


def recompute_cmvi(panel, weights, method='geometric'):
    """Recompute CMVI with custom dimension weights."""
    w = np.array(weights)
    if w.sum() == 0:
        w = np.ones(4) / 4  # fallback to equal weights
    else:
        w = w / w.sum()  # normalize
    dims = panel[['D1', 'D2', 'D3', 'D4']].values
    dims = np.clip(dims, 1e-6, 1.0)

    if method == 'geometric':
        log_dims = np.log(dims)
        cmvi = np.exp(np.average(log_dims, axis=1, weights=w))
    elif method == 'arithmetic':
        cmvi = np.average(dims, axis=1, weights=w)
    elif method == 'euclidean':
        weighted = dims * w[np.newaxis, :]
        cmvi = np.sqrt((weighted ** 2).sum(axis=1)) / np.sqrt((w ** 2).sum())
    else:
        cmvi = np.average(dims, axis=1, weights=w)

    out = panel.copy()
    out['CMVI'] = cmvi
    out['CMVI_rank'] = out.groupby('year')['CMVI'].rank(ascending=False, method='min').astype(int)
    return out


# ============================================================================
# PAGE CONFIG
# ============================================================================

st.set_page_config(
    page_title="CMVI Dashboard",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("Critical Minerals Vulnerability Index (CMVI)")

config = load_config()

# ============================================================================
# SIDEBAR
# ============================================================================

st.sidebar.header("Controls")

# Import share basis toggle (weight vs value) — placed first so data loads accordingly
_basis_choice = st.sidebar.radio(
    "Import share basis",
    ["Weight (kg)", "Value (USD)"],
    index=0,
    help="Switches between weight-based and value-based CMVI scores (fully recomputed for each basis)."
)
_hhi_basis = 'weight' if _basis_choice == "Weight (kg)" else 'value'
_share_col = 'import_share_wt' if _hhi_basis == 'weight' else 'import_share'
_share_label = 'Import Share (weight)' if _hhi_basis == 'weight' else 'Import Share (value)'
st.sidebar.caption(f"All CMVI/D2/D3 scores reflect **{_basis_choice.lower()}**-based import shares.")

# Load data for the selected basis
panel_raw = load_panel(basis=_hhi_basis, mtime=_file_mtime(RESULTS_DIR / f'cmvi_scores_{_hhi_basis}.csv'))
consumer_panels = load_all_consumer_panels(basis=_hhi_basis, mtime=_file_mtime(RESULTS_DIR / f'cmvi_all_consumers_{_hhi_basis}.csv'))
comtrade_shares = load_comtrade_shares(mtime=_file_mtime(DATA_DIR / 'comtrade_import_shares_by_stage.csv'), mtime_fallback=_file_mtime(DATA_DIR / 'comtrade_import_shares.csv'))
comtrade_indicators = load_comtrade_indicators(mtime=_file_mtime(DATA_DIR / 'comtrade_consumer_indicators.csv'))

if panel_raw is None:
    st.error("No CMVI results found. Run `python pipeline/cmvi_framework.py` first.")
    st.stop()

# Check if consumer-specific data is available
has_consumer_data = consumer_panels is not None and len(consumer_panels) > 0
available_consumers = []
if has_consumer_data:
    available_consumers = sorted(consumer_panels['consumer'].unique())

# Consumer selector
if has_consumer_data:
    consumer_options = available_consumers
    selected_consumer = st.sidebar.selectbox(
        "Consumer perspective",
        consumer_options,
        index=consumer_options.index('Global') if 'Global' in consumer_options else 0,
        help="Select a consumer to see vulnerability from their import perspective"
    )
    # Get the panel for the selected consumer
    panel_raw = consumer_panels[consumer_panels['consumer'] == selected_consumer].copy()
else:
    selected_consumer = 'Global'
    st.sidebar.caption("Consumer-specific data not available. Run `python pipeline/build_comtrade.py` + "
                       "`python pipeline/cmvi_framework.py --all-consumers` to enable.")

# Year selector
years = sorted(panel_raw['year'].unique())
selected_year = st.sidebar.select_slider("Year", options=years, value=max(years))

# Mineral selector
all_minerals = sorted(panel_raw['mineral'].unique())
selected_minerals = st.sidebar.multiselect(
    "Minerals", all_minerals, default=all_minerals
)

# Dimension weights
st.sidebar.subheader("Dimension Weights")
w1 = st.sidebar.slider("D1: Physical Availability", 0.0, 1.0, 0.25, 0.05)
w2 = st.sidebar.slider("D2: Supply Chain Fragility", 0.0, 1.0, 0.25, 0.05)
w3 = st.sidebar.slider("D3: Geopolitical Risk", 0.0, 1.0, 0.25, 0.05)
w4 = st.sidebar.slider("D4: Substitution Vuln.", 0.0, 1.0, 0.25, 0.05)
weights = [w1, w2, w3, w4]
total_w = sum(weights)
if total_w > 0:
    st.sidebar.caption(f"Normalized: {w1/total_w:.0%} / {w2/total_w:.0%} / {w3/total_w:.0%} / {w4/total_w:.0%}")

# Aggregation method
method = st.sidebar.selectbox("Aggregation", ['geometric', 'arithmetic', 'euclidean'],
                               index=0)

# Recompute CMVI with custom weights
panel = recompute_cmvi(panel_raw, weights, method)

# Filter to selected minerals
panel_filtered = panel[panel['mineral'].isin(selected_minerals)]
if len(selected_minerals) == 0:
    st.warning("No minerals selected. Use the sidebar to select at least one mineral.")
    st.stop()

# ============================================================================
# TABS
# ============================================================================

if has_consumer_data:
    tab_guide, tab_ov, tab_d1, tab_d2, tab_d3, tab_d4, tab_sec, tab_evo, tab_sens, tab_comp = st.tabs([
        "\u2139\ufe0f User Guide",
        "Overview",
        "D1: Physical Availability",
        "D2: Trade Concentration",
        "D3: Geopolitical Risk",
        "D4: Substitution",
        "Sector Vulnerability",
        "Time Evolution",
        "Sensitivity",
        "Consumer Comparison"
    ])
else:
    tab_guide, tab_ov, tab_d1, tab_d2, tab_d3, tab_d4, tab_sec, tab_evo, tab_sens = st.tabs([
        "\u2139\ufe0f User Guide",
        "Overview",
        "D1: Physical Availability",
        "D2: Trade Concentration",
        "D3: Geopolitical Risk",
        "D4: Substitution",
        "Sector Vulnerability",
        "Time Evolution",
        "Sensitivity"
    ])
    tab_comp = None

# ============================================================================
# TAB 1: Rankings & Heatmap
# ============================================================================

with tab_ov:
    st.info(
        "**CMVI (Critical Minerals Vulnerability Index)** aggregates four dimensions using a "
        "weighted geometric mean (non-compensatory): "
        "**D1** Physical Availability (reserves-to-production, by-product dependency, recycling rate); "
        "**D2** Supply Chain Fragility (4-way trade-based HHI bottleneck from UN Comtrade + import reliance); "
        "**D3** Geopolitical Risk (trade-based governance-weighted HHI, export restrictions, bilateral RISK index); "
        "**D4** Substitution Vulnerability (EU substitution index, performance penalty, end-use HHI). "
        "All sub-indicators are normalised [0,1] within each year (relative rankings). "
        "Higher scores = greater vulnerability. The **criticality matrix** plots "
        "supply risk (average of D1-D3) on the x-axis against substitution vulnerability (D4) "
        "on the y-axis; bubble size and colour encode the overall CMVI. "
        "The **Global** CMVI is the GDP-weighted average of 5 consumer-specific CMVIs "
        "(US 38.7%, China 26.9%, EU 25.5%, Japan 6.4%, Korea 2.6%)."
    )
    st.warning(
        "**Caution — relative scores, not absolute levels.** "
        "All sub-indicators are min-max normalised *within each year* across the 20 minerals. "
        "This means the most vulnerable mineral always scores 1 and the least scores 0 for every indicator. "
        "Scores measure **relative vulnerability among minerals in a given year**, not absolute risk. "
        "A mineral's score can fall even if its absolute concentration rises, provided other minerals worsen faster. "
        "Cross-year comparisons reflect **rank changes**, not absolute improvements or deteriorations."
    )
    st.caption(
        "Sources: USGS MCS 2025 (D1 reserves, import reliance); "
        "EU CRM 2023 (substitution, recycling, by-product); "
        "UN Comtrade (bilateral trade flows, D2/D3 HHI); BEC Rev.5 (raw/intermediate stage classification); "
        "World Bank WGI 2025 (governance); "
        "OECD IRMA 2024 (export restrictions); RISK bilateral index (geopolitical tensions)."
    )
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader(f"CMVI Rankings — {selected_year}")
        year_data = panel_filtered[panel_filtered['year'] == selected_year].sort_values('CMVI_rank')
        display_cols = ['CMVI_rank', 'mineral', 'CMVI', 'D1', 'D2', 'D3', 'D4']
        display_cols = [c for c in display_cols if c in year_data.columns]
        st.dataframe(
            year_data[display_cols].style.format({
                'CMVI': '{:.3f}', 'D1': '{:.3f}', 'D2': '{:.3f}',
                'D3': '{:.3f}', 'D4': '{:.3f}'
            }).background_gradient(subset=['CMVI'], cmap='RdYlGn_r'),
            use_container_width=True,
            height=600
        )

    with col2:
        st.subheader("Criticality Matrix")
        yd = panel_filtered[panel_filtered['year'] == selected_year].copy()
        yd['Supply Risk'] = (yd['D1'] + yd['D2'] + yd['D3']) / 3
        yd['Substitution Vuln.'] = yd['D4']

        fig = px.scatter(
            yd, x='Supply Risk', y='Substitution Vuln.',
            size='CMVI', color='CMVI', text='mineral',
            color_continuous_scale='RdYlGn_r',
            size_max=40, range_x=[-0.05, 1.05], range_y=[-0.05, 1.05],
            hover_data={'CMVI': ':.3f', 'D1': ':.3f', 'D2': ':.3f', 'D3': ':.3f', 'D4': ':.3f'}
        )
        fig.update_traces(textposition='top center', textfont_size=9)
        fig.add_hline(y=0.5, line_dash='dash', line_color='gray', opacity=0.3)
        fig.add_vline(x=0.5, line_dash='dash', line_color='gray', opacity=0.3)
        fig.add_annotation(x=0.75, y=0.9, text="CRITICAL", showarrow=False,
                          font=dict(size=14, color='red'), opacity=0.4)
        fig.add_annotation(x=0.25, y=0.1, text="LOW CONCERN", showarrow=False,
                          font=dict(size=14, color='green'), opacity=0.4)
        fig.update_layout(height=600, margin=dict(t=30))
        st.plotly_chart(fig, use_container_width=True)

    # Heatmap
    st.subheader("CMVI Heatmap — Minerals x Years")
    hm_data = panel_filtered.pivot_table(index='mineral', columns='year', values='CMVI')
    # Sort by latest year CMVI
    if selected_year in hm_data.columns:
        hm_data = hm_data.sort_values(selected_year, ascending=False)

    fig_hm = px.imshow(
        hm_data, color_continuous_scale='RdYlGn_r',
        labels=dict(x="Year", y="Mineral", color="CMVI"),
        aspect='auto', text_auto='.2f'
    )
    fig_hm.update_layout(height=500)
    st.plotly_chart(fig_hm, use_container_width=True)


# ============================================================================
# TAB 2: Dimension Decomposition
# ============================================================================

with tab_ov:  # Dimension Decomposition (part of Overview)
    st.subheader(f"Dimension Contributions — {selected_year}")
    st.info(
        "**Stacked bars** show the weighted contribution of each dimension to the overall CMVI "
        "(bar length = dimension score x normalised weight). "
        "The **radar chart** below compares a single mineral's dimension profile against the "
        "cross-mineral average -- spikes indicate relative strengths/weaknesses."
    )
    st.caption(
        "**D1** = physical scarcity (USGS reserves/production, EU CRM by-product fractions, EOL recycling rates). "
        "**D2** = supply chain fragility: max(import HHI raw, import HHI intermediate, export HHI raw, export HHI intermediate) "
        "from UN Comtrade bilateral trade classified by BEC Rev.5 stage + import reliance. "
        "**D3** = geopolitical risk: trade-based governance-weighted HHI (World Bank WGI x import/export shares), OECD export restrictions, bilateral RISK index. "
        "**D4** = substitution vulnerability: EU CRM 2023 substitution indices + USGS end-use sector HHI."
    )

    yd2 = panel_filtered[panel_filtered['year'] == selected_year].sort_values('CMVI', ascending=True).copy()

    fig_bar = go.Figure()
    colors = {'D1': '#e74c3c', 'D2': '#3498db', 'D3': '#f39c12', 'D4': '#9b59b6'}
    labels_d = {'D1': 'D1: Physical', 'D2': 'D2: Supply Chain',
                'D3': 'D3: Geopolitical', 'D4': 'D4: Substitution'}

    norm_w = np.array(weights)
    if norm_w.sum() > 0:
        norm_w = norm_w / norm_w.sum()

    for i, dim in enumerate(['D1', 'D2', 'D3', 'D4']):
        fig_bar.add_trace(go.Bar(
            y=yd2['mineral'], x=yd2[dim] * norm_w[i],
            name=labels_d[dim], orientation='h',
            marker_color=colors[dim], opacity=0.85
        ))

    fig_bar.update_layout(barmode='stack', height=600, xaxis_title='Weighted Contribution',
                          legend=dict(orientation='h', y=-0.1))
    st.plotly_chart(fig_bar, use_container_width=True)

    # Radar chart for selected mineral
    st.subheader("Mineral Profile (Radar)")
    mineral_choice = st.selectbox("Select mineral", all_minerals, index=0, key='radar_mineral')
    mineral_data = panel[panel['mineral'] == mineral_choice]
    yr_data = mineral_data[mineral_data['year'] == selected_year]
    avg_data = panel_filtered[panel_filtered['year'] == selected_year][['D1', 'D2', 'D3', 'D4']].mean()

    if len(yr_data) > 0:
        categories = ['D1: Physical', 'D2: Supply Chain', 'D3: Geopolitical', 'D4: Substitution']
        fig_radar = go.Figure()
        vals_m = yr_data[['D1', 'D2', 'D3', 'D4']].values[0].tolist()
        vals_m.append(vals_m[0])
        vals_a = avg_data.values.tolist()
        vals_a.append(vals_a[0])
        cats = categories + [categories[0]]

        fig_radar.add_trace(go.Scatterpolar(r=vals_m, theta=cats, name=mineral_choice,
                                             fill='toself', opacity=0.6))
        fig_radar.add_trace(go.Scatterpolar(r=vals_a, theta=cats, name='Average',
                                             fill='toself', opacity=0.3,
                                             line=dict(dash='dash')))
        fig_radar.update_layout(polar=dict(radialaxis=dict(range=[0, 1])),
                                height=400, showlegend=True)
        st.plotly_chart(fig_radar, use_container_width=True)


# ============================================================================
# TAB D1: Physical Availability
# ============================================================================

with tab_d1:
    st.subheader(f"D1: Physical Availability — {selected_year}")
    st.info(
        "**D1** captures mineral-intrinsic physical scarcity. "
        "Three sub-indicators: **Reserves-to-production ratio** (USGS) — how many years of "
        "reserves remain at current extraction rates (lower = more scarce); "
        "**By-product dependency** (EU CRM 2023) — share of global production obtained as a "
        "by-product of another mineral's extraction (higher = more supply-fragile); "
        "**Recycling rate** (EU CRM 2023 EOL-RIR) — end-of-life recycling input rate "
        "(lower recycling = more vulnerable). "
        "All sub-indicators are normalised [0,1] within each year."
    )
    st.caption(
        "Sources: **Reserves & production** = USGS Mineral Commodity Summaries 2025 + Critical Minerals "
        "Extraction database. **By-product fractions** = EU CRM 2023 Study + Nassar et al. (2015). "
        "**Recycling rates** = EU CRM 2023 EOL-RIR (End-of-Life Recycling Input Rate)."
    )

    yd1 = panel_filtered[panel_filtered['year'] == selected_year].sort_values('D1', ascending=False).copy()

    # D1 sub-indicator bar chart
    d1_sub_cols = []
    d1_labels = {}
    d1_colors = {}
    if 'rp_log_norm' in yd1.columns:
        d1_sub_cols.append('rp_log_norm')
        d1_labels['rp_log_norm'] = 'Reserves/Production (inverted)'
        d1_colors['rp_log_norm'] = '#e74c3c'
    if 'byproduct_frac_norm' in yd1.columns:
        d1_sub_cols.append('byproduct_frac_norm')
        d1_labels['byproduct_frac_norm'] = 'By-product Dependency'
        d1_colors['byproduct_frac_norm'] = '#f39c12'
    if 'eol_rir_norm' in yd1.columns:
        d1_sub_cols.append('eol_rir_norm')
        d1_labels['eol_rir_norm'] = 'Recycling Rate (inverted)'
        d1_colors['eol_rir_norm'] = '#2ecc71'

    if d1_sub_cols:
        fig_d1 = go.Figure()
        for col in d1_sub_cols:
            fig_d1.add_trace(go.Bar(
                y=yd1['mineral'], x=yd1[col],
                name=d1_labels[col], orientation='h',
                marker_color=d1_colors[col], opacity=0.85
            ))
        fig_d1.update_layout(
            barmode='group', height=max(500, len(yd1) * 28),
            xaxis_title='Normalised Score (0-1, higher = more vulnerable)',
            title=f'D1 Sub-Indicators — {selected_year}',
            yaxis={'categoryorder': 'array', 'categoryarray': yd1['mineral'].tolist()},
            legend=dict(orientation='h', y=-0.12)
        )
        st.plotly_chart(fig_d1, use_container_width=True)
    else:
        st.info("D1 sub-indicator data not available in the panel.")

    # Raw values table
    st.subheader("D1 Raw Values")
    raw_cols = ['mineral', 'D1']
    raw_labels = {'mineral': 'Mineral', 'D1': 'D1 Score'}
    for col, label in [('rp_ratio', 'R/P Ratio (years)'),
                        ('byproduct_frac', 'By-product Fraction'),
                        ('eol_rir', 'EOL Recycling Rate')]:
        if col in yd1.columns:
            raw_cols.append(col)
            raw_labels[col] = label

    raw_fmt = {'D1': '{:.3f}'}
    if 'rp_ratio' in yd1.columns:
        raw_fmt['rp_ratio'] = '{:.0f}'
    if 'byproduct_frac' in yd1.columns:
        raw_fmt['byproduct_frac'] = '{:.2f}'
    if 'eol_rir' in yd1.columns:
        raw_fmt['eol_rir'] = '{:.1%}'

    d1_table = yd1[raw_cols].sort_values('D1', ascending=False).copy()
    d1_table = d1_table.rename(columns=raw_labels)
    st.dataframe(
        d1_table.style.format({v: raw_fmt.get(k, '{:.3f}') for k, v in raw_labels.items()
                                if k in raw_fmt}),
        use_container_width=True
    )


# ============================================================================
# TAB: Time Evolution
# ============================================================================

with tab_evo:
    st.subheader("CMVI Over Time")
    st.info(
        "**Time series** of CMVI scores for the top N minerals (by latest year score). "
        "All scores are within-year normalised, so changes reflect shifts in a mineral's "
        "**relative** vulnerability compared to peers -- not absolute changes in supply conditions. "
        "The **dimension evolution** chart below breaks out D1-D4 for a single mineral, "
        "showing which dimension drives year-to-year variation."
    )
    st.warning(
        "**Caution:** Scores are relative rankings within each year. "
        "A rising line does **not** mean absolute vulnerability increased — "
        "it means the mineral worsened *relative to the other 19 minerals that year*."
    )
    st.caption(
        "Panel covers 2015-2023. D1 indicators are largely static (reserves, by-product, recycling from EU CRM 2023). "
        "D2 varies with annual UN Comtrade trade flows (4-way HHI bottleneck). "
        "D3 varies with annual WGI scores, OECD export restrictions, and monthly RISK data. "
        "D4 is mostly static (EU substitution indices, USGS end-use shares)."
    )

    # Top N selector
    top_n = st.slider("Show top N minerals", 5, 20, 10, key='top_n_evo')
    latest = panel_filtered[panel_filtered['year'] == max(years)]
    top_minerals = latest.nlargest(top_n, 'CMVI')['mineral'].tolist()
    evo_data = panel_filtered[panel_filtered['mineral'].isin(top_minerals)]

    fig_evo = px.line(
        evo_data, x='year', y='CMVI', color='mineral',
        markers=True, line_shape='linear',
        labels={'CMVI': 'CMVI Score', 'year': 'Year'}
    )
    fig_evo.update_layout(height=500, legend=dict(orientation='h', y=-0.15))
    st.plotly_chart(fig_evo, use_container_width=True)

    # Dimension evolution for selected mineral
    st.subheader("Dimension Evolution")
    mineral_evo = st.selectbox("Select mineral", all_minerals, index=0, key='evo_mineral')
    mdata = panel[panel['mineral'] == mineral_evo].sort_values('year')

    fig_dims = go.Figure()
    for dim, color in [('D1', '#e74c3c'), ('D2', '#3498db'), ('D3', '#f39c12'), ('D4', '#9b59b6')]:
        fig_dims.add_trace(go.Scatter(
            x=mdata['year'], y=mdata[dim], name=dim,
            mode='lines+markers', line=dict(color=color)
        ))
    fig_dims.update_layout(height=400, yaxis_title='Score', xaxis_title='Year',
                           title=f'{mineral_evo} — Dimension Scores Over Time')
    st.plotly_chart(fig_dims, use_container_width=True)


# ============================================================================
# TAB D3: Geopolitical Risk
# ============================================================================

with tab_d3:
    st.subheader("Geopolitical Risk (Trade-Based)")
    st.info(
        "**D3** captures governance and geopolitical risks in the supply chain, "
        "measured through trade-based concentration. Each source country's "
        "trade share is weighted by (1 - WGI), so reliance on poorly-governed states "
        "yields a higher score. The D3 bottleneck is max(import HHI-WGI raw, "
        "import HHI-WGI intermediate, export HHI-WGI raw, export HHI-WGI intermediate). "
        "**RISK** captures bilateral conflict risk between the consumer and "
        "its import sources, weighted by trade shares."
    )
    st.caption(
        "Sources: **Trade shares** = UN Comtrade bilateral imports (annual, 2015-2023), "
        "classified into raw/intermediate stages using BEC Rev.5. "
        "**WGI** = World Bank Worldwide Governance Indicators (Political Stability, Rule of Law, "
        "Regulatory Quality), 2025 release, 216 countries, 1996-2024. "
        "**Export restrictions** = OECD Industrial Raw Materials 2024, 80 countries, 57 minerals, intensity-weighted "
        "(prohibition=1.0, quota=0.8, tax=0.6, licensing=0.4). "
        "**RISK** = bilateral monthly conflict index (12-month MA), 216x219 country pairs, 2015-2026."
    )

    wgi = load_csv('country_wgi.csv', mtime=_file_mtime(DATA_DIR / 'country_wgi.csv'))
    comtrade_stage = load_comtrade_shares(mtime=_file_mtime(DATA_DIR / 'comtrade_import_shares_by_stage.csv'), mtime_fallback=_file_mtime(DATA_DIR / 'comtrade_import_shares.csv'))

    if comtrade_stage is not None and wgi is not None:
        mineral_geo = st.selectbox("Select mineral", all_minerals, index=0, key='geo_mineral')

        # Stage selector (raw / intermediate)
        stage_geo = st.radio("Trade stage", ['Raw', 'Intermediate'], horizontal=True, key='geo_stage')
        stage_key = stage_geo.lower()

        # Filter Comtrade data for selected mineral/year/stage
        has_stage_col = 'stage' in comtrade_stage.columns
        trade_data = comtrade_stage[
            (comtrade_stage['mineral'] == mineral_geo) &
            (comtrade_stage['year'] == selected_year)
        ].copy()
        if has_stage_col and len(trade_data) > 0:
            trade_data = trade_data[trade_data['stage'] == stage_key]

        # If no data for selected year, try nearest
        if len(trade_data) == 0:
            avail_yrs = comtrade_stage[comtrade_stage['mineral'] == mineral_geo]['year'].unique()
            if len(avail_yrs) > 0:
                nearest = min(avail_yrs, key=lambda x: abs(x - selected_year))
                trade_data = comtrade_stage[
                    (comtrade_stage['mineral'] == mineral_geo) &
                    (comtrade_stage['year'] == nearest)
                ].copy()
                if has_stage_col and len(trade_data) > 0:
                    trade_data = trade_data[trade_data['stage'] == stage_key]
                if len(trade_data) > 0:
                    st.caption(f"Using nearest available year: {nearest}")

        # Aggregate by consumer selection
        if len(trade_data) > 0 and 'consumer' in trade_data.columns:
            if selected_consumer != 'Global':
                trade_data = trade_data[trade_data['consumer'] == selected_consumer].copy()
            else:
                # GDP-weighted aggregation across consumers
                gdp_w = {'EU': 16.8, 'US': 25.5, 'Japan': 4.2, 'Korea': 1.7, 'China': 17.7}
                gdp_total = sum(gdp_w.values())
                gdp_w = {k: v / gdp_total for k, v in gdp_w.items()}
                # Weighted average of import shares per partner
                sc = _share_col if _share_col in trade_data.columns else 'import_share'
                trade_data['gdp_weight'] = trade_data['consumer'].map(gdp_w).fillna(0)
                trade_data['weighted_share'] = trade_data[sc] * trade_data['gdp_weight']
                agg_cols = {'weighted_share': 'sum'}
                if 'trade_value' in trade_data.columns:
                    agg_cols['trade_value'] = 'sum'
                trade_data = trade_data.groupby('partner_iso').agg(agg_cols).reset_index()
                trade_data = trade_data.rename(columns={'weighted_share': sc})

        if len(trade_data) > 0:
            # Compute share_pct from import share column (respects hhi_basis)
            sc = _share_col if _share_col in trade_data.columns else 'import_share'
            if sc in trade_data.columns:
                trade_data['share_pct'] = trade_data[sc] * 100
            else:
                trade_data['share_pct'] = 0

            # Merge WGI
            wgi_yr = wgi[wgi['year'] == min(selected_year, wgi['year'].max())][['country_iso3', 'wgi_ps', 'wgi_rl', 'wgi_rq']]
            wgi_yr['wgi_avg'] = wgi_yr[['wgi_ps', 'wgi_rl', 'wgi_rq']].mean(axis=1)
            partner_col = 'partner_iso' if 'partner_iso' in trade_data.columns else 'country_iso3'
            trade_data = trade_data.merge(wgi_yr, left_on=partner_col, right_on='country_iso3', how='left')

            col1, col2 = st.columns([1, 1])
            with col1:
                # Choropleth
                fig_map = px.choropleth(
                    trade_data, locations=partner_col,
                    color='share_pct',
                    color_continuous_scale='Reds',
                    labels={'share_pct': f'{stage_geo} Import Share (%)'},
                    title=f'{mineral_geo} {stage_geo} Sources — {selected_consumer} ({selected_year})'
                )
                fig_map.update_layout(height=400, margin=dict(t=40, b=0, l=0, r=0))
                st.plotly_chart(fig_map, use_container_width=True)

            with col2:
                # Bar chart: top sources with WGI color
                sorted_data = trade_data.nlargest(15, 'share_pct').sort_values('share_pct', ascending=True)
                fig_wgi = go.Figure()
                fig_wgi.add_trace(go.Bar(
                    y=sorted_data[partner_col],
                    x=sorted_data['share_pct'],
                    orientation='h', name=f'{stage_geo} Share (%)',
                    marker_color='steelblue'
                ))
                fig_wgi.update_layout(height=400, xaxis_title='Trade Share (%)',
                                       title=f'{mineral_geo} — Top {stage_geo} Sources')
                st.plotly_chart(fig_wgi, use_container_width=True)

            # Data table
            display_cols = [partner_col, 'share_pct', 'wgi_avg']
            if 'trade_value' in trade_data.columns:
                display_cols.insert(1, 'trade_value')
            display_cols = [c for c in display_cols if c in trade_data.columns]
            fmt_dict = {'share_pct': '{:.1f}%', 'wgi_avg': '{:.2f}'}
            if 'trade_value' in trade_data.columns:
                fmt_dict['trade_value'] = '{:,.0f}'
            st.dataframe(
                trade_data[display_cols]
                .sort_values('share_pct', ascending=False)
                .style.format(fmt_dict),
                use_container_width=True
            )
        else:
            # Check which stages are available for this mineral
            avail_stages = []
            if comtrade_stage is not None and 'stage' in comtrade_stage.columns:
                avail_stages = sorted(comtrade_stage[
                    comtrade_stage['mineral'] == mineral_geo
                ]['stage'].unique())
            if avail_stages:
                st.info(f"No {stage_key}-stage trade data for {mineral_geo}. "
                        f"Available stages: {', '.join(avail_stages)}.")
            else:
                st.info(f"No trade data for {mineral_geo} in {selected_year}. "
                        f"Run `python pipeline/build_comtrade.py` to build Comtrade data.")
    else:
        if comtrade_stage is None:
            st.info("Comtrade trade data not available. Run `python pipeline/build_comtrade.py` first.")
        if wgi is None:
            st.info("WGI governance data not available.")

    # --- D3 sub-indicators: HHI-WGI and Export Restriction Score ---
    st.subheader("Governance-Weighted Supply Concentration (HHI-WGI)")
    st.caption(
        "HHI-WGI = sum(share_c^2 x gov_risk_c) — trade shares weighted by (1-WGI). "
        "Higher values = concentrated supply from poorly-governed countries."
    )
    yd3 = panel_filtered[panel_filtered['year'] == selected_year].copy()
    if 'hhi_wgi' in yd3.columns and yd3['hhi_wgi'].notna().any():
        yd3_sorted = yd3.sort_values('hhi_wgi', ascending=True)
        fig_wgi_hhi = px.bar(
            yd3_sorted, y='mineral', x='hhi_wgi',
            orientation='h', color='hhi_wgi',
            color_continuous_scale='YlOrRd',
            labels={'hhi_wgi': 'HHI-WGI Score'},
            title=f'Governance-Weighted Concentration — {selected_consumer} ({selected_year})'
        )
        fig_wgi_hhi.update_layout(height=max(400, len(yd3_sorted) * 25), yaxis_title='')
        st.plotly_chart(fig_wgi_hhi, use_container_width=True)
    else:
        st.info("HHI-WGI data not available in the panel.")

    st.subheader("Export Restriction Score")
    st.caption(
        "OECD export restriction intensity weighted by trade shares. "
        "Higher = more supply exposed to export controls (prohibitions, quotas, taxes, licensing)."
    )
    if 'export_restriction_score' in yd3.columns and yd3['export_restriction_score'].notna().any():
        yd3_er = yd3.sort_values('export_restriction_score', ascending=True)
        fig_er = px.bar(
            yd3_er, y='mineral', x='export_restriction_score',
            orientation='h', color='export_restriction_score',
            color_continuous_scale='YlOrRd',
            labels={'export_restriction_score': 'Export Restriction Score'},
            title=f'Export Restriction Exposure — {selected_year}'
        )
        fig_er.update_layout(height=max(400, len(yd3_er) * 25), yaxis_title='')
        st.plotly_chart(fig_er, use_container_width=True)
    else:
        st.info("Export restriction data not available in the panel.")

    # RISK bilateral section
    st.subheader("Bilateral RISK-Weighted Supply Vulnerability")
    risk_path = DATA_DIR / 'monthly_2015-2026.csv'
    if risk_path.exists() and 'risk_weighted' in panel.columns:
        yr_panel = panel_filtered[panel_filtered['year'] == selected_year].dropna(subset=['risk_weighted'])
        if len(yr_panel) > 0:
            yr_sorted = yr_panel.sort_values('risk_weighted', ascending=True)
            fig_risk = px.bar(
                yr_sorted, y='mineral', x='risk_weighted',
                orientation='h', color='risk_weighted',
                color_continuous_scale='YlOrRd',
                labels={'risk_weighted': 'RISK-Weighted Score'},
                title=f'Bilateral RISK-Weighted Supply Vulnerability — {selected_year}'
            )
            fig_risk.update_layout(height=500, yaxis_title='', xaxis_title='RISK Score')
            st.plotly_chart(fig_risk, use_container_width=True)
            st.caption("Bilateral conflict risk weighted by import shares (consumer-specific) "
                       "or GDP-weighted average across 5 consumers (Global).")
        else:
            st.info(f"No RISK data available for {selected_year}")
    elif not risk_path.exists():
        st.info("RISK bilateral data not yet extracted. Run framework with `load_risk=True` to integrate.")


# ============================================================================
# TAB D2: Trade Concentration
# ============================================================================

with tab_d2:
    st.subheader("Trade Concentration (D2)")
    st.info(
        "**D2: 4-way trade-based bottleneck.** Supply chain risk is assessed across four trade "
        "channels: *import raw* (ores/concentrates), *import intermediate* "
        "(compounds/unwrought metals), *export raw*, and *export intermediate*. "
        "HHI = sum of squared market shares (0=diversified, 1=monopoly). "
        "The **binding bottleneck** is max(import HHI raw, import HHI intermediate, "
        "export HHI raw, export HHI intermediate) -- the weakest link in the supply chain. "
        "Import HHI is **consumer-specific** (each country's actual import sources). "
        "Export HHI is **global** (aggregates imports across all 5 consumers to approximate "
        "each exporter's world market share). "
        "Export HHI captures chokepoints where a few countries dominate "
        "world exports even if a consumer's own imports appear diversified. "
        "D2 = 0.75 x normalised(bottleneck HHI) + 0.25 x normalised(import reliance)."
    )
    st.caption(
        "Sources: **Import/export HHI** computed from UN Comtrade bilateral trade flows (annual, 2015-2023) "
        "for 162 HS6 codes classified into raw/intermediate stages using UN BEC Rev.5. "
        "**Import reliance** from USGS data (consumer-specific)."
    )

    # --- Primary chart: 4-way bottleneck from consumer panel ---
    if has_consumer_data:
        cons_panel = consumer_panels[
            (consumer_panels['consumer'] == selected_consumer) &
            (consumer_panels['year'] == selected_year)
        ].copy()
        if len(cons_panel) == 0:
            nearest_yr = consumer_panels[consumer_panels['consumer'] == selected_consumer]['year'].max()
            cons_panel = consumer_panels[
                (consumer_panels['consumer'] == selected_consumer) &
                (consumer_panels['year'] == nearest_yr)
            ].copy()
            if len(cons_panel) > 0:
                st.caption(f"Using nearest available year: {nearest_yr}")

        cons_panel = cons_panel[cons_panel['mineral'].isin(selected_minerals)]

        hhi_cols_4way = ['hhi_import_raw', 'hhi_import_intermediate',
                         'hhi_export_raw', 'hhi_export_intermediate']

        if len(cons_panel) > 0 and any(c in cons_panel.columns for c in hhi_cols_4way):
            # Prepare data: fill NaN with 0 for display, sort by bottleneck HHI
            plot_df = cons_panel[['mineral'] + [c for c in hhi_cols_4way if c in cons_panel.columns]].copy()
            for c in hhi_cols_4way:
                if c not in plot_df.columns:
                    plot_df[c] = np.nan
            plot_df = plot_df.set_index('mineral')

            # Warn about minerals with no HHI data at all
            all_nan_mask = plot_df[hhi_cols_4way].isna().all(axis=1)
            if all_nan_mask.any():
                missing_minerals = list(all_nan_mask[all_nan_mask].index)
                st.warning(
                    f"No trade-stage HHI data for: {', '.join(missing_minerals)}. "
                    "These minerals show as zero but actually have missing data. "
                    "Run `pipeline/build_comtrade.py --patch-download` then reprocess to fix."
                )

            # Sort by hhi_bottleneck if available, else by max across 4 channels
            if 'hhi_bottleneck' in cons_panel.columns:
                sort_order = cons_panel.set_index('mineral')['hhi_bottleneck'].reindex(plot_df.index)
            else:
                sort_order = plot_df[hhi_cols_4way].max(axis=1)
            plot_df = plot_df.loc[sort_order.sort_values(ascending=True).index]

            fig_4way = go.Figure()
            bar_specs = [
                ('hhi_import_raw', 'Import Raw', '#2ecc71'),
                ('hhi_import_intermediate', 'Import Intermediate', '#e67e22'),
                ('hhi_export_raw', 'Export Raw', '#3498db'),
                ('hhi_export_intermediate', 'Export Intermediate', '#e74c3c'),
            ]
            for col, label, color in bar_specs:
                # Use NaN (not 0) for missing data so bars simply don't appear
                vals = plot_df[col].values
                fig_4way.add_trace(go.Bar(
                    y=plot_df.index, x=vals,
                    name=label, orientation='h', marker_color=color
                ))

            fig_4way.update_layout(
                barmode='group',
                height=max(500, len(plot_df) * 32),
                xaxis_title='HHI (0-1)',
                title=f'4-Way Supply Chain HHI — {selected_consumer} ({selected_year})',
                yaxis={'categoryorder': 'array', 'categoryarray': plot_df.index.tolist()},
                legend=dict(orientation='h', y=-0.12)
            )
            fig_4way.add_vline(x=0.25, line_dash='dash', line_color='gray',
                               annotation_text='HHI=0.25 (concentrated)')
            st.plotly_chart(fig_4way, use_container_width=True)

            # --- Binding bottleneck table (all 4 stages) ---
            st.subheader("Binding Bottleneck Stage")
            st.caption("The binding stage is the trade channel with the highest HHI for each mineral.")

            btable = cons_panel[['mineral'] + [c for c in hhi_cols_4way if c in cons_panel.columns]].copy()
            for c in hhi_cols_4way:
                if c not in btable.columns:
                    btable[c] = np.nan
            btable = btable.set_index('mineral')
            # Determine binding stage (max HHI across 4 channels, ignoring NaN)
            btable['bottleneck_hhi'] = btable[hhi_cols_4way].max(axis=1)
            btable['binding_stage'] = btable[hhi_cols_4way].apply(
                lambda row: row.idxmax() if row.notna().any() else 'N/A', axis=1
            ).str.replace('hhi_', '')
            # If hhi_bottleneck / bottleneck_stage from the panel are available, prefer them
            if 'hhi_bottleneck' in cons_panel.columns:
                bn_vals = cons_panel.set_index('mineral')['hhi_bottleneck']
                btable['bottleneck_hhi'] = bn_vals.reindex(btable.index)
            if 'bottleneck_stage' in cons_panel.columns:
                bs_vals = cons_panel.set_index('mineral')['bottleneck_stage']
                btable['binding_stage'] = bs_vals.reindex(btable.index)
            btable = btable.sort_values('bottleneck_hhi', ascending=False)

            # Format: show all 4 HHI columns + binding stage
            fmt_cols = [c for c in hhi_cols_4way if c in btable.columns]
            fmt_dict = {c: '{:.3f}' for c in fmt_cols}
            fmt_dict['bottleneck_hhi'] = '{:.3f}'
            # Rename columns for display
            display_btable = btable[fmt_cols + ['bottleneck_hhi', 'binding_stage']].copy()
            display_btable.columns = [c.replace('hhi_', '').replace('_', ' ').title()
                                      for c in display_btable.columns[:-2]] + ['Bottleneck HHI', 'Binding Stage']
            st.dataframe(
                display_btable.style.format(
                    {c: '{:.3f}' for c in display_btable.columns if c not in ['Binding Stage']}
                ),
                use_container_width=True
            )
        else:
            if len(cons_panel) == 0:
                st.info(
                    f"No data for consumer '{selected_consumer}' in year {selected_year}. "
                    "Try a different year or check that `pipeline/cmvi_framework.py --all-consumers` has been run."
                )
            else:
                st.info(
                    "No 4-way bottleneck HHI columns found. "
                    "Run `pipeline/build_comtrade.py --patch-download` then `pipeline/cmvi_framework.py --all-consumers` to generate trade HHI data."
                )
    else:
        # Fallback: show D2 scores from global panel
        st.info("Consumer-specific data not available. Showing global D2 scores.")
        yd2_global = panel_filtered[panel_filtered['year'] == selected_year].sort_values('D2', ascending=False)
        if 'D2' in yd2_global.columns and len(yd2_global) > 0:
            fig_d2_fb = go.Figure(go.Bar(
                y=yd2_global['mineral'], x=yd2_global['D2'],
                orientation='h', marker_color='#3498db'))
            fig_d2_fb.update_layout(height=max(500, len(yd2_global) * 28),
                xaxis_title='D2 Score', title=f'D2: Supply Chain Fragility — {selected_year}')
            st.plotly_chart(fig_d2_fb, use_container_width=True)


# ============================================================================
# TAB D4: Substitution Vulnerability
# ============================================================================

with tab_d4:
    st.subheader(f"D4: Substitution Vulnerability — {selected_year}")
    st.info(
        "**D4** captures how difficult it is to replace a mineral with alternatives. "
        "Sub-indicators: **EU Substitution Index** (higher = harder to substitute); "
        "**Performance Penalty** (average loss of performance from best substitutes); "
        "**End-Use HHI** (sector concentration — minerals used in few sectors are more exposed). "
        "The **Sector CMVI** below is the demand-weighted average of mineral CMVIs: "
        "for each clean-energy technology, each mineral's CMVI is multiplied by that sector's "
        "demand share and summed."
    )
    st.caption(
        "Sources: **Substitution indices** = EU CRM 2023 Study (supply risk substitution index). "
        "**End-use demand shares** from USGS Mineral Commodity Summaries 2025 (US domestic consumption). "
        "Sector definitions: Solar PV, Wind, EV Batteries, Grid Storage, Electricity Networks, Hydrogen."
    )

    # D4 sub-indicator chart
    yd4 = panel_filtered[panel_filtered['year'] == selected_year].sort_values('D4', ascending=False).copy()
    d4_sub_cols = []
    d4_labels = {}
    d4_colors = {}
    for col, label, color in [
        ('eu_substitution_index_sr_norm', 'Substitution Index', '#9b59b6'),
        ('avg_performance_penalty_norm', 'Performance Penalty', '#e74c3c'),
        ('enduse_hhi_norm', 'End-Use Concentration', '#f39c12'),
    ]:
        if col in yd4.columns:
            d4_sub_cols.append(col)
            d4_labels[col] = label
            d4_colors[col] = color

    if d4_sub_cols:
        fig_d4sub = go.Figure()
        for col in d4_sub_cols:
            fig_d4sub.add_trace(go.Bar(
                y=yd4['mineral'], x=yd4[col],
                name=d4_labels[col], orientation='h',
                marker_color=d4_colors[col], opacity=0.85
            ))
        fig_d4sub.update_layout(
            barmode='group', height=max(500, len(yd4) * 28),
            xaxis_title='Normalised Score (0-1, higher = harder to substitute)',
            title=f'D4 Sub-Indicators — {selected_year}',
            yaxis={'categoryorder': 'array', 'categoryarray': yd4['mineral'].tolist()},
            legend=dict(orientation='h', y=-0.12)
        )
        st.plotly_chart(fig_d4sub, use_container_width=True)

    # D4 raw values table
    raw_d4_cols = ['mineral', 'D4']
    for col in ['eu_substitution_index_sr', 'avg_performance_penalty', 'enduse_hhi']:
        if col in yd4.columns:
            raw_d4_cols.append(col)
    if len(raw_d4_cols) > 2:
        st.subheader("D4 Raw Values")
        fmt_d4 = {'D4': '{:.3f}', 'eu_substitution_index_sr': '{:.2f}',
                  'avg_performance_penalty': '{:.2f}', 'enduse_hhi': '{:.3f}'}
        st.dataframe(
            yd4[raw_d4_cols].sort_values('D4', ascending=False)
            .style.format({c: f for c, f in fmt_d4.items() if c in raw_d4_cols}),
            use_container_width=True
        )


# ============================================================================
# TAB: Sector Vulnerability
# ============================================================================

with tab_sec:
    st.subheader("Sector Vulnerability (Demand-Side Weighting)")
    st.info(
        "**Sector CMVI** is the demand-weighted average of mineral CMVIs: "
        "for each clean-energy technology, each mineral's CMVI is multiplied by that sector's "
        "demand share for the mineral and summed. Higher values mean the sector relies heavily "
        "on minerals that are themselves highly vulnerable."
    )
    st.caption(
        "Sources: **End-use demand shares** from USGS Mineral Commodity Summaries 2025 (US domestic consumption "
        "by sector). Sector definitions: Solar PV, Wind, EV Batteries, Grid Storage, Electricity Networks, Hydrogen."
    )

    enduse = load_csv('minerals_enduse.csv', mtime=_file_mtime(DATA_DIR / 'minerals_enduse.csv'))
    if enduse is not None and len(enduse) > 0:
        # Compute sector CMVI
        yr_cmvi = panel[panel['year'] == selected_year][['mineral', 'CMVI']].copy()
        enduse_latest_yr = enduse['year'].max()
        enduse_yr = enduse[enduse['year'] == enduse_latest_yr].copy()

        sector_cmvi = enduse_yr.merge(yr_cmvi, on='mineral', how='left')
        sector_cmvi['weighted_cmvi'] = sector_cmvi['share'] * sector_cmvi['CMVI']

        sector_scores = sector_cmvi.groupby('sector').agg(
            sector_cmvi=('weighted_cmvi', 'sum'),
            n_minerals=('mineral', 'count'),
            total_demand=('demand_kt', 'sum')
        ).sort_values('sector_cmvi', ascending=False).reset_index()

        sector_labels = {
            'solar_pv': 'Solar PV',
            'wind': 'Wind',
            'ev_batteries': 'EV Batteries',
            'grid_storage': 'Grid Storage',
            'electricity_networks': 'Electricity Networks',
            'hydrogen': 'Hydrogen'
        }
        sector_scores['sector_label'] = sector_scores['sector'].map(sector_labels).fillna(sector_scores['sector'])

        col1, col2 = st.columns([1, 1])
        with col1:
            fig_sec = px.bar(
                sector_scores, x='sector_label', y='sector_cmvi',
                color='sector_cmvi', color_continuous_scale='RdYlGn_r',
                labels={'sector_label': 'Sector', 'sector_cmvi': 'Sector CMVI'},
                title='Which Clean Energy Tech is Most Vulnerable?'
            )
            fig_sec.update_layout(height=400)
            st.plotly_chart(fig_sec, use_container_width=True)

        with col2:
            sector_choice = st.selectbox("Select sector", sector_scores['sector_label'].tolist(),
                                          key='sector_choice')
            sector_key = {v: k for k, v in sector_labels.items()}.get(sector_choice, sector_choice)
            sec_data = sector_cmvi[sector_cmvi['sector'] == sector_key].sort_values('weighted_cmvi', ascending=True)

            if len(sec_data) > 0:
                fig_sec_detail = px.bar(
                    sec_data, y='mineral', x='weighted_cmvi', orientation='h',
                    color='CMVI', color_continuous_scale='RdYlGn_r',
                    labels={'weighted_cmvi': 'Weighted CMVI', 'mineral': ''},
                    title=f'{sector_choice} — Mineral Risk Contributions'
                )
                fig_sec_detail.update_layout(height=400)
                st.plotly_chart(fig_sec_detail, use_container_width=True)

        st.subheader(f"Mineral Demand by Sector (kt, {enduse_latest_yr})")
        demand_wide = enduse[enduse['year'] == enduse_latest_yr].pivot_table(
            index='mineral', columns='sector', values='demand_kt', fill_value=0
        )
        demand_wide.columns = [sector_labels.get(c, c) for c in demand_wide.columns]
        st.dataframe(demand_wide.style.format('{:.1f}'), use_container_width=True)

    else:
        st.info("No end-use data available.")


# ============================================================================
# TAB: Sensitivity Analysis
# ============================================================================

with tab_sens:
    st.subheader("Sensitivity Analysis")
    st.info(
        "**Monte Carlo weight perturbation**: dimension weights are drawn uniformly from the "
        "3-simplex (Dirichlet(1,1,1,1)) 1000 times, and rankings are recomputed for each draw. "
        "Stable minerals (low IQR) are robust to weight choice. "
        "**Dimension dropout**: each dimension is replaced by 0.5 (neutral) and the Spearman "
        "rank correlation with the baseline measures how much that dimension drives rankings "
        "(lower rho = more influential). "
        "**Aggregation comparison**: geometric vs arithmetic vs Euclidean mean -- "
        "high pairwise correlations confirm rank-robustness to aggregation method. "
        "**Live exploration**: adjust weights in the sidebar to see rankings update in real time."
    )
    st.caption(
        "Sensitivity results are pre-computed from the Global CMVI (trade-based baseline). "
        "MC uses 1000 Dirichlet draws; aggregation comparison uses Spearman rank correlation. "
        "Sidebar weight adjustments apply to the currently selected consumer."
    )

    sens = load_sensitivity(mtime=_file_mtime(RESULTS_DIR / 'sensitivity_mc_weights.csv'))

    if 'mc' in sens:
        st.subheader("Monte Carlo Weight Perturbation")
        mc = sens['mc']
        mc = mc[mc['mineral'].isin(selected_minerals)].sort_values('rank_mean')

        fig_mc = go.Figure()
        fig_mc.add_trace(go.Bar(
            y=mc['mineral'], x=mc['rank_mean'],
            orientation='h', name='Mean Rank',
            marker_color='steelblue',
            error_x=dict(type='data', array=mc['rank_std'], visible=True)
        ))
        fig_mc.update_layout(height=500, xaxis_title='Mean Rank (lower = more critical)',
                             title='Rank Stability Under Random Weights (1000 draws)')
        st.plotly_chart(fig_mc, use_container_width=True)

        # Rank range
        mc_display = mc[['mineral', 'rank_mean', 'rank_std', 'rank_q05', 'rank_q95', 'rank_iqr']].copy()
        mc_display.columns = ['Mineral', 'Mean Rank', 'Std', '5th %ile', '95th %ile', 'IQR']
        st.dataframe(mc_display.style.format({
            'Mean Rank': '{:.1f}', 'Std': '{:.1f}',
            '5th %ile': '{:.0f}', '95th %ile': '{:.0f}', 'IQR': '{:.1f}'
        }), use_container_width=True)

    if 'dropout' in sens:
        st.subheader("Dimension Dropout — Rank Correlation")
        dr = sens['dropout']
        fig_dr = px.bar(
            dr, x='dropped_dimension', y='spearman_rho',
            color='spearman_rho', color_continuous_scale='RdYlGn',
            labels={'spearman_rho': 'Spearman rho', 'dropped_dimension': 'Dropped Dimension'},
            title='How much do rankings change when dropping one dimension?'
        )
        fig_dr.update_layout(height=350, yaxis_range=[0.5, 1.05])
        st.plotly_chart(fig_dr, use_container_width=True)

    if 'aggregation' in sens:
        st.subheader("Aggregation Method Comparison")
        ag = sens['aggregation']
        st.dataframe(ag.style.format({'spearman_rho': '{:.3f}', 'p_value': '{:.4f}'}),
                     use_container_width=True)

    # Live sensitivity: custom weight exploration
    st.subheader("Live Weight Exploration")
    st.caption("Adjust weights in the sidebar to see rankings update in real time.")
    yr_sens = panel_filtered[panel_filtered['year'] == selected_year][['mineral', 'CMVI', 'CMVI_rank', 'D1', 'D2', 'D3', 'D4']].sort_values('CMVI_rank')
    st.dataframe(yr_sens.style.format({
        'CMVI': '{:.3f}', 'D1': '{:.3f}', 'D2': '{:.3f}', 'D3': '{:.3f}', 'D4': '{:.3f}'
    }), use_container_width=True)


# ============================================================================
# TAB 8: Consumer Comparison
# ============================================================================

if tab_comp is not None and has_consumer_data:
    with tab_comp:
        st.subheader("Consumer-Specific CMVI Comparison")
        st.info(
            "**Global CMVI** is the GDP-weighted average of 5 consumer-specific CMVIs "
            "(US 38.7%, China 26.9%, EU 25.5%, Japan 6.4%, Korea 2.6%). "
            "Each consumer's D2 (supply chain) and D3 (geopolitical risk) are computed from "
            "their own bilateral import shares (UN Comtrade), so the same mineral can rank differently "
            "depending on who is importing it. D1 and D4 are shared (mineral-intrinsic properties). "
            "Compare columns to spot where a consumer "
            "is more or less exposed than the global average."
        )
        st.caption(
            "Sources: **Bilateral trade** = UN Comtrade API (annual, 2015-2023, 162 HS6 codes). "
            "**Stage classification** = BEC Rev.5 (raw/intermediate). "
            "**HS-mineral mapping** = UNCTAD SDG Pulse 2025 (499 HS6 codes across 60 minerals). "
            "**Import reliance** = 1 - domestic/world output (USGS). "
            "**GDP weights** = World Bank 2023 nominal GDP."
        )

        # Consumer comparison table for selected year
        comp_data = consumer_panels[consumer_panels['year'] == selected_year].copy()

        # Apply custom weights to all consumer panels
        for consumer in comp_data['consumer'].unique():
            mask = comp_data['consumer'] == consumer
            consumer_subset = comp_data[mask].copy()
            recalc = recompute_cmvi(consumer_subset, weights, method)
            comp_data.loc[mask, 'CMVI'] = recalc['CMVI'].values
            comp_data.loc[mask, 'CMVI_rank'] = recalc['CMVI_rank'].values

        # Pivot: mineral x consumer CMVI
        col1, col2 = st.columns([1, 1])

        with col1:
            st.subheader(f"CMVI by Consumer — {selected_year}")
            cmvi_pivot = comp_data.pivot_table(
                index='mineral', columns='consumer', values='CMVI'
            ).reindex(selected_minerals).dropna(how='all')

            # Sort by Global score
            if 'Global' in cmvi_pivot.columns:
                cmvi_pivot = cmvi_pivot.sort_values('Global', ascending=False)

            st.dataframe(
                cmvi_pivot.style.format('{:.3f}')
                .background_gradient(cmap='RdYlGn_r', axis=None),
                use_container_width=True,
                height=600
            )

        with col2:
            st.subheader("CMVI Rank Differences")
            rank_pivot = comp_data.pivot_table(
                index='mineral', columns='consumer', values='CMVI_rank'
            ).reindex(selected_minerals).dropna(how='all')

            if 'Global' in rank_pivot.columns:
                rank_pivot = rank_pivot.sort_values('Global')

            st.dataframe(
                rank_pivot.style.format('{:.0f}')
                .background_gradient(cmap='RdYlGn', axis=None),
                use_container_width=True,
                height=600
            )

        # Scatter: Consumer vs Global CMVI
        st.subheader("Consumer vs Global CMVI")
        compare_consumer = st.selectbox(
            "Compare consumer", [c for c in available_consumers if c != 'Global'],
            key='compare_consumer'
        )

        if compare_consumer and 'Global' in cmvi_pivot.columns and compare_consumer in cmvi_pivot.columns:
            scatter_data = cmvi_pivot[['Global', compare_consumer]].dropna().copy()
            scatter_data['mineral'] = scatter_data.index

            fig_comp = px.scatter(
                scatter_data, x='Global', y=compare_consumer,
                text='mineral', size_max=15,
                labels={'Global': 'Global CMVI', compare_consumer: f'{compare_consumer} CMVI'},
                title=f'{compare_consumer} vs Global CMVI — {selected_year}'
            )
            fig_comp.add_shape(type='line', x0=0, y0=0, x1=1, y1=1,
                              line=dict(dash='dash', color='gray', width=1))
            fig_comp.update_traces(textposition='top center', textfont_size=9)
            fig_comp.update_layout(
                height=500,
                xaxis=dict(range=[-0.05, 1.05]),
                yaxis=dict(range=[-0.05, 1.05]),
            )
            fig_comp.add_annotation(x=0.8, y=0.2, text=f"Less critical\nfor {compare_consumer}",
                                     showarrow=False, font=dict(size=10, color='green'), opacity=0.4)
            fig_comp.add_annotation(x=0.2, y=0.8, text=f"More critical\nfor {compare_consumer}",
                                     showarrow=False, font=dict(size=10, color='red'), opacity=0.4)
            st.plotly_chart(fig_comp, use_container_width=True)

        # Dimension comparison: D2 and D3 across consumers
        st.subheader("D2 (Supply Chain) & D3 (Geopolitical) by Consumer")
        mineral_comp = st.selectbox("Select mineral for dimension comparison",
                                     selected_minerals, key='mineral_comp')

        mineral_comp_data = comp_data[comp_data['mineral'] == mineral_comp].copy()
        if len(mineral_comp_data) > 0:
            dim_fig = go.Figure()
            consumers_sorted = mineral_comp_data.sort_values('CMVI', ascending=True)['consumer'].tolist()

            for dim, color, label in [('D1', '#e74c3c', 'D1: Physical'),
                                       ('D2', '#3498db', 'D2: Supply Chain'),
                                       ('D3', '#f39c12', 'D3: Geopolitical'),
                                       ('D4', '#9b59b6', 'D4: Substitution')]:
                vals = [mineral_comp_data[mineral_comp_data['consumer'] == c][dim].values[0]
                        if len(mineral_comp_data[mineral_comp_data['consumer'] == c]) > 0 else 0
                        for c in consumers_sorted]
                dim_fig.add_trace(go.Bar(
                    y=consumers_sorted, x=vals,
                    name=label, orientation='h',
                    marker_color=color, opacity=0.85
                ))

            dim_fig.update_layout(
                barmode='group', height=400,
                xaxis_title='Score (0-1)',
                title=f'{mineral_comp} — Dimension Scores by Consumer — {selected_year}'
            )
            st.plotly_chart(dim_fig, use_container_width=True)

        # Import source details
        if comtrade_shares is not None:
            st.subheader("Import Sources by Consumer & Stage")
            st.caption(
                "Source: UN Comtrade bilateral import flows, classified by supply-chain stage "
                "using BEC Rev.5 (raw = ores/concentrates, intermediate = oxides/unwrought metals)."
            )
            imp_mineral = st.selectbox("Select mineral", selected_minerals, key='imp_mineral')
            imp_year = selected_year

            imp_data = comtrade_shares[
                (comtrade_shares['mineral'] == imp_mineral) &
                (comtrade_shares['year'] == imp_year)
            ].copy()

            if len(imp_data) > 0:
                # Stage filter if stage column exists
                has_stage = 'stage' in imp_data.columns
                if has_stage:
                    available_stages = sorted(imp_data['stage'].unique())
                    imp_stage = st.radio(
                        "Supply chain stage", ['All'] + available_stages,
                        horizontal=True, key='imp_stage'
                    )
                    if imp_stage != 'All':
                        imp_data = imp_data[imp_data['stage'] == imp_stage]

                # Top 10 partners per consumer (use weight- or value-based share)
                sc = _share_col if _share_col in imp_data.columns else 'import_share'
                top_imports = pd.concat([
                    g.nlargest(10, sc)
                    for _, g in imp_data.groupby('consumer')
                ], ignore_index=True)

                title_suffix = f' ({imp_stage})' if has_stage and imp_stage != 'All' else ''
                fig_imp = px.bar(
                    top_imports, x=sc, y='partner_iso',
                    color='consumer', barmode='group',
                    orientation='h',
                    labels={sc: _share_label, 'partner_iso': 'Partner'},
                    title=f'{imp_mineral} — Top Import Partners by Consumer{title_suffix} — {imp_year}'
                )
                fig_imp.update_layout(height=500, yaxis={'categoryorder': 'total ascending'})
                st.plotly_chart(fig_imp, use_container_width=True)
            else:
                st.info(f"No Comtrade import data for {imp_mineral} in {imp_year}.")

        # --- Import Reliance Comparison Table ---
        st.subheader(f"Import Reliance by Consumer — {selected_year}")
        st.caption("IR = net import reliance (0 = self-sufficient, 1 = fully import-dependent). "
                   "Higher values mean the consumer depends more heavily on imports for that mineral.")
        if 'import_reliance' in comp_data.columns:
            ir_pivot = comp_data.pivot_table(
                index='mineral', columns='consumer', values='import_reliance'
            ).reindex(selected_minerals).dropna(how='all')

            if 'Global' in ir_pivot.columns:
                ir_pivot = ir_pivot.sort_values('Global', ascending=False)

            if len(ir_pivot) > 0:
                st.dataframe(
                    ir_pivot.style.format('{:.3f}')
                    .background_gradient(cmap='YlOrRd', axis=None, vmin=0, vmax=1),
                    use_container_width=True,
                    height=500
                )
            else:
                st.info("No import reliance data available for the selected minerals/year.")
        else:
            st.info("Import reliance column not found in consumer panel data.")


# ============================================================================
# TAB: User Guide
# ============================================================================

with tab_guide:
    st.header("User Guide")

    st.subheader("What Is the CMVI?")
    st.markdown("""
The **Critical Minerals Vulnerability Index (CMVI)** is a composite index measuring how
vulnerable a consumer economy is to disruptions in the supply of 20 critical minerals.
It combines four dimensions:

- **D1: Physical Availability** — Can the world produce enough of this mineral in the long run?
- **D2: Supply Chain Fragility** — Is trade concentrated among a few partners, and is the consumer import-dependent?
- **D3: Geopolitical Risk** — Are key suppliers politically unstable, restrictive, or hostile?
- **D4: Substitution Vulnerability** — Can the mineral be replaced in its main industrial applications?

Each dimension is scored 0–1 (higher = more vulnerable). The four dimensions are aggregated using
a **weighted geometric mean**, which penalises minerals that are vulnerable across *all* dimensions
simultaneously (non-compensatory aggregation).

**Consumers:** EU-27, United States, Japan, South Korea, China, plus a GDP-weighted Global average.
**Time coverage:** 2015–2023. **Minerals:** 20 critical minerals.
""")

    st.subheader("Important Caveats")
    st.warning("""
**Relative scores, not absolute levels.** All sub-indicators are min-max normalised *within each year*
across the 20 minerals. The most vulnerable mineral always scores 1 and the least scores 0 for every indicator.
A mineral's score can *fall* even if its absolute concentration rises, provided other minerals worsen faster.
Cross-year comparisons reflect **rank changes**, not absolute improvements or deteriorations.
""")
    st.info("""
**Weight-based vs. value-based import shares.** Use the sidebar toggle to switch between weight (kg) and
value (USD) based CMVI scores. Both versions are fully pre-computed — the toggle loads entirely different
score files, so all CMVI, D2, and D3 values change accordingly. Weight-based shares better reflect physical
supply dependence and avoid price distortions; value-based shares capture economic exposure.
""")

    st.subheader("Tab Guide")

    with st.expander("**Overview** — Rankings, Criticality Matrix, Heatmap", expanded=False):
        st.markdown("""
- **CMVI Rankings** table for the selected year, with colour-coded scores.
- **Criticality Matrix** (bubble chart): supply risk (avg D1–D3) on x-axis vs substitution vulnerability (D4) on y-axis. Top-right = most critical.
- **CMVI Heatmap**: all minerals × all years.
- **Dimension Contributions** (stacked bar): how much each dimension contributes.
- **Radar Chart**: select a mineral to compare its D1–D4 profile against the average.
""")

    with st.expander("**D1: Physical Availability** — Reserves, By-products, Recycling", expanded=False):
        st.markdown("""
**Sub-indicators:**
- **Reserves-to-Production Ratio (RP):** Years of reserves remaining. Higher RP → lower vulnerability (inverted).
- **By-product Dependency (BP):** Fraction produced as a by-product of another metal.
- **End-of-Life Recycling Rate (EOL-RIR):** Fraction recoverable from waste. Higher → lower vulnerability (inverted).

**Sources:** USGS MCS 2025 (reserves, production); EU CRM 2023 (by-product fractions, recycling rates).

D1 is largely *static* — reserves and recycling data change slowly.
""")

    with st.expander("**D2: Trade Concentration** — Four-Way Bottleneck HHI + Import Reliance", expanded=False):
        st.markdown("""
**Sub-indicators:**
- **Four-way Bottleneck HHI:** Maximum HHI across 4 trade channels (import raw, import intermediate, export raw, export intermediate). Import channels are consumer-specific; export channels are global. Captures the *weakest link*.
- **Import Reliance (IR):** Share of consumption that must be imported.

**Sources:** UN Comtrade bilateral trade (162 HS codes, BEC Rev.5); USGS production data.

D2 is the most *dynamic* dimension — trade flows shift year-to-year.
""")

    with st.expander("**D3: Geopolitical Risk** — Governance, Export Restrictions, RISK Index", expanded=False):
        st.markdown("""
**Sub-indicators:**
- **Governance-Weighted HHI (HHI-WGI):** Import concentration weighted by supplier governance quality.
- **Export Restrictions:** OECD inventory (2024).
- **Bilateral RISK Index:** Monthly geopolitical tension (216 × 219 pairs). Import-share-weighted.

**Sources:** World Bank WGI 2025; OECD IRMA 2024; RISK bilateral index.

**EU-specific:** RISK is import-weighted across all 27 member states, capturing heterogeneous exposures.
""")

    with st.expander("**D4: Substitution Vulnerability** — Substitution Difficulty, Performance Penalty", expanded=False):
        st.markdown("""
**Sub-indicators:**
- **Substitution Index (SI):** EU CRM 2023 difficulty of replacing this mineral.
- **Performance Penalty:** Quality loss when substituting.
- **End-Use HHI:** Concentration of the mineral across industrial sectors.

**Sources:** EU CRM 2023; USGS MCS 2025 end-use shares (11 of 20 minerals).

D4 is largely *static* — substitution technology evolves on decadal timescales.
""")

    with st.expander("**Sector Vulnerability** — Industrial Sector Exposure", expanded=False):
        st.markdown("""
Sector-level vulnerability weighted by mineral consumption shares and CMVI scores.
Available for 11 of 20 minerals with USGS end-use data.
""")

    with st.expander("**Time Evolution** — Trends and Rank Dynamics", expanded=False):
        st.markdown("""
- **CMVI Over Time:** Top N minerals' scores across 2015–2023.
- **Dimension Evolution:** D1–D4 breakdown for a single mineral over time.

**Caution:** Rising lines mean the mineral worsened *relative to peers*, not in absolute terms.
""")

    with st.expander("**Sensitivity** — Weight Robustness and Aggregation Comparison", expanded=False):
        st.markdown("""
- **Monte Carlo:** 1,000 random Dirichlet weight draws. Narrow box = stable ranking.
- **Aggregation Comparison:** Spearman correlations between geometric, arithmetic, Euclidean methods.
- **Dimension Dropout:** Rankings when each dimension is removed.
""")

    with st.expander("**Consumer Comparison** — Cross-Consumer Analysis", expanded=False):
        st.markdown("""
- **Ranking Comparison:** Heatmap of CMVI by mineral and consumer.
- **Rank Divergence:** Which minerals differ most across consumers.
- **Dimension Profile:** Compare D1–D4 for a mineral across consumers.
- **Import Sources:** Top partners by import share.

D1/D4 are shared across consumers (global geology/technology). D2/D3 are consumer-specific (bilateral trade).
""")

    st.subheader("Glossary")
    glossary_data = {
        'Term': ['CMVI', 'HHI', 'Bottleneck HHI', 'WGI', 'HHI-WGI', 'RISK Index',
                 'Import Reliance', 'EOL-RIR', 'BEC Rev.5', 'Geometric Mean',
                 'Min-Max Normalisation', 'GDP-Weighted Global'],
        'Definition': [
            'Critical Minerals Vulnerability Index. Composite score (0–1) via weighted geometric mean.',
            'Herfindahl-Hirschman Index. Sum of squared market shares. 0 = perfect competition, 1 = monopoly.',
            'Max HHI across 4 trade channels (import raw/int, export raw/int). Most concentrated point in supply chain.',
            'World Governance Indicators (World Bank). Political Stability, Rule of Law, Regulatory Quality.',
            'Import HHI weighted by supplier governance risk. Imports from poorly-governed countries = higher risk.',
            'Bilateral geopolitical tension index (monthly, 216 × 219 pairs). Higher = more tension.',
            '1 − (domestic production / world production). Fraction that must be imported.',
            'End-of-Life Recycling Input Rate. Share of consumption met by recycling.',
            'UN Broad Economic Categories Rev.5. Classifies HS codes into raw, intermediate, finished.',
            'exp(Σ wᵢ ln Dᵢ). Non-compensatory: cannot offset high vulnerability in one dimension with low in another.',
            '(x − min) / (max − min). Rescales to [0,1] within each year across 20 minerals.',
            'Average of 5 consumer CMVIs: US 38.7%, China 26.9%, EU 25.5%, Japan 6.4%, Korea 2.6%.'
        ]
    }
    st.dataframe(pd.DataFrame(glossary_data), use_container_width=True, hide_index=True)

    st.subheader("Data Sources")
    sources_data = {
        'Dataset': ['Mining & reserves', 'Processing shares', 'Bilateral trade', 'Governance',
                     'Export restrictions', 'Geopolitical tensions', 'Substitution & recycling',
                     'End-use shares', 'GDP weights'],
        'Source': ['USGS MCS 2025', 'IEA (2024)', 'UN Comtrade', 'World Bank WGI 2025',
                   'OECD IRMA 2024', 'RISK bilateral index', 'EU CRM 2023', 'USGS MCS 2025',
                   'World Bank'],
        'Dimensions': ['D1, D2, D3', 'D2, D3', 'D2, D3', 'D3', 'D3', 'D3', 'D1, D4', 'D4', 'Aggregation'],
        'Coverage': ['20 minerals, 93 countries', '6 minerals', '162 HS codes, 5 consumers',
                     '216 countries', '80 countries, 57 minerals', '216 × 219 pairs, monthly',
                     '87 materials', '11 of 20 minerals', '5 consumers, 2023']
    }
    st.dataframe(pd.DataFrame(sources_data), use_container_width=True, hide_index=True)

# ============================================================================
# FOOTER
# ============================================================================

st.divider()
consumer_label = f" | Consumer: {selected_consumer}" if selected_consumer != 'Global' else ""
_scores_path = RESULTS_DIR / 'cmvi_scores.csv'
_last_computed = datetime.datetime.fromtimestamp(os.path.getmtime(_scores_path)).strftime('%Y-%m-%d %H:%M') if _scores_path.exists() else "unknown"
st.caption(
    f"CMVI Dashboard{consumer_label} | "
    f"Supply chain: 4-way trade-based bottleneck (import raw/intermediate + export raw/intermediate HHI) | "
    f"Data: USGS (D1), EU CRM 2023, OECD, World Bank WGI, UN Comtrade, RISK | "
    f"Last computed: {_last_computed}"
)
