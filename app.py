# ============================================================
# DELHI METRO ANALYTICS DASHBOARD
# MBA Applied Business Analytics
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Delhi Metro Analytics",
    page_icon="🚇",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Global CSS ────────────────────────────────────────────────
st.markdown("""
<style>
body { background-color: #0e1117; color: white; }
.big-header {
    text-align: center;
    padding: 2rem;
    background: linear-gradient(135deg, #1b0036, #3a0068, #6a0dad);
    color: white;
    border-radius: 15px;
    margin-bottom: 2rem;
    border: 1px solid #7b2fbe;
}
.big-header h1 { font-size: 2.4rem; font-weight: 900; margin: 0; color: white; }
.big-header p  { font-size: 1rem; margin: 0.5rem 0 0 0; color: #ce93d8; }
.kpi-card {
    background: linear-gradient(135deg, #1b0036, #2d1b69);
    border: 1px solid #7b2fbe;
    border-radius: 12px;
    padding: 1.2rem;
    text-align: center;
    color: white;
}
.kpi-card .kpi-value { font-size: 1.8rem; font-weight: 900; color: #ce93d8; }
.kpi-card .kpi-label { font-size: 0.85rem; color: #ab47bc; margin-top: 0.3rem; }
.state-card-green {
    background: #0a2e0a; border: 1.5px solid #2ecc71;
    border-radius: 10px; padding: 0.8rem 1rem; margin: 0.4rem 0; color: white;
}
.state-card-red {
    background: #2e0a0a; border: 1.5px solid #e74c3c;
    border-radius: 10px; padding: 0.8rem 1rem; margin: 0.4rem 0; color: white;
}
.insight-box {
    background: #1b003622;
    border-left: 4px solid #ab47bc;
    padding: 1rem 1.2rem;
    border-radius: 8px;
    margin: 0.8rem 0;
    color: white;
    font-size: 0.95rem;
}
.section-header {
    font-size: 1.3rem;
    font-weight: 700;
    color: #ce93d8;
    margin: 1rem 0 0.5rem 0;
    border-bottom: 2px solid #7b2fbe;
    padding-bottom: 0.3rem;
}
</style>
""", unsafe_allow_html=True)

PLOT_LAYOUT = dict(
    plot_bgcolor='#0e1117',
    paper_bgcolor='#0e1117',
    font=dict(color='white', size=12),
)
AXIS_STYLE = dict(gridcolor='#1f2937', color='white',
                  tickfont=dict(color='white'),
                  title_font=dict(color='white'))
LEGEND_STYLE = dict(font=dict(color='white'),
                    bgcolor='#1b0036', bordercolor='#7b2fbe')

# ── Data Loading & Cleaning ───────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv('delhi_metro_updated.csv')

    # Clean station names
    df['From_Station'] = df['From_Station'].str.strip().str.title()
    df['To_Station']   = df['To_Station'].str.strip().str.title()

    # Parse dates
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.dropna(subset=['Date'])
    df['Year']      = df['Date'].dt.year
    df['Month']     = df['Date'].dt.month
    df['Month_Name']= df['Date'].dt.strftime('%b')
    df['DayOfWeek'] = df['Date'].dt.day_name()
    df['Quarter']   = df['Date'].dt.to_period('Q').astype(str)

    # Fill / clean categoricals
    df['Ticket_Type'] = df['Ticket_Type'].fillna('Unknown').str.strip()
    df['Remarks']     = df['Remarks'].fillna('Normal').str.strip()
    df['Remarks']     = df['Remarks'].replace('', 'Normal')

    # Numeric
    df['Passengers']       = pd.to_numeric(df['Passengers'], errors='coerce')
    df['Fare']             = pd.to_numeric(df['Fare'], errors='coerce')
    df['Distance_km']      = pd.to_numeric(df['Distance_km'], errors='coerce')
    df['Cost_per_passenger']= pd.to_numeric(df['Cost_per_passenger'], errors='coerce')

    df['Revenue'] = df['Fare'] * df['Passengers'].fillna(1)

    # Route label
    df['Route'] = df['From_Station'] + ' → ' + df['To_Station']

    return df

df = load_data()

# ── Sidebar ───────────────────────────────────────────────────
st.sidebar.markdown("""
<div style='text-align:center; padding:1.2rem;
background:linear-gradient(135deg,#1b0036,#6a0dad);
border-radius:12px; margin-bottom:1rem;
border:1px solid #7b2fbe;'>
<div style='font-size:2rem;'>🚇</div>
<div style='color:white; font-size:1.1rem; font-weight:700;'>Delhi Metro</div>
<div style='color:#ce93d8; font-size:0.78rem;'>Analytics Dashboard</div>
</div>
""", unsafe_allow_html=True)

page = st.sidebar.radio(
    "Navigate",
    [
        "🏠 Overview Dashboard",
        "🚉 Station Intelligence",
        "🗺️  Route & Distance Analysis",
        "🎫 Ticket & Demand Patterns",
        "📅 Time Series Trends",
        "💡 Insights & Recommendations",
    ]
)

# ── Sidebar filters ───────────────────────────────────────────
st.sidebar.markdown("---")
st.sidebar.markdown("<div style='color:#ce93d8; font-weight:700; font-size:0.9rem;'>🔧 Global Filters</div>", unsafe_allow_html=True)

year_opts = sorted(df['Year'].dropna().unique().tolist())
sel_years = st.sidebar.multiselect("Year", year_opts, default=year_opts)

ticket_opts = sorted(df['Ticket_Type'].dropna().unique().tolist())
sel_tickets = st.sidebar.multiselect("Ticket Type", ticket_opts, default=ticket_opts)

# Apply filters
mask = (df['Year'].isin(sel_years)) & (df['Ticket_Type'].isin(sel_tickets))
dff  = df[mask].copy()

st.sidebar.markdown("---")
st.sidebar.markdown(f"""
<div style='font-size:0.8rem; color:#ce93d8; padding:0.5rem;
background:#0d001e; border-radius:8px; border:1px solid #3a0068;'>
<b style='color:#ab47bc;'>Dataset Info</b><br>
Total Records : {len(df):,}<br>
Filtered      : {len(dff):,}<br>
Date Range    : 2022 – 2024<br>
Stations      : {df['From_Station'].nunique()}<br>
Ticket Types  : {df['Ticket_Type'].nunique()}<br>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW DASHBOARD
# ══════════════════════════════════════════════════════════════
if page == "🏠 Overview Dashboard":

    st.markdown("""
    <div class='big-header'>
    <h1>🚇 Delhi Metro Analytics Dashboard</h1>
    <p>Comprehensive Ridership, Revenue & Operational Analysis (2022–2024) | MBA Applied Business Analytics</p>
    </div>
    """, unsafe_allow_html=True)

    # KPI Row
    total_trips    = len(dff)
    total_revenue  = dff['Revenue'].sum()
    avg_fare       = dff['Fare'].mean()
    avg_dist       = dff['Distance_km'].mean()
    total_pax      = dff['Passengers'].sum()
    busiest_station= dff['From_Station'].value_counts().idxmax()

    col1, col2, col3, col4, col5, col6 = st.columns(6)
    kpis = [
        (col1, f"{total_trips:,}",           "Total Trips"),
        (col2, f"₹{total_revenue/1e6:.1f}M", "Total Revenue"),
        (col3, f"₹{avg_fare:.0f}",           "Avg Fare"),
        (col4, f"{avg_dist:.1f} km",          "Avg Distance"),
        (col5, f"{total_pax/1e6:.1f}M",       "Total Passengers"),
        (col6, busiest_station,               "Busiest Origin"),
    ]
    for col, val, lbl in kpis:
        with col:
            st.markdown(f"""
            <div class='kpi-card'>
            <div class='kpi-value'>{val}</div>
            <div class='kpi-label'>{lbl}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Monthly trips trend
    col_l, col_r = st.columns([2, 1])

    with col_l:
        st.markdown("<div class='section-header'>📈 Monthly Trip Volume (2022–2024)</div>", unsafe_allow_html=True)
        monthly = (dff.groupby(['Year','Month','Month_Name'])
                      .size().reset_index(name='Trips'))
        monthly['YM'] = monthly['Year'].astype(str) + '-' + monthly['Month'].astype(str).str.zfill(2)
        monthly = monthly.sort_values('YM')

        fig = px.area(monthly, x='YM', y='Trips',
                      color_discrete_sequence=['#ab47bc'],
                      labels={'YM':'Month','Trips':'Trips'},
                      height=340)
        fig.update_layout(**PLOT_LAYOUT,
                          xaxis=dict(**AXIS_STYLE, title=''),
                          yaxis=dict(**AXIS_STYLE),
                          title=dict(text='', font=dict(color='white')))
        fig.update_traces(fillcolor='rgba(171,71,188,0.25)', line_color='#ce93d8')
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        st.markdown("<div class='section-header'>🎟️ Ticket Type Split</div>", unsafe_allow_html=True)
        tt = dff['Ticket_Type'].value_counts().reset_index()
        tt.columns = ['Ticket_Type','Count']
        fig_tt = px.pie(tt, names='Ticket_Type', values='Count',
                        color_discrete_sequence=px.colors.sequential.Purples_r,
                        height=330, hole=0.4)
        fig_tt.update_layout(**PLOT_LAYOUT,
                              legend=LEGEND_STYLE,
                              title=dict(text='', font=dict(color='white')))
        fig_tt.update_traces(textfont_color='white', textinfo='percent+label')
        st.plotly_chart(fig_tt, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Remarks breakdown + YoY Revenue
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("<div class='section-header'>⚡ Trip Condition Breakdown</div>", unsafe_allow_html=True)
        rem = dff['Remarks'].value_counts().reset_index()
        rem.columns = ['Remarks','Count']
        color_map_rem = {
            'peak'       : '#e74c3c',
            'off-peak'   : '#3498db',
            'weekend'    : '#2ecc71',
            'festival'   : '#f39c12',
            'maintenance': '#e67e22',
            'Normal'     : '#9b59b6',
        }
        fig_rem = px.bar(rem, x='Remarks', y='Count',
                         color='Remarks',
                         color_discrete_map=color_map_rem,
                         text='Count', height=320,
                         labels={'Count':'Trips','Remarks':''})
        fig_rem.update_traces(texttemplate='%{text:,}', textposition='outside',
                              textfont_color='white')
        fig_rem.update_layout(**PLOT_LAYOUT,
                               xaxis=dict(**AXIS_STYLE),
                               yaxis=dict(**AXIS_STYLE),
                               showlegend=False)
        st.plotly_chart(fig_rem, use_container_width=True)

    with col_b:
        st.markdown("<div class='section-header'>💰 Yearly Revenue Comparison</div>", unsafe_allow_html=True)
        yr_rev = (dff.groupby('Year')['Revenue']
                     .sum().reset_index()
                     .rename(columns={'Revenue':'Total_Revenue'}))
        fig_yr = px.bar(yr_rev, x='Year', y='Total_Revenue',
                        text='Total_Revenue',
                        color_discrete_sequence=['#ab47bc'],
                        height=320,
                        labels={'Total_Revenue':'Revenue (₹)','Year':''})
        fig_yr.update_traces(texttemplate='₹%{text:,.0f}', textposition='outside',
                              textfont_color='white',
                              marker_color=['#7b2fbe','#ab47bc','#ce93d8'])
        fig_yr.update_layout(**PLOT_LAYOUT,
                              xaxis=dict(**AXIS_STYLE),
                              yaxis=dict(**AXIS_STYLE))
        st.plotly_chart(fig_yr, use_container_width=True)

# ══════════════════════════════════════════════════════════════
# PAGE 2 — STATION INTELLIGENCE
# ══════════════════════════════════════════════════════════════
elif page == "🚉 Station Intelligence":

    st.markdown("""
    <div class='big-header'>
    <h1>🚉 Station Intelligence</h1>
    <p>Origin / Destination traffic, revenue, and demand heatmap by station</p>
    </div>
    """, unsafe_allow_html=True)

    # Top Origin stations
    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown("<div class='section-header'>🏆 Top 15 Origin Stations by Trips</div>", unsafe_allow_html=True)
        top_orig = (dff['From_Station'].value_counts()
                       .head(15).reset_index()
                       .rename(columns={'From_Station':'Station','count':'Trips'}))
        fig = px.bar(top_orig.sort_values('Trips'),
                     x='Trips', y='Station', orientation='h',
                     color='Trips',
                     color_continuous_scale='Purples',
                     text='Trips', height=480,
                     labels={'Trips':'Number of Trips','Station':''})
        fig.update_traces(texttemplate='%{text:,}', textposition='outside',
                          textfont_color='white')
        fig.update_layout(**PLOT_LAYOUT,
                          xaxis=dict(**AXIS_STYLE),
                          yaxis=dict(**AXIS_STYLE),
                          coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        st.markdown("<div class='section-header'>💰 Top 15 Stations by Avg Fare</div>", unsafe_allow_html=True)
        st_fare = (dff.groupby('From_Station')['Fare']
                      .mean().reset_index()
                      .rename(columns={'Fare':'Avg_Fare'})
                      .sort_values('Avg_Fare', ascending=False)
                      .head(15))
        fig2 = px.bar(st_fare.sort_values('Avg_Fare'),
                      x='Avg_Fare', y='From_Station', orientation='h',
                      color='Avg_Fare',
                      color_continuous_scale='Purples',
                      text='Avg_Fare', height=480,
                      labels={'Avg_Fare':'Avg Fare (₹)','From_Station':''})
        fig2.update_traces(texttemplate='₹%{text:.0f}', textposition='outside',
                           textfont_color='white')
        fig2.update_layout(**PLOT_LAYOUT,
                           xaxis=dict(**AXIS_STYLE),
                           yaxis=dict(**AXIS_STYLE),
                           coloraxis_showscale=False)
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Station-level revenue
    st.markdown("<div class='section-header'>🏦 Station Revenue Performance (Top 20)</div>", unsafe_allow_html=True)
    st_rev = (dff.groupby('From_Station')['Revenue']
                 .sum().reset_index()
                 .rename(columns={'Revenue':'Total_Revenue'})
                 .sort_values('Total_Revenue', ascending=False)
                 .head(20))
    medals = ['🥇','🥈','🥉'] + ['⭐']*17

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("<div class='section-header'>🏆 Top 10 Revenue Stations</div>", unsafe_allow_html=True)
        for i, row in st_rev.head(10).iterrows():
            rank = list(st_rev.index).index(i)
            st.markdown(f"""
            <div class='state-card-green'>
            <span style='font-size:1.1rem;'>{medals[rank]}</span>
            <b style='color:white;'> {row['From_Station']}</b><br>
            <span style='color:#2ecc71; font-size:0.9rem;
            font-weight:600;'>₹{row['Total_Revenue']:,.0f}</span>
            </div>""", unsafe_allow_html=True)

    with col2:
        st.markdown("<div class='section-header'>📊 Revenue Distribution</div>", unsafe_allow_html=True)
        fig_sr = px.bar(st_rev, x='From_Station', y='Total_Revenue',
                        color='Total_Revenue',
                        color_continuous_scale='Purples',
                        labels={'From_Station':'Station',
                                'Total_Revenue':'Revenue (₹)'},
                        height=480)
        fig_sr.update_layout(**PLOT_LAYOUT,
                              xaxis=dict(**AXIS_STYLE, tickangle=45),
                              yaxis=dict(**AXIS_STYLE),
                              coloraxis_showscale=False)
        st.plotly_chart(fig_sr, use_container_width=True)

    # Heatmap: station × day of week
    st.markdown("<div class='section-header'>🗓️ Station Traffic Heatmap (Top 12 × Day of Week)</div>", unsafe_allow_html=True)
    top12 = dff['From_Station'].value_counts().head(12).index.tolist()
    dow_order = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
    heat = (dff[dff['From_Station'].isin(top12)]
               .groupby(['From_Station','DayOfWeek'])
               .size().reset_index(name='Trips'))
    heat['DayOfWeek'] = pd.Categorical(heat['DayOfWeek'], categories=dow_order, ordered=True)
    heat_pivot = heat.pivot(index='From_Station', columns='DayOfWeek', values='Trips').fillna(0)

    fig_hm = px.imshow(heat_pivot,
                       color_continuous_scale='Purples',
                       aspect='auto', height=420,
                       labels={'color':'Trips'})
    fig_hm.update_layout(
    **PLOT_LAYOUT,
    xaxis=dict(color='white', tickfont=dict(color='white')),
    yaxis=dict(color='white', tickfont=dict(color='white')),
    coloraxis_colorbar=dict(
        tickfont=dict(color='white'),
        title=dict(font=dict(color='white'))   # ✅ FIXED
    )
)
    st.plotly_chart(fig_hm, use_container_width=True)

# ══════════════════════════════════════════════════════════════
# PAGE 3 — ROUTE & DISTANCE ANALYSIS
# ══════════════════════════════════════════════════════════════
elif page == "🗺️  Route & Distance Analysis":

    st.markdown("""
    <div class='big-header'>
    <h1>🗺️ Route & Distance Analysis</h1>
    <p>Top routes, distance distribution, fare vs distance, and cost efficiency</p>
    </div>
    """, unsafe_allow_html=True)

    col_l, col_r = st.columns([2,1])

    with col_l:
        st.markdown("<div class='section-header'>🔝 Top 20 Most Popular Routes</div>", unsafe_allow_html=True)
        top_routes = (dff['Route'].value_counts()
                         .head(20).reset_index()
                         .rename(columns={'Route':'Route','count':'Trips'}))
        fig = px.bar(top_routes.sort_values('Trips'),
                     x='Trips', y='Route', orientation='h',
                     color='Trips', color_continuous_scale='Purples',
                     text='Trips', height=600,
                     labels={'Route':'','Trips':'Trips'})
        fig.update_traces(texttemplate='%{text:,}', textposition='outside',
                          textfont_color='white')
        fig.update_layout(**PLOT_LAYOUT,
                          xaxis={**AXIS_STYLE, "tickfont": dict(size=10, color='white')}
                          yaxis = {**AXIS_STYLE, "tickfont": dict(size=10, color='white')}
                          coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        st.markdown("<div class='section-header'>📏 Distance Distribution</div>", unsafe_allow_html=True)
        fig_d = px.histogram(dff, x='Distance_km',
                             nbins=40,
                             color_discrete_sequence=['#ab47bc'],
                             labels={'Distance_km':'Distance (km)','count':'Trips'},
                             height=280)
        fig_d.update_layout(**PLOT_LAYOUT,
                            xaxis=dict(**AXIS_STYLE),
                            yaxis=dict(**AXIS_STYLE))
        st.plotly_chart(fig_d, use_container_width=True)

        st.markdown("<div class='section-header'>💸 Fare Distribution</div>", unsafe_allow_html=True)
        fig_f = px.histogram(dff, x='Fare', nbins=40,
                             color_discrete_sequence=['#ce93d8'],
                             labels={'Fare':'Fare (₹)','count':'Trips'},
                             height=280)
        fig_f.update_layout(**PLOT_LAYOUT,
                            xaxis=dict(**AXIS_STYLE),
                            yaxis=dict(**AXIS_STYLE))
        st.plotly_chart(fig_f, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("<div class='section-header'>📊 Fare vs Distance (Scatter)</div>", unsafe_allow_html=True)
        sample = dff.sample(min(5000, len(dff)), random_state=42)
        fig_sc = px.scatter(sample, x='Distance_km', y='Fare',
                            color='Ticket_Type',
                            opacity=0.55, height=380,
                            labels={'Distance_km':'Distance (km)','Fare':'Fare (₹)'},
                            color_discrete_sequence=px.colors.sequential.Purples_r)
        fig_sc.update_layout(**PLOT_LAYOUT,
                              xaxis=dict(**AXIS_STYLE),
                              yaxis=dict(**AXIS_STYLE),
                              legend=LEGEND_STYLE)
        st.plotly_chart(fig_sc, use_container_width=True)

    with col_b:
        st.markdown("<div class='section-header'>💰 Avg Cost Per Passenger by Ticket Type</div>", unsafe_allow_html=True)
        cpp = (dff.groupby('Ticket_Type')['Cost_per_passenger']
                  .mean().reset_index()
                  .rename(columns={'Cost_per_passenger':'Avg_CPP'})
                  .sort_values('Avg_CPP', ascending=False))
        fig_cpp = px.bar(cpp, x='Ticket_Type', y='Avg_CPP',
                         color='Ticket_Type',
                         color_discrete_sequence=px.colors.sequential.Purples_r,
                         text='Avg_CPP', height=380,
                         labels={'Ticket_Type':'','Avg_CPP':'Avg Cost/Passenger (₹)'})
        fig_cpp.update_traces(texttemplate='₹%{text:.1f}', textposition='outside',
                              textfont_color='white')
        fig_cpp.update_layout(**PLOT_LAYOUT,
                              xaxis=dict(**AXIS_STYLE),
                              yaxis=dict(**AXIS_STYLE),
                              showlegend=False)
        st.plotly_chart(fig_cpp, use_container_width=True)

    # Distance band analysis
    st.markdown("<div class='section-header'>📏 Distance Band Analysis</div>", unsafe_allow_html=True)
    dff2 = dff.copy()
    dff2['Distance_Band'] = pd.cut(dff2['Distance_km'],
                                   bins=[0,3,7,12,25],
                                   labels=['Short (0-3km)',
                                           'Medium (3-7km)',
                                           'Long (7-12km)',
                                           'Very Long (12km+)'])
    db = (dff2.groupby('Distance_Band', observed=True)
              .agg(Trips=('TripID','count'),
                   Avg_Fare=('Fare','mean'),
                   Avg_Passengers=('Passengers','mean'))
              .reset_index())

    fig_db = make_subplots(rows=1, cols=3,
                           subplot_titles=['Number of Trips',
                                           'Avg Fare (₹)',
                                           'Avg Passengers'],
                           specs=[[{"type":"bar"},{"type":"bar"},{"type":"bar"}]])

    colors = ['#7b2fbe','#ab47bc','#ce93d8','#e1bee7']
    for idx, (col_name, y_col, fmt) in enumerate([
        ('Trips','Trips','%{y:,}'),
        ('Avg Fare','Avg_Fare','₹%{y:.0f}'),
        ('Avg Pax','Avg_Passengers','%{y:.1f}')
    ]):
        fig_db.add_trace(
            go.Bar(x=db['Distance_Band'].astype(str),
                   y=db[y_col],
                   marker_color=colors,
                   text=db[y_col],
                   texttemplate=fmt,
                   textposition='outside',
                   textfont=dict(color='white'),
                   showlegend=False),
            row=1, col=idx+1
        )

    fig_db.update_layout(height=350, **PLOT_LAYOUT)
    fig_db.update_annotations(font_color='white')
    for i in range(1, 4):
        fig_db.update_xaxes(tickfont=dict(color='white', size=9),
                            color='white', row=1, col=i)
        fig_db.update_yaxes(tickfont=dict(color='white'),
                            color='white', gridcolor='#1f2937',
                            row=1, col=i)
    st.plotly_chart(fig_db, use_container_width=True)

# ══════════════════════════════════════════════════════════════
# PAGE 4 — TICKET & DEMAND PATTERNS
# ══════════════════════════════════════════════════════════════
elif page == "🎫 Ticket & Demand Patterns":

    st.markdown("""
    <div class='big-header'>
    <h1>🎫 Ticket & Demand Patterns</h1>
    <p>Ticket type performance, peak/off-peak analysis, and passenger load insights</p>
    </div>
    """, unsafe_allow_html=True)

    # Ticket type KPIs
    st.markdown("<div class='section-header'>🎟️ Ticket Type Performance</div>", unsafe_allow_html=True)
    tt_perf = (dff.groupby('Ticket_Type')
                  .agg(Trips=('TripID','count'),
                       Avg_Fare=('Fare','mean'),
                       Avg_Distance=('Distance_km','mean'),
                       Total_Revenue=('Revenue','sum'))
                  .reset_index()
                  .sort_values('Trips', ascending=False))

    col1, col2 = st.columns(2)
    with col1:
        fig_tt1 = px.bar(tt_perf, x='Ticket_Type', y='Trips',
                         color='Ticket_Type',
                         color_discrete_sequence=px.colors.sequential.Purples_r,
                         text='Trips', height=320,
                         labels={'Ticket_Type':'','Trips':'Number of Trips'})
        fig_tt1.update_traces(texttemplate='%{text:,}', textposition='outside',
                              textfont_color='white')
        fig_tt1.update_layout(**PLOT_LAYOUT,
                              xaxis=dict(**AXIS_STYLE),
                              yaxis=dict(**AXIS_STYLE),
                              showlegend=False,
                              title=dict(text='Trips by Ticket Type',
                                         font=dict(color='white',size=13), x=0.5))
        st.plotly_chart(fig_tt1, use_container_width=True)

    with col2:
        fig_tt2 = px.bar(tt_perf, x='Ticket_Type', y='Avg_Fare',
                         color='Ticket_Type',
                         color_discrete_sequence=px.colors.sequential.Purples_r,
                         text='Avg_Fare', height=320,
                         labels={'Ticket_Type':'','Avg_Fare':'Avg Fare (₹)'})
        fig_tt2.update_traces(texttemplate='₹%{text:.0f}', textposition='outside',
                              textfont_color='white')
        fig_tt2.update_layout(**PLOT_LAYOUT,
                              xaxis=dict(**AXIS_STYLE),
                              yaxis=dict(**AXIS_STYLE),
                              showlegend=False,
                              title=dict(text='Avg Fare by Ticket Type',
                                         font=dict(color='white',size=13), x=0.5))
        st.plotly_chart(fig_tt2, use_container_width=True)

    st.dataframe(tt_perf.style.format({
        'Avg_Fare'     : '₹{:.2f}',
        'Avg_Distance' : '{:.2f} km',
        'Total_Revenue': '₹{:,.0f}',
        'Trips'        : '{:,}'
    }), use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Remarks / condition analysis
    st.markdown("<div class='section-header'>⚡ Peak vs Off-Peak vs Other Conditions</div>", unsafe_allow_html=True)
    col_a, col_b = st.columns(2)

    with col_a:
        rem_fare = (dff.groupby('Remarks')['Fare']
                       .mean().reset_index()
                       .rename(columns={'Fare':'Avg_Fare'})
                       .sort_values('Avg_Fare', ascending=False))
        color_map_rem2 = {
            'peak':'#e74c3c','off-peak':'#3498db','weekend':'#2ecc71',
            'festival':'#f39c12','maintenance':'#e67e22','Normal':'#9b59b6'
        }
        fig_rem2 = px.bar(rem_fare, x='Remarks', y='Avg_Fare',
                          color='Remarks', color_discrete_map=color_map_rem2,
                          text='Avg_Fare', height=320,
                          labels={'Remarks':'Condition','Avg_Fare':'Avg Fare (₹)'})
        fig_rem2.update_traces(texttemplate='₹%{text:.0f}', textposition='outside',
                               textfont_color='white')
        fig_rem2.update_layout(**PLOT_LAYOUT,
                               xaxis=dict(**AXIS_STYLE),
                               yaxis=dict(**AXIS_STYLE),
                               showlegend=False,
                               title=dict(text='Avg Fare by Trip Condition',
                                          font=dict(color='white',size=13), x=0.5))
        st.plotly_chart(fig_rem2, use_container_width=True)

    with col_b:
        rem_pax = (dff.groupby('Remarks')['Passengers']
                      .mean().reset_index()
                      .rename(columns={'Passengers':'Avg_Passengers'})
                      .sort_values('Avg_Passengers', ascending=False))
        fig_pax = px.bar(rem_pax, x='Remarks', y='Avg_Passengers',
                         color='Remarks', color_discrete_map=color_map_rem2,
                         text='Avg_Passengers', height=320,
                         labels={'Remarks':'Condition','Avg_Passengers':'Avg Passengers'})
        fig_pax.update_traces(texttemplate='%{text:.1f}', textposition='outside',
                              textfont_color='white')
        fig_pax.update_layout(**PLOT_LAYOUT,
                              xaxis=dict(**AXIS_STYLE),
                              yaxis=dict(**AXIS_STYLE),
                              showlegend=False,
                              title=dict(text='Avg Passengers by Condition',
                                         font=dict(color='white',size=13), x=0.5))
        st.plotly_chart(fig_pax, use_container_width=True)

    # Ticket × Condition heatmap
    st.markdown("<div class='section-header'>🔥 Ticket Type × Trip Condition (Revenue Heatmap)</div>", unsafe_allow_html=True)
    tc_heat = (dff.groupby(['Ticket_Type','Remarks'])['Revenue']
                  .sum().reset_index())
    tc_pivot = tc_heat.pivot(index='Ticket_Type',
                             columns='Remarks',
                             values='Revenue').fillna(0)
    fig_tch = px.imshow(tc_pivot, color_continuous_scale='Purples',
                        aspect='auto', height=350,
                        labels={'color':'Revenue (₹)'})
    fig_tch.update_layout(**PLOT_LAYOUT,
                          xaxis=dict(color='white', tickfont=dict(color='white')),
                          yaxis=dict(color='white', tickfont=dict(color='white')),
                          coloraxis_colorbar=dict(tickfont=dict(color='white'),
                                                  titlefont=dict(color='white')))
    st.plotly_chart(fig_tch, use_container_width=True)

# ══════════════════════════════════════════════════════════════
# PAGE 5 — TIME SERIES TRENDS
# ══════════════════════════════════════════════════════════════
elif page == "📅 Time Series Trends":

    st.markdown("""
    <div class='big-header'>
    <h1>📅 Time Series Trends</h1>
    <p>Year-over-year growth, monthly seasonality, and day-of-week patterns</p>
    </div>
    """, unsafe_allow_html=True)

    # Daily trips trend
    st.markdown("<div class='section-header'>📈 Daily Trips Volume (Full Timeline)</div>", unsafe_allow_html=True)
    daily = (dff.groupby('Date').size().reset_index(name='Trips'))
    fig_d = px.line(daily, x='Date', y='Trips',
                    color_discrete_sequence=['#ab47bc'],
                    labels={'Date':'','Trips':'Trips per Day'},
                    height=300)
    fig_d.update_traces(line_width=1.5)
    fig_d.update_layout(**PLOT_LAYOUT,
                        xaxis=dict(**AXIS_STYLE),
                        yaxis=dict(**AXIS_STYLE))
    st.plotly_chart(fig_d, use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("<div class='section-header'>📆 Monthly Seasonality (Avg Trips)</div>", unsafe_allow_html=True)
        month_order = ['Jan','Feb','Mar','Apr','May','Jun',
                       'Jul','Aug','Sep','Oct','Nov','Dec']
        mon_avg = (dff.groupby('Month_Name').size()
                      .reset_index(name='Trips'))
        mon_avg['Month_Name'] = pd.Categorical(mon_avg['Month_Name'],
                                               categories=month_order, ordered=True)
        mon_avg = mon_avg.sort_values('Month_Name')
        fig_mon = px.bar(mon_avg, x='Month_Name', y='Trips',
                         color='Trips', color_continuous_scale='Purples',
                         text='Trips', height=340,
                         labels={'Month_Name':'Month','Trips':'Total Trips'})
        fig_mon.update_traces(texttemplate='%{text:,}', textposition='outside',
                              textfont_color='white')
        fig_mon.update_layout(**PLOT_LAYOUT,
                              xaxis=dict(**AXIS_STYLE),
                              yaxis=dict(**AXIS_STYLE),
                              coloraxis_showscale=False)
        st.plotly_chart(fig_mon, use_container_width=True)

    with col2:
        st.markdown("<div class='section-header'>📅 Day of Week Pattern</div>", unsafe_allow_html=True)
        dow_order = ['Monday','Tuesday','Wednesday','Thursday',
                     'Friday','Saturday','Sunday']
        dow_df = (dff.groupby('DayOfWeek').size()
                     .reset_index(name='Trips'))
        dow_df['DayOfWeek'] = pd.Categorical(dow_df['DayOfWeek'],
                                             categories=dow_order, ordered=True)
        dow_df = dow_df.sort_values('DayOfWeek')
        fig_dow = px.bar(dow_df, x='DayOfWeek', y='Trips',
                         color='Trips', color_continuous_scale='Purples',
                         text='Trips', height=340,
                         labels={'DayOfWeek':'','Trips':'Total Trips'})
        fig_dow.update_traces(texttemplate='%{text:,}', textposition='outside',
                              textfont_color='white')
        fig_dow.update_layout(**PLOT_LAYOUT,
                              xaxis=dict(**AXIS_STYLE),
                              yaxis=dict(**AXIS_STYLE),
                              coloraxis_showscale=False)
        st.plotly_chart(fig_dow, use_container_width=True)

    # YoY comparison
    st.markdown("<div class='section-header'>📊 Year-over-Year Monthly Comparison</div>", unsafe_allow_html=True)
    yoy = (dff.groupby(['Year','Month','Month_Name']).size()
              .reset_index(name='Trips'))
    yoy['Month_Name'] = pd.Categorical(yoy['Month_Name'],
                                       categories=month_order, ordered=True)
    yoy = yoy.sort_values(['Year','Month_Name'])
    fig_yoy = px.line(yoy, x='Month_Name', y='Trips',
                      color='Year', markers=True,
                      color_discrete_sequence=['#7b2fbe','#ab47bc','#ce93d8'],
                      labels={'Month_Name':'Month','Trips':'Trips','Year':'Year'},
                      height=380)
    fig_yoy.update_layout(**PLOT_LAYOUT,
                          xaxis=dict(**AXIS_STYLE),
                          yaxis=dict(**AXIS_STYLE),
                          legend=LEGEND_STYLE)
    st.plotly_chart(fig_yoy, use_container_width=True)

    # Quarterly revenue trend
    st.markdown("<div class='section-header'>📊 Quarterly Revenue Trend</div>", unsafe_allow_html=True)
    qrev = (dff.groupby('Quarter')['Revenue']
               .sum().reset_index()
               .rename(columns={'Revenue':'Total_Revenue'})
               .sort_values('Quarter'))
    fig_q = px.bar(qrev, x='Quarter', y='Total_Revenue',
                   color='Total_Revenue', color_continuous_scale='Purples',
                   text='Total_Revenue', height=340,
                   labels={'Quarter':'Quarter','Total_Revenue':'Revenue (₹)'})
    fig_q.update_traces(texttemplate='₹%{text:,.0f}', textposition='outside',
                        textfont_color='white')
    fig_q.update_layout(**PLOT_LAYOUT,
                        xaxis=dict(**AXIS_STYLE),
                        yaxis=dict(**AXIS_STYLE),
                        coloraxis_showscale=False)
    st.plotly_chart(fig_q, use_container_width=True)

# ══════════════════════════════════════════════════════════════
# PAGE 6 — INSIGHTS & RECOMMENDATIONS
# ══════════════════════════════════════════════════════════════
elif page == "💡 Insights & Recommendations":

    st.markdown("""
    <div class='big-header'>
    <h1>💡 Insights & Recommendations</h1>
    <p>Data-driven findings for DMRC, transport planners, and policy makers</p>
    </div>
    """, unsafe_allow_html=True)

    # Key stats
    busiest  = dff['From_Station'].value_counts().idxmax()
    top_route= dff['Route'].value_counts().idxmax()
    top_ticket = dff['Ticket_Type'].value_counts().idxmax()
    peak_rem = dff.groupby('Remarks')['Passengers'].mean().idxmax()
    max_fare_tt = dff.groupby('Ticket_Type')['Fare'].mean().idxmax()

    st.markdown(f"""
    <div style='background:#0a2e0a; border:2px solid #2ecc71;
    border-radius:12px; padding:1.5rem; margin:1rem 0;'>
    <div style='color:#2ecc71; font-size:1.2rem;
    font-weight:800; margin-bottom:0.8rem;'>
    🎯 Key Findings from Delhi Metro Data (2022–2024)</div>
    <div style='color:white; font-size:1rem; line-height:2;'>
    🚉 <b style='color:#2ecc71;'>Busiest Origin:</b> {busiest} — highest departure volume<br>
    🛤️ <b style='color:#2ecc71;'>Most Popular Route:</b> {top_route}<br>
    🎫 <b style='color:#2ecc71;'>Dominant Ticket Type:</b> {top_ticket} — accounts for largest share of trips<br>
    ⚡ <b style='color:#2ecc71;'>Highest Load Condition:</b> {peak_rem.title()} hours drive maximum passenger density<br>
    💰 <b style='color:#2ecc71;'>Highest Avg Fare:</b> {max_fare_tt} ticket type commands the highest average fare
    </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div style='background:#1b0036; border:1px solid #7b2fbe;
        border-radius:12px; padding:1.5rem;'>
        <div style='color:#ce93d8; font-size:1.1rem;
        font-weight:800; margin-bottom:1rem;'>
        🏢 For DMRC Operations</div>

        <div style='color:white; margin-bottom:1rem;'>
        <b style='color:#f39c12;'>1. Capacity Management at Peak Stations</b><br>
        <span style='color:#e0e0e0;'>Rajiv Chowk and New Delhi consistently show
        highest origination volumes. Deploy additional staff and
        cross-platform guidance during morning/evening peaks.</span>
        </div>

        <div style='color:white; margin-bottom:1rem;'>
        <b style='color:#f39c12;'>2. Dynamic Fare Strategy</b><br>
        <span style='color:#e0e0e0;'>Festival and peak conditions show
        different passenger load patterns. Introduce time-of-day
        dynamic pricing for Smart Card users to balance demand.</span>
        </div>

        <div style='color:white; margin-bottom:1rem;'>
        <b style='color:#f39c12;'>3. Maintenance Scheduling</b><br>
        <span style='color:#e0e0e0;'>Maintenance trips cluster on specific
        day-station combinations. Schedule planned maintenance
        during lowest-traffic windows (identified via heatmap).</span>
        </div>

        <div style='color:white;'>
        <b style='color:#f39c12;'>4. Tourist Card Monetisation</b><br>
        <span style='color:#e0e0e0;'>Tourist Card shows competitive avg fare.
        Bundle tourist packages including airport routes and heritage
        corridor stations to increase Tourist Card adoption.</span>
        </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div style='background:#1b0036; border:1px solid #7b2fbe;
        border-radius:12px; padding:1.5rem;'>
        <div style='color:#ce93d8; font-size:1.1rem;
        font-weight:800; margin-bottom:1rem;'>
        📊 For Policy & Urban Planning</div>

        <div style='color:white; margin-bottom:1rem;'>
        <b style='color:#f39c12;'>1. Weekend Ridership Growth</b><br>
        <span style='color:#e0e0e0;'>Weekend trips show competitive volumes.
        Introduce discounted weekend Smart Card passes to
        shift commuters from personal vehicles.</span>
        </div>

        <div style='color:white; margin-bottom:1rem;'>
        <b style='color:#f39c12;'>2. Short-Trip Optimisation</b><br>
        <span style='color:#e0e0e0;'>0-3km trips generate high
        cost-per-passenger. Encourage feeder bus integration
        to reduce short metro hops and improve revenue per km.</span>
        </div>

        <div style='color:white; margin-bottom:1rem;'>
        <b style='color:#f39c12;'>3. Revenue Corridor Expansion</b><br>
        <span style='color:#e0e0e0;'>Central Delhi stations (Rajiv Chowk,
        Kashmere Gate, Mandi House) generate the highest revenue.
        Prioritise interchange improvements on these hubs.</span>
        </div>

        <div style='color:white;'>
        <b style='color:#f39c12;'>4. Yearly Growth Acceleration</b><br>
        <span style='color:#e0e0e0;'>Year-on-year data reveals
        quarter-level growth patterns. Use Q4 festival season
        surge data to plan service frequency increases proactively.</span>
        </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div class='section-header'>📋 Data Quality Summary</div>", unsafe_allow_html=True)
    dq = pd.DataFrame({
        'Column'       : df.columns.tolist(),
        'Total_Records': len(df),
        'Missing'      : df.isnull().sum().values,
        'Missing_%'    : (df.isnull().mean() * 100).round(2).values,
        'Unique_Values': df.nunique().values,
    })
    st.dataframe(dq.style.format({
        'Missing_%'    : '{:.2f}%',
        'Total_Records': '{:,}',
        'Missing'      : '{:,}'
    }), use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div style='text-align:center; color:#ab47bc;
    font-size:0.85rem; padding:1rem;
    border-top:1px solid #7b2fbe;'>
    MBA Applied Business Analytics Project<br>
    Dataset: Delhi Metro Ridership (2022–2024) · 150,000 Trips<br>
    Dashboard built with Streamlit + Plotly
    </div>
    """, unsafe_allow_html=True)
