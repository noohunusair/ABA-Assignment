# ============================================================
# DELHI METRO — CROWD PREDICTION DASHBOARD
# MBA Applied Business Analytics
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, date, timedelta
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Delhi Metro Crowd Predictor",
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
    margin-bottom: 1.5rem;
    border: 1px solid #7b2fbe;
}
.big-header h1 { font-size: 2.2rem; font-weight: 900; margin: 0; color: white; }
.big-header p  { font-size: 1rem; margin: 0.5rem 0 0 0; color: #ce93d8; }
.kpi-card {
    background: linear-gradient(135deg, #1b0036, #2d1b69);
    border: 1px solid #7b2fbe;
    border-radius: 12px;
    padding: 1.2rem;
    text-align: center;
    color: white;
    margin-bottom: 0.5rem;
}
.kpi-card .kpi-value { font-size: 1.8rem; font-weight: 900; color: #ce93d8; }
.kpi-card .kpi-label { font-size: 0.82rem; color: #ab47bc; margin-top: 0.3rem; }
.crowd-low    { background: linear-gradient(135deg,#0a2e0a,#0d3b0d); border:2px solid #2ecc71; border-radius:14px; padding:1.5rem; text-align:center; }
.crowd-medium { background: linear-gradient(135deg,#2e2200,#3b2c00); border:2px solid #f39c12; border-radius:14px; padding:1.5rem; text-align:center; }
.crowd-high   { background: linear-gradient(135deg,#2e0a0a,#3b0d0d); border:2px solid #e74c3c; border-radius:14px; padding:1.5rem; text-align:center; }
.prediction-box {
    background: linear-gradient(135deg, #0d001e, #1b0036);
    border: 2px solid #7b2fbe;
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
    margin: 1rem 0;
}
.section-header {
    font-size: 1.2rem; font-weight: 700; color: #ce93d8;
    margin: 1rem 0 0.5rem 0;
    border-bottom: 2px solid #7b2fbe; padding-bottom: 0.3rem;
}
.insight-box {
    background: rgba(27,0,54,0.5);
    border-left: 4px solid #ab47bc;
    padding: 1rem 1.2rem;
    border-radius: 8px;
    margin: 0.6rem 0;
    color: white;
    font-size: 0.93rem;
}
.station-card {
    background: linear-gradient(135deg,#1b0036,#2d1b69);
    border: 1px solid #7b2fbe;
    border-radius: 10px;
    padding: 0.9rem 1.1rem;
    margin: 0.3rem 0;
    color: white;
    display: flex; align-items: center;
}
</style>
""", unsafe_allow_html=True)

PLOT_LAYOUT = dict(
    plot_bgcolor='#0e1117', paper_bgcolor='#0e1117',
    font=dict(color='white', size=12),
)
AXIS_STYLE = dict(gridcolor='#1f2937', color='white',
                  tickfont=dict(color='white'),
                  title_font=dict(color='white'))
LEGEND_STYLE = dict(font=dict(color='white'),
                    bgcolor='#1b0036', bordercolor='#7b2fbe')

# ── Data Loading ──────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv('delhi_metro_updated.csv')
    df['From_Station'] = df['From_Station'].str.strip().str.title()
    df['To_Station']   = df['To_Station'].str.strip().str.title()
    df['Date']         = pd.to_datetime(df['Date'], errors='coerce')
    df = df.dropna(subset=['Date'])
    df['Year']      = df['Date'].dt.year
    df['Month']     = df['Date'].dt.month
    df['Month_Name']= df['Date'].dt.strftime('%b')
    df['DayOfWeek'] = df['Date'].dt.day_name()
    df['Quarter']   = df['Date'].dt.to_period('Q').astype(str)
    df['Ticket_Type']= df['Ticket_Type'].fillna('Unknown').str.strip()
    df['Remarks']    = df['Remarks'].fillna('Normal').str.strip().replace('','Normal')
    for col in ['Passengers','Fare','Distance_km','Cost_per_passenger']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df['Revenue'] = df['Fare'] * df['Passengers'].fillna(1)
    df['Route']   = df['From_Station'] + ' → ' + df['To_Station']
    return df

@st.cache_data
def build_prediction_model(df):
    """
    Statistical crowd prediction model built from historical data.
    For each station × day-of-week × condition → computes expected trip volume
    (crowd index), average passengers, and derived crowd level (Low/Medium/High).
    """
    # 1. Station × condition × DOW trip volumes (crowd proxy)
    sc = df.groupby(['From_Station','Remarks','DayOfWeek']).agg(
        avg_passengers=('Passengers','mean'),
        std_passengers=('Passengers','std'),
        trip_count    =('TripID','count'),
        avg_fare      =('Fare','mean'),
    ).reset_index()
    sc['std_passengers'] = sc['std_passengers'].fillna(2.0)

    # 2. Station trip-volume rank (station importance weight)
    st_vol = df.groupby('From_Station').size().reset_index(name='total_trips')
    st_vol['station_weight'] = (st_vol['total_trips'] - st_vol['total_trips'].min()) / \
                               (st_vol['total_trips'].max() - st_vol['total_trips'].min())
    sc = sc.merge(st_vol, on='From_Station')

    # 3. Monthly seasonality factor per station
    mon_avg = df.groupby(['From_Station','Month'])['Passengers'].mean().reset_index()
    mon_avg.columns = ['From_Station','Month','monthly_avg']

    # 4. Condition multipliers (derived from real data)
    cond_mult = df.groupby('Remarks')['Passengers'].mean()
    global_mean = df['Passengers'].mean()
    cond_factor = (cond_mult / global_mean).to_dict()

    # 5. Crowd score = trip_count × condition_factor × station_weight
    # Normalise trip_count per station/DOW combination to 0–100
    max_tc = sc['trip_count'].max()
    min_tc = sc['trip_count'].min()
    sc['crowd_score'] = (
        ((sc['trip_count'] - min_tc) / (max_tc - min_tc)) * 60 +
        sc['station_weight'] * 30 +
        sc['Remarks'].map(cond_factor).fillna(1.0) * 10
    ).clip(0, 100)

    # 6. Assign crowd level labels
    def crowd_label(score):
        if score < 35:   return 'Low',    '#2ecc71', '🟢'
        elif score < 65: return 'Medium', '#f39c12', '🟡'
        else:            return 'High',   '#e74c3c', '🔴'

    sc[['crowd_level','crowd_color','crowd_emoji']] = sc['crowd_score'].apply(
        lambda s: pd.Series(crowd_label(s))
    )
    return sc, mon_avg, cond_factor

df = load_data()
sc_model, mon_avg, cond_factor = build_prediction_model(df)

ALL_STATIONS  = sorted(df['From_Station'].unique().tolist())
ALL_CONDITIONS= ['peak','off-peak','weekend','festival','maintenance']
DOW_ORDER     = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
MONTH_NAMES   = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']

# ── Sidebar ───────────────────────────────────────────────────
st.sidebar.markdown("""
<div style='text-align:center; padding:1.2rem;
background:linear-gradient(135deg,#1b0036,#6a0dad);
border-radius:12px; margin-bottom:1rem;
border:1px solid #7b2fbe;'>
<div style='font-size:2rem;'>🚇</div>
<div style='color:white; font-size:1.1rem; font-weight:700;'>Delhi Metro</div>
<div style='color:#ce93d8; font-size:0.78rem;'>Crowd Prediction System</div>
</div>
""", unsafe_allow_html=True)

page = st.sidebar.radio("Navigate", [
    "🔮 Predict Crowd",
    "🗺️ All Stations Forecast",
    "📊 Station Deep Dive",
    "📅 Weekly & Seasonal Trends",
    "🏆 Crowd Risk Ranking",
])

st.sidebar.markdown("---")
st.sidebar.markdown("<div style='color:#ce93d8;font-weight:700;font-size:0.9rem;'>ℹ️ Model Info</div>", unsafe_allow_html=True)
st.sidebar.markdown(f"""
<div style='font-size:0.8rem; color:#ce93d8; padding:0.5rem;
background:#0d001e; border-radius:8px; border:1px solid #3a0068;'>
<b style='color:#ab47bc;'>Training Data</b><br>
Records  : {len(df):,}<br>
Stations : {df['From_Station'].nunique()}<br>
Period   : 2022 – 2024<br>
Model    : Historical Aggregate +<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Condition Multipliers<br>
Accuracy : Station-level crowd<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;pattern prediction
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# PAGE 1 — PREDICT CROWD (SINGLE STATION PREDICTOR)
# ══════════════════════════════════════════════════════════════
if page == "🔮 Predict Crowd":

    st.markdown("""
    <div class='big-header'>
    <h1>🔮 Delhi Metro Crowd Predictor</h1>
    <p>Select a station, date & condition to predict expected crowd level and passenger load</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Input Form ────────────────────────────────────────────
    st.markdown("<div class='section-header'>🎛️ Prediction Inputs</div>", unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        station = st.selectbox("🚉 Station", ALL_STATIONS, index=ALL_STATIONS.index('Rajiv Chowk'))
    with c2:
        pred_date = st.date_input("📅 Date", value=date.today() + timedelta(days=1),
                                   min_value=date(2024,1,1), max_value=date(2026,12,31))
    with c3:
        condition = st.selectbox("⚡ Service Condition",
                                  ALL_CONDITIONS,
                                  format_func=lambda x: {
                                      'peak':'🔴 Peak Hour','off-peak':'🔵 Off-Peak',
                                      'weekend':'🟢 Weekend','festival':'🟡 Festival',
                                      'maintenance':'🟠 Maintenance'
                                  }.get(x, x))
    with c4:
        ticket_type = st.selectbox("🎫 Ticket Type",
                                    ['All','Smart Card','Tourist Card','Single','Return'])

    pred_dow   = pred_date.strftime('%A')
    pred_month = pred_date.month

    # ── Run Prediction ────────────────────────────────────────
    row = sc_model[
        (sc_model['From_Station'] == station) &
        (sc_model['Remarks']      == condition) &
        (sc_model['DayOfWeek']    == pred_dow)
    ]

    if row.empty:
        # Fallback: station × condition only
        row = sc_model[
            (sc_model['From_Station'] == station) &
            (sc_model['Remarks']      == condition)
        ]

    if not row.empty:
        r = row.iloc[0]
        base_pax   = r['avg_passengers']
        std_pax    = r['std_passengers']
        crowd_score= r['crowd_score']
        crowd_lvl  = r['crowd_level']
        crowd_col  = r['crowd_color']
        crowd_emoji= r['crowd_emoji']

        # Apply monthly seasonality
        mon_row = mon_avg[(mon_avg['From_Station'] == station) &
                          (mon_avg['Month']         == pred_month)]
        if not mon_row.empty:
            global_avg = df['Passengers'].mean()
            mon_factor = mon_row.iloc[0]['monthly_avg'] / global_avg
            base_pax   = base_pax * mon_factor

        pred_low  = max(4, int(base_pax - std_pax))
        pred_high = min(41, int(base_pax + std_pax))
        pred_mid  = int(base_pax)

        # Crowd CSS class
        crowd_class = {
            'Low':'crowd-low', 'Medium':'crowd-medium', 'High':'crowd-high'
        }.get(crowd_lvl, 'crowd-medium')

        wait_time  = {'Low':1,'Medium':4,'High':8}.get(crowd_lvl, 3)
        advice     = {
            'Low'    : 'Great time to travel — platforms will be comfortable.',
            'Medium' : 'Moderate crowd expected — allow a little extra time.',
            'High'   : 'High crowd expected — plan extra wait time & consider off-peak.',
        }.get(crowd_lvl, '')

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Prediction Result Card ────────────────────────────────
    col_main, col_side = st.columns([1.6, 1])

    with col_main:
        st.markdown(f"""
        <div class='prediction-box'>
            <div style='font-size:3.5rem; margin-bottom:0.5rem;'>{crowd_emoji}</div>
            <div style='font-size:1.1rem; color:#ab47bc; margin-bottom:0.3rem;'>
                {station} · {pred_dow} · {MONTH_NAMES[pred_month-1]} {pred_date.day}
            </div>
            <div style='font-size:2.8rem; font-weight:900; color:{crowd_col};
                        margin: 0.5rem 0;'>
                {crowd_lvl} Crowd
            </div>
            <div style='font-size:1.3rem; color:white; margin: 0.4rem 0;'>
                Expected Passengers: <b style='color:#ce93d8;'>
                {pred_low} – {pred_high}</b> per trip
            </div>
            <div style='background:rgba(255,255,255,0.06); border-radius:10px;
                        padding:0.6rem 1rem; margin:0.8rem auto; max-width:480px;'>
                <span style='color:#ab47bc; font-size:0.92rem;'>{advice}</span>
            </div>
            <div style='display:flex; justify-content:center; gap:2rem; margin-top:1rem;'>
                <div>
                    <div style='font-size:1.6rem; font-weight:800;
                                color:#ce93d8;'>{pred_mid}</div>
                    <div style='font-size:0.78rem; color:#9e9e9e;'>Predicted Passengers</div>
                </div>
                <div>
                    <div style='font-size:1.6rem; font-weight:800;
                                color:#f39c12;'>{crowd_score:.0f}/100</div>
                    <div style='font-size:0.78rem; color:#9e9e9e;'>Crowd Score</div>
                </div>
                <div>
                    <div style='font-size:1.6rem; font-weight:800;
                                color:#e74c3c;'>~{wait_time} min</div>
                    <div style='font-size:0.78rem; color:#9e9e9e;'>Est. Wait Time</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_side:
        # Gauge chart for crowd score
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=crowd_score,
            domain={'x':[0,1], 'y':[0,1]},
            title={'text':"Crowd Score", 'font':{'color':'white','size':16}},
            number={'font':{'color':'white','size':36}},
            gauge={
                'axis':{'range':[0,100], 'tickcolor':'white',
                        'tickfont':{'color':'white'}},
                'bar':{'color': crowd_col},
                'bgcolor':'#1b0036',
                'bordercolor':'#7b2fbe',
                'steps':[
                    {'range':[0, 35],  'color':'#0a2e0a'},
                    {'range':[35, 65], 'color':'#2e2200'},
                    {'range':[65, 100],'color':'#2e0a0a'},
                ],
                'threshold':{
                    'line':{'color':crowd_col,'width':4},
                    'thickness':0.8,
                    'value':crowd_score
                }
            }
        ))
        fig_gauge.update_layout(
            height=280, margin=dict(l=20,r=20,t=40,b=20),
            paper_bgcolor='#0e1117', font=dict(color='white')
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

        # Quick tips
        st.markdown(f"""
        <div style='background:#1b0036; border:1px solid #7b2fbe;
        border-radius:10px; padding:1rem; font-size:0.88rem; color:white;'>
        <b style='color:#ce93d8;'>📌 Quick Tips</b><br><br>
        {'✅ Best time to travel!' if crowd_lvl=='Low' else '⚠️ Moderate congestion.' if crowd_lvl=='Medium' else '🚨 High congestion period.'}<br><br>
        <b style='color:#ab47bc;'>Condition:</b> {condition.title()}<br>
        <b style='color:#ab47bc;'>Day:</b> {pred_dow}<br>
        <b style='color:#ab47bc;'>Month:</b> {MONTH_NAMES[pred_month-1]}<br>
        <b style='color:#ab47bc;'>Ticket:</b> {ticket_type}
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── 7-Day Forecast for this station ──────────────────────
    st.markdown("<div class='section-header'>📅 7-Day Crowd Forecast</div>", unsafe_allow_html=True)

    forecast_rows = []
    for i in range(7):
        fdate = pred_date + timedelta(days=i)
        fdow  = fdate.strftime('%A')
        fmon  = fdate.month
        # Auto-assign condition: weekends → weekend, otherwise use selected condition
        fcond = 'weekend' if fdow in ['Saturday','Sunday'] else condition

        fr = sc_model[
            (sc_model['From_Station'] == station) &
            (sc_model['Remarks']      == fcond) &
            (sc_model['DayOfWeek']    == fdow)
        ]
        if fr.empty:
            fr = sc_model[
                (sc_model['From_Station'] == station) &
                (sc_model['Remarks']      == fcond)
            ]
        if not fr.empty:
            rr      = fr.iloc[0]
            fp      = rr['avg_passengers']
            fs      = rr['crowd_score']
            fl, fc, fe = rr['crowd_level'], rr['crowd_color'], rr['crowd_emoji']
        else:
            fp, fs, fl, fc, fe = 20, 50, 'Medium', '#f39c12', '🟡'

        forecast_rows.append({
            'Date': fdate.strftime('%a %d %b'),
            'Passengers': int(fp),
            'Crowd_Score': round(fs, 1),
            'Crowd_Level': fl,
            'Color': fc,
            'Emoji': fe,
            'Condition': fcond,
        })

    fdf = pd.DataFrame(forecast_rows)

    # Bar chart for 7-day forecast
    fig_7d = go.Figure()
    color_map = {'Low':'#2ecc71', 'Medium':'#f39c12', 'High':'#e74c3c'}
    for lvl, grp in fdf.groupby('Crowd_Level'):
        fig_7d.add_trace(go.Bar(
            x=grp['Date'], y=grp['Crowd_Score'],
            name=f'{lvl} Crowd',
            marker_color=color_map.get(lvl,'#ab47bc'),
            text=[f"{fe} {fl}" for fe, fl in zip(grp['Emoji'], grp['Crowd_Level'])],
            textposition='outside', textfont=dict(color='white', size=11),
        ))
    fig_7d.update_layout(
        **PLOT_LAYOUT,
        height=320, barmode='stack',
        xaxis=dict(**AXIS_STYLE, title=''),
        yaxis=dict(**AXIS_STYLE, title='Crowd Score (0–100)', range=[0, 115]),
        legend=LEGEND_STYLE,
        title=dict(text=f'7-Day Crowd Score Forecast — {station}',
                   font=dict(color='white', size=14), x=0.5),
    )
    st.plotly_chart(fig_7d, use_container_width=True)

    # Table view
    display_cols = ['Date','Condition','Passengers','Crowd_Score','Crowd_Level']
    st.dataframe(
        fdf[display_cols].style
            .applymap(lambda v: f'color: {color_map.get(v, "white")}')
                      subset=['Crowd_Level'])
            .format({'Passengers':'{:.0f}','Crowd_Score':'{:.1f}'}),
        use_container_width=True, height=290
    )


# ══════════════════════════════════════════════════════════════
# PAGE 2 — ALL STATIONS FORECAST
# ══════════════════════════════════════════════════════════════
elif page == "🗺️ All Stations Forecast":

    st.markdown("""
    <div class='big-header'>
    <h1>🗺️ All Stations Crowd Forecast</h1>
    <p>Live crowd predictions across all 24 Delhi Metro stations for a selected date & condition</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        all_date = st.date_input("📅 Date",
                                  value=date.today() + timedelta(days=1),
                                  min_value=date(2024,1,1),
                                  max_value=date(2026,12,31),
                                  key='all_date')
    with col2:
        all_cond = st.selectbox("⚡ Service Condition",
                                 ALL_CONDITIONS,
                                 format_func=lambda x: {
                                     'peak':'🔴 Peak Hour','off-peak':'🔵 Off-Peak',
                                     'weekend':'🟢 Weekend','festival':'🟡 Festival',
                                     'maintenance':'🟠 Maintenance'
                                 }.get(x, x), key='all_cond')
    with col3:
        sort_by = st.selectbox("🔃 Sort By",
                                ['Crowd Score ↓','Crowd Score ↑',
                                 'Station A–Z','Passengers ↓'])

    all_dow   = all_date.strftime('%A')
    all_month = all_date.month

    # Build per-station predictions
    station_preds = []
    for stn in ALL_STATIONS:
        row = sc_model[
            (sc_model['From_Station'] == stn) &
            (sc_model['Remarks']      == all_cond) &
            (sc_model['DayOfWeek']    == all_dow)
        ]
        if row.empty:
            row = sc_model[(sc_model['From_Station']==stn) & (sc_model['Remarks']==all_cond)]
        if row.empty:
            row = sc_model[sc_model['From_Station'] == stn]

        if not row.empty:
            r  = row.iloc[0]
            fp = r['avg_passengers']
            fs = r['crowd_score']
            fl, fc, fe = r['crowd_level'], r['crowd_color'], r['crowd_emoji']
        else:
            fp, fs, fl, fc, fe = 20, 50, 'Medium', '#f39c12', '🟡'

        station_preds.append({
            'Station': stn, 'Passengers': round(fp,1),
            'Crowd_Score': round(fs,1),
            'Crowd_Level': fl, 'Color': fc, 'Emoji': fe,
        })

    spdf = pd.DataFrame(station_preds)

    # Sort
    if sort_by == 'Crowd Score ↓':
        spdf = spdf.sort_values('Crowd_Score', ascending=False)
    elif sort_by == 'Crowd Score ↑':
        spdf = spdf.sort_values('Crowd_Score')
    elif sort_by == 'Station A–Z':
        spdf = spdf.sort_values('Station')
    else:
        spdf = spdf.sort_values('Passengers', ascending=False)

    # Summary KPIs
    high_ct = (spdf['Crowd_Level']=='High').sum()
    med_ct  = (spdf['Crowd_Level']=='Medium').sum()
    low_ct  = (spdf['Crowd_Level']=='Low').sum()
    avg_score = spdf['Crowd_Score'].mean()

    k1, k2, k3, k4 = st.columns(4)
    for col, val, lbl in [
        (k1, f"{high_ct}", "🔴 High Crowd Stations"),
        (k2, f"{med_ct}",  "🟡 Medium Crowd Stations"),
        (k3, f"{low_ct}",  "🟢 Low Crowd Stations"),
        (k4, f"{avg_score:.0f}/100", "📊 Avg Crowd Score"),
    ]:
        with col:
            st.markdown(f"""
            <div class='kpi-card'>
            <div class='kpi-value'>{val}</div>
            <div class='kpi-label'>{lbl}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Horizontal bar chart — all stations
    st.markdown("<div class='section-header'>📊 Crowd Score — All Stations</div>", unsafe_allow_html=True)
    color_map2 = {'Low':'#2ecc71','Medium':'#f39c12','High':'#e74c3c'}

    fig_all = go.Figure()
    for lvl in ['High','Medium','Low']:
        sub = spdf[spdf['Crowd_Level']==lvl]
        if sub.empty: continue
        fig_all.add_trace(go.Bar(
            y=sub['Station'], x=sub['Crowd_Score'],
            orientation='h', name=f'{lvl} Crowd',
            marker_color=color_map2[lvl],
            text=[f"{e} {s:.0f}" for e, s in zip(sub['Emoji'], sub['Crowd_Score'])],
            textposition='outside', textfont=dict(color='white'),
        ))
    fig_all.update_layout(
        **PLOT_LAYOUT, height=600, barmode='overlay',
        xaxis=dict(**AXIS_STYLE, title='Crowd Score (0–100)', range=[0,115]),
        yaxis=dict(**AXIS_STYLE, title=''),
        legend=LEGEND_STYLE,
        title=dict(text=f'All Stations Crowd Forecast — {all_dow} · {all_cond.title()}',
                   font=dict(color='white', size=14), x=0.5),
    )
    st.plotly_chart(fig_all, use_container_width=True)

    # ── Station Cards Grid ────────────────────────────────────
    st.markdown("<div class='section-header'>🚉 Station-by-Station Prediction Cards</div>", unsafe_allow_html=True)

    card_cols = st.columns(3)
    for i, (_, row) in enumerate(spdf.iterrows()):
        card_class = {
            'Low':'crowd-low', 'Medium':'crowd-medium', 'High':'crowd-high'
        }.get(row['Crowd_Level'], 'crowd-medium')
        with card_cols[i % 3]:
            st.markdown(f"""
            <div class='{card_class}' style='margin-bottom:0.7rem; padding:1rem;
            border-radius:12px; text-align:center;'>
                <div style='font-size:1.5rem;'>{row['Emoji']}</div>
                <div style='font-weight:700; font-size:1rem; color:white;
                            margin:0.3rem 0;'>{row['Station']}</div>
                <div style='font-size:1.4rem; font-weight:900;
                            color:{row['Color']};'>{row['Crowd_Level']}</div>
                <div style='font-size:0.82rem; color:#aaa; margin-top:0.2rem;'>
                    Score: {row['Crowd_Score']:.0f} &nbsp;|&nbsp;
                    ~{row['Passengers']:.0f} pax/trip
                </div>
            </div>
            """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# PAGE 3 — STATION DEEP DIVE
# ══════════════════════════════════════════════════════════════
elif page == "📊 Station Deep Dive":

    st.markdown("""
    <div class='big-header'>
    <h1>📊 Station Deep Dive</h1>
    <p>Historical crowd patterns, passenger load, and condition breakdown for any station</p>
    </div>
    """, unsafe_allow_html=True)

    sel_stn = st.selectbox("🚉 Choose Station", ALL_STATIONS,
                             index=ALL_STATIONS.index('Rajiv Chowk'))

    stn_df = df[df['From_Station'] == sel_stn]

    # KPIs
    total_t   = len(stn_df)
    avg_pax   = stn_df['Passengers'].mean()
    avg_fare  = stn_df['Fare'].mean()
    total_rev = stn_df['Revenue'].sum()
    top_dest  = stn_df['To_Station'].value_counts().idxmax()

    k1,k2,k3,k4,k5 = st.columns(5)
    for col, val, lbl in [
        (k1, f"{total_t:,}",      "Total Trips"),
        (k2, f"{avg_pax:.1f}",    "Avg Passengers"),
        (k3, f"₹{avg_fare:.0f}", "Avg Fare"),
        (k4, f"₹{total_rev/1e6:.1f}M","Total Revenue"),
        (k5, top_dest,             "Top Destination"),
    ]:
        with col:
            st.markdown(f"""
            <div class='kpi-card'>
            <div class='kpi-value'>{val}</div>
            <div class='kpi-label'>{lbl}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col_l, col_r = st.columns(2)

    with col_l:
        # Crowd score by condition
        st.markdown("<div class='section-header'>⚡ Crowd Score by Condition</div>", unsafe_allow_html=True)
        stn_sc = sc_model[sc_model['From_Station'] == sel_stn].groupby('Remarks').agg(
            avg_score=('crowd_score','mean'),
            avg_pax  =('avg_passengers','mean')
        ).reset_index()
        color_cond = {
            'peak':'#e74c3c','off-peak':'#3498db','weekend':'#2ecc71',
            'festival':'#f39c12','maintenance':'#e67e22','Normal':'#9b59b6'
        }
        fig_cond = px.bar(stn_sc, x='Remarks', y='avg_score',
                          color='Remarks', color_discrete_map=color_cond,
                          text='avg_score', height=320,
                          labels={'Remarks':'Condition','avg_score':'Avg Crowd Score'})
        fig_cond.update_traces(texttemplate='%{text:.0f}', textposition='outside',
                               textfont_color='white')
        fig_cond.update_layout(**PLOT_LAYOUT, xaxis=dict(**AXIS_STYLE),
                               yaxis=dict(**AXIS_STYLE, range=[0,115]), showlegend=False)
        st.plotly_chart(fig_cond, use_container_width=True)

    with col_r:
        # Passengers by day of week
        st.markdown("<div class='section-header'>📅 Avg Passengers by Day of Week</div>", unsafe_allow_html=True)
        dow_pax = stn_df.groupby('DayOfWeek')['Passengers'].mean().reset_index()
        dow_pax['DayOfWeek'] = pd.Categorical(dow_pax['DayOfWeek'],
                                               categories=DOW_ORDER, ordered=True)
        dow_pax = dow_pax.sort_values('DayOfWeek')
        fig_dow = px.bar(dow_pax, x='DayOfWeek', y='Passengers',
                         color='Passengers', color_continuous_scale='Purples',
                         text='Passengers', height=320,
                         labels={'DayOfWeek':'','Passengers':'Avg Passengers'})
        fig_dow.update_traces(texttemplate='%{text:.1f}', textposition='outside',
                              textfont_color='white')
        fig_dow.update_layout(**PLOT_LAYOUT, xaxis=dict(**AXIS_STYLE),
                               yaxis=dict(**AXIS_STYLE), coloraxis_showscale=False)
        st.plotly_chart(fig_dow, use_container_width=True)

    # Crowd heatmap: DOW × Condition
    st.markdown("<div class='section-header'>🔥 Crowd Score Heatmap — Day × Condition</div>", unsafe_allow_html=True)
    stn_heat = sc_model[sc_model['From_Station'] == sel_stn].copy()
    stn_heat['DayOfWeek'] = pd.Categorical(stn_heat['DayOfWeek'],
                                            categories=DOW_ORDER, ordered=True)
    heat_piv = stn_heat.pivot_table(index='Remarks', columns='DayOfWeek',
                                     values='crowd_score', aggfunc='mean').fillna(0)

    fig_heat = px.imshow(heat_piv, color_continuous_scale='RdYlGn_r',
                         aspect='auto', height=300,
                         labels={'color':'Crowd Score'},
                         zmin=0, zmax=100,
                         text_auto='.0f')
    fig_heat.update_layout(**PLOT_LAYOUT,
                            xaxis=dict(color='white', tickfont=dict(color='white')),
                            yaxis=dict(color='white', tickfont=dict(color='white')),
                            coloraxis_colorbar=dict(tickfont=dict(color='white'),
                                                    titlefont=dict(color='white')))
    st.plotly_chart(fig_heat, use_container_width=True)

    # Monthly trend
    st.markdown("<div class='section-header'>📈 Monthly Passenger Trend</div>", unsafe_allow_html=True)
    mon_t = stn_df.groupby(['Year','Month','Month_Name'])['Passengers'].mean().reset_index()
    mon_t['YM'] = mon_t['Year'].astype(str) + '-' + mon_t['Month'].astype(str).str.zfill(2)
    mon_t = mon_t.sort_values('YM')

    fig_mon = px.line(mon_t, x='YM', y='Passengers',
                      color='Year',
                      color_discrete_sequence=['#7b2fbe','#ab47bc','#ce93d8'],
                      markers=True, height=320,
                      labels={'YM':'Month','Passengers':'Avg Passengers/Trip'})
    fig_mon.update_layout(**PLOT_LAYOUT,
                           xaxis=dict(**AXIS_STYLE, title=''),
                           yaxis=dict(**AXIS_STYLE), legend=LEGEND_STYLE)
    st.plotly_chart(fig_mon, use_container_width=True)


# ══════════════════════════════════════════════════════════════
# PAGE 4 — WEEKLY & SEASONAL TRENDS
# ══════════════════════════════════════════════════════════════
elif page == "📅 Weekly & Seasonal Trends":

    st.markdown("""
    <div class='big-header'>
    <h1>📅 Weekly & Seasonal Crowd Trends</h1>
    <p>System-wide crowd patterns by day of week, month, quarter, and condition</p>
    </div>
    """, unsafe_allow_html=True)

    # DOW crowd scores across all stations
    st.markdown("<div class='section-header'>📆 Average Crowd Score by Day of Week</div>", unsafe_allow_html=True)
    dow_scores = sc_model.groupby('DayOfWeek')['crowd_score'].mean().reset_index()
    dow_scores['DayOfWeek'] = pd.Categorical(dow_scores['DayOfWeek'],
                                              categories=DOW_ORDER, ordered=True)
    dow_scores = dow_scores.sort_values('DayOfWeek')

    fig_dow_cs = px.bar(dow_scores, x='DayOfWeek', y='crowd_score',
                        color='crowd_score',
                        color_continuous_scale='RdYlGn_r',
                        text='crowd_score', height=320,
                        labels={'DayOfWeek':'','crowd_score':'Avg Crowd Score'},
                        range_color=[0,100])
    fig_dow_cs.update_traces(texttemplate='%{text:.0f}', textposition='outside',
                              textfont_color='white')
    fig_dow_cs.update_layout(**PLOT_LAYOUT, xaxis=dict(**AXIS_STYLE),
                              yaxis=dict(**AXIS_STYLE, range=[0,115]),
                              coloraxis_showscale=False)
    st.plotly_chart(fig_dow_cs, use_container_width=True)

    col_l, col_r = st.columns(2)

    with col_l:
        # Monthly avg passengers
        st.markdown("<div class='section-header'>📅 Monthly Avg Passengers (System-Wide)</div>", unsafe_allow_html=True)
        mon_sys = df.groupby('Month_Name')['Passengers'].mean().reset_index()
        mon_sys['Month_Name'] = pd.Categorical(mon_sys['Month_Name'],
                                                categories=MONTH_NAMES, ordered=True)
        mon_sys = mon_sys.sort_values('Month_Name')
        fig_mon_s = px.bar(mon_sys, x='Month_Name', y='Passengers',
                           color='Passengers', color_continuous_scale='Purples',
                           text='Passengers', height=320,
                           labels={'Month_Name':'Month','Passengers':'Avg Passengers/Trip'})
        fig_mon_s.update_traces(texttemplate='%{text:.1f}', textposition='outside',
                                textfont_color='white')
        fig_mon_s.update_layout(**PLOT_LAYOUT, xaxis=dict(**AXIS_STYLE),
                                yaxis=dict(**AXIS_STYLE), coloraxis_showscale=False)
        st.plotly_chart(fig_mon_s, use_container_width=True)

    with col_r:
        # Condition crowd score
        st.markdown("<div class='section-header'>⚡ Crowd Score by Service Condition</div>", unsafe_allow_html=True)
        cond_scores = sc_model.groupby('Remarks')['crowd_score'].mean().reset_index()
        color_cond2 = {
            'peak':'#e74c3c','off-peak':'#3498db','weekend':'#2ecc71',
            'festival':'#f39c12','maintenance':'#e67e22'
        }
        fig_cs_cond = px.bar(cond_scores, x='Remarks', y='crowd_score',
                             color='Remarks', color_discrete_map=color_cond2,
                             text='crowd_score', height=320,
                             labels={'Remarks':'Condition','crowd_score':'Avg Crowd Score'})
        fig_cs_cond.update_traces(texttemplate='%{text:.0f}', textposition='outside',
                                  textfont_color='white')
        fig_cs_cond.update_layout(**PLOT_LAYOUT, xaxis=dict(**AXIS_STYLE),
                                   yaxis=dict(**AXIS_STYLE, range=[0,115]),
                                   showlegend=False)
        st.plotly_chart(fig_cs_cond, use_container_width=True)

    # Full heatmap: all stations × DOW × average crowd score
    st.markdown("<div class='section-header'>🔥 System-Wide Crowd Heatmap (Station × Day)</div>", unsafe_allow_html=True)
    sys_heat = sc_model.groupby(['From_Station','DayOfWeek'])['crowd_score'].mean().reset_index()
    sys_heat['DayOfWeek'] = pd.Categorical(sys_heat['DayOfWeek'],
                                            categories=DOW_ORDER, ordered=True)
    sys_pivot = sys_heat.pivot(index='From_Station', columns='DayOfWeek',
                                values='crowd_score').fillna(0)
    # Sort by mean crowd score
    sys_pivot = sys_pivot.loc[sys_pivot.mean(axis=1).sort_values(ascending=False).index]

    fig_sys_heat = px.imshow(sys_pivot,
                             color_continuous_scale='RdYlGn_r',
                             aspect='auto', height=500,
                             labels={'color':'Crowd Score'},
                             zmin=0, zmax=100,
                             text_auto='.0f')
    fig_sys_heat.update_layout(**PLOT_LAYOUT,
                                xaxis=dict(color='white', tickfont=dict(color='white')),
                                yaxis=dict(color='white', tickfont=dict(color='white')),
                                coloraxis_colorbar=dict(tickfont=dict(color='white'),
                                                        titlefont=dict(color='white')))
    st.plotly_chart(fig_sys_heat, use_container_width=True)

    # YoY passenger trend
    st.markdown("<div class='section-header'>📈 Year-over-Year Passenger Load</div>", unsafe_allow_html=True)
    yoy = df.groupby(['Year','Month','Month_Name'])['Passengers'].mean().reset_index()
    yoy['Month_Name'] = pd.Categorical(yoy['Month_Name'],
                                        categories=MONTH_NAMES, ordered=True)
    yoy = yoy.sort_values(['Year','Month_Name'])
    fig_yoy = px.line(yoy, x='Month_Name', y='Passengers',
                      color='Year', markers=True,
                      color_discrete_sequence=['#7b2fbe','#ab47bc','#ce93d8'],
                      labels={'Month_Name':'Month','Passengers':'Avg Passengers/Trip','Year':'Year'},
                      height=340)
    fig_yoy.update_layout(**PLOT_LAYOUT, xaxis=dict(**AXIS_STYLE),
                           yaxis=dict(**AXIS_STYLE), legend=LEGEND_STYLE)
    st.plotly_chart(fig_yoy, use_container_width=True)


# ══════════════════════════════════════════════════════════════
# PAGE 5 — CROWD RISK RANKING
# ══════════════════════════════════════════════════════════════
elif page == "🏆 Crowd Risk Ranking":

    st.markdown("""
    <div class='big-header'>
    <h1>🏆 Crowd Risk Station Ranking</h1>
    <p>Stations ranked by average crowd score — identify where overcrowding risk is highest</p>
    </div>
    """, unsafe_allow_html=True)

    # Average crowd score per station across all conditions
    risk = sc_model.groupby('From_Station').agg(
        avg_crowd_score  =('crowd_score','mean'),
        max_crowd_score  =('crowd_score','max'),
        avg_passengers   =('avg_passengers','mean'),
        high_risk_periods=('crowd_score', lambda x: (x >= 65).sum()),
    ).reset_index().sort_values('avg_crowd_score', ascending=False)

    risk['Rank'] = range(1, len(risk)+1)
    risk['Risk_Level'] = risk['avg_crowd_score'].apply(
        lambda s: '🔴 High' if s>=65 else ('🟡 Medium' if s>=35 else '🟢 Low')
    )

    # KPIs
    k1,k2,k3 = st.columns(3)
    high_risk_stn = (risk['avg_crowd_score']>=65).sum()
    med_risk_stn  = ((risk['avg_crowd_score']>=35) & (risk['avg_crowd_score']<65)).sum()
    low_risk_stn  = (risk['avg_crowd_score']<35).sum()

    for col, val, lbl in [
        (k1, f"{high_risk_stn}", "🔴 High Risk Stations"),
        (k2, f"{med_risk_stn}",  "🟡 Medium Risk Stations"),
        (k3, f"{low_risk_stn}",  "🟢 Low Risk Stations"),
    ]:
        with col:
            st.markdown(f"""
            <div class='kpi-card'>
            <div class='kpi-value'>{val}</div>
            <div class='kpi-label'>{lbl}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Ranked bar chart
    st.markdown("<div class='section-header'>📊 Station Crowd Risk Ranking (All Conditions Average)</div>", unsafe_allow_html=True)

    def risk_color(score):
        if score >= 65: return '#e74c3c'
        elif score >= 35: return '#f39c12'
        return '#2ecc71'

    risk['bar_color'] = risk['avg_crowd_score'].apply(risk_color)

    fig_rank = go.Figure(go.Bar(
        y=risk['From_Station'],
        x=risk['avg_crowd_score'],
        orientation='h',
        marker_color=risk['bar_color'],
        text=[f"{rc} {s:.0f}" for rc, s in zip(risk['Risk_Level'], risk['avg_crowd_score'])],
        textposition='outside',
        textfont=dict(color='white', size=11),
    ))
    fig_rank.update_layout(
        **PLOT_LAYOUT, height=600,
        xaxis=dict(**AXIS_STYLE, title='Avg Crowd Score (0–100)', range=[0,115]),
        yaxis=dict(**AXIS_STYLE, title='', autorange='reversed'),
        title=dict(text='Station Crowd Risk Ranking (Higher = More Crowded)',
                   font=dict(color='white', size=14), x=0.5),
    )
    st.plotly_chart(fig_rank, use_container_width=True)

    # Ranked table
    st.markdown("<div class='section-header'>📋 Full Risk Table</div>", unsafe_allow_html=True)
    display_risk = risk[['Rank','From_Station','avg_crowd_score',
                          'max_crowd_score','avg_passengers',
                          'high_risk_periods','Risk_Level']].copy()
    display_risk.columns = ['Rank','Station','Avg Score','Peak Score',
                             'Avg Passengers','High-Risk Slots','Risk Level']

    st.dataframe(
        display_risk.style.format({
            'Avg Score': '{:.1f}', 'Peak Score': '{:.1f}',
            'Avg Passengers': '{:.1f}', 'High-Risk Slots': '{:.0f}'
        }),
        use_container_width=True, height=500
    )

    # Recommendations
    top5_high = risk.head(5)['From_Station'].tolist()
    top5_low  = risk.tail(5)['From_Station'].tolist()

    st.markdown(f"""
    <div style='background:#0a2e0a; border:2px solid #2ecc71;
    border-radius:12px; padding:1.5rem; margin-top:1rem;'>
    <div style='color:#2ecc71; font-size:1.1rem; font-weight:800;
    margin-bottom:0.8rem;'>💡 Crowd Management Recommendations</div>
    <div style='color:white; line-height:1.9; font-size:0.95rem;'>
    🔴 <b style='color:#e74c3c;'>Highest-Risk Stations</b> — {', '.join(top5_high)}<br>
    &nbsp;&nbsp;&nbsp;&nbsp;→ Deploy additional platform staff during peak & festival conditions.<br>
    &nbsp;&nbsp;&nbsp;&nbsp;→ Enable real-time crowd announcements and dynamic platform routing.<br><br>
    🟢 <b style='color:#2ecc71;'>Lowest-Risk Stations</b> — {', '.join(top5_low)}<br>
    &nbsp;&nbsp;&nbsp;&nbsp;→ These stations can absorb overflow passengers from high-risk hubs.<br>
    &nbsp;&nbsp;&nbsp;&nbsp;→ Promote interchange routing via these stations during peak hours.
    </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style='text-align:center; color:#ab47bc; font-size:0.82rem;
    padding:1rem; border-top:1px solid #7b2fbe; margin-top:1.5rem;'>
    MBA Applied Business Analytics · Delhi Metro Crowd Prediction System<br>
    Model trained on 150,000 trips (2022–2024) · Built with Streamlit + Plotly
    </div>
    """, unsafe_allow_html=True)
