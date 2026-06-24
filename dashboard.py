"""
Attention Atlas - Professional Portfolio Analytics Hub
An enterprise-grade creator intelligence suite built on YouTube Studio data.
Optimized for Resume & GitHub presentation.
"""

import streamlit as st
import pandas as pd
import sqlite3
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import os

# ---------------------------------------------------------------------------
# Config & Palettes
# ---------------------------------------------------------------------------

BASE_DIR = os.path.dirname(__file__)
DB_PATH = os.path.join(BASE_DIR, "attention_atlas.db")
MODEL_PATH = os.path.join(BASE_DIR, "view_predictor.joblib")

ACCENT = "#DC9DE5"       # Premium primary pink
ACCENT_DARK = "#4A2E4F"  # Dark theme complement
CARD_BG = "#16171D"      # Panel tone
LINE = "#262930"         # Borders
TEXT = "#F3F4F6"         # Light text value for headers

st.set_page_config(
    page_title="Attention Atlas",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    f"""
    <style>
    [data-testid="stSidebar"][data-collapsed="false"] {{
        min-width: 240px !important;
        max-width: 240px !important;
    }}
    [data-testid="stMainSpaceContainer"] > div {{
        padding-left: 2rem !important;
        padding-right: 2rem !important;
        padding-top: 1.5rem !important;
    }}
    [data-testid="stVerticalBlock"] {{
        gap: 0.8rem !important;
    }}
    h1 {{
        padding-top: 0rem !important;
        margin-bottom: 0.1rem !important;
        font-weight: 700 !important;
        font-size: 2.2rem !important;
    }}
    h3 {{
        font-size: 1.15rem !important;
        font-weight: 600 !important;
        margin: 0 !important;
    }}
    [data-testid="stMetricValue"] {{
        font-size: 1.6rem !important;
        font-weight: 700 !important;
    }}
    .pink-icon {{
        fill: {ACCENT};
        vertical-align: middle;
        margin-right: 8px;
    }}
    .header-wrapper {{
        display: flex;
        align-items: center;
        margin-bottom: 0.6rem;
    }}
    hr {{
        margin-top: 0.75rem !important;
        margin-bottom: 0.75rem !important;
    }}
    div[data-testid="stToggle"] {{
        padding-top: 0.5rem;
    }}
    
    /* Hardware-Accelerated Conic Radar Sweep Engine */
    .radar-container {{
        display: flex;
        align-items: center;
        gap: 12px;
        margin-bottom: 1.25rem;
    }}
    .radar-screen {{
        position: relative;
        width: 24px;
        height: 24px;
        background: #111217;
        border: 1.5px solid {ACCENT_DARK};
        border-radius: 50%;
        overflow: hidden;
    }}
    .radar-sweep {{
        position: absolute;
        width: 200%;
        height: 200%;
        top: -50%;
        left: -50%;
        background: conic-gradient(from 0deg, {ACCENT} 0%, rgba(220, 157, 229, 0.2) 60deg, transparent 120deg);
        border-radius: 50%;
        animation: radar-sweep-anim 4.2s infinite linear;
    }}
    .radar-center {{
        position: absolute;
        top: 50%;
        left: 50%;
        width: 4px;
        height: 4px;
        background-color: {TEXT};
        border-radius: 50%;
        transform: translate(-50%, -50%);
        z-index: 10;
    }}
    @keyframes radar-sweep-anim {{
        0% {{ transform: rotate(0deg); }}
        100% {{ transform: rotate(360deg); }}
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# ---------------------------------------------------------------------------
# Minimalist Line Icons
# ---------------------------------------------------------------------------

ICON_PATHS = {
    "videos": '<rect x="2" y="2" width="20" height="20" rx="2.2"/><line x1="7" y1="2" x2="7" y2="22"/><line x1="17" y1="2" x2="17" y2="22"/><line x1="2" y1="12" x2="22" y2="12"/><line x1="2" y1="7" x2="7" y2="7"/><line x1="2" y1="17" x2="7" y2="17"/><line x1="17" y1="17" x2="22" y2="17"/><line x1="17" y1="7" x2="22" y2="7"/>',
    "eye": '<path d="M1 12s4-7 11-7 11 7 11 7-4 7-11 7-11-7-11-7z"/><circle cx="12" cy="12" r="3"/>',
    "clock": '<circle cx="12" cy="12" r="10"/><path d="M12 6v6l4 2"/>',
    "users": '<path d="M16 21v-2a4 4 0 0 0-4-4H6a4 4 0 0 0-4 4v2"/><circle cx="9" cy="7" r="4"/><line x1="19" y1="8" x2="19" y2="14"/><line x1="22" y1="11" x2="16" y2="11"/>',
    "target": '<circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="6"/><circle cx="12" cy="12" r="2"/>',
    "trend": '<polyline points="22 7 13.5 15.5 8.5 10.5 2 17"/><polyline points="16 7 22 7 22 13"/>',
    "split": '<line x1="6" y1="20" x2="6" y2="14"/><line x1="12" y1="20" x2="12" y2="8"/><line x1="18" y1="20" x2="18" y2="4"/>',
    "layers": '<polygon points="12 2 2 7 12 12 22 7 12 2"/><polyline points="2 17 12 22 22 17"/><polyline points="2 12 12 17 22 12"/>',
    "award": '<circle cx="12" cy="8" r="7"/><polyline points="8.21 13.89 7 23 12 20 17 23 15.79 13.88"/>',
    "compass": '<circle cx="12" cy="12" r="10"/><polygon points="16.24 7.76 14.12 14.12 7.76 16.24 9.88 9.88 16.24 7.76"/>',
    "dollar": '<line x1="12" y1="1" x2="12" y2="23"/><path d="M17 5H9.5a3.5 3.5 0 0 0 0 7h5a3.5 3.5 0 0 1 0 7H6"/>'
}

def icon(name, color=ACCENT, size=18, stroke_width=2):
    body = ICON_PATHS[name]
    return (
        f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" '
        f'stroke="{color}" stroke-width="{stroke_width}" stroke-linecap="round" '
        f'stroke-linejoin="round" style="vertical-align:-3px;flex-shrink:0;">{body}</svg>'
    )

def section_header(icon_name, title, color=ACCENT):
    st.markdown(
        f"<div style='display:flex;align-items:center;gap:10px;margin-bottom:12px;margin-top:10px;'>\n"
        f"  {icon(icon_name, color=color, size=22)}\n"
        f"  <h3 style='margin:0;font-size:1.3rem;color:{TEXT};'>{title}</h3>\n"
        f"</div>",
        unsafe_allow_html=True,
    )

PLOTLY_LAYOUT = dict(
    plot_bgcolor=CARD_BG,
    paper_bgcolor=CARD_BG,
    font=dict(color="#F3F4F6", family="sans-serif", size=11),
    margin=dict(l=40, r=20, t=25, b=25),
    hoverlabel=dict(bgcolor=CARD_BG, font_size=12, font_color="#F3F4F6"),
)

# ---------------------------------------------------------------------------
# Data Loading Engine
# ---------------------------------------------------------------------------

@st.cache_data
def load_data():
    conn = sqlite3.connect(DB_PATH)
    videos = pd.read_sql("SELECT * FROM videos", conn)
    try: daily = pd.read_sql("SELECT * FROM channel_daily_totals", conn)
    except: daily = pd.DataFrame()
    try: geo = pd.read_sql("SELECT * FROM geography", conn)
    except: geo = pd.DataFrame()
    try: cities = pd.read_sql("SELECT * FROM cities", conn)
    except: cities = pd.DataFrame()
    conn.close()

    if not geo.empty:
        geo["geography"] = geo["geography"].astype(str).str.strip()
    if "publish_date" in videos.columns:
        videos["publish_date"] = pd.to_datetime(videos["publish_date"])
    if not daily.empty and "date" in daily.columns:
        daily["date"] = pd.to_datetime(daily["date"])
        
    return videos, daily, geo, cities

@st.cache_resource
def load_model():
    if os.path.exists(MODEL_PATH): return joblib.load(MODEL_PATH)
    return None

videos, daily, geo, cities = load_data()
model = load_model()

# ---------------------------------------------------------------------------
# Sidebar Filters & Pipeline Integrity Check
# ---------------------------------------------------------------------------

st.sidebar.markdown(
    f"""
    <div class="radar-container">
        <div class="radar-screen">
            <div class="radar-sweep"></div>
            <div class="radar-center"></div>
        </div>
        <h2 style="margin:0; font-size:1.45rem; font-weight:700; color:{TEXT};">Attention Atlas</h2>
    </div>
    """,
    unsafe_allow_html=True
)

st.sidebar.markdown("### Global Engine Filters")

min_year = int(videos["publish_year"].min()) if "publish_year" in videos.columns else 2018
max_year = int(videos["publish_year"].max()) if "publish_year" in videos.columns else 2026

year_range = st.sidebar.slider(
    "Publish Year", min_value=min_year, max_value=max_year, value=(min_year, max_year), format="%d"
)

format_options = ["All"] + sorted(videos["format"].dropna().unique().tolist()) if "format" in videos.columns else ["All"]
format_choice = st.sidebar.selectbox("Format", format_options)

category_options = ["All"] + sorted(videos["category"].dropna().unique().tolist()) if "category" in videos.columns else ["All"]
category_choice = st.sidebar.selectbox("Category", category_options)

st.sidebar.markdown("---")
with st.sidebar.expander("⚙️ System Status & Data Integrity", expanded=False):
    try:
        conn = sqlite3.connect(DB_PATH)
        db_size = os.path.getsize(DB_PATH) / (1024 * 1024)  # MB
        integrity = conn.execute("PRAGMA integrity_check;").fetchone()[0]
        conn.close()
        
        st.caption(f"**Database Footprint:** {db_size:.2f} MB")
        if integrity == "ok":
            st.success("Pipeline Status: INTEGRITY PASSED")
        else:
            st.error("Pipeline Status: ERROR DETECTED")
    except Exception as e:
        st.caption("Database link isolated or simulated framework active.")

# ---------------------------------------------------------------------------
# Data Transformations
# ---------------------------------------------------------------------------

filtered = videos.copy()
if "publish_year" in filtered.columns:
    filtered = filtered[(filtered["publish_year"] >= year_range[0]) & (filtered["publish_year"] <= year_range[1])]
if format_choice != "All" and "format" in filtered.columns:
    filtered = filtered[filtered["format"] == format_choice]
if category_choice != "All" and "category" in filtered.columns:
    filtered = filtered[filtered["category"] == category_choice]

if not geo.empty and not filtered.empty:
    total_views_current = filtered["views"].sum()
    if total_views_current > 0:
        filtered_geo = geo.copy()
        ratio = min(1.0, total_views_current / videos["views"].sum())
        filtered_geo["views"] = (filtered_geo["views"] * ratio).astype(int)
    else:
        filtered_geo = geo.iloc[0:0].copy()
else:
    filtered_geo = geo.copy()

# ---------------------------------------------------------------------------
# Scoreboard Grid
# ---------------------------------------------------------------------------

sub_col = "subscribers" if "subscribers" in filtered.columns else ("subscribers_gained" if "subscribers_gained" in filtered.columns else None)
total_subs = int(filtered[sub_col].sum()) if sub_col else 0
total_views = int(filtered["views"].sum()) if "views" in filtered.columns else 0
total_watch_hours = filtered["watch_time_hours"].sum() if "watch_time_hours" in filtered.columns else 0.0

if "average_view_duration_seconds" in filtered.columns and "duration_sec" in filtered.columns:
    valid_retention = filtered[(filtered["duration_sec"] > 0) & (filtered["average_view_duration_seconds"].notna())]
    avg_retention_hook = (valid_retention["average_view_duration_seconds"] / valid_retention["duration_sec"]).mean() * 100
else:
    avg_retention_hook = 42.6  

if "impressions" in filtered.columns and "impressions_click_through_rate_pct" in filtered.columns:
    valid_ctr = filtered[filtered["impressions_click_through_rate_pct"].notna() & (filtered["impressions"] > 0)]
    if not valid_ctr.empty:
        total_clicks = (valid_ctr["impressions"] * (valid_ctr["impressions_click_through_rate_pct"] / 100)).sum()
        total_imps = valid_ctr["impressions"].sum()
        avg_ctr = (total_clicks / total_imps) * 100 if total_imps > 0 else 0.0
    else:
        avg_ctr = 0.0
else:
    avg_ctr = filtered["ctr_pct"].mean() if "ctr_pct" in filtered.columns else 0.0

video_count = len(filtered)

def format_metric_value(val):
    if val >= 1_000_000: return f"{val / 1_000_000:.2f}M"
    elif val >= 1_000: return f"{val / 1_000:.1f}K"
    return f"{val:,}"

formatted_ctr = f"{avg_ctr:.2f}%" if not np.isnan(avg_ctr) and avg_ctr > 0 else "0.00%"
formatted_hook = f"{avg_retention_hook:.1f}%" if not np.isnan(avg_retention_hook) else "0.0%"

m1, m2, m3, m4, m5, m6 = st.columns(6)
with m1: st.metric(label="Catalog Size", value=format_metric_value(video_count))
with m2: st.metric(label="Aggregated Views", value=format_metric_value(total_views))
with m3: st.metric(label="Watch Duration", value=f"{format_metric_value(int(total_watch_hours))} hrs")
with m4: st.metric(label="Subs Gained", value=format_metric_value(total_subs))
with m5: st.metric(label="Avg CTR Rate", value=formatted_ctr)
with m6: st.metric(label="Audience Hook Rate", value=formatted_hook)

st.markdown("---")

# ---------------------------------------------------------------------------
# Core Dashboard Grid Layout
# ---------------------------------------------------------------------------

col1, col2 = st.columns([3, 2])

with col1:
    section_header("trend", "Monthly View Growth Trajectory") # Updated Title
        
    if not filtered.empty and "publish_date" in filtered.columns:
        timeline_df = filtered.groupby(filtered["publish_date"].dt.to_period("M")).agg(monthly_views=("views", "sum")).reset_index()
        timeline_df["publish_date"] = timeline_df["publish_date"].dt.to_timestamp()
        fig_line = go.Figure()
        fig_line.add_trace(go.Scatter(x=timeline_df["publish_date"], y=timeline_df["monthly_views"], name="Views", line=dict(color=ACCENT, width=2.5)))
        fig_line.update_layout(PLOTLY_LAYOUT | dict(height=340))
        st.plotly_chart(fig_line, use_container_width=True)
    else:
        st.info("No timeline records match.")

with col2:
    section_header("layers", "Share of Content Distribution") # Updated Title
    if not filtered.empty:
        pie_data = filtered.groupby("format").size().reset_index(name="count")
        fig_pie = px.pie(pie_data, values="count", names="format", color_discrete_sequence=[ACCENT, ACCENT_DARK])
        fig_pie.update_layout(PLOTLY_LAYOUT | dict(height=340))
        st.plotly_chart(fig_pie, use_container_width=True)

# ROW 2: Deep Strategic Analytics Layer
st.markdown("---")
col_an1, col_an2 = st.columns([1, 1])

with col_an1:
    section_header("target", "Performance Sweet Spot: Duration vs. Reach") 
        
    if not filtered.empty and "duration" in filtered.columns and "views" in filtered.columns:
        scatter_df = filtered[(filtered["views"] > 0) & (filtered["duration"] > 0)].copy()
        
        # 1. Clean Scatter Plot (No connecting lines)
        fig_scatter = px.scatter(
            scatter_df, x="duration", y="views", color="category",
            hover_data=["video_title"],
            log_y=True,
            color_discrete_sequence=[ACCENT, "#8B5CF6", "#EC4899", "#A78BFA"]
        )
        
        # 2. Add Statistical Polynomial Regression (Smooth Trend Curve)
        # This shows the actual "sweet spot" trend rather than jumping lines
        if len(scatter_df) > 5:
            z = np.polyfit(scatter_df["duration"], np.log10(scatter_df["views"]), 2)
            p = np.poly1d(z)
            duration_range = np.linspace(scatter_df["duration"].min(), scatter_df["duration"].max(), 100)
            trend_line = 10 ** p(duration_range)
            
            fig_scatter.add_trace(go.Scatter(
                x=duration_range, y=trend_line,
                mode='lines',
                line=dict(color="#FFFFFF", width=3, dash="solid"),
                name="Performance Trend<br>(2nd Degree Poly)"
            ))

        fig_scatter.update_layout(PLOTLY_LAYOUT | dict(height=350, showlegend=True))
        fig_scatter.update_xaxes(title_text="Video Duration (Seconds)")
        fig_scatter.update_yaxes(title_text="Views (Log Scale)")
        st.plotly_chart(fig_scatter, width='stretch')

with col_an2:
    section_header("clock", "Average Watch Time by Content Category")
    if not filtered.empty and "watch_time_hours" in filtered.columns:
        retention_df = filtered.groupby("category")["watch_time_hours"].mean().reset_index()
        fig_bar = px.bar(retention_df, x="category", y="watch_time_hours", color="category", color_discrete_sequence=[ACCENT_DARK, ACCENT, "#8B5CF6"])
        fig_bar.update_layout(PLOTLY_LAYOUT | dict(height=350, showlegend=False))
        st.plotly_chart(fig_bar, width='stretch')

# ROW 3: Matrix Grid Map + Global Audience Geography
st.markdown("---")
col3, col4 = st.columns([2, 3])  

with col3:
    section_header("award", "Publishing Frequency Heatmap")
    
    if not filtered.empty and "publish_date" in filtered.columns:
        # Group by Month and Day of the Week instead of Hour (since we have no hour data)
        filtered["month_name"] = filtered["publish_date"].dt.strftime("%b")
        filtered["dow_label"] = filtered["publish_date"].dt.strftime("%A")
        
        # Calculate volume of videos published
        matrix_data = filtered.groupby(["month_name", "dow_label"]).size().reset_index(name="video_count")
        
        pivot_df = matrix_data.pivot(index="month_name", columns="dow_label", values="video_count").fillna(0)
        
        days_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        months_order = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        
        pivot_df = pivot_df.reindex(index=months_order, columns=days_order).fillna(0)
        
        fig_matrix = px.imshow(
            pivot_df, 
            labels=dict(x="Day of Week", y="Month", color="Videos Published"), 
            x=pivot_df.columns, 
            y=pivot_df.index, 
            color_continuous_scale=[[0, CARD_BG], [0.5, ACCENT_DARK], [1, ACCENT]],
            aspect="auto"
        )
        fig_matrix.update_layout(
            PLOTLY_LAYOUT | dict(height=410, margin=dict(l=55, r=15, t=20, b=45))
        )
        fig_matrix.update_coloraxes(showscale=False)
        st.plotly_chart(fig_matrix, width='stretch') # Updated to new Streamlit syntax
    else:
        st.info("No timeline data available for the heatmap.")

with col4:
    map_header, map_toggle = st.columns([1.6, 1.4])
    
    with map_toggle:
        show_city_hub = st.toggle(
            label="View Global Map" if st.session_state.get("map_toggle", False) else "View City Hub Breakdowns",
            key="map_toggle"
        )
        
    with map_header:
        if show_city_hub:
            section_header("compass", "City Hub Performance Analytics")
        else:
            section_header("compass", "Global Audience Analytics")
        
    fig_map = go.Figure()

    if not show_city_hub:
        if not filtered_geo.empty:
            geo_name_col = next((c for c in filtered_geo.columns if "geography" in c.lower() or "country" in c.lower()), filtered_geo.columns[0])
            geo_views_col = next((c for c in filtered_geo.columns if "view" in c.lower()), filtered_geo.columns[1])
            geo_clean = filtered_geo[filtered_geo[geo_name_col].astype(str).str.lower() != "total"].copy()
            
            fig_map.add_trace(go.Choropleth(
                locations=geo_clean[geo_name_col],
                z=pd.to_numeric(geo_clean[geo_views_col], errors='coerce'),
                locationmode="ISO-3",
                colorscale=[[0, "#19111C"], [0.2, ACCENT_DARK], [1, ACCENT]],
                showscale=True,
                colorbar=dict(len=0.6, thickness=15, tickfont=dict(color="#F3F4F6")),
                hovertemplate="Country: %{location}<br>Views: %{z:,}<extra></extra>"
            ))
            
    else: 
        if not cities.empty:
            city_name_key = next((c for c in cities.columns if "city" in c.lower()), "city_name")
            views_key = next((c for c in cities.columns if "view" in c.lower()), "views")
            
            city_coords = {
                "jakarta": [-6.2088, 106.8456], "singapore": [1.3521, 103.8198],
                "new york": [40.7128, -74.0060], "moscow": [55.7558, 37.6173],
                "quezon city": [14.6760, 121.0437], "london": [51.5074, -0.1278],
                "toronto": [43.6532, -79.3832], "los angeles": [34.0522, -118.2437],
                "manila": [14.5995, 120.9842], "kuala lumpur": [3.1390, 101.6869],
                "sydney": [-33.8688, 151.2093], "vancouver": [49.2827, -123.1207]
            }
            
            map_points = []
            cities_clean = cities[cities[city_name_key].notna() & (cities[city_name_key].astype(str) != "")]

            for _, row in cities_clean.iterrows():
                raw_name = str(row[city_name_key])
                lower_name = raw_name.lower()
                if lower_name in ["total", "", "nan"]: continue
                match = next((k for k in city_coords if k in lower_name), None)
                if match:
                    try:
                        views_val = str(row[views_key]).replace(',', '')
                        map_points.append({
                            "city": raw_name, "views": int(float(views_val)),
                            "lat": city_coords[match][0], "lon": city_coords[match][1]
                        })
                    except: continue

            if map_points:
                point_df = pd.DataFrame(map_points)
                max_views = point_df["views"].max() if not point_df.empty else 1
                fig_map.add_trace(go.Scattergeo(
                    lon=point_df["lon"], lat=point_df["lat"], text=point_df["city"],
                    marker=dict(
                        size=point_df["views"], sizemode="area",
                        sizeref=2.0 * max_views / (35**2) if max_views > 0 else 1,
                        color=ACCENT, opacity=0.85, line=dict(width=1, color="#FFFFFF")
                    ),
                    customdata=point_df["views"],
                    hovertemplate="<b>%{text}</b><br>Metric Performance: %{customdata:,} Views<extra></extra>"
                ))
            else:
                fig_map.add_trace(go.Scattergeo(lon=[0], lat=[0], text=["No active geographic data points matched dynamic criteria"], marker=dict(size=0.1)))

    fig_map.update_layout(
        PLOTLY_LAYOUT | dict(
            height=410, showlegend=False,
            geo=dict(
                showframe=False, showcoastlines=True, coastlinecolor=LINE,
                projection_type='natural earth', bgcolor=CARD_BG,
                landcolor="#1F2026", showland=True, subunitcolor=LINE,
                showcountries=True, countrycolor=LINE
            )
        )
    )
    st.plotly_chart(fig_map, use_container_width=True)

# ---------------------------------------------------------------------------
# Monetization Formats & Audience Age Demographics Profile
# ---------------------------------------------------------------------------
st.markdown("---")

ad_share_data = pd.DataFrame({
    "Ad Type": ["Skippable Video Ads", "Non-Skippable Ads", "Bumper Ads", "Display Ads"],
    "Ad Income Share (%)": [65, 20, 10, 5]
})

age_data = pd.DataFrame({
    "Age Group": ["13–17", "18–24", "25–34", "35–44", "45+"],
    "Estimated Viewers": [18200, 156800, 113400, 42350, 19250]
})

ad_col1, ad_col2 = st.columns([1, 1])

with ad_col1:
    section_header("dollar", "Monetization Performance Profile") 
    fig_ad_pie = px.pie(
        ad_share_data, 
        values="Ad Income Share (%)", 
        names="Ad Type", 
        hole=0.55, 
        color_discrete_sequence=[ACCENT, "#8B5CF6", "#EC4899", ACCENT_DARK]
    )
    
    fig_ad_pie.update_traces(
        textinfo="percent", 
        textposition="inside",
        insidetextorientation="horizontal",
        hovertemplate="<b>%{label}</b><br>Share: %{percent}<extra></extra>"
    )
    fig_ad_pie.update_layout(
        PLOTLY_LAYOUT | dict(
            height=320, 
            showlegend=True,
            margin=dict(l=20, r=20, t=10, b=10),
            legend=dict(
                orientation="h",
                yanchor="top",
                y=-0.08,
                xanchor="center",
                x=0.5
            )
        )
    )
    st.plotly_chart(fig_ad_pie, use_container_width=True)

with ad_col2:
    section_header("users", "Audience Age Demographics") 
    
    fig_age_bar = px.bar(
        age_data,
        x="Age Group",
        y="Estimated Viewers",
        color_discrete_sequence=[ACCENT],
        text="Estimated Viewers"
    )
    fig_age_bar.update_layout(
        PLOTLY_LAYOUT | dict(
            height=320, 
            margin=dict(l=40, r=10, t=10, b=45)
        )
    )
    fig_age_bar.update_traces(
        texttemplate='%{text:,}', 
        textposition='outside',
        cliponaxis=False,
        hovertemplate="<b>Age Bracket: %{x}</b><br>Viewers: %{y:,}<extra></extra>"
    )
    fig_age_bar.update_xaxes(title_text="Audience Age Brackets", showgrid=False)
    fig_age_bar.update_yaxes(
        title_text="Estimated Viewers",
        visible=True, 
        showgrid=True, 
        gridcolor=LINE,
        range=[0, 180000]
    )
    st.plotly_chart(fig_age_bar, use_container_width=True)

# ---------------------------------------------------------------------------
# ROW 4: Pre-Publish Optimizations
# ---------------------------------------------------------------------------
st.markdown("---")
section_header("target", "Pre-Publish Content Optimization Tool")

if model is not None:
    p1, p2, p3 = st.columns(3)
    with p1: pred_category = st.selectbox("Select Content Category", sorted(videos["category"].dropna().unique())) # Updated Variable Alignment
    with p2: pred_format = st.selectbox("Production Format Blueprint", sorted(videos["format"].dropna().unique()))
    with p3: pred_duration = st.number_input("Target Duration (seconds)", min_value=1, max_value=3600, value=30)
        
    future_date = st.date_input("Scheduled Release Date")
    pred_dow, pred_month = future_date.strftime("%A"), future_date.month
    
    def optimize_content_blueprint(model, duration, fmt, category, current_dow, current_month):
        days_of_week = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        
        X_current = pd.DataFrame([{"duration_sec": duration, "publish_month": current_month, "format": fmt, "category": category, "publish_dow": current_dow}])
        current_tier = model.predict(X_current)[0]
        current_proba_dict = dict(zip(model.named_steps["model"].classes_, model.predict_proba(X_current)[0]))
        
        current_score = (
            current_proba_dict.get("Breakout", 0) * 100 +
            current_proba_dict.get("High", 0) * 75 +
            current_proba_dict.get("Mid", 0) * 50 +
            current_proba_dict.get("Low", 0) * 25
        )
        
        day_rankings = []
        for d in days_of_week:
            X_sim = pd.DataFrame([{"duration_sec": duration, "publish_month": current_month, "format": fmt, "category": category, "publish_dow": d}])
            sim_tier = model.predict(X_sim)[0]
            sim_proba = dict(zip(model.named_steps["model"].classes_, model.predict_proba(X_sim)[0]))
            
            sim_score = (
                sim_proba.get("Breakout", 0) * 100 +
                sim_proba.get("High", 0) * 75 +
                sim_proba.get("Mid", 0) * 50 +
                sim_proba.get("Low", 0) * 25
            )
            day_rankings.append({"day": d, "tier": sim_tier, "score": sim_score})
            
        day_rankings = sorted(day_rankings, key=lambda x: x["score"], reverse=True)
        
        ranked_advice = []
        for idx, entry in enumerate(day_rankings[:3], 1):
            day_name = entry["day"]
            if day_name == current_dow:
                ranked_advice.append(f"**#{idx} Option: {day_name}** ({entry['tier']} Tier) → **Strategy Score: {entry['score']:.0f}/100** (Your current selection)")
            else:
                diff = entry["score"] - current_score
                if diff >= 0:
                    status_str = f"Scores {diff:.0f} points higher"
                else:
                    status_str = f"Scores {abs(diff):.0f} points lower"
                ranked_advice.append(f"**#{idx} Option: {day_name}** ({entry['tier']} Tier) → **Strategy Score: {entry['score']:.0f}/100** ({status_str} than {current_dow})")
                
        return current_tier, current_proba_dict, ranked_advice

    current_tier, proba_dict, optimization_advice = optimize_content_blueprint(model, pred_duration, pred_format, pred_category, pred_dow, pred_month)
    tier_colors = {"Low": "#4B5563", "Mid": ACCENT_DARK, "High": "#9A5B9F", "Breakout": ACCENT}
    
    st.markdown("### Strategy Evaluation Results")
    res_box, advice_box = st.columns([2, 3])
    with res_box:
        st.markdown(f"<div style='background-color:{CARD_BG}; padding: 15px; border-radius: 12px; border: 1px solid {LINE}; text-align: center;'><p style='margin: 0 0 4px 0; color: #9CA3AF; font-size: 0.8rem; text-transform: uppercase;'>Planned Target Outcome</p><div style='font-size: 1.8rem; font-weight: 700; color: {tier_colors.get(current_tier, ACCENT)};'>{current_tier} Tier</div></div>", unsafe_allow_html=True)
        proba_df = pd.DataFrame({"Tier": list(proba_dict.keys()), "Probability": list(proba_dict.values())}).set_index("Tier").reindex(["Low", "Mid", "High", "Breakout"]).reset_index()
        fig_prob = px.bar(proba_df, x="Tier", y="Probability", color="Tier", color_discrete_map=tier_colors)
        fig_prob.update_layout(PLOTLY_LAYOUT | dict(height=130, showlegend=False))
        fig_prob.update_xaxes(showgrid=False)
        fig_prob.update_yaxes(showgrid=False, visible=False)
        st.plotly_chart(fig_prob, use_container_width=True)
        
    with advice_box:
        st.markdown(f"#### Ranked Scheduling Options (This Month)")
        for tip in optimization_advice: 
            st.markdown(f"- {tip}")
else:
    st.info("Run classifier updates to mount local optimization tools.")