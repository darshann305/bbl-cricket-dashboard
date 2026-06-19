import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pickle
import os
import time
from streamlit_option_menu import option_menu

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="BBL Cricket Dashboard",
    page_icon="🏏",
    layout="wide"
)

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE      = r"C:\Users\HP\OneDrive\Desktop\bbl-cricket-dashboard"
PROCESSED = os.path.join(BASE, "data", "processed")
MODEL_PATH = os.path.join(BASE, "app", "model.pkl")

# ── Load data ─────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df           = pd.read_parquet(os.path.join(PROCESSED, "bbl_final.parquet"))
    batter_stats = pd.read_parquet(os.path.join(PROCESSED, "batter_stats.parquet"))
    bowler_stats = pd.read_parquet(os.path.join(PROCESSED, "bowler_stats.parquet"))
    season_stats = pd.read_parquet(os.path.join(PROCESSED, "season_stats.parquet"))
    phase_bowler = pd.read_parquet(os.path.join(PROCESSED, "phase_bowler_stats.parquet"))
    model_df     = pd.read_parquet(os.path.join(PROCESSED, "model_df.parquet"))
    return df, batter_stats, bowler_stats, season_stats, phase_bowler, model_df

@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)

df, batter_stats, bowler_stats, season_stats, phase_bowler, model_df = load_data()
model = load_model()

FEATURES = [
    "cum_total_runs", "cum_wickets", "balls_remaining",
    "current_rr", "runs_needed", "required_rr", "rr_pressure", "target"
]

# ── Session state for animation ───────────────────────────────────────────────
if "win_prob_playing"  not in st.session_state:
    st.session_state.win_prob_playing  = False
if "win_prob_ball"     not in st.session_state:
    st.session_state.win_prob_ball     = 1
if "selected_match_id" not in st.session_state:
    st.session_state.selected_match_id = None

# ── Helper: normalize to 0-100 ────────────────────────────────────────────────
def normalize(val, min_val, max_val):
    if max_val == min_val:
        return 50
    return (val - min_val) / (max_val - min_val) * 100

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image(
        "https://upload.wikimedia.org/wikipedia/en/thumb/2/20/"
        "Big_Bash_League_logo.svg/200px-Big_Bash_League_logo.svg.png",
        width=160
    )
    st.markdown("## BBL Analytics Dashboard")
    st.markdown("*15 seasons • 662 matches • 153,250 deliveries*")
    st.markdown("---")

    page = option_menu(
        menu_title=None,
        options=["Overview", "Batter Analysis", "Bowler Analysis",
                 "Win Probability", "Season Trends"],
        icons=["house", "person", "bullseye", "graph-up", "calendar"],
        default_index=0
    )

    st.markdown("---")
    st.markdown("Built by **Darshan Nagaraja**")
    st.markdown("Data: [Cricsheet.org](https://cricsheet.org)")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if page == "Overview":
    st.title("🏏 BBL Cricket Analytics Dashboard")
    st.markdown("### The Big Bash League — 15 seasons of data, one dashboard")
    st.markdown("---")

    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("🏟️ Total Matches",  f"{df['match_id'].nunique():,}")
    col2.metric("🏃 Total Runs",     f"{df['runs_off_bat'].sum():,}")
    col3.metric("💥 Total Sixes",    f"{df['is_six'].sum():,}")
    col4.metric("🎯 Total Wickets",  f"{df['is_wicket'].sum():,}")

    st.markdown("---")

    # ── Animated bar chart race ───────────────────────────────────────────────
    st.subheader("🏆 Team Runs Race — Cumulative Across Seasons")
    st.markdown("*Press ▶ Play to watch teams accumulate runs across BBL history*")

    seasons = sorted(df["season"].unique())
    teams   = sorted(df["batting_team"].unique())

    cum_data   = []
    cumulative = {team: 0 for team in teams}

    for season in seasons:
        season_df = df[df["season"] == season]
        for team in teams:
            cumulative[team] += season_df[season_df["batting_team"] == team]["runs_off_bat"].sum()
        for team in teams:
            cum_data.append({"season": season, "team": team,
                             "cumulative_runs": cumulative[team]})

    cum_df = pd.DataFrame(cum_data)

    fig = px.bar(
        cum_df,
        x="cumulative_runs", y="team",
        animation_frame="season",
        orientation="h",
        color="team",
        title="Cumulative Runs by Team — Season by Season",
        labels={"cumulative_runs": "Cumulative Runs", "team": "Team"},
        range_x=[0, cum_df["cumulative_runs"].max() * 1.1]
    )
    fig.update_layout(
        height=420, showlegend=False,
        updatemenus=[{
            "type": "buttons", "showactive": False,
            "y": 1.15, "x": 0.5, "xanchor": "center",
            "buttons": [
                {"label": "▶ Play", "method": "animate",
                 "args": [None, {"frame": {"duration": 800, "redraw": True},
                                 "fromcurrent": True}]},
                {"label": "⏸ Pause", "method": "animate",
                 "args": [[None], {"frame": {"duration": 0, "redraw": False},
                                   "mode": "immediate"}]}
            ]
        }]
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # ── Scoring heatmap ───────────────────────────────────────────────────────
    st.subheader("🔥 Run Rate Heatmap — Team × Over")
    st.markdown("*Which team scores the most in which overs?*")

    team_over = df.groupby(["batting_team", "over"])["total_runs"].mean().reset_index()
    team_over["run_rate"] = (team_over["total_runs"] * 6).round(2)
    heatmap_pivot = team_over.pivot(index="batting_team", columns="over", values="run_rate")

    fig = px.imshow(
        heatmap_pivot,
        color_continuous_scale="RdYlGn",
        title="Average Run Rate by Team and Over",
        labels={"x": "Over", "y": "Team", "color": "Run Rate"},
        aspect="auto"
    )
    fig.update_layout(height=380)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # ── Toss analysis ─────────────────────────────────────────────────────────
    st.subheader("🪙 Toss Analysis")

    toss_df = df.drop_duplicates("match_id")[
        ["match_id", "winner", "toss_winner", "toss_decision"]
    ].dropna()
    toss_df["toss_helped"] = (toss_df["winner"] == toss_df["toss_winner"])

    col1, col2 = st.columns(2)

    with col1:
        fig = go.Figure(go.Pie(
            labels=["Toss Winner Won", "Toss Winner Lost"],
            values=[toss_df["toss_helped"].sum(),
                    (~toss_df["toss_helped"]).sum()],
            hole=0.45,
            marker_colors=["#2ecc71", "#e74c3c"],
            textinfo="label+percent"
        ))
        fig.update_layout(title="Does Winning the Toss Help?", height=350)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        dw = toss_df.groupby("toss_decision")["toss_helped"].mean().reset_index()
        dw.columns = ["decision", "win_rate"]
        dw["win_rate"] = (dw["win_rate"] * 100).round(1)

        fig = go.Figure(go.Bar(
            x=dw["decision"], y=dw["win_rate"],
            marker_color=["#3498db", "#e67e22"],
            text=dw["win_rate"].astype(str) + "%",
            textposition="outside"
        ))
        fig.update_layout(
            title="Win Rate by Toss Decision (Bat vs Field)",
            yaxis_title="Win Rate %",
            yaxis_range=[0, 70],
            height=350
        )
        st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — BATTER ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Batter Analysis":
    st.title("🏏 Batter Analysis")
    st.markdown("*Minimum 20 innings to qualify*")
    st.markdown("---")

    # ── Top 10 horizontal bar ─────────────────────────────────────────────────
    st.subheader("Top 10 Run Scorers in BBL History")
    top10 = batter_stats.sort_values("runs", ascending=False).head(10)

    fig = px.bar(
        top10.sort_values("runs"),
        x="runs", y="striker",
        orientation="h",
        color="strike_rate",
        color_continuous_scale="RdYlGn",
        hover_data=["average", "strike_rate", "sixes", "fours"],
        text="runs",
        labels={"striker": "Batter", "runs": "Total Runs"}
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(height=420, yaxis_title="")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # ── Head to head ──────────────────────────────────────────────────────────
    st.subheader("⚔️ Head-to-Head Batter Comparison")
    st.markdown("*Select two players to compare on a radar chart*")

    col1, col2 = st.columns(2)
    all_batters = sorted(batter_stats["striker"].unique())
    with col1:
        p1_name = st.selectbox("Batter 1", all_batters, index=0)
    with col2:
        p2_name = st.selectbox("Batter 2", all_batters, index=1)

    if p1_name != p2_name:
        p1 = batter_stats[batter_stats["striker"] == p1_name].iloc[0]
        p2 = batter_stats[batter_stats["striker"] == p2_name].iloc[0]

        cats   = ["Average", "Strike Rate", "Boundary %", "Sixes", "Fours"]
        fields = ["average", "strike_rate", "boundary_pct", "sixes", "fours"]

        p1_vals = [normalize(p1[f], batter_stats[f].min(), batter_stats[f].max()) for f in fields]
        p2_vals = [normalize(p2[f], batter_stats[f].min(), batter_stats[f].max()) for f in fields]

        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=p1_vals + [p1_vals[0]], theta=cats + [cats[0]],
            fill="toself", name=p1_name,
            line_color="#3498db", fillcolor="rgba(52,152,219,0.25)"
        ))
        fig.add_trace(go.Scatterpolar(
            r=p2_vals + [p2_vals[0]], theta=cats + [cats[0]],
            fill="toself", name=p2_name,
            line_color="#e74c3c", fillcolor="rgba(231,76,60,0.25)"
        ))
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
            title=f"{p1_name} vs {p2_name} — Skill Radar",
            height=450
        )
        st.plotly_chart(fig, use_container_width=True)

        # Side by side metrics with delta
        cols = st.columns(5)
        for col, field, label in zip(
            cols,
            ["runs", "average", "strike_rate", "sixes", "fours"],
            ["Runs", "Average", "Strike Rate", "Sixes", "Fours"]
        ):
            v1 = round(float(p1[field]), 1)
            v2 = round(float(p2[field]), 1)
            col.metric(label, f"{v1:,}", delta=f"vs {v2:,} ({p2_name})")

    else:
        st.warning("Please select two different batters.")

    st.markdown("---")

    # ── Individual profile ────────────────────────────────────────────────────
    st.subheader("🔍 Individual Batter Profile")
    selected = st.selectbox("Select a Batter", all_batters, key="batter_profile")
    pr = batter_stats[batter_stats["striker"] == selected].iloc[0]

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Runs",         f"{pr['runs']:,}")
    col2.metric("Average",      f"{pr['average']:.1f}")
    col3.metric("Strike Rate",  f"{pr['strike_rate']:.1f}")
    col4.metric("Sixes",        f"{int(pr['sixes'])}")
    col5.metric("Fours",        f"{int(pr['fours'])}")

    col1, col2 = st.columns(2)

    with col1:
        ps = df[df["striker"] == selected].groupby("season")["runs_off_bat"].sum().reset_index()
        ps.columns = ["season", "runs"]
        fig = px.bar(ps, x="season", y="runs",
                     title=f"{selected} — Runs by Season",
                     color="runs", color_continuous_scale="Blues",
                     text="runs")
        fig.update_traces(textposition="outside")
        fig.update_layout(xaxis_tickangle=-30)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        cats   = ["Average", "Strike Rate", "Boundary %", "Sixes", "Fours"]
        fields = ["average", "strike_rate", "boundary_pct", "sixes", "fours"]
        vals   = [normalize(pr[f], batter_stats[f].min(), batter_stats[f].max()) for f in fields]

        fig = go.Figure(go.Scatterpolar(
            r=vals + [vals[0]], theta=cats + [cats[0]],
            fill="toself", name=selected,
            line_color="#2ecc71", fillcolor="rgba(46,204,113,0.25)"
        ))
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
            title=f"{selected} — Skill Profile",
            height=380
        )
        st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — BOWLER ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Bowler Analysis":
    st.title("🎯 Bowler Analysis")
    st.markdown("*Minimum 20 matches to qualify*")
    st.markdown("---")

    # ── Top 10 horizontal ─────────────────────────────────────────────────────
    st.subheader("Top 10 Wicket Takers in BBL History")
    top10b = bowler_stats.sort_values("wickets", ascending=False).head(10)

    fig = px.bar(
        top10b.sort_values("wickets"),
        x="wickets", y="bowler",
        orientation="h",
        color="economy",
        color_continuous_scale="RdYlGn_r",
        hover_data=["economy", "average", "matches"],
        text="wickets",
        labels={"bowler": "Bowler", "wickets": "Total Wickets"}
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(height=420, yaxis_title="")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # ── Phase heatmap ─────────────────────────────────────────────────────────
    st.subheader("🌡️ Economy Heatmap — Top 15 Bowlers by Phase")
    st.markdown("*Green = economical, Red = expensive*")

    top15 = bowler_stats.sort_values("wickets", ascending=False).head(15)["bowler"].tolist()
    phase_heat = phase_bowler[phase_bowler["bowler"].isin(top15)].copy()
    phase_pivot = phase_heat.pivot_table(
        index="bowler", columns="phase", values="economy", aggfunc="mean"
    )

    phase_order = [p for p in ["Powerplay", "Middle", "Death"] if p in phase_pivot.columns]
    phase_pivot = phase_pivot[phase_order]

    fig = px.imshow(
        phase_pivot,
        color_continuous_scale="RdYlGn_r",
        title="Economy Rate by Phase — Top 15 Wicket Takers",
        labels={"x": "Phase", "y": "Bowler", "color": "Economy"},
        text_auto=".1f",
        aspect="auto"
    )
    fig.update_layout(height=480)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # ── Phase specialists ─────────────────────────────────────────────────────
    st.subheader("⚡ Phase Specialists")
    sel_phase = st.radio("Select Phase", ["Powerplay", "Middle", "Death"], horizontal=True)

    pd_data = phase_bowler[phase_bowler["phase"] == sel_phase].sort_values("economy").head(10)

    fig = go.Figure(go.Bar(
        x=pd_data["bowler"], y=pd_data["economy"],
        marker=dict(color=pd_data["wickets"], colorscale="Blues",
                    showscale=True, colorbar=dict(title="Wickets")),
        text=pd_data["economy"].round(2),
        textposition="outside",
        customdata=pd_data[["wickets", "balls"]].values,
        hovertemplate="<b>%{x}</b><br>Economy: %{y}<br>Wickets: %{customdata[0]}<br>Balls: %{customdata[1]}<extra></extra>"
    ))
    fig.update_layout(
        title=f"Best {sel_phase} Bowlers — Economy Rate",
        yaxis_title="Economy Rate",
        xaxis_tickangle=-30,
        height=420
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # ── Individual bowler ─────────────────────────────────────────────────────
    st.subheader("🔍 Individual Bowler Profile")
    sel_bowler = st.selectbox("Select a Bowler", sorted(bowler_stats["bowler"].unique()))
    br = bowler_stats[bowler_stats["bowler"] == sel_bowler].iloc[0]

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Wickets",     f"{int(br['wickets'])}")
    col2.metric("Economy",     f"{br['economy']:.2f}")
    col3.metric("Average",     f"{br['average']:.1f}")
    col4.metric("Strike Rate", f"{br['sr']:.1f}")
    col5.metric("Matches",     f"{int(br['matches'])}")

    col1, col2 = st.columns(2)

    with col1:
        bs = df[df["bowler"] == sel_bowler].groupby("season").agg(
            wickets=("is_wicket", "sum"),
            runs=("runs_off_bat", "sum"),
            balls=("is_legal", "sum")
        ).reset_index()
        bs["economy"] = ((bs["runs"] / bs["balls"].replace(0, 1)) * 6).round(2)

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=bs["season"], y=bs["wickets"],
            name="Wickets", marker_color="#e74c3c", yaxis="y"
        ))
        fig.add_trace(go.Scatter(
            x=bs["season"], y=bs["economy"],
            name="Economy", line=dict(color="#3498db", width=2.5),
            yaxis="y2"
        ))
        fig.update_layout(
            title=f"{sel_bowler} — Wickets & Economy by Season",
            yaxis=dict(title="Wickets"),
            yaxis2=dict(title="Economy Rate", overlaying="y", side="right"),
            xaxis_tickangle=-30,
            height=380,
            legend=dict(orientation="h")
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Bowling radar (invert economy/average — lower is better)
        econ_s = 100 - normalize(br["economy"], bowler_stats["economy"].min(), bowler_stats["economy"].max())
        avg_s  = 100 - normalize(br["average"], bowler_stats["average"].min(), bowler_stats["average"].max())
        wkt_s  = normalize(br["wickets"],  bowler_stats["wickets"].min(),  bowler_stats["wickets"].max())
        sr_s   = 100 - normalize(br["sr"], bowler_stats["sr"].min(),       bowler_stats["sr"].max())
        exp_s  = normalize(br["matches"],  bowler_stats["matches"].min(),  bowler_stats["matches"].max())

        bcats = ["Economy", "Average", "Wickets", "Strike Rate", "Experience"]
        bvals = [econ_s, avg_s, wkt_s, sr_s, exp_s]

        fig = go.Figure(go.Scatterpolar(
            r=bvals + [bvals[0]], theta=bcats + [bcats[0]],
            fill="toself", name=sel_bowler,
            line_color="#e74c3c", fillcolor="rgba(231,76,60,0.25)"
        ))
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
            title=f"{sel_bowler} — Bowling Profile",
            height=380
        )
        st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — WIN PROBABILITY
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Win Probability":
    st.title("📈 Win Probability — Ball by Ball")
    st.markdown("*Model accuracy: 82.0% | AUC: 0.902*")
    st.markdown("---")

    # Match selector
    avail = model_df.drop_duplicates("match_id")[
        ["match_id", "batting_team", "bowling_team", "season", "target", "winner"]
    ].dropna()

    avail["label"] = (
        avail["season"] + " | " +
        avail["batting_team"] + " vs " +
        avail["bowling_team"] + " (Target: " +
        avail["target"].astype(int).astype(str) + ")"
    )

    sel_label = st.selectbox("Select a Match", avail["label"].tolist())
    sel_row   = avail[avail["label"] == sel_label].iloc[0]
    sel_id    = sel_row["match_id"]

    # Reset animation when match changes
    if st.session_state.selected_match_id != sel_id:
        st.session_state.selected_match_id = sel_id
        st.session_state.win_prob_ball      = 1
        st.session_state.win_prob_playing   = False

    # Get full match data with predictions
    match_data = model_df[model_df["match_id"] == sel_id].copy().reset_index(drop=True)
    match_data["win_prob"] = model.predict_proba(match_data[FEATURES])[:, 1]
    total_balls = len(match_data)

    # Match header
    col1, col2, col3 = st.columns(3)
    col1.metric("🏏 Chasing Team",  sel_row["batting_team"])
    col2.metric("🎯 Target",        int(sel_row["target"]))
    col3.metric("🏆 Actual Winner", sel_row["winner"])

    st.markdown("---")

    # Controls
    col1, col2, col3, col4 = st.columns([1, 1, 1, 3])

    with col1:
        play_clicked = st.button(
            "▶ Play" if not st.session_state.win_prob_playing else "⏸ Pause"
        )
    with col2:
        reset_clicked = st.button("⏮ Reset")
    with col3:
        end_clicked = st.button("⏭ Jump to End")
    with col4:
        speed = st.select_slider(
            "Playback Speed",
            options=["0.5x", "1x", "2x", "4x"],
            value="1x"
        )

    speed_map  = {"0.5x": 0.2, "1x": 0.1, "2x": 0.05, "4x": 0.02}
    sleep_time = speed_map[speed]

    if play_clicked:
        st.session_state.win_prob_playing = not st.session_state.win_prob_playing
    if reset_clicked:
        st.session_state.win_prob_ball    = 1
        st.session_state.win_prob_playing = False
    if end_clicked:
        st.session_state.win_prob_ball    = total_balls
        st.session_state.win_prob_playing = False

    # Scrub slider
    current_ball = st.slider(
        "Drag to any ball",
        min_value=1, max_value=total_balls,
        value=st.session_state.win_prob_ball
    )
    st.session_state.win_prob_ball = current_ball

    # Placeholders for live update
    metrics_ph = st.empty()
    chart_ph   = st.empty()

    def render_frame(ball_idx):
        cur  = match_data.iloc[:ball_idx]
        last = match_data.iloc[ball_idx - 1]
        wp   = float(last["win_prob"])

        # Live metrics
        with metrics_ph.container():
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Runs Scored",    int(last["cum_total_runs"]))
            m2.metric("Wickets",        int(last["cum_wickets"]))
            m3.metric("Runs Needed",    int(last["runs_needed"]))
            m4.metric("Required RR",    f"{last['required_rr']:.1f}")
            m5.metric("Win Probability", f"{wp*100:.1f}%")

        # Colour by who's winning
        is_winning  = wp >= 0.5
        fill_color  = "rgba(46,204,113,0.2)"  if is_winning else "rgba(231,76,60,0.2)"
        line_color  = "#2ecc71"               if is_winning else "#e74c3c"

        fig = go.Figure()

        # Probability area
        fig.add_trace(go.Scatter(
            x=cur["balls_bowled"], y=cur["win_prob"],
            mode="lines",
            fill="tozeroy",
            fillcolor=fill_color,
            line=dict(color=line_color, width=2.5),
            name="Win Probability"
        ))

        # Current ball marker
        fig.add_trace(go.Scatter(
            x=[last["balls_bowled"]], y=[wp],
            mode="markers",
            marker=dict(size=12, color=line_color, symbol="circle",
                        line=dict(width=2, color="white")),
            showlegend=False
        ))

        # Reference lines
        fig.add_hline(y=0.5, line_dash="dash", line_color="gray",
                      annotation_text="50/50")
        fig.add_vline(x=36, line_dash="dot", line_color="#f1c40f",
                      line_width=1, annotation_text="Powerplay ends",
                      annotation_textangle=-90)
        fig.add_vline(x=90, line_dash="dot", line_color="#e67e22",
                      line_width=1, annotation_text="Death overs",
                      annotation_textangle=-90)

        fig.update_layout(
            title=(f"{'🟢' if is_winning else '🔴'} "
                   f"{sel_row['batting_team']} chasing {int(sel_row['target'])} "
                   f"vs {sel_row['bowling_team']}"),
            xaxis_title="Balls Bowled",
            yaxis_title="Win Probability",
            yaxis=dict(tickformat=".0%", range=[0, 1]),
            xaxis=dict(range=[0, total_balls]),
            height=420,
            showlegend=False
        )

        with chart_ph.container():
            st.plotly_chart(fig, use_container_width=True)

    # Render current frame
    render_frame(st.session_state.win_prob_ball)

    # Animation loop
    if st.session_state.win_prob_playing:
        for ball in range(st.session_state.win_prob_ball, total_balls):
            if not st.session_state.win_prob_playing:
                break
            st.session_state.win_prob_ball = ball + 1
            render_frame(ball + 1)
            time.sleep(sleep_time)
            if ball + 1 >= total_balls:
                st.session_state.win_prob_playing = False
                break
        st.rerun()

    st.markdown("---")

    # Over by over table
    st.subheader("📊 Over-by-Over Summary")
    disp = match_data[
        ["balls_bowled", "cum_total_runs", "cum_wickets",
         "runs_needed", "required_rr", "win_prob"]
    ].copy()
    disp["win_prob"] = (disp["win_prob"] * 100).round(1).astype(str) + "%"
    disp.columns = ["Balls", "Runs", "Wickets", "Runs Needed", "Req RR", "Win Prob"]
    st.dataframe(disp.iloc[::6].reset_index(drop=True), use_container_width=True)

    st.markdown("---")
    st.info(
        "💡 **How it works:** Gradient Boosting Classifier trained on 662 BBL matches. "
        "Required run rate is the single strongest predictor (50.3% importance). "
        "Chart turns 🟢 green when the chasing team is favoured, 🔴 red when they're under pressure."
    )

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — SEASON TRENDS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Season Trends":
    st.title("📅 Season Trends")
    st.markdown("*How has BBL evolved across 15 seasons?*")
    st.markdown("---")

    # Season range slider
    seasons_list = sorted(season_stats["season"].unique())
    season_range = st.select_slider(
        "🎛️ Filter Season Range",
        options=seasons_list,
        value=(seasons_list[0], seasons_list[-1])
    )

    start_i  = seasons_list.index(season_range[0])
    end_i    = seasons_list.index(season_range[1])
    filtered = season_stats[season_stats["season"].isin(seasons_list[start_i:end_i + 1])]

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        fig = px.line(filtered, x="season", y="runs_per_match",
                      title="Average Runs Per Match",
                      markers=True,
                      labels={"runs_per_match": "Runs / Match"})
        fig.update_traces(line_color="#3498db", line_width=2.5,
                          marker=dict(size=8, color="#3498db"))
        fig.update_layout(xaxis_tickangle=-30)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.bar(filtered, x="season", y="sixes_per_match",
                     title="Average Sixes Per Match",
                     color="sixes_per_match",
                     color_continuous_scale="Reds",
                     text="sixes_per_match",
                     labels={"sixes_per_match": "Sixes / Match"})
        fig.update_traces(textposition="outside", texttemplate="%{text:.1f}")
        fig.update_layout(xaxis_tickangle=-30)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # Metric selector with filled area
    st.subheader("📈 Deep Dive — Pick Any Metric")
    metric = st.radio(
        "Select Metric",
        ["runs_per_match", "sixes_per_match", "fours_per_match", "matches"],
        format_func=lambda x: {
            "runs_per_match":  "Runs per Match",
            "sixes_per_match": "Sixes per Match",
            "fours_per_match": "Fours per Match",
            "matches":         "Number of Matches"
        }[x],
        horizontal=True
    )

    fig = go.Figure(go.Scatter(
        x=filtered["season"],
        y=filtered[metric],
        mode="lines+markers+text",
        fill="tozeroy",
        fillcolor="rgba(52,152,219,0.15)",
        line=dict(color="#3498db", width=2.5),
        marker=dict(size=8),
        text=filtered[metric].round(1),
        textposition="top center"
    ))
    fig.update_layout(
        xaxis_tickangle=-30,
        height=400,
        yaxis_title=metric.replace("_", " ").title()
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # Insight callouts
    st.subheader("📌 Key Insights")
    col1, col2, col3 = st.columns(3)
    col1.success(
        f"🏏 **Peak scoring:** "
        f"{filtered.loc[filtered['runs_per_match'].idxmax(), 'season']} "
        f"({filtered['runs_per_match'].max():.0f} runs/match)"
    )
    col2.success(
        f"💥 **Six-hitting peak:** "
        f"{filtered.loc[filtered['sixes_per_match'].idxmax(), 'season']} "
        f"({filtered['sixes_per_match'].max():.1f} sixes/match)"
    )
    col3.success(
        f"📈 **Sixes growth:** "
        f"{((filtered['sixes_per_match'].iloc[-1] / filtered['sixes_per_match'].iloc[0]) - 1) * 100:.0f}% "
        f"more sixes vs first selected season"
    )

    # Full table
    st.subheader("Full Season Stats Table")
    disp = filtered[["season", "matches", "runs_per_match",
                      "sixes_per_match", "fours_per_match"]].copy()
    disp.columns = ["Season", "Matches", "Runs/Match", "Sixes/Match", "Fours/Match"]
    st.dataframe(disp, use_container_width=True, hide_index=True)