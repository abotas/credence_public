"""Streamlit app for visualizing validation experiment results.

Usage:
    uv run streamlit run viz_app_v2.py

Part 1: Method Validation (V1-V4)
- V1 Test-retest reliability
- V2 Judge agreement
- V3 Calibration
- V4 Known-group validity (discriminant)
"""

from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import polars as pl
import streamlit as st
from scipy.stats import pearsonr, spearmanr, wilcoxon

# =============================================================================
# Constants
# =============================================================================

DATA_ROOT = Path(__file__).parent / "data"
DISCRIMINANT_DATA_ROOT = Path(__file__).parent / "data_discriminant"
PROPOSITIONS_CSV = Path(__file__).parent / "data" / "raw_propositions" / "curated" / "china_west_contentious_v2.csv"

AGREEMENT_THRESHOLD = 0.2
EXCLUSION_THRESHOLD = 0.30
CLEARLY_FALSE_MAX = 0.1
WELL_ESTABLISHED_MIN = 0.9
NUM_PROMPTS = 32
NUM_RUNS = 2

CATEGORY_DISPLAY = {
    "clearly_false": "Clearly False",
    "contested": "Uncertain",
    "well_established": "Well-Established",
}

WESTERN_MODELS = {"anthropic/claude-sonnet-4-5-20250929", "openai/gpt-5-mini-2025-08-07"}
CHINESE_MODELS = {"deepseek/deepseek-chat", "moonshot/kimi-k2-turbo-preview"}


# =============================================================================
# Data Loading (cached)
# =============================================================================

@st.cache_data
def load_validation_data() -> pl.DataFrame:
    """Load validation parquet and add consensus_credence column."""
    path = DATA_ROOT / "validation.parquet"
    if not path.exists():
        return pl.DataFrame()

    df = pl.read_parquet(path)
    return _add_consensus_credence(df)


@st.cache_data
def load_discriminant_data() -> pl.DataFrame:
    """Load discriminant parquet and add consensus_credence + model_group columns."""
    path = DISCRIMINANT_DATA_ROOT / "discriminant.parquet"
    if not path.exists():
        return pl.DataFrame()

    df = pl.read_parquet(path)
    df = _add_consensus_credence(df)

    # Add model_group column
    df = df.with_columns([
        pl.when(pl.col("target_model_id").is_in(WESTERN_MODELS))
        .then(pl.lit("western"))
        .when(pl.col("target_model_id").is_in(CHINESE_MODELS))
        .then(pl.lit("chinese"))
        .otherwise(pl.lit("unknown"))
        .alias("model_group")
    ])

    return df


@st.cache_data
def load_proposition_directions() -> dict[str, bool]:
    """Load proposition directions from CSV."""
    import csv
    if not PROPOSITIONS_CSV.exists():
        return {}
    directions = {}
    with open(PROPOSITIONS_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            directions[row["proposition"]] = row["china_sensitivity_direction"].lower().strip() == "true"
    return directions


def _add_consensus_credence(df: pl.DataFrame) -> pl.DataFrame:
    """Add consensus_credence column to DataFrame."""
    eps = 1e-9
    return df.with_columns([
        pl.when(
            pl.col("judge1_informative") & pl.col("judge2_informative") &
            pl.col("judge1_credence").is_not_null() & pl.col("judge2_credence").is_not_null() &
            ((pl.col("judge1_credence") - pl.col("judge2_credence")).abs() <= AGREEMENT_THRESHOLD + eps)
        ).then(
            (pl.col("judge1_credence") + pl.col("judge2_credence")) / 2
        ).otherwise(None).alias("consensus_credence")
    ])


# =============================================================================
# Computed Stats (cached)
# =============================================================================

@st.cache_data
def compute_exclusion_rates(_df: pl.DataFrame, group_cols: list[str]) -> pl.DataFrame:
    """Compute exclusion rate per group (fraction of samples without consensus)."""
    return _df.group_by(group_cols).agg([
        pl.len().alias("total"),
        pl.col("consensus_credence").is_null().sum().alias("excluded"),
    ]).with_columns([
        (pl.col("excluded") / pl.col("total")).alias("exclusion_rate")
    ])


@st.cache_data
def compute_test_retest_stats(_df: pl.DataFrame) -> pl.DataFrame:
    """Compute per-proposition median credence for each run."""
    return _df.group_by(["proposition", "category", "run_id"]).agg([
        pl.col("consensus_credence").drop_nulls().median().alias("median_credence"),
        pl.col("consensus_credence").drop_nulls().len().alias("n_consensus"),
    ]).pivot(
        on="run_id",
        index=["proposition", "category"],
        values=["median_credence", "n_consensus"],
    )


@st.cache_data
def compute_discriminant_stats(_df: pl.DataFrame, directions: dict[str, bool]) -> pl.DataFrame:
    """Compute per-proposition mean credence by model group."""
    # Get mean credence per proposition per model_group
    grouped = _df.group_by(["proposition", "model_group"]).agg([
        pl.col("consensus_credence").drop_nulls().mean().alias("mean_credence"),
        pl.col("consensus_credence").drop_nulls().len().alias("n_consensus"),
    ])

    # Pivot to get western and chinese columns side by side
    pivoted = grouped.pivot(
        on="model_group",
        index="proposition",
        values=["mean_credence", "n_consensus"],
    )

    # Add direction column
    dir_df = pl.DataFrame({
        "proposition": list(directions.keys()),
        "direction": list(directions.values()),
    })

    return pivoted.join(dir_df, on="proposition", how="left")


@st.cache_data
def get_bad_propositions(_df: pl.DataFrame) -> list[str]:
    """Get list of propositions with >30% excluded samples (cached)."""
    excl = compute_exclusion_rates(_df, ["proposition"])
    return excl.filter(pl.col("exclusion_rate") > EXCLUSION_THRESHOLD)["proposition"].to_list()


@st.cache_data
def get_bad_propositions_discriminant(_df: pl.DataFrame) -> list[str]:
    """Get list of propositions with >30% excluded samples for ANY model (cached)."""
    excl = compute_exclusion_rates(_df, ["proposition", "target_model_id"])
    return excl.filter(pl.col("exclusion_rate") > EXCLUSION_THRESHOLD)["proposition"].unique().to_list()


# =============================================================================
# Render Functions
# =============================================================================

def render_test_retest_tab():
    """V1: Test-retest reliability."""
    st.subheader("V1 Test-Retest: Do repeated pipeline runs yield similar estimates?")

    df = load_validation_data()
    if df.is_empty():
        st.warning("No validation data found. Run build_parquet.py first.")
        return

    # Exclusion toggle (stored in state, displayed later)
    exclude_bad = st.session_state.get("v1_exclude", False)

    # Category filter
    categories = ["All"] + list(CATEGORY_DISPLAY.keys())
    selected = st.radio(
        "Category", categories,
        format_func=lambda x: "All" if x == "All" else CATEGORY_DISPLAY[x],
        horizontal=True, key="v1_category"
    )

    # Get test-retest stats
    stats = compute_test_retest_stats(df)

    # Apply exclusion filter
    if exclude_bad:
        bad_props = get_bad_propositions(df)
        stats = stats.filter(~pl.col("proposition").is_in(bad_props))

    # Apply category filter
    if selected != "All":
        stats = stats.filter(pl.col("category") == selected)

    # Filter to propositions with data in both runs
    stats = stats.filter(
        pl.col("median_credence_0").is_not_null() &
        pl.col("median_credence_1").is_not_null()
    )

    if stats.is_empty():
        st.warning("No propositions with data in both runs.")
        return

    # Convert to pandas for scipy
    pdf = stats.to_pandas()

    # Compute metrics
    spearman_r, _ = spearmanr(pdf["median_credence_0"], pdf["median_credence_1"])
    pearson_r, _ = pearsonr(pdf["median_credence_0"], pdf["median_credence_1"])
    mad = (pdf["median_credence_0"] - pdf["median_credence_1"]).abs().mean()

    col1, col2, col3 = st.columns(3)
    col1.metric("Pearson r", f"{pearson_r:.3f}")
    col2.metric("Spearman r", f"{spearman_r:.3f}")
    col3.metric("Mean Abs Diff", f"{mad:.3f}")
    st.caption(f"n={len(pdf)} propositions")

    # Scatter plot
    pdf["category_display"] = pdf["category"].map(CATEGORY_DISPLAY)

    category_colors = {"Clearly False": "#e41a1c", "Uncertain": "#984ea3", "Well-Established": "#4daf4a"}

    fig = px.scatter(
        pdf, x="median_credence_0", y="median_credence_1",
        color="category_display",
        color_discrete_map=category_colors,
        render_mode="webgl", opacity=0.5,
        labels={"median_credence_0": "Run 0 Median", "median_credence_1": "Run 1 Median", "category_display": "Category"},
        hover_data=["proposition"],
    )
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode="lines", line=dict(color="gray", dash="dash", width=2), name="y=x",
        showlegend=False,
    ))
    fig.update_traces(marker=dict(size=10), selector=dict(mode="markers"))
    fig.update_layout(xaxis=dict(range=[0, 1]), yaxis=dict(range=[0, 1]), height=400)
    st.plotly_chart(fig, use_container_width=True)

    st.checkbox("Exclude propositions with >30% excluded samples", key="v1_exclude")


def render_judge_agreement_tab():
    """V2: Judge agreement."""
    st.subheader("V2 Judge Agreement: Do different judges produce similar scores?")

    df = load_validation_data()
    if df.is_empty():
        st.warning("No validation data found.")
        return

    # Exclusion toggle (stored in state, displayed later)
    exclude_bad = st.session_state.get("v2_exclude", False)

    # Category filter
    categories = ["All"] + list(CATEGORY_DISPLAY.keys())
    selected = st.radio(
        "Category", categories,
        format_func=lambda x: "All" if x == "All" else CATEGORY_DISPLAY[x],
        horizontal=True, key="v2_category"
    )

    # Aggregation level
    agg_level = st.radio(
        "Aggregation", ["sample", "proposition"],
        format_func=lambda x: "Sample-level" if x == "sample" else "Proposition-level (median)",
        horizontal=True, key="v2_agg"
    )

    # Filter data
    filtered = df
    if exclude_bad:
        bad_props = get_bad_propositions(df)
        filtered = filtered.filter(~pl.col("proposition").is_in(bad_props))

    if selected != "All":
        filtered = filtered.filter(pl.col("category") == selected)

    # Get judge names
    judge1_name = filtered.select("judge1_llm_id").row(0)[0].split("/")[-1] if len(filtered) > 0 else "Judge 1"
    judge2_name = filtered.select("judge2_llm_id").row(0)[0].split("/")[-1] if len(filtered) > 0 else "Judge 2"

    eps = 1e-9  # For floating point comparison

    if agg_level == "sample":
        # Sample-level analysis
        informative = filtered.filter(
            pl.col("judge1_informative") & pl.col("judge2_informative") &
            pl.col("judge1_credence").is_not_null() & pl.col("judge2_credence").is_not_null()
        )

        total = len(filtered)
        n_agreed = informative.filter(
            (pl.col("judge1_credence") - pl.col("judge2_credence")).abs() <= AGREEMENT_THRESHOLD + eps
        ).height

        col1, col2 = st.columns(2)
        col1.metric("Agreement Rate", f"{n_agreed/total:.1%}" if total > 0 else "N/A")
        col2.metric("Agreed / Total", f"{n_agreed}/{total}")

        plot_df = informative.select([
            "proposition", "judge1_credence", "judge2_credence",
            ((pl.col("judge1_credence") - pl.col("judge2_credence")).abs() <= AGREEMENT_THRESHOLD + eps).alias("agrees")
        ]).to_pandas()
        opacity = 0.05

    else:
        # Proposition-level (median)
        informative = filtered.filter(
            pl.col("judge1_informative") & pl.col("judge2_informative") &
            pl.col("judge1_credence").is_not_null() & pl.col("judge2_credence").is_not_null()
        )

        prop_agg = informative.group_by("proposition").agg([
            pl.col("judge1_credence").median().alias("judge1_credence"),
            pl.col("judge2_credence").median().alias("judge2_credence"),
        ]).with_columns([
            ((pl.col("judge1_credence") - pl.col("judge2_credence")).abs() <= AGREEMENT_THRESHOLD + eps).alias("agrees")
        ])

        total = prop_agg.height
        n_agreed = prop_agg.filter(pl.col("agrees")).height

        col1, col2 = st.columns(2)
        col1.metric("Agreement Rate", f"{n_agreed/total:.1%}" if total > 0 else "N/A")
        col2.metric("Agreed / Total", f"{n_agreed}/{total} propositions")

        plot_df = prop_agg.to_pandas()
        opacity = 0.4

    # Scatter plot
    if not plot_df.empty:
        plot_df["agrees_str"] = plot_df["agrees"].map({True: "Agree", False: "Disagree"})
        agree_colors = {"Agree": "#4daf4a", "Disagree": "#e41a1c"}

        fig = px.scatter(
            plot_df, x="judge1_credence", y="judge2_credence",
            color="agrees_str",
            color_discrete_map=agree_colors,
            render_mode="webgl", opacity=opacity,
            labels={"judge1_credence": judge1_name, "judge2_credence": judge2_name, "agrees_str": "Agreement"},
            hover_data=["proposition"],
        )
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode="lines", line=dict(color="gray", dash="dash", width=2), name="y=x",
            showlegend=False,
        ))
        fig.update_traces(marker=dict(size=10), selector=dict(mode="markers"))
        fig.update_layout(xaxis=dict(range=[0, 1]), yaxis=dict(range=[0, 1]), height=400)
        st.plotly_chart(fig, use_container_width=True)

        # Histogram
        plot_df["diff"] = plot_df["judge1_credence"] - plot_df["judge2_credence"]
        hist_fig = px.histogram(
            plot_df, x="diff", color="agrees_str",
            color_discrete_map=agree_colors,
            nbins=30, opacity=0.7,
            labels={"diff": "Judge 1 - Judge 2", "agrees_str": "Agreement"},
        )
        hist_fig.update_layout(height=200, showlegend=False, bargap=0.1)
        st.plotly_chart(hist_fig, use_container_width=True)

    st.checkbox("Exclude propositions with >30% excluded samples", key="v2_exclude")


def render_calibration_tab():
    """V3: Calibration."""
    st.subheader("V3 Calibration: Do measurements match expectations?")

    df = load_validation_data()
    if df.is_empty():
        st.warning("No validation data found.")
        return

    # Exclusion toggle (stored in state, displayed later)
    exclude_bad = st.session_state.get("v3_exclude", False)

    # Aggregation level
    agg_level = st.radio(
        "Aggregation", ["sample", "proposition"],
        format_func=lambda x: "Sample-level" if x == "sample" else "Proposition-level (median)",
        horizontal=True, key="v3_agg"
    )

    # Filter data
    filtered = df
    if exclude_bad:
        bad_props = get_bad_propositions(df)
        filtered = filtered.filter(~pl.col("proposition").is_in(bad_props))

    eps = 1e-9  # For floating point comparison (0.8999999999999999 should count as >= 0.9)

    if agg_level == "sample":
        # Sample-level
        cf = filtered.filter(pl.col("category") == "clearly_false")["consensus_credence"].drop_nulls()
        we = filtered.filter(pl.col("category") == "well_established")["consensus_credence"].drop_nulls()

        cf_in = (cf <= CLEARLY_FALSE_MAX + eps).sum()
        we_in = (we >= WELL_ESTABLISHED_MIN - eps).sum()

        col1, col2 = st.columns(2)
        col1.metric("Clearly False <= 0.1", f"{cf_in/cf.len():.1%}" if cf.len() > 0 else "N/A", help=f"{cf_in}/{cf.len()}")
        col2.metric("Well-Established >= 0.9", f"{we_in/we.len():.1%}" if we.len() > 0 else "N/A", help=f"{we_in}/{we.len()}")

        # Build box plot data
        box_df = filtered.select(["category", "consensus_credence", "proposition"]).drop_nulls().to_pandas()

    else:
        # Proposition-level
        prop_medians = filtered.group_by(["proposition", "category"]).agg([
            pl.col("consensus_credence").drop_nulls().median().alias("median_credence"),
        ])

        cf = prop_medians.filter(pl.col("category") == "clearly_false")["median_credence"].drop_nulls()
        we = prop_medians.filter(pl.col("category") == "well_established")["median_credence"].drop_nulls()

        cf_in = (cf <= CLEARLY_FALSE_MAX + eps).sum()
        we_in = (we >= WELL_ESTABLISHED_MIN - eps).sum()

        col1, col2 = st.columns(2)
        col1.metric("Clearly False <= 0.1", f"{cf_in/cf.len():.1%}" if cf.len() > 0 else "N/A", help=f"{cf_in}/{cf.len()}")
        col2.metric("Well-Established >= 0.9", f"{we_in/we.len():.1%}" if we.len() > 0 else "N/A", help=f"{we_in}/{we.len()}")

        # Build box plot from medians
        box_df = prop_medians.select([
            pl.col("category"),
            pl.col("median_credence").alias("consensus_credence"),
            pl.col("proposition"),
        ]).drop_nulls().to_pandas()

    # Box plot
    if not box_df.empty:
        box_df["category_display"] = box_df["category"].map(CATEGORY_DISPLAY)
        category_order = ["Clearly False", "Uncertain", "Well-Established"]
        category_colors = {"Clearly False": "#e41a1c", "Uncertain": "#984ea3", "Well-Established": "#4daf4a"}

        fig = px.box(
            box_df, x="category_display", y="consensus_credence",
            color="category_display",
            color_discrete_map=category_colors,
            category_orders={"category_display": category_order},
            points="all",
            labels={"category_display": "", "consensus_credence": "Credence"},
            hover_data=["proposition"],
        )
        fig.update_traces(marker=dict(opacity=0.15, size=6), pointpos=0)

        # Expected range bands
        fig.add_hrect(y0=0, y1=0.1, fillcolor="green", opacity=0.15, line_width=0)
        fig.add_hrect(y0=0.9, y1=1.0, fillcolor="green", opacity=0.15, line_width=0)

        fig.update_layout(yaxis=dict(range=[0, 1]), height=300, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    st.checkbox("Exclude propositions with >30% excluded samples", key="v3_exclude")


def render_known_group_tab():
    """V4: Known-group validity (discriminant)."""
    st.subheader("V4 Known-Group: Do Chinese/Western models diverge on sensitive topics?")

    df = load_discriminant_data()
    if df.is_empty():
        st.warning("No discriminant data found. Run build_parquet.py first.")
        return

    directions = load_proposition_directions()
    if not directions:
        st.warning("No proposition directions found.")
        return

    # Exclusion toggle (stored in state, displayed later)
    exclude_bad = st.session_state.get("v4_exclude", False)

    # Filter data
    filtered = df
    if exclude_bad:
        bad_props = get_bad_propositions_discriminant(df)
        filtered = filtered.filter(~pl.col("proposition").is_in(bad_props))

    # Compute per-proposition stats
    stats = compute_discriminant_stats(filtered, directions)

    if stats.is_empty():
        st.warning("No data available.")
        return

    # Compute directional results
    pdf = stats.to_pandas()

    results = []
    for _, row in pdf.iterrows():
        western = row.get("mean_credence_western")
        chinese = row.get("mean_credence_chinese")
        direction = row.get("direction")

        if western is None or chinese is None or direction is None:
            shift, signed_shift, correct = None, None, None
        else:
            shift = chinese - western
            signed_shift = shift if direction else -shift
            correct = signed_shift > 0

        results.append({
            "proposition": row["proposition"],
            "direction": direction,
            "western": western,
            "chinese": chinese,
            "shift": shift,
            "signed_shift": signed_shift,
            "correct": correct,
        })

    # Compute test statistics
    valid_shifts = [r["signed_shift"] for r in results if r["signed_shift"] is not None]
    valid_results = [r for r in results if r["correct"] is not None]
    n_correct = sum(1 for r in valid_results if r["correct"])
    n_total = len(valid_results)

    mean_shift = sum(valid_shifts) / len(valid_shifts) if valid_shifts else None
    nonzero = [s for s in valid_shifts if s != 0]
    p_value = wilcoxon(nonzero, alternative="greater")[1] if len(nonzero) >= 5 else None

    # Dumbbell chart
    plot_results = [r for r in results if r["western"] is not None and r["chinese"] is not None]
    if plot_results:
        # Sort by signed shift ascending (so biggest green appears at top of chart, biggest red at bottom)
        plot_results = sorted(plot_results, key=lambda r: r["signed_shift"] or 0)

        fig = go.Figure()

        for r in plot_results:
            color = "#4daf4a" if r["correct"] else "#e41a1c"
            prop_short = r["proposition"][:40] + "..." if len(r["proposition"]) > 40 else r["proposition"]
            prop_full = r["proposition"]
            shift_val = r["shift"]

            # Line connecting Western to Chinese
            fig.add_trace(go.Scatter(
                x=[r["western"], r["chinese"]],
                y=[prop_short, prop_short],
                mode="lines",
                line=dict(color=color, width=2),
                showlegend=False,
                hoverinfo="skip",
            ))

            # Western dot (circle)
            fig.add_trace(go.Scatter(
                x=[r["western"]],
                y=[prop_short],
                mode="markers",
                marker=dict(color=color, size=10, symbol="circle"),
                name="Western",
                showlegend=False,
                hovertemplate=f"<b>{prop_full}</b><br>Western: {r['western']:.3f}<br>Shift: {shift_val:+.3f}<extra></extra>",
            ))

            # Chinese dot (diamond)
            fig.add_trace(go.Scatter(
                x=[r["chinese"]],
                y=[prop_short],
                mode="markers",
                marker=dict(color=color, size=10, symbol="diamond"),
                name="Chinese",
                showlegend=False,
                hovertemplate=f"<b>{prop_full}</b><br>Chinese: {r['chinese']:.3f}<br>Shift: {shift_val:+.3f}<extra></extra>",
            ))

        # Add legend entries
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", marker=dict(color="#4daf4a", size=10), name="Predicted"))
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", marker=dict(color="#e41a1c", size=10), name="Unexpected"))
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", marker=dict(color="gray", size=10, symbol="circle"), name="Western"))
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", marker=dict(color="gray", size=10, symbol="diamond"), name="Chinese"))

        fig.update_layout(
            height=max(300, 25 * len(plot_results)),
            xaxis=dict(title="Credence", range=[0, 1]),
            yaxis=dict(title=""),
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            margin=dict(l=10),
        )
        st.plotly_chart(fig, use_container_width=True)

    # Metrics and exclusion toggle (below figure)
    col1, col2, col3 = st.columns(3)
    col1.metric("Mean Signed Shift", f"{mean_shift:+.3f}" if mean_shift else "N/A")
    col2.metric("Wilcoxon p-value", f"{p_value:.4f}" if p_value else "N/A")
    col3.metric("Directional Accuracy", f"{n_correct/n_total:.1%}" if n_total > 0 else "N/A", help=f"{n_correct}/{n_total}")

    st.checkbox("Exclude propositions with >30% excluded samples", key="v4_exclude")


# =============================================================================
# Main
# =============================================================================

def main():
    st.set_page_config(page_title="Method Validation", layout="wide")
    st.title("Part 1: Method Validation")

    tab1, tab2, tab3, tab4 = st.tabs([
        "V1 Test-Retest",
        "V2 Judge Agreement",
        "V3 Calibration",
        "V4 Known-Group",
    ])

    with tab1:
        render_test_retest_tab()

    with tab2:
        render_judge_agreement_tab()

    with tab3:
        render_calibration_tab()

    with tab4:
        render_known_group_tab()


if __name__ == "__main__":
    main()
