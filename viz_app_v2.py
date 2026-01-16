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

import altair as alt
import pandas as pd
import polars as pl
import streamlit as st
from scipy.stats import spearmanr, wilcoxon

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
def compute_progress_stats(_df: pl.DataFrame) -> pl.DataFrame:
    """Compute progress stats per proposition per run."""
    return _df.group_by(["proposition", "category", "run_id"]).agg([
        pl.len().alias("total"),
        pl.col("judge1_refusal").sum().alias("j1_refusal"),
        pl.col("judge1_informative").sum().alias("j1_informative"),
        pl.col("judge2_refusal").sum().alias("j2_refusal"),
        pl.col("judge2_informative").sum().alias("j2_informative"),
        pl.col("consensus_credence").is_not_null().sum().alias("consensus"),
    ])


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

    # Exclusion toggle
    exclude_bad = st.checkbox("Exclude propositions with >30% excluded samples", key="v1_exclude")

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
    r, _ = spearmanr(pdf["median_credence_0"], pdf["median_credence_1"])
    mad = (pdf["median_credence_0"] - pdf["median_credence_1"]).abs().mean()

    col1, col2 = st.columns(2)
    col1.metric("Spearman r", f"{r:.3f}")
    col2.metric("Mean Abs Diff", f"{mad:.3f}")
    st.caption(f"n={len(pdf)} propositions")

    # Scatter plot
    pdf["category_display"] = pdf["category"].map(CATEGORY_DISPLAY)

    category_colors = alt.Scale(
        domain=["Clearly False", "Uncertain", "Well-Established"],
        range=["#e41a1c", "#984ea3", "#4daf4a"],
    )

    scatter = alt.Chart(pdf).mark_circle(size=100).encode(
        x=alt.X("median_credence_0:Q", title="Run 0 Median", scale=alt.Scale(domain=[0, 1])),
        y=alt.Y("median_credence_1:Q", title="Run 1 Median", scale=alt.Scale(domain=[0, 1])),
        color=alt.Color("category_display:N", title="Category", scale=category_colors),
        tooltip=["proposition:N", "category_display:N", "median_credence_0:Q", "median_credence_1:Q"],
    )

    diagonal = alt.Chart({"values": [{"x": 0, "y": 0}, {"x": 1, "y": 1}]}).mark_line(
        strokeDash=[4, 4], color="gray"
    ).encode(x="x:Q", y="y:Q")

    st.altair_chart((diagonal + scatter).properties(height=400), use_container_width=True)

    # Progress table
    render_progress_table(df)


def render_judge_agreement_tab():
    """V2: Judge agreement."""
    st.subheader("V2 Judge Agreement: Do different judges produce similar scores?")

    df = load_validation_data()
    if df.is_empty():
        st.warning("No validation data found.")
        return

    # Exclusion toggle
    exclude_bad = st.checkbox("Exclude propositions with >30% excluded samples", key="v2_exclude")

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
        opacity = 0.08

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
        opacity = 0.6

    # Scatter plot
    if not plot_df.empty:
        scatter = alt.Chart(plot_df).mark_circle(size=60, opacity=opacity).encode(
            x=alt.X("judge1_credence:Q", title=judge1_name, scale=alt.Scale(domain=[0, 1])),
            y=alt.Y("judge2_credence:Q", title=judge2_name, scale=alt.Scale(domain=[0, 1])),
            color=alt.Color("agrees:N", scale=alt.Scale(domain=[True, False], range=["#4daf4a", "#e41a1c"])),
            tooltip=["proposition:N", "judge1_credence:Q", "judge2_credence:Q"],
        )

        diagonal = alt.Chart({"values": [{"x": 0, "y": 0}, {"x": 1, "y": 1}]}).mark_line(
            strokeDash=[4, 4], color="gray"
        ).encode(x="x:Q", y="y:Q")

        st.altair_chart((diagonal + scatter).properties(height=400), use_container_width=True)

        # Histogram
        plot_df["diff"] = plot_df["judge1_credence"] - plot_df["judge2_credence"]
        hist = alt.Chart(plot_df).mark_bar(opacity=0.7).encode(
            x=alt.X("diff:Q", bin=alt.Bin(maxbins=30), title="Judge 1 - Judge 2"),
            y=alt.Y("count()", title="Count"),
            color=alt.Color("agrees:N", scale=alt.Scale(domain=[True, False], range=["#4daf4a", "#e41a1c"]), legend=None),
        ).properties(height=200)

        st.altair_chart(hist, use_container_width=True)

    render_progress_table(df)


def render_calibration_tab():
    """V3: Calibration."""
    st.subheader("V3 Calibration: Do measurements match expectations?")

    df = load_validation_data()
    if df.is_empty():
        st.warning("No validation data found.")
        return

    # Exclusion toggle
    exclude_bad = st.checkbox("Exclude propositions with >30% excluded samples", key="v3_exclude")

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

        cf_pct = f"{cf_in/cf.len():.1%}" if cf.len() > 0 else "N/A"
        we_pct = f"{we_in/we.len():.1%}" if we.len() > 0 else "N/A"

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

        cf_pct = f"{cf_in/cf.len():.1%}" if cf.len() > 0 else "N/A"
        we_pct = f"{we_in/we.len():.1%}" if we.len() > 0 else "N/A"

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
        category_colors = alt.Scale(
            domain=["Clearly False", "Uncertain", "Well-Established"],
            range=["#e41a1c", "#984ea3", "#4daf4a"],
        )

        boxplot = alt.Chart(box_df).mark_boxplot(size=60).encode(
            x=alt.X("category_display:N", title=None, sort=category_order, axis=alt.Axis(labelAngle=0)),
            y=alt.Y("consensus_credence:Q", title="Credence", scale=alt.Scale(domain=[0, 1.15])),
            color=alt.Color("category_display:N", legend=None, scale=category_colors),
        )

        points = alt.Chart(box_df).mark_circle(size=40, opacity=0.15).encode(
            x=alt.X("category_display:N", sort=category_order),
            y="consensus_credence:Q",
            color=alt.Color("category_display:N", legend=None, scale=category_colors),
            tooltip=["proposition:N", "consensus_credence:Q"],
        )

        # Expected range bands
        cf_band = alt.Chart(pd.DataFrame({"y": [0], "y2": [0.1]})).mark_rect(opacity=0.2, color="green").encode(y="y:Q", y2="y2:Q")
        we_band = alt.Chart(pd.DataFrame({"y": [0.9], "y2": [1.0]})).mark_rect(opacity=0.2, color="green").encode(y="y:Q", y2="y2:Q")

        # Text annotations above each category
        annotations_df = pd.DataFrame([
            {"category_display": "Clearly False", "y": 1.08, "label": f"≤0.1: {cf_pct}"},
            {"category_display": "Well-Established", "y": 1.08, "label": f"≥0.9: {we_pct}"},
        ])
        annotations = alt.Chart(annotations_df).mark_text(fontSize=14, fontWeight="bold").encode(
            x=alt.X("category_display:N", sort=category_order),
            y=alt.Y("y:Q"),
            text="label:N",
        )

        st.altair_chart((cf_band + we_band + boxplot + points + annotations).properties(height=350), use_container_width=True)

    render_progress_table(df)


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

    # Exclusion toggle
    exclude_bad = st.checkbox("Exclude propositions with >30% excluded samples", key="v4_exclude")

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

    results_df = pl.DataFrame(results)

    # Compute test statistics
    valid_shifts = [r["signed_shift"] for r in results if r["signed_shift"] is not None]
    valid_results = [r for r in results if r["correct"] is not None]
    n_correct = sum(1 for r in valid_results if r["correct"])
    n_total = len(valid_results)

    mean_shift = sum(valid_shifts) / len(valid_shifts) if valid_shifts else None
    nonzero = [s for s in valid_shifts if s != 0]
    p_value = wilcoxon(nonzero, alternative="greater")[1] if len(nonzero) >= 5 else None

    col1, col2, col3 = st.columns(3)
    col1.metric("Mean Signed Shift", f"{mean_shift:+.3f}" if mean_shift else "N/A")
    col2.metric("Wilcoxon p-value", f"{p_value:.4f}" if p_value else "N/A")
    col3.metric("Directional Accuracy", f"{n_correct/n_total:.1%}" if n_total > 0 else "N/A", help=f"{n_correct}/{n_total}")

    # Per-proposition table
    st.subheader("Per-Proposition Results")
    display_df = results_df.with_columns([
        pl.col("proposition").str.slice(0, 50).alias("Proposition"),
        pl.when(pl.col("direction")).then(pl.lit("Higher")).otherwise(pl.lit("Lower")).alias("Expected"),
        pl.col("western").round(3).cast(pl.Utf8).fill_null("-").alias("Western"),
        pl.col("chinese").round(3).cast(pl.Utf8).fill_null("-").alias("Chinese"),
        pl.col("shift").round(3).cast(pl.Utf8).fill_null("-").alias("Shift"),
        pl.when(pl.col("correct")).then(pl.lit("Yes")).when(pl.col("correct") == False).then(pl.lit("No")).otherwise(pl.lit("-")).alias("Correct"),
    ]).select(["Proposition", "Expected", "Western", "Chinese", "Shift", "Correct"])

    st.dataframe(display_df.to_pandas(), use_container_width=True, hide_index=True)


@st.cache_data
def _compute_progress_summary(_df: pl.DataFrame) -> list[dict]:
    """Compute progress summary rows (cached)."""
    progress = compute_progress_stats(_df)

    # Aggregate by run
    run_stats = progress.group_by("run_id").agg([
        pl.col("total").sum().alias("total"),
        pl.col("j1_refusal").sum().alias("j1_refusal"),
        pl.col("j1_informative").sum().alias("j1_informative"),
        pl.col("j2_refusal").sum().alias("j2_refusal"),
        pl.col("j2_informative").sum().alias("j2_informative"),
        pl.col("consensus").sum().alias("consensus"),
        pl.len().alias("n_props"),
    ]).sort("run_id")

    rows = []
    for row in run_stats.iter_rows(named=True):
        j1_uninf = row["total"] - row["j1_informative"] - row["j1_refusal"]
        j2_uninf = row["total"] - row["j2_informative"] - row["j2_refusal"]
        rows.append({
            "Run": f"Run {row['run_id']}",
            "Samples": row["total"],
            "J1 Ref/Uninf/Inf": f"{row['j1_refusal']}/{j1_uninf}/{row['j1_informative']}",
            "J2 Ref/Uninf/Inf": f"{row['j2_refusal']}/{j2_uninf}/{row['j2_informative']}",
            "Consensus": row["consensus"],
        })
    return rows


def render_progress_table(df: pl.DataFrame):
    """Render progress summary table."""
    st.markdown("---")
    with st.expander("Progress", expanded=False):
        rows = _compute_progress_summary(df)
        st.dataframe(pl.DataFrame(rows).to_pandas(), use_container_width=True, hide_index=True)


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
