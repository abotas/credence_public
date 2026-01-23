"""Streamlit app for visualizing validation experiment results.

Usage:
    uv run streamlit run viz_app_v2.py

Part 1: Method Validation
- V1 Test-retest reliability
- V2 Judge agreement
- V3 Calibration
- V4 Known-group validity (discriminant)
- Inspect: proposition inspection
"""

import gzip
import hashlib
import json
import re
from pathlib import Path

import plotly.express as px
import plotly.graph_objects as go
import polars as pl
import streamlit as st
from scipy.stats import pearsonr, spearmanr, wilcoxon


def slugify(text: str, max_length: int = 50) -> str:
    """Convert text to lowercase alphanumeric slug with hash suffix for uniqueness."""
    text_hash = hashlib.md5(text.encode()).hexdigest()[:8]
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    prefix_max = max_length - len(text_hash) - 1
    if len(slug) > prefix_max:
        slug = slug[:prefix_max].rsplit("-", 1)[0]
    return f"{slug}-{text_hash}"


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


@st.cache_data
def load_judged_jsonl(data_root: str, subpath: str, slug: str) -> list[dict]:
    """Load full judged.jsonl.gz for a specific proposition (on-demand)."""
    path = Path(data_root) / subpath / slug / "judged.jsonl.gz"
    if not path.exists():
        return []
    samples = []
    with gzip.open(path, "rt") as f:
        for line in f:
            samples.append(json.loads(line))
    return samples


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
    st.caption("We test 150 propositions (50 clearly false, 50 well-established, 50 uncertain) by running the full pipeline twice. Each run generates 32 fresh prompts per proposition using two prompt models (Claude Haiku 4.5 and GPT-5-mini), collects responses from the test model (GPT-5-mini), and has judges score each response. Comparing median credences across runs tells us if our measurements are stable or noisy, at provider-default temperatures.")

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
    st.caption("Two judges (Claude Sonnet 4.5 and GPT-5) independently evaluate every proposition-prompt-response triplet, estimating the test model's credence from its response. Here we measure how often judges agree within 0.2 across 150 propositions (50 clearly false, 50 well-established, 50 uncertain).")

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
    st.caption("We check whether 'clearly false' propositions score low credence values, 'well-established' propositions score high, and 'uncertain' score somewhere in between, to establish whether our measured values track expectation.")

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
    st.caption("Here we check whether our methodology can detect a known cross-model difference. We compare Chinese models (DeepSeek, Kimi) to Western models (Claude, GPT-5-mini) on a set of politically sensitive propositions across which we expect measurable divergence.")

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


def render_inspect_tab():
    """V5 Inspect: Deep dive into individual propositions."""
    st.subheader("Inspect: Examine individual propositions and samples")

    # Dataset selector
    dataset = st.radio(
        "",
        ["Clearly False", "Well-Established", "Uncertain", "China Higher", "West Higher"],
        horizontal=True,
        key="inspect_dataset"
    )

    # Load appropriate data
    is_china = dataset in ("China Higher", "West Higher")

    if is_china:
        df = load_discriminant_data()
        directions = load_proposition_directions()
        direction_val = dataset == "China Higher"
        props_with_dir = [p for p, d in directions.items() if d == direction_val]
        df = df.filter(pl.col("proposition").is_in(props_with_dir))

        # Model filter
        all_models = df["target_model_id"].unique().sort().to_list()
        selected_models = st.multiselect("Filter models", all_models, default=all_models, key="inspect_models")
        df = df.filter(pl.col("target_model_id").is_in(selected_models))

        data_root = DISCRIMINANT_DATA_ROOT
    else:
        df = load_validation_data()
        category_map = {"Clearly False": "clearly_false", "Well-Established": "well_established", "Uncertain": "contested"}
        df = df.filter(pl.col("category") == category_map[dataset])
        data_root = DATA_ROOT

    if df.is_empty():
        st.warning("No data for selected filters.")
        return

    # Compute per-proposition stats (sorted by median descending, then by proposition for stability)
    prop_stats = df.group_by("proposition").agg([
        pl.col("consensus_credence").drop_nulls().median().alias("median"),
        pl.len().alias("n_total"),
        pl.col("consensus_credence").is_null().sum().alias("n_excluded"),
        (~pl.col("judge1_informative") | ~pl.col("judge2_informative")).sum().alias("n_uninformative"),
        (pl.col("judge1_refusal") | pl.col("judge2_refusal")).sum().alias("n_refusals"),
    ]).sort(["median", "proposition"], descending=[True, False])

    props = prop_stats["proposition"].to_list()
    medians = prop_stats["median"].to_list()

    # Build formatted labels
    prop_to_label = {}
    for p, m in zip(props, medians):
        m_str = f"{m:.2f}" if m is not None else "N/A"
        p_short = p[:55] + "..." if len(p) > 55 else p
        prop_to_label[p] = f"{m_str} - {p_short}"

    widget_key = f"v5_prop_{dataset.replace('-', '_').replace(' ', '_')}"

    selected_prop = st.selectbox(
        "Proposition (by median credence)",
        props,
        format_func=lambda p: prop_to_label[p],
        key=widget_key,
    )

    # Stats row
    st.markdown(f"**{selected_prop}**")

    if is_china:
        # Compute stats separately for Western and Chinese models
        prop_df_stats = df.filter(pl.col("proposition") == selected_prop)
        west_df = prop_df_stats.filter(pl.col("model_group") == "western")
        china_df = prop_df_stats.filter(pl.col("model_group") == "chinese")

        def get_stats(sub_df: pl.DataFrame) -> dict:
            return {
                "median": sub_df["consensus_credence"].drop_nulls().median(),
                "total": len(sub_df),
                "excluded": sub_df["consensus_credence"].is_null().sum(),
                "uninformative": (~sub_df["judge1_informative"] | ~sub_df["judge2_informative"]).sum(),
                "refusals": (sub_df["judge1_refusal"] | sub_df["judge2_refusal"]).sum(),
            }

        west_stats = get_stats(west_df)
        china_stats = get_stats(china_df)

        def fmt_split(west_val, china_val, is_float: bool = False) -> str:
            if is_float:
                w = f"{west_val:.2f}" if west_val is not None else "N/A"
                c = f"{china_val:.2f}" if china_val is not None else "N/A"
            else:
                w, c = str(west_val), str(china_val)
            return f'<span style="color:blue">{w}</span> / <span style="color:red">{c}</span>'

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.markdown("**Median**")
        c1.markdown(fmt_split(west_stats["median"], china_stats["median"], is_float=True), unsafe_allow_html=True)
        c2.markdown("**Total**")
        c2.markdown(fmt_split(west_stats["total"], china_stats["total"]), unsafe_allow_html=True)
        c3.markdown("**Excluded**")
        c3.markdown(fmt_split(west_stats["excluded"], china_stats["excluded"]), unsafe_allow_html=True)
        c4.markdown("**Uninformative**")
        c4.markdown(fmt_split(west_stats["uninformative"], china_stats["uninformative"]), unsafe_allow_html=True)
        c5.markdown("**Refusals**")
        c5.markdown(fmt_split(west_stats["refusals"], china_stats["refusals"]), unsafe_allow_html=True)
    else:
        row = prop_stats.filter(pl.col("proposition") == selected_prop).row(0, named=True)
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Median", f"{row['median']:.2f}" if row['median'] is not None else "N/A")
        c2.metric("Total", row['n_total'])
        c3.metric("Excluded", row['n_excluded'])
        c4.metric("Uninformative", row['n_uninformative'])
        c5.metric("Refusals", row['n_refusals'])

    # Distribution histogram with fixed bins: 0-0.05, 0.05-0.1, ..., 0.95-1.0
    prop_df = df.filter(pl.col("proposition") == selected_prop)
    if is_china:
        # Separate histograms for Chinese (red) and Western (blue) models
        west_credences = prop_df.filter(pl.col("model_group") == "western")["consensus_credence"].drop_nulls().to_list()
        china_credences = prop_df.filter(pl.col("model_group") == "chinese")["consensus_credence"].drop_nulls().to_list()
        if west_credences or china_credences:
            fig = go.Figure()
            if west_credences:
                fig.add_trace(go.Histogram(
                    x=west_credences,
                    xbins=dict(start=0, end=1, size=0.05),
                    name="Western",
                    marker_color="blue",
                    opacity=0.6,
                ))
            if china_credences:
                fig.add_trace(go.Histogram(
                    x=china_credences,
                    xbins=dict(start=0, end=1, size=0.05),
                    name="Chinese",
                    marker_color="red",
                    opacity=0.6,
                ))
            fig.update_layout(
                height=150,
                margin=dict(l=0, r=0, t=10, b=30),
                xaxis=dict(range=[0, 1], title="Credence"),
                yaxis=dict(title=None),
                barmode="overlay",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        prop_credences = prop_df["consensus_credence"].drop_nulls().to_list()
        if prop_credences:
            fig = go.Figure(go.Histogram(
                x=prop_credences,
                xbins=dict(start=0, end=1, size=0.05),
            ))
            fig.update_layout(
                height=120,
                margin=dict(l=0, r=0, t=10, b=30),
                xaxis=dict(range=[0, 1], title="Credence"),
                yaxis=dict(title=None),
                bargap=0.1,
            )
            st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # Load JSONL samples
    slug = slugify(selected_prop)
    if is_china:
        all_samples = []
        for model in selected_models:
            model_slug = model.replace("/", "_")
            samples = load_judged_jsonl(str(data_root), model_slug, slug)
            all_samples.extend(samples)
    else:
        all_samples = load_judged_jsonl(str(data_root), "run_0", slug)

    # Sort samples: consensus first (descending), then non-consensus
    def sample_sort_key(s: dict) -> tuple:
        cc = s.get("consensus_credence")
        if cc is not None:
            return (0, -cc)
        return (1, 0)

    all_samples_sorted = sorted(all_samples, key=sample_sort_key)

    st.markdown(f"**Samples** ({len(all_samples_sorted)} total)")

    for s in all_samples_sorted:
        cc = s.get("consensus_credence")
        is_consensus = cc is not None
        model_name = s.get("test_llm_id", "unknown").split("/")[-1]
        prompt_preview = s.get("text", "")[:80].replace("\n", " ")
        if len(s.get("text", "")) > 80:
            prompt_preview += "..."

        j1_inf = s.get("judge1_informative", True)
        j2_inf = s.get("judge2_informative", True)
        j1_c = s.get("judge1_credence")
        j2_c = s.get("judge2_credence")
        j1_ref = s.get("judge1_refusal", False)
        j2_ref = s.get("judge2_refusal", False)

        def judge_str(credence, informative: bool, refusal: bool) -> str:
            if informative and credence is not None:
                return f"{credence:.2f}"
            if refusal:
                return "Refusal"
            return "non-inf"

        header = f"**{model_name}** | "
        if is_consensus:
            header += f"{cc:.2f}"
        else:
            header += "No consensus ("
            header += f"J1={judge_str(j1_c, j1_inf, j1_ref)}, J2={judge_str(j2_c, j2_inf, j2_ref)})"
        header += f" | {prompt_preview}"

        with st.expander(header, expanded=False):
            st.markdown("**Prompt:**")
            st.code(s.get("text", "")[:1000], language=None)

            st.markdown("**Response:**")
            st.code(s.get("response_text", "")[:1500], language=None)

            j1_name = s.get("judge1_llm_id", "Judge 1").split("/")[-1]
            j2_name = s.get("judge2_llm_id", "Judge 2").split("/")[-1]

            j1_label = f"{j1_c:.2f}" if j1_c is not None else "N/A"
            if not j1_inf:
                j1_label += " (Refusal)" if j1_ref else " (non-informative)"
            st.markdown(f"**{j1_name}**: {j1_label}")
            st.text(s.get("judge1_explanation", "")[:500])

            j2_label = f"{j2_c:.2f}" if j2_c is not None else "N/A"
            if not j2_inf:
                j2_label += " (Refusal)" if j2_ref else " (non-informative)"
            st.markdown(f"**{j2_name}**: {j2_label}")
            st.text(s.get("judge2_explanation", "")[:500])


# =============================================================================
# Main
# =============================================================================

def main():
    st.set_page_config(page_title="Method Validation", layout="wide")
    st.title("Part 1: Method Validation")

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "V1 Test-Retest",
        "V2 Judge Agreement",
        "V3 Calibration",
        "V4 Known-Group",
        "Inspect",
    ])

    with tab1:
        render_test_retest_tab()

    with tab2:
        render_judge_agreement_tab()

    with tab3:
        render_calibration_tab()

    with tab4:
        render_known_group_tab()

    with tab5:
        render_inspect_tab()


if __name__ == "__main__":
    main()
