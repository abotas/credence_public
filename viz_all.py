"""Combined Streamlit app for visualizing credence experiment results.

Usage:
    uv run streamlit run viz_all.py

Part 1: Method Validation
- V1 Test-retest reliability
- V2 Judge agreement
- V3 Calibration
- V4 Known-group validity (discriminant)
- Inspect: proposition inspection

Part 2: Exploration
- What Models Believe
- User Deference
- Credence Distributions
- Model Agreement
- Prompt Attributes
"""

import gzip
import hashlib
import json
import math
import re
from pathlib import Path
from statistics import median

import numpy as np
import pandas as pd
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

# Shared
AGREEMENT_THRESHOLD = 0.2

# Prompt attribute names (from prompt_judge)
ATTRIBUTE_NAMES = [
    "user_valence",
    "sophistication",
    "apparent_stakes",
    "emotionality",
    "user_authority",
    "confrontationality",
    "clarity",
]

# Consensus threshold for prompt attribute agreement
CONSENSUS_THRESHOLD = 0.2

# Validation paths
VALIDATION_DATA_ROOT = Path(__file__).parent / "data"
DISCRIMINANT_DATA_ROOT = Path(__file__).parent / "data_discriminant"
PROPOSITIONS_CSV = Path(__file__).parent / "data" / "raw_propositions" / "curated" / "china_west_contentious_v2.csv"

# Exploration paths
EXPLORATION_PARQUET = Path(__file__).parent / "data_exploration" / "exploration.parquet"

# Validation constants
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
# Validation Data Loading (cached)
# =============================================================================

@st.cache_data
def load_validation_data() -> pl.DataFrame:
    """Load validation parquet and add consensus_credence column."""
    path = VALIDATION_DATA_ROOT / "validation.parquet"
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
# Validation Computed Stats (cached)
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
# Exploration Data Loading (cached)
# =============================================================================

def _friendly_model_name(model_id: str) -> str:
    """Convert model ID to human-readable name."""
    name = model_id.split("/")[-1]
    parts = name.split("-")
    for i, part in enumerate(parts):
        if len(part) == 4 and part.isdigit() and part.startswith("20"):
            name = "-".join(parts[:i])
            break
        if len(part) == 8 and part.isdigit() and part.startswith("20"):
            name = "-".join(parts[:i])
            break
    if name.startswith("gpt"):
        name = "GPT" + name[3:]
    elif name.startswith("claude"):
        parts = name[7:].split("-")
        model_type = parts[0].title()
        version = ".".join(parts[1:]) if len(parts) > 1 else ""
        name = f"Claude {model_type} {version}".strip()
    return name


@st.cache_data
def load_exploration_parquet() -> pl.DataFrame:
    """Load exploration data from parquet file (cached)."""
    if not EXPLORATION_PARQUET.exists():
        return pl.DataFrame()
    return pl.read_parquet(EXPLORATION_PARQUET)


@st.cache_data
def add_computed_columns(_df: pl.DataFrame) -> pl.DataFrame:
    """Add consensus columns for credences and prompt attributes."""
    if _df.is_empty():
        return _df

    df = _df

    # Add credence consensus
    df = df.with_columns([
        pl.when(
            pl.col("judge1_informative") & pl.col("judge2_informative") &
            pl.col("judge1_credence").is_not_null() & pl.col("judge2_credence").is_not_null() &
            ((pl.col("judge1_credence") - pl.col("judge2_credence")).abs() <= AGREEMENT_THRESHOLD + 1e-9)
        ).then(
            (pl.col("judge1_credence") + pl.col("judge2_credence")) / 2
        ).otherwise(None).alias("consensus_credence")
    ])

    # Add prompt attribute consensus values (average if within threshold)
    for attr in ATTRIBUTE_NAMES:
        j1_col = f"prompt_judge1_{attr}"
        j2_col = f"prompt_judge2_{attr}"
        if j1_col in df.columns and j2_col in df.columns:
            df = df.with_columns([
                pl.when(
                    pl.col(j1_col).is_not_null() & pl.col(j2_col).is_not_null() &
                    ((pl.col(j1_col) - pl.col(j2_col)).abs() <= CONSENSUS_THRESHOLD + 1e-9)
                ).then(
                    (pl.col(j1_col) + pl.col(j2_col)) / 2
                ).otherwise(None).alias(f"consensus_{attr}")
            ])

    return df


# =============================================================================
# Exploration Helper Functions
# =============================================================================


EXPLORATION_DATA_ROOT = Path(__file__).parent / "data_exploration"


def _load_exploration_samples(proposition: str) -> list[dict]:
    """Load all judged samples for a proposition across all models."""
    slug = slugify(proposition)
    all_samples = []

    for model_dir in EXPLORATION_DATA_ROOT.iterdir():
        if not model_dir.is_dir() or model_dir.name in ("prompts", "config.json"):
            continue
        # Look for gzipped jsonl files
        jsonl_path = model_dir / slug / "judged.jsonl.gz"
        if jsonl_path.exists():
            with gzip.open(jsonl_path, "rt") as f:
                for line in f:
                    all_samples.append(json.loads(line))

    return all_samples


def normalize_model_id(model_id: str) -> str:
    """Normalize model IDs (parquet uses gpt-5-2, jsonl uses gpt-5.2)."""
    return model_id.replace("gpt-5.2", "gpt-5-2")


def sig_stars(p: float) -> str:
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def fisher_z_ci(r: float, n: int) -> tuple[float, float]:
    if n < 4:
        return (r, r)
    z = 0.5 * np.log((1 + r) / (1 - r + 1e-10))
    se_z = 1 / np.sqrt(n - 3)
    z_lower = z - 1.96 * se_z
    z_upper = z + 1.96 * se_z
    return np.tanh(z_lower), np.tanh(z_upper)


def bootstrap_corr_ci(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    rng = np.random.default_rng(42)
    n = len(x)
    boot_rs = []
    for _ in range(200):
        idx = rng.choice(n, size=n, replace=True)
        boot_rs.append(spearmanr(x[idx], y[idx])[0])
    return np.percentile(boot_rs, 2.5), np.percentile(boot_rs, 97.5)


def format_shift(shift: float) -> str:
    if abs(shift) < 0.005:
        return ""
    color = "#2ca02c" if shift > 0 else "#d62728"  # green/red
    return f' <span style="color:{color}">({shift:+.2f})</span>'


ATTR_DISPLAY = {
    "user_valence": "User Valence",
    "sophistication": "Sophistication",
    "apparent_stakes": "Stakes",
    "emotionality": "Emotionality",
    "user_authority": "Authority",
    "confrontationality": "Confrontationality",
    "clarity": "Clarity",
}


# =============================================================================
# Validation Render Functions
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
        "Dataset",
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
        data_root = VALIDATION_DATA_ROOT

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
# Exploration Render Functions
# =============================================================================

def render_prompt_sensitivity_tab(df: pl.DataFrame):
    """Render prompt sensitivity tab - user valence correlation and distributions."""
    if df.is_empty():
        st.warning("No data found.")
        return

    if "consensus_user_valence" not in df.columns:
        st.warning("Prompt attributes not yet scored.")
        return

    # Domain filter
    st.markdown("**Domain filter**")
    all_domains = sorted(df["domain"].unique().to_list())
    selected_domains = st.multiselect(
        "Domain filter",
        all_domains,
        default=all_domains,
        key="sensitivity_domains",
        label_visibility="collapsed",
    )

    if not selected_domains:
        st.info("Select at least one domain.")
        return

    filtered_df = df.filter(pl.col("domain").is_in(selected_domains))
    selected_models = sorted(filtered_df["target_model_id"].unique().to_list())

    # Calculate % of neutral prompts
    total_prompts = len(filtered_df.filter(pl.col("consensus_user_valence").is_not_null()))
    neutral_prompts = len(filtered_df.filter(
        (pl.col("consensus_user_valence").is_not_null()) &
        ((pl.col("consensus_user_valence") - 0.5).abs() < 1e-9)
    ))
    neutral_pct = neutral_prompts / total_prompts * 100 if total_prompts > 0 else 0

    # Checkbox value (widget rendered later, but value needed now)
    discard_neutral = st.session_state.get("sensitivity_discard_neutral", True)
    baseline_df = filtered_df
    if discard_neutral:
        baseline_df = baseline_df.filter((pl.col("consensus_user_valence") - 0.5).abs() > 1e-9)

    # --- Section 1: Correlation Forest Plot ---
    st.markdown("**User Valence vs Credence Correlation**")
    st.caption("Each prompt has a 'user valence' score (0-1) indicating whether the prompt suggests the user doubts (0) or believes (1) the proposition. This is correlated with the model's credence (also 0-1). A positive correlation means the model tends to express higher credence when the user seems to believe the proposition, suggesting sensitivity to user framing.")

    # Compute correlation for each model
    results = []
    for model_id in selected_models:
        model_data = baseline_df.filter(pl.col("target_model_id") == model_id)
        valid_data = model_data.select(["consensus_credence", "consensus_user_valence"]).drop_nulls().to_pandas()

        if len(valid_data) < 10:
            continue

        x, y = valid_data["consensus_user_valence"].values, valid_data["consensus_credence"].values
        r, p = pearsonr(x, y)
        ci = fisher_z_ci(r, len(x))

        results.append({
            "model_id": model_id,
            "model": _friendly_model_name(model_id),
            "r": r,
            "p": p,
            "ci_low": ci[0],
            "ci_high": ci[1],
        })

    if results:
        results_df = pd.DataFrame(results)

        # Custom model order
        model_order = [
            "openai/gpt-4o-2024-11-20",
            "openai/gpt-5-nano-2025-08-07",
            "openai/gpt-5-mini-2025-08-07",
            "openai/gpt-5-2-chat-latest",
            "openai/gpt-5-2-2025-12-11",
            "anthropic/claude-haiku-4-5-20251001",
            "anthropic/claude-sonnet-4-5-20250929",
        ]
        model_rank = {m: i for i, m in enumerate(model_order)}
        results_df["_sort_key"] = results_df["model_id"].map(lambda x: model_rank.get(x, 999))
        results_df = results_df.sort_values("_sort_key").drop(columns=["_sort_key"])

        fig_corr = go.Figure()
        for _, row in results_df.iterrows():
            if row["model_id"].startswith("openai/"):
                color = "#10a37f"
            elif row["model_id"].startswith("anthropic/"):
                color = "#d97757"
            else:
                color = "#1f77b4"

            sig = sig_stars(row["p"])
            fig_corr.add_trace(go.Scatter(
                x=[row["r"]],
                y=[row["model"]],
                mode="markers",
                marker=dict(size=10, color=color),
                error_x=dict(
                    type="data",
                    symmetric=False,
                    array=[row["ci_high"] - row["r"]],
                    arrayminus=[row["r"] - row["ci_low"]],
                    thickness=2,
                    width=6,
                    color=color,
                ),
                hovertemplate=f"r = {row['r']:.3f}{sig}<br>95% CI: [{row['ci_low']:.3f}, {row['ci_high']:.3f}]<extra></extra>",
                showlegend=False,
            ))

        fig_corr.update_layout(
            height=max(150, len(results_df) * 30),
            margin=dict(l=10, r=10, t=10, b=55),
            xaxis=dict(title="Pearson r", range=[0, 1]),
            yaxis=dict(title=""),
            annotations=[
                dict(
                    text="Points = correlation coefficient. Whiskers = 95% CI.",
                    xref="paper", yref="paper",
                    x=0, y=-0.35,
                    xanchor="left", yanchor="top",
                    showarrow=False,
                    font=dict(size=10, color="#666"),
                )
            ],
        )
        st.plotly_chart(fig_corr, use_container_width=True)
        st.checkbox(
            f"Discard neutral user valence (0.5) ({neutral_pct:.1f}% of prompts)",
            value=True,
            key="sensitivity_discard_neutral",
        )

    # --- Section 2: Distribution Histograms ---
    st.markdown("**Credence Distributions**")
    st.caption("Use the slider to filter prompts by user valence range. Compare how credence distributions shift when prompts imply the user doubts (low valence) vs believes (high valence) the proposition.")
    valence_range = st.slider(
        "User Valence",
        min_value=0.0,
        max_value=1.0,
        value=(0.0, 1.0),
        step=0.05,
        key="sensitivity_valence",
    )

    # Filter by valence range
    attr_filtered_df = baseline_df
    if valence_range[0] > 0.0 or valence_range[1] < 1.0:
        attr_filtered_df = attr_filtered_df.filter(
            (pl.col("consensus_user_valence") >= valence_range[0] - 1e-9) &
            (pl.col("consensus_user_valence") <= valence_range[1] + 1e-9)
        )

    cols = st.columns(2)
    for i, model_id in enumerate(selected_models):
        baseline_model_df = baseline_df.filter(pl.col("target_model_id") == model_id)
        baseline_credences = baseline_model_df["consensus_credence"].drop_nulls().to_list()

        model_df = attr_filtered_df.filter(pl.col("target_model_id") == model_id)
        credences = model_df["consensus_credence"].drop_nulls().to_list()
        model_name = _friendly_model_name(model_id)

        if model_id.startswith("openai/"):
            bar_color = "#10a37f"
        elif model_id.startswith("anthropic/"):
            bar_color = "#d97757"
        else:
            bar_color = "#1f77b4"

        n_baseline = len(baseline_credences)
        n_filtered = len(credences)

        baseline_mean = sum(baseline_credences) / len(baseline_credences) if baseline_credences else 0
        baseline_median = median(baseline_credences) if baseline_credences else 0

        fig = go.Figure()

        if credences:
            mean_cred = sum(credences) / len(credences)
            median_cred = median(credences)
            mean_shift = mean_cred - baseline_mean
            median_shift = median_cred - baseline_median

            fig.add_trace(go.Histogram(
                x=credences,
                xbins=dict(start=0, end=1, size=0.1),
                histnorm="percent",
                marker=dict(color=bar_color, opacity=0.6),
                name="Filtered",
                showlegend=False,
            ))

            legend_text = (
                f"n={n_filtered}/{n_baseline}<br>"
                f"mean={mean_cred:.2f}{format_shift(mean_shift)}<br>"
                f"median={median_cred:.2f}{format_shift(median_shift)}"
            )
        else:
            legend_text = f"n=0/{n_baseline}"

        if baseline_credences:
            fig.add_trace(go.Histogram(
                x=baseline_credences,
                xbins=dict(start=0, end=1, size=0.1),
                histnorm="percent",
                marker=dict(color="rgba(0,0,0,0)", line=dict(color="#888888", width=1.5)),
                name="All",
                showlegend=False,
            ))

        fig.update_layout(
            height=160,
            title=model_name,
            title_font_size=12,
            xaxis=dict(range=[0, 1], title="Credence", title_font_size=10),
            yaxis=dict(title="%", title_font_size=10),
            margin=dict(l=40, r=90, t=35, b=30),
            bargap=0.1,
            barmode="overlay",
            annotations=[
                dict(
                    text=legend_text,
                    xref="paper", yref="paper",
                    x=1.02, y=0.95,
                    xanchor="left", yanchor="top",
                    showarrow=False,
                    font=dict(size=10),
                    align="left",
                )
            ],
        )

        with cols[i % 2]:
            st.plotly_chart(fig, use_container_width=True)

    # --- Section 3: User Valence Distribution ---
    st.markdown("**User Valence Distribution**")
    valence_values = baseline_df["consensus_user_valence"].drop_nulls().to_list()
    if valence_values:
        # 20 bins: [0-0.05), [0.05-0.10), ..., [0.95-1.0]
        bin_edges = [i * 0.05 for i in range(21)]  # 0.0, 0.05, 0.10, ..., 1.0
        counts, _ = np.histogram(valence_values, bins=bin_edges)
        pcts = [100 * c / len(valence_values) for c in counts]
        bin_centers = [(bin_edges[i] + bin_edges[i + 1]) / 2 for i in range(20)]

        fig_valence = go.Figure(go.Bar(
            x=bin_centers,
            y=pcts,
            width=0.045,
            marker_color="#636EFA",
        ))
        fig_valence.update_layout(
            height=120,
            margin=dict(l=40, r=10, t=10, b=30),
            xaxis=dict(range=[0, 1], title="User Valence", dtick=0.1),
            yaxis=dict(title="%"),
            bargap=0.1,
        )
        st.plotly_chart(fig_valence, use_container_width=True)


def render_dispersion_tab(df: pl.DataFrame):
    """Render what models believe tab - overview/deep dive toggle."""
    if df.is_empty():
        st.warning("No data found.")
        return

    # Header with segmented control
    col_title, col_toggle = st.columns([3, 2])
    with col_title:
        st.subheader("What Models Believe")
    with col_toggle:
        view_mode = st.segmented_control(
            "View",
            ["Overview", "Deep Dive"],
            default="Overview",
            key="beliefs_view_mode",
            label_visibility="collapsed",
        )

    if view_mode == "Overview":
        _render_beliefs_overview(df)
    else:
        _render_beliefs_deep_dive(df)


def _render_beliefs_overview(df: pl.DataFrame):
    """Render extremity and dispersion overview charts."""
    # Model ordering - OpenAI together, then Anthropic
    model_order = [
        "openai/gpt-4o-2024-11-20",
        "openai/gpt-5-nano-2025-08-07",
        "openai/gpt-5-mini-2025-08-07",
        "openai/gpt-5-2-2025-12-11",
        "openai/gpt-5-2-chat-latest",
        "openai/gpt-5.2-2025-12-11",
        "openai/gpt-5.2-chat-latest",
        "anthropic/claude-haiku-4-5-20251001",
        "anthropic/claude-sonnet-4-5-20250929",
    ]
    model_rank = {m: i for i, m in enumerate(model_order)}

    domain_colors = {
        "contested_social_science": "#e377c2",  # pink
        "frontier_natural_science": "#2ca02c",  # green
        "prediction_market": "#ff7f0e",  # orange
        "ai_claims": "#9467bd",  # purple
        "historical_facts": "#8c564b",  # brown
        "moral_claims": "#d62728",  # red
        "nutrition_health": "#17becf",  # cyan
        "paranormal_claims": "#7f7f7f",  # gray
        "philosophical_propositions": "#bcbd22",  # olive
        "politically_polarizing": "#1f77b4",  # blue
    }

    # ===========================================
    # EXTREMITY: mean |credence - 0.5| per proposition, then aggregated
    # ===========================================
    st.markdown("**Extremity** — How far from 0.5 are model credences? Higher values indicate stronger positions; lower values indicate more hedging.")

    # Compute mean credence per (model, proposition, domain), then extremity
    extremity_data = df.filter(
        pl.col("consensus_credence").is_not_null()
    ).group_by(["target_model_id", "proposition", "domain"]).agg([
        pl.col("consensus_credence").mean().alias("mean_credence"),
    ])

    if extremity_data.is_empty():
        st.warning("No data for extremity calculation.")
    else:
        extremity_pdf = extremity_data.to_pandas()
        extremity_pdf["extremity"] = (extremity_pdf["mean_credence"] - 0.5).abs()
        extremity_pdf["model"] = extremity_pdf["target_model_id"].apply(_friendly_model_name)
        extremity_pdf["_model_rank"] = extremity_pdf["target_model_id"].map(lambda x: model_rank.get(x, 999))

        col1, col2 = st.columns(2)

        # --- Extremity by Model ---
        with col1:
            st.markdown("### By Model")

            ext_model_stats = extremity_pdf.groupby(["target_model_id", "model", "_model_rank"])["extremity"].mean().reset_index()
            ext_model_stats.columns = ["target_model_id", "model", "_model_rank", "mean_extremity"]
            ext_model_stats = ext_model_stats.sort_values("_model_rank", ascending=False)

            ext_model_stats["color"] = ext_model_stats["target_model_id"].apply(
                lambda x: "#10a37f" if x.startswith("openai/") else "#d97757" if x.startswith("anthropic/") else "#1f77b4"
            )

            fig_ext_model = go.Figure()
            fig_ext_model.add_trace(go.Bar(
                y=ext_model_stats["model"],
                x=ext_model_stats["mean_extremity"],
                orientation="h",
                marker_color=ext_model_stats["color"],
                text=[f"{v:.2f}" for v in ext_model_stats["mean_extremity"]],
                textposition="outside",
            ))
            fig_ext_model.update_layout(
                height=max(250, len(ext_model_stats) * 35),
                margin=dict(l=10, r=40, t=10, b=40),
                xaxis=dict(title="Mean Extremity", range=[0, ext_model_stats["mean_extremity"].max() * 1.25]),
                yaxis=dict(title=""),
            )
            st.plotly_chart(fig_ext_model, use_container_width=True)

        # --- Extremity by Domain ---
        with col2:
            st.markdown("### By Domain")

            ext_domain_stats = extremity_pdf.groupby("domain")["extremity"].mean().reset_index()
            ext_domain_stats.columns = ["domain", "mean_extremity"]
            ext_domain_stats = ext_domain_stats.sort_values("mean_extremity", ascending=True)
            ext_domain_stats["domain_display"] = ext_domain_stats["domain"].apply(lambda x: x.replace("_", " ").title())
            ext_domain_stats["color"] = ext_domain_stats["domain"].apply(lambda x: domain_colors.get(x, "#636EFA"))

            fig_ext_domain = go.Figure()
            fig_ext_domain.add_trace(go.Bar(
                y=ext_domain_stats["domain_display"],
                x=ext_domain_stats["mean_extremity"],
                orientation="h",
                marker_color=ext_domain_stats["color"],
                text=[f"{v:.2f}" for v in ext_domain_stats["mean_extremity"]],
                textposition="outside",
            ))
            fig_ext_domain.update_layout(
                height=max(250, len(ext_domain_stats) * 35),
                margin=dict(l=10, r=40, t=10, b=40),
                xaxis=dict(title="Mean Extremity", range=[0, ext_domain_stats["mean_extremity"].max() * 1.25]),
                yaxis=dict(title=""),
            )
            st.plotly_chart(fig_ext_domain, use_container_width=True)

    # ===========================================
    # DISPERSION: IQR across prompts for same proposition
    # ===========================================
    st.markdown("**Dispersion** — How much does credence vary across prompts for the same proposition? Higher IQR indicates more sensitivity to prompt framing.")

    # Compute IQR for each (model, proposition) pair
    iqr_data = df.filter(
        pl.col("consensus_credence").is_not_null()
    ).group_by(["target_model_id", "proposition", "domain"]).agg([
        (pl.col("consensus_credence").quantile(0.75) - pl.col("consensus_credence").quantile(0.25)).alias("iqr"),
        pl.col("consensus_credence").len().alias("n_samples"),
    ]).filter(pl.col("n_samples") >= 5)  # Need enough samples for meaningful IQR

    if iqr_data.is_empty():
        st.warning("Not enough data per proposition to compute IQR.")
        return

    iqr_pdf = iqr_data.to_pandas()
    iqr_pdf["model"] = iqr_pdf["target_model_id"].apply(_friendly_model_name)
    iqr_pdf["_model_rank"] = iqr_pdf["target_model_id"].map(lambda x: model_rank.get(x, 999))

    col1, col2 = st.columns(2)

    # --- By Model: Horizontal bar chart ---
    with col1:
        st.markdown("### By Model")

        model_stats = iqr_pdf.groupby(["target_model_id", "model", "_model_rank"])["iqr"].mean().reset_index()
        model_stats.columns = ["target_model_id", "model", "_model_rank", "mean_iqr"]
        model_stats = model_stats.sort_values("_model_rank", ascending=False)  # Reverse for horizontal

        # Assign colors by provider
        model_stats["color"] = model_stats["target_model_id"].apply(
            lambda x: "#10a37f" if x.startswith("openai/") else "#d97757" if x.startswith("anthropic/") else "#1f77b4"
        )

        fig_model = go.Figure()
        fig_model.add_trace(go.Bar(
            y=model_stats["model"],
            x=model_stats["mean_iqr"],
            orientation="h",
            marker_color=model_stats["color"],
            text=[f"{v:.2f}" for v in model_stats["mean_iqr"]],
            textposition="outside",
        ))
        fig_model.update_layout(
            height=max(250, len(model_stats) * 35),
            margin=dict(l=10, r=40, t=10, b=40),
            xaxis=dict(title="Mean IQR", range=[0, model_stats["mean_iqr"].max() * 1.25]),
            yaxis=dict(title=""),
        )
        st.plotly_chart(fig_model, use_container_width=True)

    # --- By Domain: Horizontal bar chart ---
    with col2:
        st.markdown("### By Domain")

        domain_stats = iqr_pdf.groupby("domain")["iqr"].mean().reset_index()
        domain_stats.columns = ["domain", "mean_iqr"]
        domain_stats = domain_stats.sort_values("mean_iqr", ascending=True)  # Ascending for horizontal
        domain_stats["domain_display"] = domain_stats["domain"].apply(lambda x: x.replace("_", " ").title())
        domain_stats["color"] = domain_stats["domain"].apply(lambda x: domain_colors.get(x, "#636EFA"))

        fig_domain = go.Figure()
        fig_domain.add_trace(go.Bar(
            y=domain_stats["domain_display"],
            x=domain_stats["mean_iqr"],
            orientation="h",
            marker_color=domain_stats["color"],
            text=[f"{v:.2f}" for v in domain_stats["mean_iqr"]],
            textposition="outside",
        ))
        fig_domain.update_layout(
            height=max(250, len(domain_stats) * 35),
            margin=dict(l=10, r=40, t=10, b=40),
            xaxis=dict(title="Mean IQR", range=[0, domain_stats["mean_iqr"].max() * 1.25]),
            yaxis=dict(title=""),
        )
        st.plotly_chart(fig_domain, use_container_width=True)


def _render_beliefs_deep_dive(df: pl.DataFrame):
    """Render proposition-level inspection view."""
    # Domain selector
    all_domains = sorted(df["domain"].unique().to_list())
    selected_domain = st.selectbox("Domain", all_domains, key="beliefs_domain")
    domain_df = df.filter(pl.col("domain") == selected_domain)

    # Model selector - get available models from the data
    all_models = sorted(domain_df["target_model_id"].unique().to_list())
    all_model_names = [_friendly_model_name(m) for m in all_models]
    name_to_id = {_friendly_model_name(m): m for m in all_models}

    selected_model_names = st.multiselect(
        "Models",
        all_model_names,
        default=all_model_names,
        key="beliefs_models",
    )
    selected_models = [name_to_id[n] for n in selected_model_names]

    # Filter by selected models
    if selected_models:
        domain_df = domain_df.filter(pl.col("target_model_id").is_in(selected_models))

    # Compute per-proposition stats, sorted by median credence
    prop_stats = domain_df.group_by("proposition").agg([
        pl.col("consensus_credence").drop_nulls().median().alias("median"),
        pl.len().alias("n_total"),
        pl.col("consensus_credence").is_null().sum().alias("n_excluded"),
    ]).sort(["median", "proposition"], descending=[True, False])

    if prop_stats.is_empty():
        st.warning("No propositions match the selected filters.")
        return

    # Proposition selector with median in label
    props = prop_stats["proposition"].to_list()
    medians = prop_stats["median"].to_list()

    def format_prop_label(p: str) -> str:
        idx = props.index(p)
        m = medians[idx]
        prefix = f"{m:.2f}" if m is not None else "N/A"
        truncated = p[:55] + "..." if len(p) > 55 else p
        return f"{prefix} - {truncated}"

    selected_prop = st.selectbox("Proposition", props, format_func=format_prop_label, key="beliefs_prop")

    # Show full proposition text
    st.caption(selected_prop)

    # Histogram - colored by model, with median in legend
    prop_df = domain_df.filter(pl.col("proposition") == selected_prop)

    # Model colors - use plotly's qualitative palette for distinct colors
    color_palette = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    ]
    model_colors = {m: color_palette[i % len(color_palette)] for i, m in enumerate(selected_models)}

    fig = go.Figure()
    for model_id in selected_models:
        model_credences = prop_df.filter(
            pl.col("target_model_id") == model_id
        )["consensus_credence"].drop_nulls().to_list()

        if model_credences:
            model_median = median(model_credences)
            legend_name = f"{_friendly_model_name(model_id)} (med: {model_median:.2f})"
            fig.add_trace(go.Histogram(
                x=model_credences,
                xbins=dict(start=0, end=1, size=0.05),
                name=legend_name,
                marker_color=model_colors[model_id],
                opacity=0.7,
            ))

    fig.update_layout(
        height=180,
        margin=dict(l=0, r=0, t=10, b=30),
        xaxis=dict(range=[0, 1], title="Credence"),
        yaxis=dict(title=None),
        barmode="overlay",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # Load and display samples
    all_samples = _load_exploration_samples(selected_prop)

    # Filter by selected models (with normalization)
    selected_models_set = set(selected_models)
    all_samples = [s for s in all_samples if normalize_model_id(s.get("test_llm_id", "")) in selected_models_set]

    # Categorize samples
    def categorize_sample(s: dict) -> str:
        if s.get("consensus_credence") is not None:
            return "consensus"
        j1_ref = s.get("judge1_refusal", False)
        j2_ref = s.get("judge2_refusal", False)
        if j1_ref or j2_ref:
            return "refusal"
        return "non_informative"

    # Compute counts by model
    model_counts = {m: {"consensus": 0, "non_informative": 0, "refusal": 0} for m in selected_models}
    for s in all_samples:
        model_id = normalize_model_id(s.get("test_llm_id", ""))
        if model_id in model_counts:
            model_counts[model_id][categorize_sample(s)] += 1

    # Bar charts by model - percentages
    models_display = [_friendly_model_name(m) for m in selected_models]

    def pct(count: int, total: int) -> float:
        return 100 * count / total if total > 0 else 0

    model_totals = {m: sum(model_counts[m].values()) for m in selected_models}
    consensus_pcts = [pct(model_counts[m]["consensus"], model_totals[m]) for m in selected_models]
    non_inf_pcts = [pct(model_counts[m]["non_informative"], model_totals[m]) for m in selected_models]
    refusal_pcts = [pct(model_counts[m]["refusal"], model_totals[m]) for m in selected_models]

    fig_counts = go.Figure()
    fig_counts.add_trace(go.Bar(
        name="Consensus",
        x=models_display,
        y=consensus_pcts,
        marker_color="#2ca02c",
        text=[f"{v:.0f}%" for v in consensus_pcts],
        textposition="auto",
    ))
    fig_counts.add_trace(go.Bar(
        name="Non-informative",
        x=models_display,
        y=non_inf_pcts,
        marker_color="#ff7f0e",
        text=[f"{v:.0f}%" for v in non_inf_pcts],
        textposition="auto",
    ))
    fig_counts.add_trace(go.Bar(
        name="Refusal",
        x=models_display,
        y=refusal_pcts,
        marker_color="#d62728",
        text=[f"{v:.0f}%" for v in refusal_pcts],
        textposition="auto",
    ))
    fig_counts.update_layout(
        height=150,
        margin=dict(l=0, r=0, t=10, b=30),
        barmode="group",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        yaxis=dict(title="%", range=[0, 105]),
    )
    st.plotly_chart(fig_counts, use_container_width=True)

    st.divider()

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


def render_correlations_detail_tab(df: pl.DataFrame):
    """Render correlations tab - baseline user deference only."""

    if df.is_empty():
        st.warning("No data found.")
        return

    if "consensus_user_valence" not in df.columns:
        st.warning("Prompt attributes not yet scored.")
        return

    # Get filter options
    all_domains = sorted(df["domain"].unique().to_list()) if "domain" in df.columns else []
    all_models_list = sorted(df["target_model_id"].unique().to_list())
    all_model_names = [_friendly_model_name(m) for m in all_models_list]
    name_to_id = {_friendly_model_name(m): m for m in all_models_list}

    with st.expander("Filters", expanded=False):
        col1, col2, col3 = st.columns(3)
        with col1:
            selected_domains = st.multiselect(
                "Domains",
                all_domains,
                default=all_domains,
                key="detail_domains",
            )
        with col2:
            selected_model_names = st.multiselect(
                "Models",
                all_model_names,
                default=all_model_names,
                key="detail_models",
            )
        with col3:
            corr_type = st.radio(
                "Correlation type",
                ["Pearson", "Spearman"],
                index=0,
                key="detail_corr_type",
            )

    use_spearman = corr_type == "Spearman"
    corr_func = spearmanr if use_spearman else pearsonr
    corr_label = "Spearman r" if use_spearman else "Pearson r"

    selected_models = [name_to_id[name] for name in selected_model_names if name in name_to_id]

    filtered_df = df
    if selected_domains:
        filtered_df = filtered_df.filter(pl.col("domain").is_in(selected_domains))
    if selected_models:
        filtered_df = filtered_df.filter(pl.col("target_model_id").is_in(selected_models))

    if filtered_df.is_empty():
        st.warning("No data matches the selected filters.")
        return

    all_models = sorted(filtered_df["target_model_id"].unique().to_list())

    # Compute valence-credence correlation for each model
    results = []
    for model_id in all_models:
        model_data = filtered_df.filter(pl.col("target_model_id") == model_id)
        valid = model_data.select(["consensus_credence", "consensus_user_valence"]).drop_nulls().to_pandas()

        # Directional only (exclude valence == 0.5)
        dir_data = model_data.filter(
            (pl.col("consensus_user_valence") - 0.5).abs() > 1e-9
        ).select(["consensus_credence", "consensus_user_valence"]).drop_nulls().to_pandas()

        if len(valid) < 10 or len(dir_data) < 10:
            continue

        row = {"model_id": model_id, "model": _friendly_model_name(model_id)}

        # All prompts
        x_all, y_all = valid["consensus_user_valence"].values, valid["consensus_credence"].values
        r_all, p_all = corr_func(x_all, y_all)
        ci_all = bootstrap_corr_ci(x_all, y_all) if use_spearman else fisher_z_ci(r_all, len(x_all))
        row.update({"valence_r_all": r_all, "valence_p_all": p_all, "valence_ci_low_all": ci_all[0], "valence_ci_high_all": ci_all[1], "n_all": len(valid)})

        # Directional only
        x_dir, y_dir = dir_data["consensus_user_valence"].values, dir_data["consensus_credence"].values
        r_dir, p_dir = corr_func(x_dir, y_dir)
        ci_dir = bootstrap_corr_ci(x_dir, y_dir) if use_spearman else fisher_z_ci(r_dir, len(x_dir))
        row.update({"valence_r_dir": r_dir, "valence_p_dir": p_dir, "valence_ci_low_dir": ci_dir[0], "valence_ci_high_dir": ci_dir[1], "n_dir": len(dir_data)})

        results.append(row)

    if not results:
        st.warning("Not enough data for analysis.")
        return

    results_df = pd.DataFrame(results)

    # Custom model order
    model_order = [
        "openai/gpt-4o-2024-11-20",
        "openai/gpt-5-nano-2025-08-07",
        "openai/gpt-5-mini-2025-08-07",
        "openai/gpt-5-2-chat-latest",
        "openai/gpt-5-2-2025-12-11",
        "anthropic/claude-haiku-4-5-20251001",
        "anthropic/claude-sonnet-4-5-20250929",
    ]
    model_rank = {m: i for i, m in enumerate(model_order)}
    results_df["_sort_key"] = results_df["model_id"].map(lambda x: model_rank.get(x, 999))
    results_df = results_df.sort_values("_sort_key").drop(columns=["_sort_key"])

    # Calculate % of prompts with exactly v==0.5
    total_prompts = len(filtered_df.filter(pl.col("consensus_user_valence").is_not_null()))
    neutral_prompts = len(filtered_df.filter(
        (pl.col("consensus_user_valence").is_not_null()) &
        ((pl.col("consensus_user_valence") - 0.5).abs() < 1e-9)
    ))
    neutral_pct = neutral_prompts / total_prompts * 100 if total_prompts > 0 else 0

    st.subheader("Does the user's apparent belief affect the model's answer?")
    st.markdown("""
    We score each prompt on "user valence" - does the prompt suggest the user believes (1) or doubts (0) the proposition?
    A positive correlation between user valence and model credence indicates the model shifts its credence toward the user's implied position.
    """)

    discard_neutral_s1 = st.checkbox(
        f"Discard samples where prompt user valence is neutral, v==0.5 ({neutral_pct:.1f}% of prompts)",
        value=True,
        key="discard_neutral_s1",
    )
    s1_suffix = "_dir" if discard_neutral_s1 else "_all"

    # Forest plot with model colors
    plot_data = results_df[["model_id", "model", f"valence_r{s1_suffix}", f"valence_p{s1_suffix}", f"valence_ci_low{s1_suffix}", f"valence_ci_high{s1_suffix}"]].copy()
    plot_data.columns = ["model_id", "model", "r", "p", "ci_low", "ci_high"]

    fig = go.Figure()

    for _, row in plot_data.iterrows():
        # Color by provider
        if row["model_id"].startswith("openai/"):
            color = "#10a37f"  # OpenAI teal
        elif row["model_id"].startswith("anthropic/"):
            color = "#d97757"  # Anthropic coral
        else:
            color = "#1f77b4"

        sig = sig_stars(row["p"])
        fig.add_trace(go.Scatter(
            x=[row["r"]],
            y=[row["model"]],
            mode="markers",
            marker=dict(size=10, color=color),
            error_x=dict(
                type="data",
                symmetric=False,
                array=[row["ci_high"] - row["r"]],
                arrayminus=[row["r"] - row["ci_low"]],
                thickness=2,
                width=6,
                color=color,
            ),
            hovertemplate=f"r = {row['r']:.3f}{sig}<br>95% CI: [{row['ci_low']:.3f}, {row['ci_high']:.3f}]<extra></extra>",
            showlegend=False,
        ))

    fig.update_layout(
        height=max(200, len(plot_data) * 35),
        margin=dict(l=10, r=10, t=10, b=40),
        xaxis=dict(title=corr_label, range=[0, 1]),
        yaxis=dict(title=""),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Points = correlation coefficient. Whiskers = 95% CI.")


def render_explorer_tab(df: pl.DataFrame):
    """Render interactive explorer tab with filters and histogram."""
    st.subheader("Distributions Explorer")
    st.caption("Filter samples by model, domain, and prompt attributes to explore credence distributions.")

    if df.is_empty():
        st.warning("No data found.")
        return

    # Check for required columns
    has_attrs = "consensus_user_valence" in df.columns
    if not has_attrs:
        st.warning("Prompt attributes not yet scored. Run run_exploration.py first.")
        return

    # Get filter options
    all_domains = sorted(df["domain"].unique().to_list()) if "domain" in df.columns else []
    all_models_list = sorted(df["target_model_id"].unique().to_list())
    all_model_names = [_friendly_model_name(m) for m in all_models_list]
    name_to_id = {_friendly_model_name(m): m for m in all_models_list}

    # Model and domain filters (collapsible)
    with st.expander("Filters", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            selected_model_names = st.multiselect(
                "Models",
                all_model_names,
                default=all_model_names,
                key="explorer_models",
            )
        with col2:
            selected_domains = st.multiselect(
                "Domains",
                all_domains,
                default=all_domains,
                key="explorer_domains",
            )

    selected_models = [name_to_id[name] for name in selected_model_names if name in name_to_id]

    # Attribute filter selection
    all_attr_options = [ATTR_DISPLAY[attr] for attr in ATTRIBUTE_NAMES]
    display_to_attr = {ATTR_DISPLAY[attr]: attr for attr in ATTRIBUTE_NAMES}

    selected_attr_names = st.multiselect(
        "Prompt Attribute Filters",
        all_attr_options,
        default=["User Valence"],
        key="explorer_attr_select",
    )
    selected_attrs = [display_to_attr[name] for name in selected_attr_names]

    # Show sliders for selected attributes
    attr_ranges = {}
    if selected_attrs:
        n_attrs = len(selected_attrs)
        cols = st.columns(min(n_attrs, 4))
        for i, attr in enumerate(selected_attrs):
            with cols[i % min(n_attrs, 4)]:
                attr_ranges[attr] = st.slider(
                    ATTR_DISPLAY[attr],
                    min_value=0.0,
                    max_value=1.0,
                    value=(0.0, 1.0),
                    step=0.05,
                    key=f"explorer_{attr}",
                )

    # Apply model/domain filters (baseline for histogram outline)
    baseline_df = df
    if selected_models:
        baseline_df = baseline_df.filter(pl.col("target_model_id").is_in(selected_models))
    if selected_domains:
        baseline_df = baseline_df.filter(pl.col("domain").is_in(selected_domains))

    # Apply attribute range filters
    filtered_df = baseline_df
    for attr, (low, high) in attr_ranges.items():
        consensus_col = f"consensus_{attr}"
        if consensus_col in filtered_df.columns:
            # Only filter if range is not full [0, 1]
            if low > 0.0 or high < 1.0:
                filtered_df = filtered_df.filter(
                    (pl.col(consensus_col) >= low - 1e-9) &
                    (pl.col(consensus_col) <= high + 1e-9)
                )

    # Histograms in 2-column grid
    if not selected_models:
        st.info("Select at least one model to see histograms.")
        return

    # Create 2-column layout
    cols = st.columns(2)
    for i, model_id in enumerate(selected_models):
        # Baseline (unfiltered by attributes) for outline
        baseline_model_df = baseline_df.filter(pl.col("target_model_id") == model_id)
        baseline_credences = baseline_model_df["consensus_credence"].drop_nulls().to_list()

        # Filtered data
        model_df = filtered_df.filter(pl.col("target_model_id") == model_id)
        credences = model_df["consensus_credence"].drop_nulls().to_list()
        model_name = _friendly_model_name(model_id)

        # Color by provider: OpenAI brand teal, Anthropic brand coral
        if model_id.startswith("openai/"):
            bar_color = "#10a37f"  # OpenAI teal
        elif model_id.startswith("anthropic/"):
            bar_color = "#d97757"  # Anthropic coral
        else:
            bar_color = "#1f77b4"  # default blue

        n_baseline = len(baseline_credences)
        n_filtered = len(credences)

        # Baseline stats
        baseline_mean = sum(baseline_credences) / len(baseline_credences) if baseline_credences else 0
        baseline_median = median(baseline_credences) if baseline_credences else 0
        baseline_extremity = sum(abs(c - 0.5) for c in baseline_credences) / len(baseline_credences) if baseline_credences else 0

        fig = go.Figure()

        # Filtered histogram first (with opacity so outline shows through)
        if credences:
            mean_cred = sum(credences) / len(credences)
            median_cred = median(credences)
            extremity = sum(abs(c - 0.5) for c in credences) / len(credences)

            # Calculate shifts
            mean_shift = mean_cred - baseline_mean
            median_shift = median_cred - baseline_median
            extremity_shift = extremity - baseline_extremity

            fig.add_trace(go.Histogram(
                x=credences,
                xbins=dict(start=0, end=1, size=0.1),
                histnorm="percent",
                marker=dict(color=bar_color, opacity=0.6),
                name="Filtered",
                showlegend=False,
            ))

            legend_text = (
                f"n={n_filtered}/{n_baseline}<br>"
                f"mean={mean_cred:.2f}{format_shift(mean_shift)}<br>"
                f"median={median_cred:.2f}{format_shift(median_shift)}<br>"
                f"extremity={extremity:.2f}{format_shift(extremity_shift)}"
            )
        else:
            legend_text = f"n=0/{n_baseline}"

        # Baseline histogram as outline on top (gray, unfilled)
        if baseline_credences:
            fig.add_trace(go.Histogram(
                x=baseline_credences,
                xbins=dict(start=0, end=1, size=0.1),
                histnorm="percent",
                marker=dict(color="rgba(0,0,0,0)", line=dict(color="#888888", width=1.5)),
                name="All",
                showlegend=False,
            ))

        fig.update_layout(
            height=180,
            title=model_name,
            title_font_size=12,
            xaxis=dict(range=[0, 1], title="Credence", title_font_size=10),
            yaxis=dict(title="Percent", title_font_size=10),
            margin=dict(l=40, r=100, t=35, b=30),
            bargap=0.1,
            barmode="overlay",
            annotations=[
                dict(
                    text=legend_text,
                    xref="paper", yref="paper",
                    x=1.02, y=0.95,
                    xanchor="left", yanchor="top",
                    showarrow=False,
                    font=dict(size=10),
                    align="left",
                )
            ],
        )

        with cols[i % 2]:
            st.plotly_chart(fig, use_container_width=True)

    st.caption("Extremity = mean |credence - 0.5|. Higher values indicate credences closer to 0 or 1; lower values indicate credences closer to 0.5.")


def render_model_agreement_tab(df: pl.DataFrame):
    """Render model agreement heatmap tab."""
    st.subheader("Model Agreement")

    if df.is_empty():
        st.warning("No data found.")
        return

    # Get filter options
    all_models_list = sorted(df["target_model_id"].unique().to_list())
    all_model_names = [_friendly_model_name(m) for m in all_models_list]
    name_to_id = {_friendly_model_name(m): m for m in all_models_list}
    all_domains = sorted(df["domain"].unique().to_list()) if "domain" in df.columns else []

    # Filters
    with st.expander("Filters", expanded=False):
        col1, col2, col3 = st.columns([2, 2, 1])
        with col1:
            selected_model_names = st.multiselect(
                "Models",
                all_model_names,
                default=all_model_names,
                key="agreement_models",
            )
        with col2:
            selected_domains = st.multiselect(
                "Domains",
                all_domains,
                default=all_domains,
                key="agreement_domains",
            )
        with col3:
            metric_type = st.radio("Metric", ["Pearson", "Spearman", "MAE"], horizontal=True, key="agreement_metric_type")

    use_mae = metric_type == "MAE"
    selected_models = [name_to_id[name] for name in selected_model_names if name in name_to_id]

    # Apply filters
    filtered_df = df
    if selected_models:
        filtered_df = filtered_df.filter(pl.col("target_model_id").is_in(selected_models))
    if selected_domains:
        filtered_df = filtered_df.filter(pl.col("domain").is_in(selected_domains))

    if len(selected_models) < 2:
        st.warning("Select at least 2 models to compare.")
        return

    # Pivot data to get credences per prompt per model
    pivot_df = filtered_df.filter(
        pl.col("consensus_credence").is_not_null()
    ).select(["prompt_id", "target_model_id", "consensus_credence"]).to_pandas()

    if len(pivot_df) == 0:
        st.warning("No consensus credences found.")
        return

    pivot_wide = pivot_df.pivot(index="prompt_id", columns="target_model_id", values="consensus_credence")

    # Custom model order for display
    model_order_preference = [
        "anthropic/claude-haiku-4-5-20251001",
        "anthropic/claude-sonnet-4-5-20250929",
        "openai/gpt-4o-2024-11-20",
        "openai/gpt-5-nano-2025-08-07",
        "openai/gpt-5-mini-2025-08-07",
        "openai/gpt-5-2-2025-12-11",
        "openai/gpt-5-2-chat-latest",
    ]
    model_rank = {m: i for i, m in enumerate(model_order_preference)}
    selected_models_ordered = sorted(selected_models, key=lambda m: model_rank.get(m, 999))

    # Calculate pairwise agreement metrics
    n_models = len(selected_models_ordered)
    metric_matrix = np.full((n_models, n_models), np.nan)

    for i, m1 in enumerate(selected_models_ordered):
        for j, m2 in enumerate(selected_models_ordered):
            if i == j:
                metric_matrix[i, j] = 0.0 if use_mae else 1.0
            elif m1 in pivot_wide.columns and m2 in pivot_wide.columns:
                valid = pivot_wide[[m1, m2]].dropna()
                if len(valid) >= 10:
                    if use_mae:
                        mae = np.abs(valid[m1].values - valid[m2].values).mean()
                        metric_matrix[i, j] = mae
                    else:
                        corr_func = spearmanr if metric_type == "Spearman" else pearsonr
                        r, _ = corr_func(valid[m1].values, valid[m2].values)
                        metric_matrix[i, j] = r

    # Create friendly names for display
    friendly_names = [_friendly_model_name(m) for m in selected_models_ordered]

    # Collect unique pairs (i < j only to avoid duplicates)
    pair_metrics = []
    for i in range(n_models):
        for j in range(i + 1, n_models):
            if not math.isnan(metric_matrix[i, j]):
                pair_metrics.append({
                    "Model 1": friendly_names[i],
                    "Model 2": friendly_names[j],
                    "value": metric_matrix[i, j],
                })

    if pair_metrics:
        # Sort: for MAE lower=better (ascending), for correlation higher=better (descending)
        pair_metrics.sort(key=lambda x: x["value"], reverse=not use_mae)

        # Auto-scale color range
        values = [p["value"] for p in pair_metrics]
        z_min = min(values) - 0.02
        z_max = max(values) + 0.02
    else:
        z_min, z_max = (0, 0.5) if use_mae else (0, 1)

    # Format text labels
    text_labels = []
    for i in range(n_models):
        text_row = []
        for j in range(n_models):
            v = metric_matrix[i, j]
            if math.isnan(v):
                text_row.append("")
            else:
                text_row.append(f"{v:.2f}")
        text_labels.append(text_row)

    # Build heatmap figure
    # For MAE: lower=better, so use RdBu (red=high=bad, blue=low=good)
    # For correlation: higher=better, so use RdBu_r (red=high=good)
    colorscale = "RdBu" if use_mae else "RdBu_r"
    colorbar_title = "MAE" if use_mae else "r"

    fig_heatmap = go.Figure(data=go.Heatmap(
        z=metric_matrix,
        x=friendly_names,
        y=friendly_names,
        text=text_labels,
        texttemplate="%{text}",
        colorscale=colorscale,
        zmin=z_min,
        zmax=z_max,
        colorbar=dict(title=colorbar_title),
    ))
    fig_heatmap.update_layout(
        height=400,
        margin=dict(l=120, r=20, t=20, b=120),
        xaxis=dict(tickangle=45),
    )

    # Build domain strip plot if we have domain data
    fig_strip = None
    if "domain" in filtered_df.columns and selected_domains:
        domain_points = []
        domain_means = {}

        for domain in selected_domains:
            domain_df = filtered_df.filter(pl.col("domain") == domain)
            domain_pivot_df = domain_df.filter(
                pl.col("consensus_credence").is_not_null()
            ).select(["prompt_id", "target_model_id", "consensus_credence"]).to_pandas()

            if len(domain_pivot_df) < 10:
                continue

            domain_pivot = domain_pivot_df.pivot(index="prompt_id", columns="target_model_id", values="consensus_credence")

            domain_values = []
            for i, m1 in enumerate(selected_models_ordered):
                for j, m2 in enumerate(selected_models_ordered):
                    if i < j and m1 in domain_pivot.columns and m2 in domain_pivot.columns:
                        valid = domain_pivot[[m1, m2]].dropna()
                        if len(valid) >= 10:
                            if use_mae:
                                val = np.abs(valid[m1].values - valid[m2].values).mean()
                            else:
                                corr_func = spearmanr if metric_type == "Spearman" else pearsonr
                                val, _ = corr_func(valid[m1].values, valid[m2].values)
                            domain_values.append(val)
                            domain_points.append({
                                "Domain": domain,
                                "value": val,
                                "model1": _friendly_model_name(m1),
                                "model2": _friendly_model_name(m2),
                            })

            if domain_values:
                domain_means[domain] = sum(domain_values) / len(domain_values)

        if domain_points:
            points_df = pd.DataFrame(domain_points)
            # Sort: for MAE lower=better (ascending), for correlation higher=better (descending)
            sorted_domains = sorted(domain_means.keys(), key=lambda d: domain_means[d], reverse=not use_mae)

            fig_strip = go.Figure()

            metric_label = "MAE" if use_mae else "r"
            for domain in sorted_domains:
                domain_data = points_df[points_df["Domain"] == domain]
                domain_vals = domain_data["value"].values
                domain_m1 = domain_data["model1"].values
                domain_m2 = domain_data["model2"].values
                jitter = np.random.uniform(-0.2, 0.2, len(domain_vals))
                y_positions = [sorted_domains.index(domain) + j for j in jitter]

                fig_strip.add_trace(go.Scatter(
                    x=domain_vals,
                    y=y_positions,
                    mode="markers",
                    marker=dict(size=8, opacity=0.6),
                    name=domain,
                    showlegend=False,
                    customdata=list(zip(domain_m1, domain_m2)),
                    hovertemplate=f"%{{customdata[0]}} & %{{customdata[1]}}<br>{metric_label}=%{{x:.2f}}<extra></extra>",
                ))
                fig_strip.add_trace(go.Scatter(
                    x=[domain_means[domain]],
                    y=[sorted_domains.index(domain)],
                    mode="markers",
                    marker=dict(size=14, symbol="diamond", color="black", line=dict(width=2, color="white")),
                    showlegend=False,
                    hovertemplate=f"Mean: %{{x:.2f}}<extra></extra>",
                ))

            x_title = "Mean Absolute Error" if use_mae else "Pairwise Correlation (r)"
            x_range = [0, max(points_df["value"]) + 0.05] if use_mae else [0, 1]
            fig_strip.update_layout(
                height=400,
                margin=dict(l=120, r=20, t=20, b=40),
                xaxis=dict(title=x_title, range=x_range),
                yaxis=dict(
                    tickmode="array",
                    tickvals=list(range(len(sorted_domains))),
                    ticktext=sorted_domains,
                    title="",
                ),
            )

    # Display heatmap, then strip plot below
    st.markdown("**Pairwise Model Agreement**")
    if use_mae:
        st.caption("For each pair of models, we compute the mean absolute error (MAE) between their credences across all propositions. Lower MAE means the models assign similar credence values.")
    else:
        st.caption(f"For each pair of models, we compute the {metric_type} correlation between their credences across all propositions. Higher correlation means the models tend to agree on which propositions are more/less credible.")
    st.plotly_chart(fig_heatmap, use_container_width=True)

    if pair_metrics:
        most_similar = pair_metrics[0]
        least_similar = pair_metrics[-1]
        if use_mae:
            st.caption(
                f"Most similar: **{most_similar['Model 1']}** & **{most_similar['Model 2']}** (MAE={most_similar['value']:.2f}) | "
                f"Least similar: **{least_similar['Model 1']}** & **{least_similar['Model 2']}** (MAE={least_similar['value']:.2f})"
            )
        else:
            st.caption(
                f"Most similar: **{most_similar['Model 1']}** & **{most_similar['Model 2']}** (r={most_similar['value']:.2f}) | "
                f"Least similar: **{least_similar['Model 1']}** & **{least_similar['Model 2']}** (r={least_similar['value']:.2f})"
            )

    if fig_strip:
        st.markdown("**Agreement by Domain**")
        st.caption("Same pairwise agreement metric, but computed separately for each domain. This reveals whether models agree more on certain topics than others.")
        st.plotly_chart(fig_strip, use_container_width=True)
        st.caption("Each dot = one model pair. Diamonds = domain mean.")

    # --- Disagreement Outliers ---
    st.markdown("**Disagreement Outliers**")
    st.caption("Propositions ranked by cross-model disagreement. For each proposition, we compute each model's median credence, then measure the standard deviation across models. High std = models disagree about this proposition.")

    # Compute per-proposition, per-model median credence
    prop_model_medians = filtered_df.filter(
        pl.col("consensus_credence").is_not_null()
    ).group_by(["proposition", "domain", "target_model_id"]).agg(
        pl.col("consensus_credence").median().alias("model_median")
    )

    # Compute disagreement stats per proposition (std of model medians)
    prop_disagreement = prop_model_medians.group_by(["proposition", "domain"]).agg([
        pl.col("model_median").std().alias("std"),
        pl.col("model_median").min().alias("min"),
        pl.col("model_median").max().alias("max"),
        pl.col("model_median").median().alias("overall_median"),
        pl.len().alias("n_models"),
    ]).filter(
        pl.col("n_models") >= 2
    ).sort("std", descending=True)

    all_disagreement = prop_disagreement.to_dicts()
    total_props = len(all_disagreement)
    page_size = 5
    total_pages = (total_props + page_size - 1) // page_size

    if all_disagreement:
        # Pagination controls
        if "disagreement_page" not in st.session_state or not isinstance(st.session_state.disagreement_page, int):
            st.session_state.disagreement_page = 0

        col1, col2, col3 = st.columns([1, 2, 1])
        with col1:
            if st.button("Prev", disabled=st.session_state.disagreement_page == 0, key="disagreement_prev"):
                st.session_state.disagreement_page -= 1
                st.rerun()
        with col2:
            start = st.session_state.disagreement_page * page_size + 1
            end = min(start + page_size - 1, total_props)
            st.markdown(f"**{start}-{end}** of {total_props}")
        with col3:
            if st.button("Next", disabled=st.session_state.disagreement_page >= total_pages - 1, key="disagreement_next"):
                st.session_state.disagreement_page += 1
                st.rerun()

        start_idx = st.session_state.disagreement_page * page_size
        page_props = all_disagreement[start_idx:start_idx + page_size]

        for rank, prop_stats in enumerate(page_props, start=start_idx + 1):
            prop = prop_stats["proposition"]
            std = prop_stats["std"]

            # Show proposition with rank and std
            st.markdown(f"**{rank}. [std={std:.2f}]** {prop}")

            # Get per-model breakdown for this proposition
            prop_data = prop_model_medians.filter(
                pl.col("proposition") == prop
            ).sort("model_median", descending=True).to_dicts()

            # Horizontal dot plot of model medians
            fig_dots = go.Figure()

            # Draw the number line
            fig_dots.add_shape(
                type="line",
                x0=0, x1=1, y0=0, y1=0,
                line=dict(color="gray", width=2),
            )

            # Sort by median to assign staggered label positions
            sorted_data = sorted(prop_data, key=lambda d: d["model_median"])

            # Assign y-offsets to avoid label overlap (stagger when close)
            label_positions = []
            y_levels = [20, 45, 70, 95]
            for d in sorted_data:
                level = 0
                for prev_d, prev_level in label_positions:
                    if abs(d["model_median"] - prev_d["model_median"]) < 0.12:
                        if prev_level >= level:
                            level = prev_level + 1
                label_positions.append((d, level % len(y_levels)))

            # Add dots and labels
            for d, level in label_positions:
                model_id = d["target_model_id"]
                color = "#10a37f" if model_id.startswith("openai/") else "#d97757" if model_id.startswith("anthropic/") else "#1f77b4"
                fig_dots.add_trace(go.Scatter(
                    x=[d["model_median"]],
                    y=[0],
                    mode="markers",
                    marker=dict(size=12, color=color),
                    hovertemplate=f"{_friendly_model_name(model_id)}: %{{x:.2f}}<extra></extra>",
                    showlegend=False,
                ))
                fig_dots.add_annotation(
                    x=d["model_median"],
                    y=0,
                    text=_friendly_model_name(model_id),
                    textangle=-45,
                    showarrow=True,
                    arrowhead=0,
                    arrowwidth=1,
                    arrowcolor="#ccc",
                    ay=-y_levels[level],
                    ax=0,
                    font=dict(size=10, color=color),
                )

            fig_dots.update_layout(
                height=180,
                margin=dict(l=20, r=20, t=80, b=20),
                xaxis=dict(title="", range=[-0.02, 1.02], dtick=0.2),
                yaxis=dict(visible=False, range=[-0.3, 0.3]),
                showlegend=False,
            )
            st.plotly_chart(fig_dots, use_container_width=True, key=f"disagreement_plot_{rank}")


def render_prompt_attributes_tab(df: pl.DataFrame):
    """Render prompt attributes tab - judge agreement and distributions."""
    st.subheader("Prompt Attributes")

    if df.is_empty():
        st.warning("No data found.")
        return

    # Check if prompt attributes exist
    has_attrs = f"prompt_judge1_{ATTRIBUTE_NAMES[0]}" in df.columns
    if not has_attrs:
        st.warning("Prompt attributes not yet scored. Run run_exploration.py first.")
        return

    # Display names for attributes
    attr_display = {
        "user_valence": "user valence",
        "sophistication": "sophistication",
        "apparent_stakes": "stakes",
        "emotionality": "emotionality",
        "user_authority": "authority",
        "confrontationality": "confrontationality",
        "clarity": "clarity",
    }

    # Tooltips for attributes
    attr_tooltips = {
        "user_valence": "Implied user prior belief (skeptical to credulous)",
        "sophistication": "Apparent education level and writing quality",
        "apparent_stakes": "How consequential the decision/situation is",
        "emotionality": "How emotionally charged the prompt is",
        "user_authority": "How much expertise/confidence user projects",
        "confrontationality": "Cooperative vs adversarial stance toward AI",
        "clarity": "How specific and unambiguous the request is",
    }

    # Consistent colors for attributes
    colors = px.colors.qualitative.Plotly
    attr_colors = {attr_display[attr]: colors[i % len(colors)] for i, attr in enumerate(ATTRIBUTE_NAMES)}

    # === Judge Agreement Section ===
    st.markdown("**Judge Agreement**")
    st.caption("Each prompt is scored by two judges (GPT-5 Mini and Claude Haiku 4.5). Agreement = |J1 - J2| <= 0.2.")

    # Judge agreement rates per attribute - bar chart with labels
    agreement_data = []
    for attr in ATTRIBUTE_NAMES:
        j1_col = f"prompt_judge1_{attr}"
        j2_col = f"prompt_judge2_{attr}"
        if j1_col in df.columns and j2_col in df.columns:
            valid = df.filter(pl.col(j1_col).is_not_null() & pl.col(j2_col).is_not_null())
            if len(valid) > 0:
                agreed = valid.filter(
                    (pl.col(j1_col) - pl.col(j2_col)).abs() <= CONSENSUS_THRESHOLD + 1e-9
                )
                agreement_rate = len(agreed) / len(valid)
                agreement_data.append({
                    "Attribute": attr_display[attr],
                    "Agreement Rate": agreement_rate,
                    "Label": f"{agreement_rate:.0%}",
                    "Tooltip": attr_tooltips[attr],
                })

    if agreement_data:
        agreement_df = pd.DataFrame(agreement_data).sort_values("Agreement Rate", ascending=False)
        bar_fig = px.bar(
            agreement_df, x="Attribute", y="Agreement Rate",
            text="Label",
            color="Attribute",
            color_discrete_map=attr_colors,
            hover_data={"Tooltip": True, "Agreement Rate": False, "Attribute": False, "Label": False},
        )
        bar_fig.update_layout(
            height=250,
            yaxis=dict(range=[0, 1.18], tickformat=".0%"),
            showlegend=False,
        )
        bar_fig.update_traces(textposition="outside", hovertemplate="%{customdata[0]}<extra></extra>")
        st.plotly_chart(bar_fig, use_container_width=True)

    # === Attribute Distributions ===
    st.markdown("**Distributions**")

    # Attribute distribution histograms - 4 on top, 3 centered below
    row1_cols = st.columns(4)
    row2_spacer_left, row2_c1, row2_c2, row2_c3, row2_spacer_right = st.columns([0.5, 1, 1, 1, 0.5])
    row2_cols = [row2_c1, row2_c2, row2_c3]
    all_cols = row1_cols + row2_cols

    for i, attr in enumerate(ATTRIBUTE_NAMES):
        consensus_col = f"consensus_{attr}"
        display_name = attr_display[attr]

        if consensus_col in df.columns:
            values = df[consensus_col].drop_nulls().to_pandas()

            if not values.empty:
                hist_df = pd.DataFrame({"value": values})
                hist_fig = px.histogram(
                    hist_df, x="value",
                    opacity=0.7,
                    range_x=[0, 1],
                )
                hist_fig.update_traces(
                    xbins=dict(start=0, end=1, size=0.1),
                    marker_color=attr_colors[display_name],
                )
                hist_fig.update_layout(
                    height=150,
                    bargap=0.1,
                    title=display_name,
                    title_font_size=11,
                    margin=dict(l=30, r=10, t=30, b=30),
                    xaxis_title="",
                    yaxis_title="",
                )
                with all_cols[i]:
                    st.plotly_chart(hist_fig, use_container_width=True)


# =============================================================================
# Overview Page
# =============================================================================

def _nav_link(label: str, section: str, tab: str) -> str:
    """Generate a markdown link that navigates to a section/tab."""
    return f"[{label}](?section={section}&tab={tab})"


def render_overview_page():
    """Render the overview/about page."""
    st.title("Surveying AI Belief")

    st.markdown("""
    What do AI models actually believe? We built an automated pipeline to measure AI-expressed
    beliefs towards arbitrary propositions. 
    We surveyed frontier models 500 propositions across a range of domains to produce initial results:
    1) validating that our pipeline produces stable results that track reality.
    2) exploring initial results characterizing model beliefs, sensitivity to prompt framings, and cross-model differences.
    """)

    st.subheader("Method")
    st.markdown("""
    Each proposition is run through our 4 step pipeline:

    1. **Prompt generation**: Elicitor models generate varied prompts about the proposition.
       Prompts are designed to be credence-sensitive, so a model's response reveals its stance; realistic,
       so a user would plausibly ask them; and diverse across a range of dimensions.
    2. **Prompt attribute scoring**: We score each prompt for attributes like clarity, emotionality,
       and implied user valence - whether the user seems to believe or doubt the proposition.
    3. **Test model response**: The model under test responds to each prompt.
    4. **Credence judging**: Two judge models independently estimate the implied credence from
       each response on a 0-1 scale. We use consensus scores where judges agree.
    """)

    st.divider()

    st.subheader("What's in This App")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Part 1: Validation")
        st.markdown("Checks that determine if the method produces stable credences in-line with expectations. We ran four (preregistered) validation checks. They all passed with no adjustments needed to our pipeline.")
        st.markdown(f"""
| Tab | What you'll see |
|-----|-----------------|
| {_nav_link("V1 Test-Retest", "validation", "test_retest")} | Do repeated end-to-end runs of our pipeline (with different prompts, responses, and judgments) yield stable credences? |
| {_nav_link("V2 Judge Agreement", "validation", "judge_agreement")} | Do our AI judges generally score prompt-response interactions similarly for credences? |
| {_nav_link("V3 Calibration", "validation", "calibration")} | Do known-true propositions score high, known-false score low, and uncertain propositions land in between? |
| {_nav_link("V4 Known-Group", "validation", "known_group")} | Do Chinese vs Western models diverge in expected directions on sensitive topics that are known to be manipulated? |
        """)

    with col2:
        st.markdown("#### Part 2: Exploration")
        st.markdown("An exploration of the extremity and dispersion of model beliefs, model sensitivity to prompt framing, and cross-model agreement.")
        st.markdown(f"""
| Tab | What you'll see |
|-----|-----------------|
| {_nav_link("What Models Believe", "exploration", "dispersion")} | Extremity and dispersion of model credences |
| {_nav_link("Prompt Sensitivity", "exploration", "sensitivity")} | How models shift credence based on user's implied belief |
| {_nav_link("Cross-Model Agreement", "exploration", "model_agreement")} | Which models agree and disagree the most |
        """)


# =============================================================================
# Main
# =============================================================================

# Tab mappings for deep linking
VALIDATION_TABS = {
    "test_retest": ("V1 Test-Retest", render_test_retest_tab),
    "judge_agreement": ("V2 Judge Agreement", render_judge_agreement_tab),
    "calibration": ("V3 Calibration", render_calibration_tab),
    "known_group": ("V4 Known-Group", render_known_group_tab),
    "inspect": ("Inspect", render_inspect_tab),
}

EXPLORATION_TABS = {
    "dispersion": ("What Models Believe", lambda df: render_dispersion_tab(df)),
    "sensitivity": ("Prompt Sensitivity", lambda df: render_prompt_sensitivity_tab(df)),
    "model_agreement": ("Cross-Model Agreement", lambda df: render_model_agreement_tab(df)),
}


def main():
    st.set_page_config(page_title="Credence Visualization", layout="wide")

    # Read query params for deep linking
    params = st.query_params
    section_param = params.get("section")
    tab_param = params.get("tab")

    # Determine initial section from query params
    if section_param == "validation":
        default_section = "Part 1: Validation"
    elif section_param == "exploration":
        default_section = "Part 2: Exploration"
    else:
        default_section = "Overview"

    sections = ["Overview", "Part 1: Validation", "Part 2: Exploration"]
    part = st.sidebar.radio("Section", sections, index=sections.index(default_section))

    # Clear query params when user manually navigates via sidebar
    if part != default_section:
        st.query_params.clear()
        tab_param = None

    if part == "Overview":
        render_overview_page()

    elif part == "Part 1: Validation":
        st.title("Part 1: Method Validation")

        # If deep-linked to a specific tab, render just that tab
        if tab_param and tab_param in VALIDATION_TABS:
            tab_name, render_fn = VALIDATION_TABS[tab_param]
            st.caption(f"Showing: {tab_name} [(back to overview)](?)")
            render_fn()
        else:
            # Normal tabs view
            tab_names = [name for name, _ in VALIDATION_TABS.values()]
            tabs = st.tabs(tab_names)
            for i, (_, render_fn) in enumerate(VALIDATION_TABS.values()):
                with tabs[i]:
                    render_fn()

    else:
        st.title("Part 2: Exploration")
        # Load exploration data
        raw_df = load_exploration_parquet()
        df = add_computed_columns(raw_df)

        # If deep-linked to a specific tab, render just that tab
        if tab_param and tab_param in EXPLORATION_TABS:
            tab_name, render_fn = EXPLORATION_TABS[tab_param]
            st.caption(f"Showing: {tab_name} [(back to overview)](?)")
            render_fn(df)
        else:
            # Normal tabs view
            tab_names = [name for name, _ in EXPLORATION_TABS.values()]
            tabs = st.tabs(tab_names)
            for i, (_, render_fn) in enumerate(EXPLORATION_TABS.values()):
                with tabs[i]:
                    render_fn(df)


if __name__ == "__main__":
    main()
