"""Streamlit app for visualizing exploration experiment results.

Usage:
    uv run streamlit run viz_exploration.py

Explores credence measurements across domains, models, and prompt characteristics.
"""

import math
from pathlib import Path
from statistics import median

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import polars as pl
import streamlit as st
from scipy.stats import spearmanr, pearsonr


# =============================================================================
# Constants
# =============================================================================

DATA_ROOT = Path(__file__).parent / "data_exploration"
EXPLORATION_PARQUET = DATA_ROOT / "exploration.parquet"

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

# Credence judge agreement threshold
CREDENCE_AGREEMENT_THRESHOLD = 0.2


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
            ((pl.col("judge1_credence") - pl.col("judge2_credence")).abs() <= CREDENCE_AGREEMENT_THRESHOLD + 1e-9)
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


def render_completeness_tab(df: pl.DataFrame):
    """Render completeness/progress tab."""
    st.subheader("Completeness")

    if df.is_empty():
        st.warning("No data found. Run the exploration pipeline first.")
        st.code("uvr experiments/exploration/run_exploration.py")
        return

    # Check for prompt attributes
    has_prompt_attrs = "consensus_user_valence" in df.columns

    # Summary metrics
    n_samples = len(df)
    n_propositions = df["proposition"].n_unique()
    n_domains = df["domain"].n_unique() if "domain" in df.columns else 0
    n_models = df["target_model_id"].n_unique()

    # Credence consensus
    n_credence_consensus = df["consensus_credence"].drop_nulls().len()
    credence_consensus_rate = n_credence_consensus / n_samples if n_samples > 0 else 0

    # Prompt judge consensus (use user_valence as proxy)
    n_prompt_consensus = df["consensus_user_valence"].drop_nulls().len() if has_prompt_attrs else 0
    prompt_consensus_rate = n_prompt_consensus / n_samples if n_samples > 0 else 0

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Samples", f"{n_samples:,}")
    col2.metric("Propositions", n_propositions)
    col3.metric("Domains", n_domains)
    col4.metric("Models", n_models)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Credence Consensus", f"{n_credence_consensus:,}")
    col2.metric("Credence Consensus Rate", f"{credence_consensus_rate:.1%}")
    col3.metric("Prompt Judge Consensus", f"{n_prompt_consensus:,}")
    col4.metric("Prompt Judge Consensus Rate", f"{prompt_consensus_rate:.1%}")

    # Progress by domain
    st.subheader("By Domain")
    if "domain" in df.columns:
        agg_cols = [
            pl.col("proposition").n_unique().alias("propositions"),
            pl.len().alias("samples"),
            pl.col("consensus_credence").is_not_null().sum().alias("credence_consensus"),
        ]
        if has_prompt_attrs:
            agg_cols.append(pl.col("consensus_user_valence").is_not_null().sum().alias("prompt_consensus"))

        domain_stats = df.group_by("domain").agg(agg_cols).sort("domain")
        domain_df = domain_stats.to_pandas()
        domain_df["credence_rate"] = (domain_df["credence_consensus"] / domain_df["samples"]).apply(lambda x: f"{x:.1%}")
        if has_prompt_attrs:
            domain_df["prompt_rate"] = (domain_df["prompt_consensus"] / domain_df["samples"]).apply(lambda x: f"{x:.1%}")
            st.dataframe(domain_df[["domain", "propositions", "samples", "credence_consensus", "credence_rate", "prompt_consensus", "prompt_rate"]],
                         use_container_width=True, hide_index=True)
        else:
            st.dataframe(domain_df[["domain", "propositions", "samples", "credence_consensus", "credence_rate"]],
                         use_container_width=True, hide_index=True)

    # Progress by model
    st.subheader("By Model")
    agg_cols = [
        pl.col("proposition").n_unique().alias("propositions"),
        pl.len().alias("samples"),
        pl.col("consensus_credence").is_not_null().sum().alias("credence_consensus"),
    ]
    if has_prompt_attrs:
        agg_cols.append(pl.col("consensus_user_valence").is_not_null().sum().alias("prompt_consensus"))

    model_stats = df.group_by("target_model_id").agg(agg_cols).sort("target_model_id")
    model_df = model_stats.to_pandas()
    model_df["model"] = model_df["target_model_id"].apply(_friendly_model_name)
    model_df["credence_rate"] = (model_df["credence_consensus"] / model_df["samples"]).apply(lambda x: f"{x:.1%}")

    if has_prompt_attrs:
        model_df["prompt_rate"] = (model_df["prompt_consensus"] / model_df["samples"]).apply(lambda x: f"{x:.1%}")
        st.dataframe(model_df[["model", "propositions", "samples", "credence_consensus", "credence_rate", "prompt_consensus", "prompt_rate"]],
                     use_container_width=True, hide_index=True)
    else:
        st.dataframe(model_df[["model", "propositions", "samples", "credence_consensus", "credence_rate"]],
                     use_container_width=True, hide_index=True)


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

    # Attribute display names
    attr_display = {
        "user_valence": "User Valence",
        "sophistication": "Sophistication",
        "apparent_stakes": "Stakes",
        "emotionality": "Emotionality",
        "user_authority": "Authority",
        "confrontationality": "Confrontationality",
        "clarity": "Clarity",
    }

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
    all_attr_options = [attr_display[attr] for attr in ATTRIBUTE_NAMES]
    display_to_attr = {attr_display[attr]: attr for attr in ATTRIBUTE_NAMES}

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
                    attr_display[attr],
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

    # Helper to format shift with color
    def format_shift(shift):
        if abs(shift) < 0.005:
            return ""
        color = "#2ca02c" if shift > 0 else "#d62728"  # green/red
        return f' <span style="color:{color}">({shift:+.2f})</span>'

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
        st.plotly_chart(fig_strip, use_container_width=True)
        st.caption("Each dot = one model pair. Diamonds = domain means.")


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


def main():
    st.set_page_config(page_title="Exploration Results", layout="wide")
    st.title("Part 2: Exploration Results")

    # Load data
    raw_df = load_exploration_parquet()
    df = add_computed_columns(raw_df)

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "User Deference",
        "Credence Distributions",
        "Model Agreement",
        "Prompt Attributes",
        "Completeness",
    ])

    with tab1:
        render_correlations_detail_tab(df)

    with tab2:
        render_explorer_tab(df)

    with tab3:
        render_model_agreement_tab(df)

    with tab4:
        render_prompt_attributes_tab(df)

    with tab5:
        render_completeness_tab(df)


if __name__ == "__main__":
    main()
