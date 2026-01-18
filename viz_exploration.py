"""Streamlit app for visualizing exploration experiment results.

Usage:
    uv run streamlit run viz_exploration.py

Explores credence measurements across domains, models, and prompt characteristics.
"""

from pathlib import Path
from statistics import median

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import polars as pl
import streamlit as st
from scipy.stats import spearmanr, pearsonr
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score


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


def _get_prompt_key_cols(df: pl.DataFrame) -> list[str]:
    """Get list of columns that together uniquely identify a prompt across models."""
    return ["prompt_id"]


@st.cache_data
def get_complete_consensus_prompts(_df: pl.DataFrame, selected_models: list[str]) -> pl.DataFrame:
    """Find prompts where ALL selected models have consensus credence and prompt attributes.

    Returns DataFrame with prompt key columns for prompts that pass the filter.
    Use with semi-join to filter the main DataFrame.
    """
    if _df.is_empty() or not selected_models:
        return pl.DataFrame()

    key_cols = _get_prompt_key_cols(_df)
    n_models = len(selected_models)

    # Filter to selected models only
    filtered = _df.filter(pl.col("target_model_id").is_in(selected_models))

    # Group by prompt key (proposition + attributes identifies unique prompts across models)
    # Count how many models have consensus_credence AND consensus_user_valence for each prompt
    prompt_stats = filtered.group_by(key_cols).agg([
        pl.col("target_model_id").n_unique().alias("n_models_total"),
        (pl.col("consensus_credence").is_not_null() & pl.col("consensus_user_valence").is_not_null())
        .sum().alias("n_models_with_consensus"),
    ])

    # Keep prompts where ALL models have consensus
    complete_prompts = prompt_stats.filter(
        (pl.col("n_models_total") == n_models) &
        (pl.col("n_models_with_consensus") == n_models)
    )

    return complete_prompts.select(key_cols)


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


def render_prompt_attributes_tab(df: pl.DataFrame):
    """Render prompt attributes analysis tab."""
    st.subheader("Prompt Attributes")

    st.markdown(
        "Each prompt is scored on these attributes by two judges (GPT-5 Mini and Claude Haiku 4.5).  \n"
        "Judge agreement is defined as |J1 - J2| <= 0.2 for these attributes which are scored [0, 1]."
    )

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

    # Consistent colors for attributes (same as correlations tab)
    colors = px.colors.qualitative.Plotly
    attr_colors = {attr_display[attr]: colors[i % len(colors)] for i, attr in enumerate(ATTRIBUTE_NAMES)}

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
            height=280,
            yaxis=dict(range=[0, 1.12], tickformat=".0%"),
            title="Prompt Judge Agreement Rate by Attribute",
            showlegend=False,
        )
        bar_fig.update_traces(textposition="outside", hovertemplate="%{customdata[0]}<extra></extra>")
        st.plotly_chart(bar_fig, use_container_width=True)

    # Attribute distribution histograms - 4 on top, 3 centered below
    st.markdown("**Attribute Distributions (Consensus Values)**")
    row1_cols = st.columns(4)
    # Center 3 columns beneath 4: use spacers on sides (0.5 : 1 : 1 : 1 : 0.5)
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
                    xbins=dict(start=0, end=1, size=0.05),
                    marker_color=attr_colors[display_name],
                )
                hist_fig.update_layout(
                    height=180,
                    bargap=0.1,
                    title=display_name,
                    title_font_size=12,
                    margin=dict(l=30, r=10, t=30, b=30),
                    xaxis_title="",
                    yaxis_title="",
                )
                with all_cols[i]:
                    st.plotly_chart(hist_fig, use_container_width=True)


def render_correlations_detail_tab(df: pl.DataFrame):
    """Render correlations tab with breakdown of predictability components."""

    if df.is_empty():
        st.warning("No data found.")
        return

    # Check for required columns
    if "consensus_user_valence" not in df.columns:
        st.warning("Prompt attributes not yet scored.")
        return

    # Get filter options
    all_domains = sorted(df["domain"].unique().to_list()) if "domain" in df.columns else []
    all_models_list = sorted(df["target_model_id"].unique().to_list())
    all_model_names = [_friendly_model_name(m) for m in all_models_list]
    name_to_id = {_friendly_model_name(m): m for m in all_models_list}

    # Filters at the top
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
                help="Pearson: linear, assumes normality. Spearman: rank-based, robust to outliers."
            )

    use_spearman = corr_type == "Spearman"
    corr_func = spearmanr if use_spearman else pearsonr
    corr_label = "Spearman r" if use_spearman else "Pearson r"

    selected_models = [name_to_id[name] for name in selected_model_names if name in name_to_id]

    # Apply filters
    filtered_df = df
    if selected_domains:
        filtered_df = filtered_df.filter(pl.col("domain").is_in(selected_domains))
    if selected_models:
        filtered_df = filtered_df.filter(pl.col("target_model_id").is_in(selected_models))

    if filtered_df.is_empty():
        st.warning("No data matches the selected filters.")
        return

    st.markdown("""
    **Goal**: Understand how credence varies with prompt attributes.
    Theory: 
    - **User valence** correlates directly with credence because models have some tendency to defer to the user's apparent belief
    - **Other prompt attributes** modulate user deference. They correlate with *agreement*, how much the model defers to the user's apparent belief. agreement = credence when valence > 0.5, else (1 - credence)
    
    By combining these features we can understand how much of the variation in credence is explained by these prompt attributes.
    """)

    # Get models from filtered data
    all_models = sorted(filtered_df["target_model_id"].unique().to_list())
    model_names = {m: _friendly_model_name(m) for m in all_models}

    non_valence_attrs = [a for a in ATTRIBUTE_NAMES if a != "user_valence"]
    attr_display = {
        "user_valence": "user valence",
        "sophistication": "sophistication",
        "apparent_stakes": "stakes",
        "emotionality": "emotionality",
        "user_authority": "authority",
        "confrontationality": "confrontationality",
        "clarity": "clarity",
    }

    def sig_stars(p: float, threshold: float = 0.05) -> str:
        """Return significance stars based on p-value."""
        if p < threshold / 50:  # 0.001 equivalent
            return "***"
        if p < threshold / 5:   # 0.01 equivalent
            return "**"
        if p < threshold:
            return "*"
        return ""

    def apply_fdr_correction(p_values: list[float]) -> list[float]:
        """Apply Benjamini-Hochberg FDR correction. Returns adjusted p-values."""
        n = len(p_values)
        if n == 0:
            return []
        indexed = sorted(enumerate(p_values), key=lambda x: x[1])
        adjusted = [0.0] * n
        min_so_far = 1.0
        for rank, (orig_idx, p) in enumerate(reversed(indexed), 1):
            adjusted_p = min(p * n / (n - rank + 1), 1.0)
            min_so_far = min(min_so_far, adjusted_p)
            adjusted[orig_idx] = min_so_far
        return adjusted

    def fisher_z_ci(r: float, n: int, ci: float = 0.95) -> tuple[float, float]:
        """Compute analytic CI for Pearson correlation using Fisher z-transform."""
        if n < 4:
            return (r, r)
        # Fisher z-transform
        z = 0.5 * np.log((1 + r) / (1 - r + 1e-10))
        se_z = 1 / np.sqrt(n - 3)
        # CI in z-space
        z_crit = 1.96 if ci == 0.95 else 2.576  # 95% or 99%
        z_lower = z - z_crit * se_z
        z_upper = z + z_crit * se_z
        # Transform back to r-space
        lower = np.tanh(z_lower)
        upper = np.tanh(z_upper)
        return lower, upper

    def bootstrap_corr_ci(x: np.ndarray, y: np.ndarray, n_boot: int = 200, ci: float = 0.95) -> tuple[float, float]:
        """Compute bootstrap CI for Spearman correlation (reduced iterations for speed)."""
        rng = np.random.default_rng(42)
        n = len(x)
        boot_rs = []
        for _ in range(n_boot):
            idx = rng.choice(n, size=n, replace=True)
            r, _ = spearmanr(x[idx], y[idx])
            boot_rs.append(r)
        alpha = 1 - ci
        lower = np.percentile(boot_rs, 100 * alpha / 2)
        upper = np.percentile(boot_rs, 100 * (1 - alpha / 2))
        return lower, upper

    def compute_corr_ci(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
        """Compute CI using analytic method for Pearson, bootstrap for Spearman."""
        if use_spearman:
            return bootstrap_corr_ci(x, y, n_boot=200)
        else:
            r, _ = pearsonr(x, y)
            return fisher_z_ci(r, len(x))

    # Compute stats for each model
    # FDR correction strategy:
    # - Section 1 (valence -> credence): No correction, single hypothesis per model
    # - Section 2 (agreement ~ attributes): FDR across all (models × attributes)
    results = []
    section2_p_values = []  # Only agreement correlations for FDR
    section2_p_keys = []

    for model_id in all_models:
        model_data = filtered_df.filter(pl.col("target_model_id") == model_id)

        # Get valid data (all prompts)
        all_cols = ["consensus_credence", "consensus_user_valence"] + [f"consensus_{a}" for a in non_valence_attrs]
        all_valid = model_data.select(all_cols).drop_nulls().to_pandas()

        # Get directional data (exclude valence == 0.5)
        directional = model_data.filter(
            (pl.col("consensus_user_valence") - 0.5).abs() > 1e-9
        )
        dir_valid = directional.select(all_cols).drop_nulls().to_pandas()

        if len(all_valid) < 10 or len(dir_valid) < 10:
            continue

        row_idx = len(results)
        row = {"model_id": model_id, "model": model_names[model_id]}

        # 1. Raw valence-credence correlation (ALL prompts)
        # No FDR correction for Section 1 - single hypothesis per model
        x_all = all_valid["consensus_user_valence"].values
        y_all_cred = all_valid["consensus_credence"].values
        r_val_all, p_val_all = corr_func(x_all, y_all_cred)
        ci_low_all, ci_high_all = compute_corr_ci(x_all, y_all_cred)
        row["valence_r_all"] = r_val_all
        row["valence_p_all"] = p_val_all  # Raw p-value used directly for stars
        row["valence_ci_low_all"] = ci_low_all
        row["valence_ci_high_all"] = ci_high_all
        row["n_all"] = len(all_valid)

        # 2. Raw valence-credence correlation (DIRECTIONAL only)
        x_dir = dir_valid["consensus_user_valence"].values
        y_dir_cred = dir_valid["consensus_credence"].values
        r_val_dir, p_val_dir = corr_func(x_dir, y_dir_cred)
        ci_low_dir, ci_high_dir = compute_corr_ci(x_dir, y_dir_cred)
        row["valence_r_dir"] = r_val_dir
        row["valence_p_dir"] = p_val_dir  # Raw p-value used directly for stars
        row["valence_ci_low_dir"] = ci_low_dir
        row["valence_ci_high_dir"] = ci_high_dir
        row["n_dir"] = len(dir_valid)

        # 3. Direction-corrected correlations (agreement ~ attr) for DIRECTIONAL prompts
        # These get FDR correction across all (models × attributes)
        agreement = np.where(
            dir_valid["consensus_user_valence"].values > 0.5,
            dir_valid["consensus_credence"].values,
            1 - dir_valid["consensus_credence"].values
        )
        for attr in non_valence_attrs:
            col = f"consensus_{attr}"
            attr_vals = dir_valid[col].values
            r, p = corr_func(attr_vals, agreement)
            ci_low, ci_high = compute_corr_ci(attr_vals, agreement)
            row[f"{attr}_r_agree"] = r
            row[f"{attr}_p_agree"] = p
            row[f"{attr}_ci_low_agree"] = ci_low
            row[f"{attr}_ci_high_agree"] = ci_high
            section2_p_values.append(p)
            section2_p_keys.append((row_idx, f"{attr}_q_agree"))

        # 4. R² models - DIRECTIONAL prompts
        # Model: credence ~ β₀ + β₁(v-0.5) + β₂(v-0.5)×attr₁ + ...
        valence_vals = dir_valid["consensus_user_valence"].values
        valence_centered = valence_vals - 0.5
        y = dir_valid["consensus_credence"].values
        interaction_cols = [valence_centered * dir_valid[f"consensus_{attr}"].values for attr in non_valence_attrs]

        # Valence-only baseline: credence ~ β₀ + β₁(v-0.5)
        X_valence = valence_centered.reshape(-1, 1)
        row["valence_only_r2_dir"] = LinearRegression().fit(X_valence, y).score(X_valence, y)
        n_dir_samples = len(y)
        # Cross-validated R² (3-fold)
        if n_dir_samples >= 20:
            row["valence_only_cv_r2_dir"] = cross_val_score(LinearRegression(), X_valence, y, cv=3, scoring='r2').mean()
        else:
            row["valence_only_cv_r2_dir"] = None

        # Interactions model: credence ~ β₀ + β₁(v-0.5) + β₂(v-0.5)×attr₁ + ...
        X_interact = np.column_stack([valence_centered] + interaction_cols)
        row["interactions_r2_dir"] = LinearRegression().fit(X_interact, y).score(X_interact, y)
        if n_dir_samples >= 20:
            row["interactions_cv_r2_dir"] = cross_val_score(LinearRegression(), X_interact, y, cv=3, scoring='r2').mean()
        else:
            row["interactions_cv_r2_dir"] = None

        # 5. R² models - ALL prompts
        valence_all = all_valid["consensus_user_valence"].values
        valence_all_centered = valence_all - 0.5
        y_all = all_valid["consensus_credence"].values
        interaction_cols_all = [valence_all_centered * all_valid[f"consensus_{attr}"].values for attr in non_valence_attrs]
        n_all_reg = len(y_all)

        # Valence-only baseline
        X_valence_all = valence_all_centered.reshape(-1, 1)
        row["valence_only_r2_all"] = LinearRegression().fit(X_valence_all, y_all).score(X_valence_all, y_all)
        if n_all_reg >= 20:
            row["valence_only_cv_r2_all"] = cross_val_score(LinearRegression(), X_valence_all, y_all, cv=3, scoring='r2').mean()
        else:
            row["valence_only_cv_r2_all"] = None

        # Interactions model
        X_interact_all = np.column_stack([valence_all_centered] + interaction_cols_all)
        row["interactions_r2_all"] = LinearRegression().fit(X_interact_all, y_all).score(X_interact_all, y_all)
        if n_all_reg >= 20:
            row["interactions_cv_r2_all"] = cross_val_score(LinearRegression(), X_interact_all, y_all, cv=3, scoring='r2').mean()
        else:
            row["interactions_cv_r2_all"] = None

        results.append(row)

    if not results:
        st.warning("Not enough data for analysis.")
        return

    # Apply FDR correction only to Section 2 (agreement correlations)
    adjusted_p = apply_fdr_correction(section2_p_values)
    for i, (row_idx, key) in enumerate(section2_p_keys):
        results[row_idx][key] = adjusted_p[i]

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

    # === Section 1: Baseline User Deference ===
    st.subheader("1. Baseline User Deference: credence ~ user_valence")
    st.markdown(f"Direct {corr_type} correlation between user's apparent belief and model's expressed credence.")

    discard_neutral_s1 = st.checkbox(
        f"Discard samples where prompt user valence is neutral, v==0.5 ({neutral_pct:.1f}% of prompts)",
        value=True,
        key="discard_neutral_s1",
    )
    if discard_neutral_s1:
        s1_suffix, s1_n = "_dir", "n_dir"
    else:
        s1_suffix, s1_n = "_all", "n_all"

    # Section 1 uses raw p-values (no FDR correction - single hypothesis per model)
    tbl1 = results_df[["model", f"valence_r{s1_suffix}", f"valence_p{s1_suffix}", f"valence_ci_low{s1_suffix}", f"valence_ci_high{s1_suffix}", s1_n]].copy()
    tbl1[corr_label] = tbl1.apply(
        lambda row: f"{row[f'valence_r{s1_suffix}']:+.3f}{sig_stars(row[f'valence_p{s1_suffix}'])}  [95% CI: {row[f'valence_ci_low{s1_suffix}']:.3f}, {row[f'valence_ci_high{s1_suffix}']:.3f}]",
        axis=1
    )
    tbl1 = tbl1[["model", corr_label, s1_n]]
    tbl1.columns = ["Model", corr_label, "N"]
    st.dataframe(tbl1, use_container_width=True, hide_index=True)
    st.caption("\\* p<0.05, \\*\\* p<0.01, \\*\\*\\* p<0.001 (uncorrected - single test per model)")

    # === Section 2: User Deference Modulation (always agreement-based) ===
    st.subheader("2. User Deference Modulation: agreement ~ attributes")
    st.markdown(f"""
    Direction-corrected {corr_type} correlations: does the attribute predict model *agreement* with user's apparent belief?
    - `agreement = credence` when user seems to believe (valence > 0.5)
    - `agreement = 1 - credence` when user seems skeptical (valence < 0.5)
    - Positive r = attribute increases user deference
    - *Directional prompts only (v != 0.5)*
    """)

    # Heatmap for agreement correlations with significance stars and CI hover
    heatmap_data = []
    for _, row in results_df.iterrows():
        for attr in non_valence_attrs:
            r_val = row[f"{attr}_r_agree"]
            q_val = row[f"{attr}_q_agree"]
            ci_low = row[f"{attr}_ci_low_agree"]
            ci_high = row[f"{attr}_ci_high_agree"]
            heatmap_data.append({
                "Model": row["model"],
                "Attribute": attr_display[attr],
                "value": r_val,
                "text": f"{r_val:.2f}{sig_stars(q_val)}",
                "hover": f"r = {r_val:.3f}{sig_stars(q_val)}<br>95% CI: [{ci_low:.3f}, {ci_high:.3f}]",
            })
    heatmap_df = pd.DataFrame(heatmap_data)
    heatmap_pivot = heatmap_df.pivot(index="Model", columns="Attribute", values="value")
    heatmap_text_pivot = heatmap_df.pivot(index="Model", columns="Attribute", values="text")
    heatmap_hover_pivot = heatmap_df.pivot(index="Model", columns="Attribute", values="hover")

    # Reorder columns by mean value
    col_order = heatmap_pivot.mean().sort_values(ascending=False).index.tolist()
    heatmap_pivot = heatmap_pivot[col_order]
    heatmap_text_pivot = heatmap_text_pivot[col_order]
    heatmap_hover_pivot = heatmap_hover_pivot[col_order]
    # Reorder rows to match results_df order
    heatmap_pivot = heatmap_pivot.reindex(results_df["model"].tolist())
    heatmap_text_pivot = heatmap_text_pivot.reindex(results_df["model"].tolist())
    heatmap_hover_pivot = heatmap_hover_pivot.reindex(results_df["model"].tolist())

    max_abs = max(0.1, heatmap_pivot.abs().max().max())
    fig2 = go.Figure(data=go.Heatmap(
        z=heatmap_pivot.values,
        x=heatmap_pivot.columns.tolist(),
        y=heatmap_pivot.index.tolist(),
        text=heatmap_text_pivot.values,
        texttemplate="%{text}",
        customdata=heatmap_hover_pivot.values,
        hovertemplate="%{y}<br>%{x}<br>%{customdata}<extra></extra>",
        colorscale="RdBu_r",
        zmin=-max_abs,
        zmax=max_abs,
        colorbar=dict(title="r"),
    ))
    fig2.update_layout(
        height=300,
        xaxis_title="",
        yaxis_title="",
    )
    st.plotly_chart(fig2, use_container_width=True)
    st.caption("\\* q<0.05, \\*\\* q<0.01, \\*\\*\\* q<0.001 (FDR-corrected)")

    # === Section 3: Overall Predictability ===
    st.subheader("3. Overall Predictability")
    st.markdown("""
    Model: `credence ~ β₀ + β₁(v-0.5) + β₂(v-0.5)×attr₁ + β₃(v-0.5)×attr₂ + ...`

    Attributes modulate user deference but have no direct effect on credence when user is neutral (v=0.5).
    At v=0.5, all interaction terms zero out and predicted credence = β₀.
    """)

    discard_neutral_s3 = st.checkbox(
        f"Discard samples where prompt user valence is neutral, v==0.5 ({neutral_pct:.1f}% of prompts)",
        value=True,
        key="discard_neutral_s3",
    )
    if discard_neutral_s3:
        s3_suffix, s3_n = "_dir", "n_dir"
    else:
        s3_suffix, s3_n = "_all", "n_all"

    tbl3 = results_df[["model", f"valence_only_r2{s3_suffix}", f"valence_only_cv_r2{s3_suffix}", f"interactions_r2{s3_suffix}", f"interactions_cv_r2{s3_suffix}", s3_n]].copy()
    tbl3.columns = ["Model", "val_r2", "val_cv", "int_r2", "int_cv", "N"]

    def fmt_r2(r2, cv):
        if cv is not None:
            return f"{r2:.3f} (CV: {cv:.3f})"
        return f"{r2:.3f}"

    tbl3["Valence-only R²"] = tbl3.apply(lambda row: fmt_r2(row["val_r2"], row["val_cv"]), axis=1)
    tbl3["+ Interactions R²"] = tbl3.apply(lambda row: fmt_r2(row["int_r2"], row["int_cv"]), axis=1)
    tbl3 = tbl3[["Model", "Valence-only R²", "+ Interactions R²", "N"]]
    st.dataframe(tbl3, use_container_width=True, hide_index=True)
    st.caption("Format: R² (CV: 3-fold cross-validated R²). Valence-only = baseline. + Interactions = with (v-0.5)×attr terms.")


def main():
    st.set_page_config(page_title="Exploration Results", layout="wide")
    st.title("Part 2: Exploration Results")

    # Load data
    raw_df = load_exploration_parquet()
    df = add_computed_columns(raw_df)

    tab1, tab2, tab3 = st.tabs([
        "Correlations",
        "Prompt Attributes",
        "Completeness",
    ])

    with tab1:
        render_correlations_detail_tab(df)

    with tab2:
        render_prompt_attributes_tab(df)

    with tab3:
        render_completeness_tab(df)


if __name__ == "__main__":
    main()
