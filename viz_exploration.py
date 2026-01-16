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
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression


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

# Preregistered exclusion threshold
PROPOSITION_EXCLUSION_THRESHOLD = 0.30


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


def render_prompt_attributes_tab(df: pl.DataFrame):
    """Render prompt attributes analysis tab."""
    st.subheader("Prompt Attributes")

    st.markdown(
        "Each prompt is scored on 7 attributes by two judge models: **GPT-5 Mini** and **Claude Haiku 4.5**. "
        "Judges agree when their scores are within **0.2** of each other (on a 0-1 scale)."
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
    # Also map with ⓘ suffix for bar chart
    attr_colors_with_icon = {attr_display[attr] + " ⓘ": colors[i % len(colors)] for i, attr in enumerate(ATTRIBUTE_NAMES)}

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
                    "Attribute": attr_display[attr] + " ⓘ",
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
            color_discrete_map=attr_colors_with_icon,
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


def render_correlations_tab(df: pl.DataFrame):
    """Render attribute-credence correlations tab."""
    st.markdown("**How does expressed credence from the test model vary with prompt attributes?**")

    # Smaller table text
    st.markdown("""<style>
        [data-testid="stDataFrame"] { font-size: 0.85rem; }
    </style>""", unsafe_allow_html=True)

    if df.is_empty():
        st.warning("No data found.")
        return

    # Check if prompt attributes exist
    has_attrs = f"consensus_{ATTRIBUTE_NAMES[0]}" in df.columns
    if not has_attrs:
        st.warning("Prompt attributes not yet scored. Run run_exploration.py first.")
        return

    # Get filter options
    all_domains = sorted(df["domain"].unique().to_list()) if "domain" in df.columns else []
    all_models = sorted(df["target_model_id"].unique().to_list())
    all_model_names = [_friendly_model_name(m) for m in all_models]
    name_to_id = {_friendly_model_name(m): m for m in all_models}

    # Read filter values from session_state (displayed above table)
    display_mode = st.session_state.get("corr_display_mode", "Spearman Correlation")
    show_r2 = display_mode == "R²"
    selected_domains = st.session_state.get("corr_domains", all_domains)
    selected_model_names = st.session_state.get("corr_models", all_model_names)
    selected_models = [name_to_id[name] for name in selected_model_names if name in name_to_id]

    # Apply filters
    filtered = df
    if selected_domains:
        filtered = filtered.filter(pl.col("domain").is_in(selected_domains))
    if selected_models:
        filtered = filtered.filter(pl.col("target_model_id").is_in(selected_models))

    attrs_to_show = list(ATTRIBUTE_NAMES)

    # Display names and tooltips for attributes
    attr_display = {
        "user_valence": "user valence",
        "sophistication": "sophistication",
        "apparent_stakes": "stakes",
        "emotionality": "emotionality",
        "user_authority": "authority",
        "confrontationality": "confrontationality",
        "clarity": "clarity",
    }
    attr_tooltips = {
        "user_valence": "Implied user prior belief (skeptical to credulous) in the prompt",
        "sophistication": "Apparent education level and writing quality from the prompt",
        "apparent_stakes": "How consequential the decision/situation is from the prompt",
        "emotionality": "How emotionally charged the prompt is",
        "user_authority": "How much expertise/confidence user projects in the prompt",
        "confrontationality": "Cooperative vs adversarial stance from the user in the prompt",
        "clarity": "How specific and unambiguous the request is in the prompt",
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
        # Sort p-values and track original indices
        indexed = sorted(enumerate(p_values), key=lambda x: x[1])
        adjusted = [0.0] * n
        min_so_far = 1.0
        for rank, (orig_idx, p) in enumerate(reversed(indexed), 1):
            adjusted_p = min(p * n / (n - rank + 1), 1.0)
            min_so_far = min(min_so_far, adjusted_p)
            adjusted[orig_idx] = min_so_far
        return adjusted

    sensitivity_col = "Overall Sensitivity (R²)"

    # Table: rows = models, columns = attributes + Overall Sensitivity
    if attrs_to_show and selected_models:
        # First pass: collect all correlations and p-values
        attr_r_values: dict[str, list[float]] = {attr: [] for attr in attrs_to_show}
        model_results: list[dict] = []  # [{model_id, attr, r, p, r2_sensitivity}, ...]

        for model_id in selected_models:
            model_data = filtered.filter(pl.col("target_model_id") == model_id)
            model_entry = {"model_id": model_id, "correlations": {}, "sensitivity_r2": None}

            for attr in attrs_to_show:
                consensus_col = f"consensus_{attr}"
                if consensus_col in model_data.columns:
                    valid = model_data.select([consensus_col, "consensus_credence"]).drop_nulls().to_pandas()
                    if len(valid) >= 3:
                        r, p = spearmanr(valid[consensus_col], valid["consensus_credence"])
                        model_entry["correlations"][attr] = {"r": r, "p": p}
                        attr_r_values[attr].append(abs(r))

            # Compute Overall Sensitivity R²
            attr_cols = [f"consensus_{attr}" for attr in attrs_to_show]
            available_cols = [c for c in attr_cols if c in model_data.columns]
            if available_cols and "consensus_credence" in model_data.columns:
                reg_data = model_data.select(available_cols + ["consensus_credence"]).drop_nulls().to_pandas()
                n = len(reg_data)
                k = len(available_cols)
                if n >= k + 2:
                    X = reg_data[available_cols].values
                    y = reg_data["consensus_credence"].values
                    model_reg = LinearRegression().fit(X, y)
                    model_entry["sensitivity_r2"] = model_reg.score(X, y)

            model_results.append(model_entry)

        # Collect all p-values and apply FDR correction
        all_p_values = []
        p_value_keys = []  # (model_idx, attr) tuples to map back
        for model_idx, entry in enumerate(model_results):
            for attr, corr in entry["correlations"].items():
                all_p_values.append(corr["p"])
                p_value_keys.append((model_idx, attr))

        if all_p_values:
            adjusted_p = apply_fdr_correction(all_p_values)
            for i, (model_idx, attr) in enumerate(p_value_keys):
                model_results[model_idx]["correlations"][attr]["q"] = adjusted_p[i]

        # Build table rows
        model_corr_rows = []
        for entry in model_results:
            row = {"Test Model": _friendly_model_name(entry["model_id"])}
            for attr in attrs_to_show:
                if attr in entry["correlations"]:
                    corr = entry["correlations"][attr]
                    r = corr["r"]
                    q = corr["q"]
                    if show_r2:
                        row[attr] = f"{r**2:.3f}{sig_stars(q)}"
                    else:
                        row[attr] = f"{r:+.3f}{sig_stars(q)}"
                else:
                    row[attr] = "-"

            if entry["sensitivity_r2"] is not None:
                row[sensitivity_col] = f"{entry['sensitivity_r2']:.3f}"
            else:
                row[sensitivity_col] = "-"

            model_corr_rows.append(row)

        if model_corr_rows:
            # Sort attributes by mean |r| descending
            attr_mean_abs_r = {attr: (sum(vals) / len(vals) if vals else 0) for attr, vals in attr_r_values.items()}
            sorted_attrs = sorted(attrs_to_show, key=lambda a: attr_mean_abs_r[a], reverse=True)

            model_corr_df = pd.DataFrame(model_corr_rows)
            model_corr_df = model_corr_df[["Test Model"] + sorted_attrs + [sensitivity_col]]

            # Rename columns to display names
            rename_map = {attr: attr_display[attr] for attr in sorted_attrs}
            model_corr_df = model_corr_df.rename(columns=rename_map)

            # Build column config with tooltips (add ⓘ to hint hover)
            col_config = {
                "Test Model": st.column_config.TextColumn("Test Model", help="Model under test"),
                sensitivity_col: st.column_config.TextColumn(sensitivity_col + " ⓘ", help="R² from regressing credence on all prompt attributes"),
            }
            for attr in sorted_attrs:
                display_name = attr_display[attr]
                col_config[display_name] = st.column_config.TextColumn(display_name + " ⓘ", help=attr_tooltips[attr])

            # Controls row: display mode toggle + filters
            ctrl_col1, ctrl_col2, ctrl_col3 = st.columns([1, 1.5, 1.5])
            with ctrl_col1:
                st.radio("Display mode", ["Spearman Correlation", "R²"], horizontal=True, key="corr_display_mode", label_visibility="collapsed")
            with ctrl_col2:
                with st.expander("Domains", expanded=False):
                    st.multiselect("Select domains", all_domains, default=all_domains, key="corr_domains", label_visibility="collapsed")
            with ctrl_col3:
                with st.expander("Test Models", expanded=False):
                    st.multiselect("Select models", all_model_names, default=all_model_names, key="corr_models", label_visibility="collapsed")

            # Table
            st.dataframe(model_corr_df, use_container_width=True, hide_index=True, column_config=col_config)
            st.caption("\\* q<0.05, \\*\\* q<0.01, \\*\\*\\* q<0.001 (FDR-corrected). Overall Sensitivity: variance explained by all prompt attributes.")

    # Coefficient plot: dots showing r or r² values per model, colored by attribute
    if attrs_to_show and selected_models and model_corr_rows:
        coef_data = []
        for entry in model_results:
            model_name = _friendly_model_name(entry["model_id"])
            for attr in sorted_attrs:
                if attr in entry["correlations"]:
                    corr = entry["correlations"][attr]
                    coef_data.append({
                        "Model": model_name,
                        "Attribute": attr_display[attr],
                        "value": corr["r"]**2 if show_r2 else corr["r"],
                        "significant": corr["q"] < 0.05,
                    })

        if coef_data:
            coef_df = pd.DataFrame(coef_data)
            x_label = "r²" if show_r2 else "Spearman r"
            # Compute dynamic range based on actual data
            min_val = coef_df["value"].min()
            max_val = coef_df["value"].max()
            padding = (max_val - min_val) * 0.1 if max_val > min_val else 0.1
            if show_r2:
                x_range = [0, max(0.5, max_val + padding)]
            else:
                x_range = [min(-0.5, min_val - padding), max(0.5, max_val + padding)]
            # Build plot manually for cleaner legend
            fig = go.Figure()
            colors = px.colors.qualitative.Plotly
            attr_colors = {attr_display[attr]: colors[i % len(colors)] for i, attr in enumerate(sorted_attrs)}

            for attr in sorted_attrs:
                attr_name = attr_display[attr]
                color = attr_colors[attr_name]
                attr_data = coef_df[coef_df["Attribute"] == attr_name]

                # Significant (filled)
                sig_data = attr_data[attr_data["significant"]]
                if not sig_data.empty:
                    fig.add_trace(go.Scatter(
                        x=sig_data["value"], y=sig_data["Model"],
                        mode="markers",
                        marker=dict(size=10, color=color, symbol="circle"),
                        name=attr_name,
                        legendgroup=attr_name,
                        showlegend=True,
                    ))

                # Not significant (open)
                nonsig_data = attr_data[~attr_data["significant"]]
                if not nonsig_data.empty:
                    fig.add_trace(go.Scatter(
                        x=nonsig_data["value"], y=nonsig_data["Model"],
                        mode="markers",
                        marker=dict(size=10, color=color, symbol="circle-open", line=dict(width=2, color=color)),
                        name=attr_name,
                        legendgroup=attr_name,
                        showlegend=sig_data.empty,  # Only show in legend if no filled version
                    ))

            if not show_r2:
                fig.add_vline(x=0, line_dash="dash", line_color="gray")
            fig.update_layout(
                height=250 + 30 * len(selected_models),
                xaxis=dict(range=x_range, title=x_label),
                yaxis=dict(title=""),
                legend_title_text="",
            )
            st.plotly_chart(fig, use_container_width=True)
            st.caption("Filled = significant (q < 0.05, FDR-corrected).")


def main():
    st.set_page_config(page_title="Exploration Results", layout="wide")
    st.title("Part 2: Exploration Results")

    # Load data
    raw_df = load_exploration_parquet()
    df = add_computed_columns(raw_df)

    tab1, tab2, tab3 = st.tabs([
        "Attribute-Credence Correlations",
        "Prompt Attributes",
        "Completeness",
    ])

    with tab1:
        render_correlations_tab(df)

    with tab2:
        render_prompt_attributes_tab(df)

    with tab3:
        render_completeness_tab(df)


if __name__ == "__main__":
    main()
