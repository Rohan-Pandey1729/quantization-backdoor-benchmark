"""
Visualization utilities for generating paper figures.

This script generates:
1. Heatmaps of detection rates (defense x quantization)
2. Bar charts comparing defenses
3. Accuracy vs quantization plots
4. Summary tables in LaTeX format
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional
import argparse


def set_paper_style():
    """Set matplotlib style for paper-quality figures."""
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        'font.size': 12,
        'font.family': 'serif',
        'axes.labelsize': 14,
        'axes.titlesize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 11,
        'figure.figsize': (8, 6),
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight'
    })


def load_results(results_path: str) -> pd.DataFrame:
    """Load results from CSV or JSON."""
    if results_path.endswith('.csv'):
        return pd.read_csv(results_path)
    elif results_path.endswith('.json'):
        return pd.read_json(results_path)
    else:
        raise ValueError("Results file must be .csv or .json")


def plot_detection_heatmap(
    df: pd.DataFrame,
    output_path: str,
    title: str = "Defense Detection Rate Under Quantization"
):
    """
    Create a heatmap of detection rates.
    Rows: Defenses
    Columns: Quantization schemes
    """
    # Pivot table
    pivot = df.pivot_table(
        values='detected',
        index='defense',
        columns='quant_scheme',
        aggfunc='mean'
    )
    
    # Reorder columns
    col_order = ['fp32', 'int8_dynamic', 'int8_static', 'int4_simulated']
    pivot = pivot[[c for c in col_order if c in pivot.columns]]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Heatmap
    sns.heatmap(
        pivot,
        annot=True,
        fmt='.2f',
        cmap='RdYlGn',
        vmin=0,
        vmax=1,
        ax=ax,
        cbar_kws={'label': 'Detection Rate'}
    )
    
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.set_xlabel('Quantization Scheme', fontsize=14)
    ax.set_ylabel('Defense Method', fontsize=14)
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    
    print(f"Saved heatmap to {output_path}")


def plot_defense_comparison(
    df: pd.DataFrame,
    output_path: str,
    title: str = "Defense Performance Comparison"
):
    """
    Grouped bar chart comparing defenses across quantization schemes.
    """
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Aggregate
    agg = df.groupby(['defense', 'quant_scheme'])['detected'].mean().reset_index()
    
    defenses = sorted(agg['defense'].unique())
    quant_schemes = ['fp32', 'int8_dynamic', 'int4_simulated']
    quant_schemes = [q for q in quant_schemes if q in agg['quant_scheme'].unique()]
    
    x = np.arange(len(defenses))
    width = 0.25
    
    colors = ['#27ae60', '#2980b9', '#c0392b']
    
    for i, scheme in enumerate(quant_schemes):
        data = agg[agg['quant_scheme'] == scheme]
        rates = []
        for d in defenses:
            val = data[data['defense'] == d]['detected'].values
            rates.append(val[0] if len(val) > 0 else 0)
        
        bars = ax.bar(x + i * width, rates, width, label=scheme, color=colors[i % len(colors)])
        
        # Add value labels on bars
        for bar, rate in zip(bars, rates):
            height = bar.get_height()
            ax.annotate(f'{rate:.2f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=9)
    
    ax.set_xlabel('Defense Method', fontsize=14)
    ax.set_ylabel('Detection Rate', fontsize=14)
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels([d.upper() for d in defenses])
    ax.legend(title='Quantization', loc='upper right')
    ax.set_ylim(0, 1.2)
    
    # Add horizontal lines
    ax.axhline(y=1.0, color='gray', linestyle='-', alpha=0.3)
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    
    print(f"Saved defense comparison to {output_path}")


def plot_quantization_impact(
    df: pd.DataFrame,
    output_path: str,
    title: str = "Impact of Quantization on Detection"
):
    """
    Line plot showing how detection rate changes with quantization.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Aggregate by defense and quantization
    agg = df.groupby(['defense', 'quant_scheme'])['detected'].mean().reset_index()
    
    defenses = agg['defense'].unique()
    quant_order = ['fp32', 'int8_dynamic', 'int8_static', 'int4_simulated']
    quant_order = [q for q in quant_order if q in agg['quant_scheme'].unique()]
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(defenses)))
    
    for i, defense in enumerate(defenses):
        data = agg[agg['defense'] == defense]
        rates = []
        for q in quant_order:
            val = data[data['quant_scheme'] == q]['detected'].values
            rates.append(val[0] if len(val) > 0 else np.nan)
        
        ax.plot(quant_order, rates, 'o-', label=defense.upper(), 
                color=colors[i], linewidth=2, markersize=8)
    
    ax.set_xlabel('Quantization Scheme', fontsize=14)
    ax.set_ylabel('Detection Rate', fontsize=14)
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.legend(title='Defense', loc='best', ncol=2)
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    
    print(f"Saved quantization impact to {output_path}")


def generate_latex_table(
    df: pd.DataFrame,
    output_path: str
):
    """
    Generate a LaTeX table summarizing results.
    """
    # Pivot table
    pivot = df.pivot_table(
        values='detected',
        index='defense',
        columns='quant_scheme',
        aggfunc='mean'
    )
    
    # Format for LaTeX
    latex_str = pivot.to_latex(
        float_format="%.3f",
        caption="Detection rates of backdoor defenses under different quantization schemes. "
                "Higher values indicate better defense performance.",
        label="tab:detection_rates",
        bold_rows=True
    )
    
    with open(output_path, 'w') as f:
        f.write(latex_str)
    
    print(f"Saved LaTeX table to {output_path}")
    
    return latex_str


def generate_all_figures(results_path: str, output_dir: str):
    """Generate all paper figures from results."""
    
    set_paper_style()
    os.makedirs(output_dir, exist_ok=True)
    
    # Load results
    df = load_results(results_path)
    
    print(f"Loaded {len(df)} results")
    print(f"Defenses: {df['defense'].unique()}")
    print(f"Quantization schemes: {df['quant_scheme'].unique()}")
    print(f"Attacks: {df['attack'].unique()}")
    
    # Generate figures
    plot_detection_heatmap(
        df, 
        os.path.join(output_dir, 'fig_heatmap.pdf')
    )
    
    plot_defense_comparison(
        df,
        os.path.join(output_dir, 'fig_defense_comparison.pdf')
    )
    
    plot_quantization_impact(
        df,
        os.path.join(output_dir, 'fig_quant_impact.pdf')
    )
    
    # Generate LaTeX table
    generate_latex_table(
        df,
        os.path.join(output_dir, 'table_results.tex')
    )
    
    print(f"\nAll figures saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Generate paper figures')
    parser.add_argument('--results', default='./results/results.csv',
                        help='Path to results file')
    parser.add_argument('--output_dir', default='./figures',
                        help='Output directory for figures')
    
    args = parser.parse_args()
    
    generate_all_figures(args.results, args.output_dir)


if __name__ == "__main__":
    main()