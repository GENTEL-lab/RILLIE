import json
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib import font_manager
import matplotlib.colors as mcolors
from matplotlib import rcParams

# Load custom fonts if necessary
font_dirs = ['./fonts']
font_files = font_manager.findSystemFonts(fontpaths=font_dirs)
for font_file in font_files:
    font_manager.fontManager.addfont(font_file)

rcParams['font.family'] = 'Arial'
rcParams['font.sans-serif'] = ['Arial']

def plot_bar_chart(data_file, focus, bar_colors):
    np.random.seed(42)

    with open(data_file, 'r') as file:
        data = json.load(file)

    if focus not in data:
        raise ValueError("Focus must be either 'Spearman' or 'Pearson'")

    x_categories = [
        "rho_score", "AIDO_score", "rinalmo_score", "rnafm_score",
        "rnamsm_score", "rnabert_score", "evo_score", "nt_score", "gena_score", "grover_score"
    ]

    if len(bar_colors) != len(x_categories):
        raise ValueError("The number of bar colors must match the number of selected x-axis categories.")

    averages = []
    scatter_data = {}
    for category in x_categories:
        values = [abs(data[focus][key][category]) for key in data[focus].keys()]
        averages.append(np.mean(values))
        scatter_data[category] = values

    fig, ax = plt.subplots(figsize=(6, 6))
    bars = plt.bar(x_categories, averages, color=bar_colors, label="Average", zorder=1)

    for i, category in enumerate(x_categories):
        y_values = scatter_data[category]
        x_values = np.random.normal(i, 0.05, len(y_values))
        plt.scatter(x_values, y_values, color="black", s=8, alpha=1, zorder=2)

    ylabel_text = "|Spearman r|" if focus == 'Spearman' else "|Pearson r|"
    plt.ylabel(ylabel_text, fontsize=12, fontweight='semibold')
    plt.title("Zero-shot ncRNA Fitness Prediction", fontsize=14, fontweight='semibold')

    x_label_mapping = {
        "rho_score": "RhoDesign\n2024 | 37.1M",
        "AIDO_score": "AIDO\n2024 | 1.6B",
        "rinalmo_score": "Rinalmo\n2024 | 651M",
        "rnafm_score": "RNAFM\n2022 | 99.5M",
        "rnamsm_score": "RNAMSM\n2023 | 96.5M",
        "rnabert_score": "RNABERT\n2022 | 0.53M",
        "evo_score": "Evo\n2024 | 6.45B",
        "nt_score": "Nucl. Trans.\n2023 | 500M",
        "gena_score": "GENA\n2023 | 110M",
        "grover_score": "Grover\n2023 | 86.5M"
    }

    mapped_labels = []
    for label in x_categories:
        original_label = x_label_mapping.get(label, label)
        parts = original_label.split('\n')
        if len(parts) == 2:
            formatted_label = f"$\\bf{{{parts[0]}}}$\n{parts[1]}"
            mapped_labels.append(formatted_label)
        else:
            mapped_labels.append(label)

    original_positions = np.arange(len(x_categories))
    ax.set_xticks(original_positions)
    ax.set_xticklabels([])

    shifted_positions = original_positions + 0.5
    for x, label in zip(shifted_positions, mapped_labels):
        ax.text(x, -0.22, label, fontsize=9, ha='right', rotation=45, fontweight='normal')

    ax.text(1.015, 0.1, "Better", transform=ax.transAxes,
            rotation=90, fontsize=12, color="black", ha='left', va='center')

    ax.annotate('', xy=(1.03, 1.0), xytext=(1.03, 0.2),
                xycoords=ax.transAxes,
                arrowprops=dict(arrowstyle='->', color='black', lw=1.5))

    ax.yaxis.grid(True, linestyle='--', alpha=0.7, zorder=0)

    legend_colors = ['#FF6D15', '#FE9D3E', '#77A9D3']
    legend_labels = ['Structure model', 'RNA Language model', 'DNA language model']
    legend_handles = [plt.Rectangle((0, 0), 1, 1, color=color) for color in legend_colors]
    ax.legend(legend_handles, legend_labels, loc='upper center', bbox_to_anchor=(0.5, -0.35), ncol=3, frameon=False, fontsize=10)
    ax.text(0.35, -0.53, "● Experimental Study", transform=ax.transAxes,
            fontsize=10, color="black", ha='left', va='center')

    plt.subplots_adjust(bottom=0.45, right=0.9)
    plt.yticks(fontsize=14, fontweight='semibold')

    output_file = "./figure.png"
    plt.savefig(output_file, dpi=300)
    print(f"Figure saved as {output_file}")

    plt.show()

# Main function to execute the script
if __name__ == "__main__":
    plot_bar_chart(
        "./RILLIE/data/ncRNA_fitness_prediction.json",  # Path to data file (change as needed)
        "Spearman",  # "Pearson" can also be used
        ["#FF6D15", "#FE9D3E", "#FFAE60", "#FFCA9B", "#FFE0C1", "#FFEDDB", "#77A9D3", "#98BBDE", "#BED6EA", "#D8E7F3"]
    )

