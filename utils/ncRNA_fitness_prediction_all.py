import matplotlib.pyplot as plt
import numpy as np
import json
from matplotlib import rcParams
from matplotlib import font_manager

# Load custom fonts if necessary
font_dirs = ['./fonts']
font_files = font_manager.findSystemFonts(fontpaths=font_dirs)
for font_file in font_files:
    font_manager.fontManager.addfont(font_file)

rcParams['font.family'] = 'Arial'
rcParams['font.sans-serif'] = ['Arial']

# Load JSON data from file
file_path = "./data.json"
with open(file_path, "r") as file:
    data = json.load(file)

# Choose "Spearman" or "Pearson"
selection = "Spearman"

# Extract selected data
selected_data = data[selection]

# Define colors
colors = ["#FF6D15", "#FE9D3E", "#FFAE60", "#FFCA9B", "#FFE0C1", "#FFEDDB", "#77A9D3", "#98BBDE", "#BED6EA", "#D8E7F3"]

# Extract groups and keys
groups = list(selected_data.keys())
keys = list(selected_data[groups[0]].keys())
values_per_group = [[abs(selected_data[group][key]) for key in keys] for group in groups]

# Adjust figure layout
plt.figure(figsize=(20, 5))
bar_width = 0.85 / len(keys)
x_positions = np.linspace(0, len(groups) - 1, len(groups))

# Draw bars
for i, key in enumerate(keys):
    group_values = [values[i] for values in values_per_group]
    plt.bar(
        x_positions + i * bar_width - bar_width * len(keys) / 2,
        group_values,
        width=bar_width,
        color=colors[i],
        edgecolor='black',
        label=key
    )

# Define x-axis labels
key_names = ['RhoDesign', 'Rinalmo', 'AIDO', 'Rnafm', 'Rnabert', 'Rnamsm', "Grover", "GENA", "NT-Transformer", "Evo"]
key_labels = [key for _ in groups for key in key_names]
xtick_positions = [x + i * bar_width - bar_width * len(keys) / 2 for x in x_positions for i in range(len(keys))]
plt.xticks(xtick_positions, key_labels, rotation=45, fontsize=12, ha='right', fontweight='bold')
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Add group labels
for idx, group in enumerate(groups):
    plt.text(
        x_positions[idx], 0.63, group, fontsize=16, ha='center', fontweight='bold'
    )

# Set axis labels and title
ylabel_text = "|Spearman r|" if selection == 'Spearman' else "|Pearson r|"
plt.ylabel(ylabel_text, fontsize=18, fontweight='semibold')
plt.title("Zero-shot Performance on Different Datasets", fontsize=20, fontweight='semibold')

# Hide legend
plt.legend().set_visible(False)

# Adjust x-axis limits
plt.xlim(-0.58, len(groups) - 0.5)

plt.yticks(fontsize=16, fontweight='bold')

# Adjust layout
plt.subplots_adjust(bottom=0.4)

# Save and show the figure
output_file = "./figure.png"
plt.savefig(output_file, dpi=300)
print(f"Figure saved as {output_file}")

plt.show()
