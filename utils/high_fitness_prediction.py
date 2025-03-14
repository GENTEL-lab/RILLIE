import matplotlib.pyplot as plt
from matplotlib import rcParams
import numpy as np
from matplotlib import font_manager

# Load custom fonts if necessary
font_dirs = ['./fonts']
font_files = font_manager.findSystemFonts(fontpaths=font_dirs)
for font_file in font_files:
    font_manager.fontManager.addfont(font_file)

rcParams['font.family'] = 'Arial'

def read_data(file_path):
    """Reads and processes data from the given file."""
    data = {}
    with open(file_path, 'r') as file:
        lines = file.readlines()
        current_section = None
        for line in lines:
            if line.strip() in ["glms_obs", "Okra", "Pairwise_tRNA", "Pepper", "tRNA_single", "Clivia"]:
                current_section = line.strip()
                data[current_section] = []
            elif line.strip():
                parts = line.split(', ')
                x = parts[0].split(': ')[1]
                y = parts[1].split(': ')[1]
                if x == 'rnabert_score' or y == 'rnabert_score':
                    continue  # Skip rnabert data
                percentage = float(parts[2].split(': ')[1])
                ratio = float(parts[3].split(': ')[1])
                data[current_section].append((x, y, percentage, ratio))
    return data

# Pearson correlation data
spearman_data = {
    "rho_score": [0.0978216, 0.307343294, 0.439637654, 0.199604854, 0.122149955, 0.317324268],
    "AIDO_score": [0.2057622, 0.275325162, 0.500541418, 0.06363412, 0.160210364, 0.243335808],
    "rinalmo_score": [0.0661000, 0.2921, 0.3934, 0.222, 0.6507, 0.0416],
    "rnafm_score": [0.0454310, 0.378104239, 0.251248924, 0.002348292, 0.100511318, 0.152384901],
    "rnamsm_score": [0.1216481, 0.166603246, 0.247556944, 0.004420315, 0.152933324, 0.071953862],
    "grover_score": [0.039158938, 0.040726532, 0.026057338, 0.048024882, 0.01213999, 0.161168481],
    "gena_score": [0.042445373, 0.328228481, 0.048344694, 0.048209062, 0.091981907, 0.019711988],
    "nt_score": [0.00684666, 0.203175197, 0.052962511, 0.149784219, 0.111867722, 0.204519423],
    "evo_score": [0.0518933, 0.055814437, 0.420698212, 0.024058486, 0.002277491, 0.128434325]
}

def plot_performance(data, section_titles):
    mapping = {
        "rho_score": "RhoDesign",
        "AIDO_score": "AIDO",
        "rinalmo_score": "Rinalmo",
        "rnafm_score": "RNAFM",
        "rnamsm_score": "RNAMSM",
        "evo_score": "Evo",
        "nt_score": "Nucl. Trans.",
        "gena_score": "GENA",
        "grover_score": "Grover"
    }

    fig, axes = plt.subplots(2, 3, figsize=(11, 5))
    axes = axes.ravel()

    for idx, (section, values) in enumerate(data.items()):
        model_names = []
        base_ratios = []
        rho_ratios = {}

        for v in values:
            if v[0] == v[1]:
                model_names.append(v[0])
                base_ratios.append(v[3])

        for v in values:
            if v[0] == 'rho_score' and v[1] in model_names:
                rho_ratios[v[1]] = v[3]

        ax = axes[idx]
        y_pos = range(len(model_names))

        bars = ax.barh(y_pos, base_ratios, height=0.4, color='#0072b2', alpha=1)

        for i, model in enumerate(model_names):
            if model in rho_ratios:
                if rho_ratios[model] > base_ratios[i]:
                    ax.barh(i, rho_ratios[model] - base_ratios[i],
                            height=0.4, left=base_ratios[i],
                            color='#fd7c1a', alpha=1)
                else:
                    ax.barh(i, base_ratios[i] - rho_ratios[model],
                            height=0.4, left=rho_ratios[model],
                            color='#fcb93e', alpha=0.6)

            if model in spearman_data:
                pearson_value = spearman_data[model][idx]
                ax.barh(i, pearson_value, height=0.4, color='#f7ec44', alpha=0.5)

        display_names = [mapping.get(name, name) for name in model_names]

        ax.set_yticks(y_pos)
        ax.set_yticklabels(display_names, fontsize=12, fontweight='bold')

        ax.set_title(section_titles.get(section, section), fontweight='bold', fontsize=14)
        ax.tick_params(axis='x', labelsize=12)
        for label in ax.get_xticklabels():
            label.set_fontweight('bold')

        for v in values:
            if v[0] == 'rho_score' and v[1] == 'rho_score':
                ax.axvline(x=v[3], color='gray', linestyle='--', alpha=0.5)
                break

        ax.grid(False)

    plt.tight_layout()
    plt.savefig('./horizontal_bars.png', dpi=600, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    file_path = "./RILLIE/data/high_fitness_precision.txt"
    data = read_data(file_path)

    section_titles = {
        "glms_obs": "Glms\n(ribozyme)",
        "Okra": "Okra\n(fluorescent aptamer)",
        "Pairwise_tRNA": "Pair_tRNA\n(tRNA)",
        "Pepper": "Pepper\n(fluorescent aptamer)",
        "tRNA_single": "Single_tRNA\n(tRNA)",
        "Clivia": "Clivia\n(fluorescent aptamer)"
    }

    plot_performance(data, section_titles)
