import json
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from matplotlib import font_manager
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import rcParams
from matplotlib.ticker import MultipleLocator

# 设置字体
rcParams['font.size'] = 10
rcParams['font.family'] = 'Arial'
rcParams['font.sans-serif'] = ['Arial']
rcParams['font.weight'] = 'bold'
font_dirs = ['/home/zaitpub04/hyj/conda/conda/envs/RhoDesign2/lib/python3.9/site-packages/matplotlib/mpl-data/fonts']
font_files = font_manager.findSystemFonts(fontpaths=font_dirs)
for font_file in font_files:
    font_manager.fontManager.addfont(font_file)

# 从JSON文件中读取突变数据
def read_mutations_from_json(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

# 解析突变数据，提取位点和突变信息
def parse_mutations(mutation_data):
    # 创建一个字典来存储每个位点和每个突变的brightness值
    mutation_map = defaultdict(lambda: {'A': [], 'U': [], 'C': [], 'G': []})
    mutated_positions = set()  # 用于记录发生突变的位点

    for mutation in mutation_data:
        name = mutation.get('name', '')
        brightness_brightness = mutation.get('brightness', 0)  # Use 'brightness' field for brightness value

        mutations = name.split('_')  # 按','分割每个位点的突变
        for mut in mutations:
            original = mut[0]  # 原始核苷酸（如 A）
            position = int(mut[1:-1]) - 1  # 位点位置
            mutated = mut[-1]  # 突变后的核苷酸（如 G）

            # 将对应突变的brightness值加入该位置和突变核苷酸的列表
            mutation_map[position][mutated].append(brightness_brightness)  # Normalize brightness
            mutated_positions.add(position)  # 记录发生突变的位点
    #print(mutation_map)
    #print(mutated_positions)
    return mutation_map, mutated_positions
def add_mutations(mutation_data,mutation_map,mutated_positions,original_sequence):
    for mutation in mutation_data:
        name = mutation.get('name', '')
        brightness_brightness = mutation.get('brightness', 0)  # Use 'brightness' field for brightness value
        mutations = name.split('_')  # 按','分割每个位点的突变
        mut_info=set()
        for mut in mutations:
            original = mut[0]
            position = int(mut[1:-1]) - 1
            mutated = mut[-1]
            mut_info.add(position)
        #print('mutated_positions:',mutated_positions)
        #print('mut_info:',mut_info)
        mut_add=mutated_positions-mut_info
        #print('mut_add:',mut_add)
        #print('mut_add:',mut_add)
        for position in mut_add:
            original = original_sequence[position]
            #print('p,o:',position,original)
            mutation_map[position][original].append(brightness_brightness)
    #print(mutation_map)
    return mutation_map, mutated_positions

# 创建热图数据
def create_heatmap_data(mutation_map, total_sequences, mutated_positions, original_sequence):
    # 计算每个位置的AUCG突变的brightness平均值
    heatmap_data = []
    for position in sorted(mutated_positions):
        row = []
        for i, nucleotide in enumerate(['A', 'U', 'C', 'G']):
            # 计算该位置、该核苷酸的平均brightness
            brightness_values = mutation_map[position][nucleotide]
            if brightness_values:
                avg_brightness = np.mean(brightness_values)
            else:
                avg_brightness = np.nan  # 如果没有突变数据，设置为nan，表示无数据
                  
            row.append(avg_brightness)
        heatmap_data.append(row)

    return np.array(heatmap_data)
def plot_mutation_heatmap(heatmap_data, total_positions, mutated_positions, mutation_map, original_sequence, highlight_indices):
    """
    绘制突变热图，支持根据输入索引列表高亮X轴标签。

    参数:
        heatmap_data: 热图数据 (2D array)。
        total_positions: 总位置数。
        mutated_positions: 突变位置集合。
        mutation_map: 突变映射。
        original_sequence: 原始RNA序列。
        highlight_indices: 高亮显示的索引列表 (1-based index)。
    """
    # 动态调整图像尺寸
    num_rows, num_cols = heatmap_data.T.shape  # 行数和列数
    fig_width = num_cols * 0.5  # 每列占固定宽度
    fig_height = num_rows * 0.5  # 每行占固定高度
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    # 创建自定义蓝色渐变色，从浅蓝到深蓝
    cmap = LinearSegmentedColormap.from_list("green_gradient", ["#00481D", "#80CA80","#E8F6E3" ])
    #cmap = LinearSegmentedColormap.from_list("green_gradient", ["#E8F6E3", "#80CA80","#00481D"])
    # 使用自定义颜色映射绘制热图
    cax = ax.imshow(heatmap_data.T, cmap=cmap, aspect='equal')  # aspect='equal' 保证每个子格是正方形

    # 设置X轴标签，显示原始序列中对应位置的核苷酸和索引
    x_labels = []
    for position in sorted(mutated_positions):
        original_nucleotide = original_sequence[position]  # 获取原始序列中对应位置的核苷酸
        x_labels.append(f"{original_nucleotide}_{position+1}")

    ax.set_xticks(np.arange(len(mutated_positions)))

    # 设置高亮颜色逻辑
    xtick_labels = []
    for idx, label in enumerate(x_labels):
        position = sorted(mutated_positions)[idx] + 1  # 获取1-based索引
        if position in highlight_indices:  # 如果索引在高亮列表中
            xtick_labels.append((label, "#80CA80"))  # 高亮颜色
        else:
            xtick_labels.append((label, "black"))  # 默认颜色

    # 设置X轴标签和颜色
    for tick, (_, color) in zip(ax.get_xticklabels(), xtick_labels):
        tick.set_color(color)
    ax.set_xticklabels([label for label, _ in xtick_labels], rotation=45, fontsize=9)

    # 设置Y轴标签
    ax.set_ylabel("Mutations", fontsize=12, fontweight="bold")
    ax.set_xlabel("Residue Positions", fontsize=12, fontweight="bold")
    ax.set_yticks(np.arange(4))
    ax.set_yticklabels(["A", "U", "C", "G"])

    # 添加数值标签，不论是0还是1，都显示在热图中
    for i, row in enumerate(heatmap_data):
        for j, value in enumerate(row):
            if not np.isnan(value):  # 如果值不是nan，添加数值标签
                color = "white" if value <=0.6 else "black"  # 如果数值大于0.8，字体颜色为白色
                ax.text(i, j, f"{value:.2f}", ha="center", va="center", color=color)

    # 添加colorbar
    cbar = fig.colorbar(cax)
    cbar.set_label("Fluorescence Change\n(DFHBI-1T)", fontweight="bold")

    # 调整布局
    #plt.tight_layout(pad=1.0)

    # 保存图片
    plt.savefig(
        "/home/zaitpub04/hyj/RhoDesign/benchmark/paint/3/heatmap_mut/broccoli_heat_horizontal_fake2——3.12.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()


# 主函数：从文件中读取数据并绘制热点图
def main():
    # 输入文件路径
    file_path = '/home/zaitpub04/hyj/RhoDesign/benchmark/paint/3/heatmap_mut/broccoli1.26.json'  # 这里使用你上传的JSON文件路径

    # 输入原始序列
    original_sequence = "GAGACGGUCGGGUCCAGAUAUUCGUAUCUGUCGAGUAGAGUGUGGGCUC"  # 请替换为您的原始序列

    # 高亮索引列表
    highlight_indices = []  # 第一个列表

    # 读取突变数据
    mutation_data = read_mutations_from_json(file_path)

    # 解析突变信息
    mutation_map, mutated_positions = parse_mutations(mutation_data)
    added_mutation_map,mutated_positions = add_mutations(mutation_data,mutation_map,mutated_positions,original_sequence)

    # 计算热图数据
    total_sequences = len(mutation_data)  # 假设每个数据条目代表一个突变序列
    heatmap_data = create_heatmap_data(added_mutation_map, total_sequences, mutated_positions, original_sequence)

    # 绘制频率热点图
    plot_mutation_heatmap(heatmap_data, len(mutated_positions), mutated_positions, mutation_map, original_sequence, highlight_indices)


# 运行主函数
if __name__ == '__main__':
    main()
