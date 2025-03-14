import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import rcParams
from matplotlib import font_manager

# 设置字体
font_dirs = ['/home/zaitpub04/hyj/conda/conda/envs/RhoDesign2/lib/python3.9/site-packages/matplotlib/mpl-data/fonts']
font_files = font_manager.findSystemFonts(fontpaths=font_dirs)
for font_file in font_files:
    font_manager.fontManager.addfont(font_file)
rcParams['font.family'] = 'Arial'
rcParams['font.sans-serif'] = ['Arial']
rcParams['font.weight'] = 'bold'

# 读取RNA序列文件的函数
def read_rna_sequences(file_path):
    # 读取xlsx文件
    df = pd.read_excel(file_path)
    print("Dataframe loaded:")
    print(df.head())  # 打印数据前几行，查看内容
    return df['name'], df['seq']  # 假设文件中第一列是序列名，第二列是RNA序列

# 计算每个点的AUCG频率，频率为该点某核苷酸的数量/序列总数
def calculate_nucleotide_frequencies(sequences, wildtype_seq):
    seq_length = len(wildtype_seq)  # 假设所有序列长度相同
    sequence_count = len(sequences)  # 总序列数
    frequency_map = np.zeros((4, seq_length), dtype=float)  # 4行分别表示A, U, C, G，列表示序列的不同位置

    nucleotide_dict = {'A': 0, 'U': 1, 'C': 2, 'G': 3}

    # 遍历所有序列并记录每个位点的核苷酸频率
    for seq in sequences:
        for i in range(seq_length):
            nucleotide = seq[i]
            if nucleotide in nucleotide_dict:
                frequency_map[nucleotide_dict[nucleotide], i] += 1

    # 计算频率：每个点的频率为该核苷酸在该位置出现的次数 / 总序列数
    frequency_map /= sequence_count
    print("Frequency map:",frequency_map)
    return frequency_map

# 绘制频率热点图
def plot_mutation_heatmap(frequency_map, wildtype_seq, reference_seq_length, highlight_indices):
    """
    绘制突变热图，支持根据输入索引列表高亮X轴标签。

    参数:
        frequency_map: 热图数据 (2D array)。
        wildtype_seq: 野生型RNA序列。
        reference_seq_length: 参考序列长度。
        highlight_indices: 高亮显示的索引列表 (1-based index)。
    """
    # 创建自定义颜色渐变，从白色到蓝色
    #cmap = LinearSegmentedColormap.from_list("green_gradient", ["#00481D", "#80CA80","#E8F6E3"])
    cmap = LinearSegmentedColormap.from_list("green_gradient", ["#E8F6E3", "#80CA80","#00481D" ])
    # 筛选掉频率为0的位点，且只保留频率大于0的列
    non_zero_mask = np.sum(frequency_map > 0, axis=0) > 1  # 如果一列只有一个1，其它全为0，忽略该列
    frequency_map = frequency_map[:, non_zero_mask]  # 只保留频率大于0的列
    wildtype_seq = ''.join([wildtype_seq[i] for i in range(len(wildtype_seq)) if non_zero_mask[i]])  # 更新wildtype_seq

    # 计算X轴标注（序号 - wildtype该点位的核苷酸）
    x_labels = [f"{i + 1}-{wildtype_seq[i]}" for i in range(len(wildtype_seq))]

    # 创建热图
    fig, ax = plt.subplots(figsize=(len(wildtype_seq) / 3, 9/2))
    cax = ax.imshow(frequency_map, cmap=cmap, aspect='equal', interpolation='nearest')  # 使用自定义颜色映射绘制热图

    # 设置X轴为序号和wildtype对应核苷酸
    ax.set_xticks(np.arange(len(wildtype_seq)))
    ax.set_xticklabels(x_labels, rotation=45, ha="right")

    # 设置高亮颜色的逻辑
    xtick_labels = []
    for idx, label in enumerate(x_labels):
        if idx + 1 in highlight_indices:  # 检查是否在高亮索引列表中 (1-based)
            xtick_labels.append((label, "#4896C8"))  # 高亮颜色
        else:
            xtick_labels.append((label, "black"))  # 默认颜色

    # 设置标签颜色
    for tick, (_, color) in zip(ax.get_xticklabels(), xtick_labels):
        tick.set_color(color)

    # 设置Y轴标签
    ax.set_yticks(np.arange(4))
    ax.set_yticklabels(['A', 'U', 'C', 'G'])

    # 对每个位置进行检查并添加文本
    for i in range(frequency_map.shape[0]):
        for j in range(frequency_map.shape[1]):
            value = frequency_map[i, j]
            
            if value == 0:  # 如果是0值，将其背景色设置为自定义颜色
                plt.gca().add_patch(plt.Rectangle((j-0.5, i-0.5), 0.97, 0.97, color='#ffffff'))  # 自定义颜色
            elif value > 0:  # 如果有数值，显示数值标签
                text_color = "black" if value <=0.5 else "white"
                plt.text(j, i, f'{value:.2f}', ha='center', va='center', color=text_color, fontsize=8)

    plt.tight_layout()  # 自动调整子图的参数，使之填充整个图像区域
    plt.show()

    # 保存图片
    plt.savefig('/home/zaitpub04/hyj/RhoDesign/benchmark/paint/3/heatmap_mut/mut_hot/mut_hot_b_0203', dpi=300)
# 主函数：读取文件、计算频率、绘制图像
def main():
    # 输入文件路径和wildtype序列
    file_path = '/home/zaitpub04/hyj/RhoDesign/benchmark/paint/3/heatmap_mut/mut_hot/broccoli-hot.xlsx'
    wildtype_seq = "GAGACGGUCGGGUCCAGAUAUUCGUAUCUGUCGAGUAGAGUGUGGGCUC"  # 直接指定wildtype序列

    # 高亮索引列表 (1-based index)
    highlight_indices = []# 用户直接输入的高亮索引列表
#[6, 7, 10, 11, 12, 13, 14, 15, 31, 33, 34, 35, 36, 38, 39, 40, 42] 
    # 读取Excel文件
    seq_names, rna_sequences = read_rna_sequences(file_path)

    # 打印查看序列数据
    print(f"Sequences loaded: {len(rna_sequences)} sequences found.")

    # 检查是否有序列数据
    if len(rna_sequences) == 0:
        print("Error: No sequences found in the file.")
        return

    # 直接使用用户输入的wildtype序列
    sequences = rna_sequences  # 使用所有序列，假设没有wildtype序列在文件中

    # 计算序列的AUCG频率
    frequency_map = calculate_nucleotide_frequencies(sequences, wildtype_seq)

    # 绘制频率热点图
    plot_mutation_heatmap(frequency_map, wildtype_seq, len(wildtype_seq), highlight_indices)

# 运行主函数
if __name__ == '__main__':
    main()
