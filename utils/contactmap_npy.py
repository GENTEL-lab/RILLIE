import numpy as np

def ct_to_adjacency_matrix(ct_file):
    with open(ct_file, 'r') as file:
        lines = file.readlines()
    
    # 获取碱基数量
    num_bases = int(lines[0].split()[0])
    
    # 初始化邻接矩阵
    adjacency_matrix = np.zeros((num_bases, num_bases), dtype=int)
    
    # 解析每一行并填充邻接矩阵
    for line in lines[1:]:
        parts = line.split()
        index = int(parts[0]) - 1
        paired_index = int(parts[4]) - 1
        
        if paired_index >= 0:
            adjacency_matrix[index][paired_index] = 1
            adjacency_matrix[paired_index][index] = 1
    
    return adjacency_matrix

def save_adjacency_matrix(adjacency_matrix, output_file):
    np.save(output_file, adjacency_matrix)

# 使用示例
ct_file = '/home/zaitpub04/hyj/RhoDesign/RhoDesign/src/choose_mutation/bro_structure/bro/ss.ct'  # 替换为你的 .ct 文件路径
output_file = '/home/zaitpub04/hyj/RhoDesign/RhoDesign/data/npy/rho_bro.npy'  # 替换为你想要保存的 .npy 文件路径

adjacency_matrix = ct_to_adjacency_matrix(ct_file)
save_adjacency_matrix(adjacency_matrix, output_file)

print(f"邻接矩阵已保存为 {output_file}")

