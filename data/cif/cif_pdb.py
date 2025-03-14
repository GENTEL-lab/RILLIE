import os
from Bio.PDB import MMCIFParser, PDBIO

def convert_cif_to_pdb(cif_file, pdb_file):
    # 创建一个 MMCIFParser 对象
    parser = MMCIFParser()
    
    # 解析 CIF 文件
    structure = parser.get_structure('structure', cif_file)
    
    # 创建一个 PDBIO 对象
    io = PDBIO()
    # 设置要保存的结构
    io.set_structure(structure)
    
    # 获取PDB文件的目录
    pdb_dir = os.path.dirname(pdb_file)
    
    # 检查目录是否存在，如果不存在则创建
    if not os.path.exists(pdb_dir):
        os.makedirs(pdb_dir)
    
    # 保存为 PDB 文件
    io.save(pdb_file)

# 示例用法
cif_file = '/home/zaitpub04/hyj/3d/3d/B12.cif'
pdb_file = '/home/zaitpub04/hyj/3d/3d/B12_good.pdb'
convert_cif_to_pdb(cif_file, pdb_file)