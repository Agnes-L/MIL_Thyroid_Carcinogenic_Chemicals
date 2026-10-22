import argparse
import torch
import numpy as np
import pandas as pd
import os
import glob
from tqdm import tqdm
from torch.utils.data import DataLoader
from rdkit import Chem

# 只需要 dataset_graph 里的这两个函数来处理数据
from dataset_graph import construct_dataset, mol_collate_func
from transformer_graph_finetune import make_model


# ==========================================
# 动态获取模型参数 (终极真相版)
# ==========================================
def get_model_config(model_name):
    # 默认的主流配置 (完美匹配 DIO1/2, NIS 等 9 个最新模型)
    config = {
        'd_atom': 115,
        'd_edge': 13,
        'd_model': 256,
        'N': 8,              
        'h': 4,              
        'N_dense': 2,            
        'n_generator_layers': 4, # <--- 绝大多数模型都是 4 层
        'n_output': 1,       
        'leaky_relu_slope': 0.1,
        'dense_output_nonlinearity': 'mish',
        'distance_matrix_kernel': 'exp',
        'aggregation_type': 'gru',
        'scale_norm': True,      
        'dropout': 0.0,
        'num_labels': 1,     
    }
    
    # 针对 TPO 进行精准打击：它的输出在 proj.8，说明它只有 3 层生成器
    if model_name == 'TPO':
        print(f"  [Info] Detected TPO checkpoint. Adjusting n_generator_layers to 3...")
        config['n_generator_layers'] = 3

    return config

def predict_all(args):
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # --- 1. 读取数据并解析分子 ---
    print(f"Reading SMILES from: {args.input_path}")
    df = pd.read_csv(args.input_path)
    
    if 'smiles' in df.columns: smiles = df['smiles'].tolist()
    elif 'SMILES' in df.columns: smiles = df['SMILES'].tolist()
    else: smiles = df.iloc[:, 0].tolist()

    data_mol = []
    valid_smiles = []
    for s in smiles:
        m = Chem.MolFromSmiles(s)
        if m:
            data_mol.append(m)
            valid_smiles.append(s)
    
    print(f"Valid molecules: {len(data_mol)}")
    
    # --- 2. 统一构建特征数据集 (提到了循环外，大幅提升效率！) ---
    print("Constructing graph dataset...")
    fake_labels = np.zeros((len(data_mol), 1))
    max_len = max([m.GetNumAtoms() for m in data_mol])
    
    # 使用所有模型通用的 d_atom 和 d_edge 进行构建
    dataset = construct_dataset(data_mol, fake_labels, d_atom=115, d_edge=13, max_length=max_len)
    test_loader = DataLoader(dataset, batch_size=args.batch_size, 
                             collate_fn=mol_collate_func, shuffle=False)

    # --- 3. 准备结果 DataFrame ---
    final_df = pd.DataFrame({'SMILES': valid_smiles})

    # --- 4. 查找并加载所有模型进行预测 ---
    model_files = glob.glob(os.path.join(args.model_dir, "*.pt"))
    if not model_files:
        raise ValueError(f"No .pt files found in {args.model_dir}")
    
    print(f"\nFound {len(model_files)} models. Starting inference...")

    for model_path in sorted(model_files):
        model_name = os.path.basename(model_path).replace('.pt', '')
        print(f"\n>> Running Model: {model_name}")
        
        # 获取针对该模型的专属配置，并加上 max_length
        current_config = get_model_config(model_name)
        current_config['max_length'] = max_len
        
        # 初始化对应结构的模型
        model = make_model(**current_config)
        
        # 加载权重文件
        checkpoint = torch.load(model_path, map_location=device)
        state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
        
        # 清洗 key 名称
        new_state_dict = {}
        for k, v in state_dict.items():
            name = k.replace("module.", "")
            if 'pos_embed.pe.weight' in name: continue
            new_state_dict[name] = v
            
        # 严苛的匹配检查：任何维度对不上都会立刻报错，拒绝静默失败！
        model.load_state_dict(new_state_dict, strict=False) 
        model.to(device)
        model.eval()
        
        # 推理阶段
        preds = []
        with torch.no_grad():
            for batch in tqdm(test_loader, leave=False):
                _, adj, node, edge, _ = batch
                adj, node, edge = adj.to(device), node.to(device), edge.to(device)
                
                # 构建 mask
                mask = torch.sum(torch.abs(node), dim=-1) != 0
                
                # 模型前向传播并经过 sigmoid
                logits, _ = model(node, mask, adj, edge)
                probs = torch.sigmoid(logits).cpu().numpy()
                preds.append(probs)
        
        # 拼接结果并存入 DataFrame 对应的一列
        preds = np.concatenate(preds, axis=0).flatten()
        final_df[model_name] = preds

    # --- 5. 保存最终结果 ---
    final_df.to_csv(args.output_path, index=False)
    print(f"\nAll predictions successfully saved to: {args.output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True, help="Path to input csv")
    parser.add_argument("--model_dir", type=str, required=True, help="Folder containing all .pt model files")
    parser.add_argument("--output_path", type=str, default="ensemble_predictions.csv")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--gpu", type=str, default='0')
    
    args = parser.parse_args()
    predict_all(args)