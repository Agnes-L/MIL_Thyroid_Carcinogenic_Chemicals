import argparse
import torch
import numpy as np
import pandas as pd
import pickle as pkl
import shutil  # ?????????
import os
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from dataset_graph import construct_dataset, mol_collate_func
from transformer_graph_finetune import make_model
from utils_for_multiple_tasks_focal_loss2 import ScheduledOptim, get_options, get_loss, cal_loss, evaluate, scaffold_split
from collections import defaultdict


def model_train(model, train_dataset, valid_dataset, model_params, train_params, dataset_name, experiment_name, fold):
    # build data loader
    train_loader = DataLoader(dataset=train_dataset, batch_size=train_params['batch_size'], collate_fn=mol_collate_func,
                              shuffle=True, drop_last=True, num_workers=0, pin_memory=False)

    valid_loader = DataLoader(dataset=valid_dataset, batch_size=train_params['batch_size'], collate_fn=mol_collate_func,
                              shuffle=True, drop_last=True, num_workers=0, pin_memory=False)

    # [Modified] Build loss function with alpha for Focal Loss
    alpha_val = train_params.get('alpha', None)
    pos_weight = train_params.get('pos_weight', None)
    criterion = get_loss(train_params['loss_function'], pos_weight=pos_weight, alpha=alpha_val)

    # build optimizer
    optimizer = ScheduledOptim(torch.optim.Adam(model.parameters(), lr=0),
                               train_params['warmup_factor'], model_params['d_model'],
                               train_params['total_warmup_steps'])

    best_valid_metric = float('inf') if train_params['task'] == 'regression' else float('-inf')
    best_epoch = -1
    best_valid_result, best_valid_bedding = None, None

    for epoch in range(train_params['total_epochs']):
        # train
        train_loss = list()
        model.train()
        for batch in tqdm(train_loader, leave=False, desc=f"Epoch {epoch+1}"):
            smile_list, adjacency_matrix, node_features, edge_features, y_true = batch
            adjacency_matrix = adjacency_matrix.to(train_params['device'])
            node_features = node_features.to(train_params['device'])
            edge_features = edge_features.to(train_params['device'])
            y_true = y_true.to(train_params['device'])
            batch_mask = torch.sum(torch.abs(node_features), dim=-1) != 0
            
            y_pred, _ = model(node_features, batch_mask, adjacency_matrix, edge_features)
            loss = cal_loss(y_true, y_pred, train_params['loss_function'], criterion,
                            0, 1, train_params['device'])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step_and_update_lr()
            train_loss.append(loss.detach().item())

        # valid
        model.eval()
        with torch.no_grad():
            valid_true, valid_pred, valid_smile, valid_embedding = list(), list(), list(), list()
            for batch in tqdm(valid_loader, leave=False, desc="Validating"):
                smile_list, adjacency_matrix, node_features, edge_features, y_true = batch
                adjacency_matrix = adjacency_matrix.to(train_params['device'])
                node_features = node_features.to(train_params['device'])
                edge_features = edge_features.to(train_params['device'])
                batch_mask = torch.sum(torch.abs(node_features), dim=-1) != 0
                
                y_pred, y_embedding = model(node_features, batch_mask, adjacency_matrix, edge_features)

                y_true = y_true.numpy()
                y_pred = y_pred.detach().cpu().numpy()
                y_embedding = y_embedding.detach().cpu().numpy()

                valid_true.append(y_true)
                valid_pred.append(y_pred)
                valid_smile.append(smile_list)
                valid_embedding.append(y_embedding)

            valid_true, valid_pred = np.concatenate(valid_true, axis=0), np.concatenate(valid_pred, axis=0)
            valid_smile, valid_embedding = np.concatenate(valid_smile, axis=0), np.concatenate(valid_embedding, axis=0)

        valid_result = evaluate(valid_true, valid_pred, valid_smile,
                                requirement=['sample', train_params['loss_function'], train_params['metric']],
                                data_mean=0, data_std=1, data_task=train_params['task'])

        # Logic to save the best model to a temporary file
        is_best = False
        if train_params['task'] == 'regression':
            if valid_result[train_params['metric']] < best_valid_metric: is_best = True
        else:
            if valid_result[train_params['metric']] > best_valid_metric: is_best = True

        if is_best:
            best_valid_metric = valid_result[train_params['metric']]
            best_epoch = epoch + 1
            best_valid_result = valid_result
            best_valid_bedding = valid_embedding
            torch.save({'state_dict': model.state_dict(),
                        'best_epoch': best_epoch,
                        f'best_valid_{train_params["metric"]}': best_valid_metric},
                       f'./Model/{dataset_name}/best_model_{experiment_name}_{dataset_name}_fold_{fold}.pt')

        # early stop
        if abs(best_epoch - epoch) >= 20:
            break

    print(f"Finished training. Best Valid {train_params['metric']}: {best_valid_metric:.4f} at epoch {best_epoch}")
    return best_valid_result, best_valid_bedding


def model_test(checkpoint, test_dataset, model_params, train_params):
    test_loader = DataLoader(dataset=test_dataset, batch_size=train_params['batch_size'], collate_fn=mol_collate_func,
                             shuffle=False, drop_last=True, num_workers=0, pin_memory=False)

    model = make_model(**model_params)
    model.to(train_params['device'])
    model.load_state_dict(checkpoint['state_dict'])

    model.eval()
    with torch.no_grad():
        test_true, test_pred, test_smile, test_embedding = list(), list(), list(), list()
        for batch in tqdm(test_loader, leave=False, desc="Testing"):
            smile_list, adjacency_matrix, node_features, edge_features, y_true = batch
            adjacency_matrix = adjacency_matrix.to(train_params['device'])
            node_features = node_features.to(train_params['device'])
            edge_features = edge_features.to(train_params['device'])
            batch_mask = torch.sum(torch.abs(node_features), dim=-1) != 0
            
            y_pred, y_embedding = model(node_features, batch_mask, adjacency_matrix, edge_features)

            y_true = y_true.numpy()
            y_pred = y_pred.detach().cpu().numpy()
            y_embedding = y_embedding.detach().cpu().numpy()

            test_true.append(y_true)
            test_pred.append(y_pred)
            test_smile.append(smile_list)
            test_embedding.append(y_embedding)
        test_true, test_pred = np.concatenate(test_true, axis=0), np.concatenate(test_pred, axis=0)
        test_smile, test_embedding = np.concatenate(test_smile, axis=0), np.concatenate(test_embedding, axis=0)
    test_result = evaluate(test_true, test_pred, test_smile,
                           requirement=['sample', train_params['loss_function'], train_params['metric']],
                           data_mean=0, data_std=1, data_task=train_params['task'])

    print("Final Test {}: {:.4f}".format(train_params['metric'], test_result[train_params['metric']]))

    return test_result, test_embedding


# [Modified] 辅助函数：计算 Focal Loss 的动态 Alpha 权重
def calculate_alpha_weight(dataset, data_label):
    if isinstance(dataset, torch.utils.data.Subset):
        indices = dataset.indices
    else:
        indices = range(len(dataset))
        
    y_train = np.array(data_label)[indices]
    
    num_pos = np.sum(y_train == 1)
    num_neg = np.sum(y_train == 0)
    total_valid = num_pos + num_neg
    
    if total_valid > 0 and num_pos > 0:
        alpha = num_neg / total_valid
    else:
        alpha = 0.5 
        
    print(f"  [Focal-Alpha] Train Set Stats: Pos={num_pos}, Neg={num_neg}, Alpha={alpha:.4f}")
    return float(alpha)


if __name__ == '__main__':
    # init args
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, help="base random seed", default=np.random.randint(10000))
    parser.add_argument("--gpu", type=str, help='gpu', default=-1)
    parser.add_argument("--fold", type=int, help='For Random split: Model Runs per Seed. For CV: Number of folds.', default=5)
    parser.add_argument("--num_seeds", type=int, help='For Random split: Number of data split seeds.', default=10)
    parser.add_argument("--dataset", type=str, help='choose a dataset', default='esol')
    parser.add_argument("--split", type=str, help="choose the split type", default='random',
                        choices=['random', 'scaffold', 'cv', 'stratified'])
    parser.add_argument("--ckpt_path", type=str, help="ckpt-path", default=None)
    parser.add_argument("--experiment_name", type=str, help="experiment_nam", default=None)
    args = parser.parse_args()

    # load options
    model_params, train_params = get_options(args.dataset)

    # init device and seed
    print(f"Base Seed: {args.seed}")
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        train_params['device'] = torch.device(f'cuda:{args.gpu}')
        torch.cuda.manual_seed(args.seed)
    else:
        train_params['device'] = torch.device('cpu')

    # load data
    if train_params['task'] == 'regression':
        with open(f'./Data/{args.dataset}/preprocess/{args.dataset}.pickle', 'rb') as f:
            [data_mol, data_label, data_mean, data_std] = pkl.load(f)
    else:
        with open(f'./Data/{args.dataset}/preprocess/{args.dataset}.pickle', 'rb') as f:
            [data_mol, data_label] = pkl.load(f)

    # calculate the padding
    model_params['max_length'] = max([data.GetNumAtoms() for data in data_mol])
    print(f"Max padding length is: {model_params['max_length']}")

    # construct dataset
    print('=' * 20 + ' construct dataset ' + '=' * 20)
    dataset = construct_dataset(data_mol, data_label, model_params['d_atom'], model_params['d_edge'], model_params['max_length'])
    total_metrics = defaultdict(list)

    if args.split == 'random':
        import os
        os.makedirs(f'./Result/{args.dataset}', exist_ok=True)
        os.makedirs(f'./Model/{args.dataset}', exist_ok=True)
        
        print(f"Random Split Mode: Running {args.num_seeds} Data Splits. For each split, finding BEST of {args.fold} Runs.")

        seed_records = []  

        for seed_idx in range(args.num_seeds):
            current_data_seed = args.seed + seed_idx * 1000 
            
            print('\n' + '#' * 60)
            print(f'###  Data Split Seed {seed_idx + 1}/{args.num_seeds} (Seed={current_data_seed})  ###')
            print('#' * 60)

            # Data Split
            train_valid_dataset, test_dataset = train_test_split(dataset, test_size=0.1, random_state=current_data_seed)
            train_dataset, valid_dataset = train_test_split(train_valid_dataset, test_size=len(test_dataset),
                                                            random_state=current_data_seed)
            
            # [Modified] Calculate Alpha for this specific split
            if train_params['task'] == 'classification':
                 dynamic_alpha = calculate_alpha_weight(train_dataset, data_label)
                 train_params['alpha'] = dynamic_alpha

            print(f"Split Done. Train: {len(train_dataset)}, Valid: {len(valid_dataset)}, Test: {len(test_dataset)}")

            if train_params['task'] == 'regression':
                best_seed_valid_metric = float('inf')
            else:
                best_seed_valid_metric = float('-inf')
            
            best_seed_test_metric = 0.0 
            
            for run_idx in range(args.fold):
                print(f"  > Run {run_idx + 1}/{args.fold} (Init Seed {current_data_seed + run_idx})...")
                
                current_model_seed = current_data_seed + run_idx
                np.random.seed(current_model_seed)
                torch.manual_seed(current_model_seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed(current_model_seed)

                train_params['total_warmup_steps'] = \
                    int(len(train_dataset) / train_params['batch_size']) * train_params['total_warmup_epochs']

                model = make_model(**model_params).to(train_params['device'])
                
                # [Modified] 预训练模型诊断式加载
                if args.ckpt_path is not None:
                    ckpt = torch.load(args.ckpt_path, map_location=train_params['device'])['state_dict']
                    for k in list(ckpt.keys()):
                        if k == 'pos_embed.pe.weight':
                            del ckpt[k]
                            
                    incompatible = model.load_state_dict(ckpt, strict=False)

                     # 过滤掉那些本来就该重新初始化的下游任务专属层
                    expected_missing_prefixes = ['pos_embed', 'generator']
                    real_missing_keys = [k for k in incompatible.missing_keys if not any(p in k for p in expected_missing_prefixes)]

                    if real_missing_keys:
                        print(f"\n[⚠️ 严重警告] 骨干网络有 {len(real_missing_keys)} 个参数未加载预训练权重!")
                        for k in real_missing_keys[:10]: print(f"    - {k}")
                    else:
                        print('\n[✅ 完美成功] 骨干网络 (Encoder) 预训练权重 100% 加载完毕！预测头已按预期重新初始化。')

                valid_result, _ = model_train(model, train_dataset, valid_dataset, model_params, train_params, args.dataset, args.experiment_name, run_idx + 1)
                
                current_metric = valid_result[train_params['metric']]
                
                is_best_run = False
                if train_params['task'] == 'regression':
                    if current_metric < best_seed_valid_metric: is_best_run = True
                else:
                    if current_metric > best_seed_valid_metric: is_best_run = True
                
                if is_best_run:
                    print(f"    [!] New Best Run for Seed {current_data_seed}! Valid {train_params['metric']}: {current_metric:.4f}")
                    best_seed_valid_metric = current_metric
                    
                    best_valid_csv = pd.DataFrame.from_dict({
                        'smile': valid_result['smile'], 
                        'actual': valid_result['label'], 
                        'predict': valid_result['prediction']
                    })
                    valid_csv_name = f'./Result/{args.dataset}/best_valid_result_{args.experiment_name}_{args.dataset}_seed_{seed_idx + 1}_best.csv'
                    best_valid_csv.to_csv(valid_csv_name, sep=',', index=False, encoding='UTF-8')

                    temp_ckpt_path = f'./Model/{args.dataset}/best_model_{args.experiment_name}_{args.dataset}_fold_{run_idx + 1}.pt'
                    checkpoint = torch.load(temp_ckpt_path, map_location=train_params['device'])
                    test_result, _ = model_test(checkpoint, test_dataset, model_params, train_params)
                    best_seed_test_metric = test_result[train_params['metric']]
                    
                    test_csv = pd.DataFrame.from_dict({
                        'smile': test_result['smile'], 
                        'actual': test_result['label'], 
                        'predict': test_result['prediction']
                    })
                    test_csv_name = f'./Result/{args.dataset}/best_test_result_{args.experiment_name}_{args.dataset}_seed_{seed_idx + 1}_best.csv'
                    test_csv.to_csv(test_csv_name, sep=',', index=False, encoding='UTF-8')

                    final_model_path = f'./Model/{args.dataset}/best_model_{args.experiment_name}_{args.dataset}_seed_{seed_idx + 1}_best.pt'
                    shutil.copy(temp_ckpt_path, final_model_path)
                else:
                    print(f"    Run {run_idx + 1} did not beat current best ({best_seed_valid_metric:.4f})")

            total_metrics['valid'].append(best_seed_valid_metric)
            total_metrics['test'].append(best_seed_test_metric)
            seed_records.append((current_data_seed, best_seed_valid_metric, best_seed_test_metric))
            print(f"--> Seed {seed_idx+1} Finished. Best Valid: {best_seed_valid_metric:.4f}, Test: {best_seed_test_metric:.4f}")
       
        print('=' * 20 + ' summary ' + '=' * 20)
        print(f"Total Data Seeds: {args.num_seeds} (Each selected best from {args.fold} runs)")
        
        for i, (seed_val, v_score, t_score) in enumerate(seed_records):
            print(f"Split {i+1} (Seed {seed_val}): Valid {train_params['metric']} = {v_score:.4f}, Test {train_params['metric']} = {t_score:.4f}")
        
        print('-' * 40)
        print('Average across all seeds:')
        print('valid {} = {:.4f} ± {:.4f}, test {} = {:.4f} ± {:.4f}'
              .format(train_params['metric'], np.nanmean(total_metrics['valid']), np.nanstd(total_metrics['valid']),
                      train_params['metric'], np.nanmean(total_metrics['test']), np.nanstd(total_metrics['test']),
                      ))
        print('=' * 20 + " finished! " + '=' * 20)

        summary_df = pd.DataFrame(seed_records, columns=['Seed', f'Valid_{train_params["metric"]}', f'Test_{train_params["metric"]}'])
        summary_df.loc[len(summary_df)] = ['Mean', np.nanmean(total_metrics['valid']), np.nanmean(total_metrics['test'])]
        summary_df.loc[len(summary_df)] = ['Std', np.nanstd(total_metrics['valid']), np.nanstd(total_metrics['test'])]
        
        save_path = f'./Result/{args.dataset}/summary_{args.experiment_name}_{args.dataset}_random.csv'
        summary_df.to_csv(save_path, index=False, encoding='UTF-8')
        print(f"Summary saved to: {save_path}")

    elif args.split == 'scaffold':
         import os
         os.makedirs(f'./Result/{args.dataset}', exist_ok=True)
         os.makedirs(f'./Model/{args.dataset}', exist_ok=True)    
         for idx in range(args.fold):
            print('=' * 20 + f' train on fold {idx + 1} ' + '=' * 20)
            print(f"Seed: {args.seed+ idx}")
            np.random.seed(args.seed+idx)
            torch.manual_seed(args.seed+idx)
            if torch.cuda.is_available():
                train_params['device'] = torch.device(f'cuda:{args.gpu}')
                torch.cuda.manual_seed(args.seed+idx)
            # get dataset
            train_index, valid_index, test_index = scaffold_split(data_mol, frac=[0.8, 0.1, 0.1], balanced=True,
                                                                  include_chirality=False, ramdom_state=args.seed + idx)
            train_dataset, valid_dataset, test_dataset = dataset[train_index], dataset[valid_index], dataset[test_index]

            # [Modified] Calculate Alpha Weight
            if train_params['task'] == 'classification':
                 dynamic_alpha = calculate_alpha_weight(train_dataset, data_label)
                 train_params['alpha'] = dynamic_alpha

            # calculate total warmup steps
            train_params['total_warmup_steps'] = \
                int(len(train_dataset) / train_params['batch_size']) * train_params['total_warmup_epochs']
            print('train warmup step is: {}'.format(train_params['total_warmup_steps']))

            if train_params['task'] == 'regression':
                train_params['mean'] = np.mean(np.array(data_label)[train_index])
                train_params['std'] = np.std(np.array(data_label)[train_index])
            else:
                train_params['mean'], train_params['std'] = 0, 1

            # define a model
            model = make_model(**model_params).to(train_params['device'])

            # [Modified] 预训练模型诊断式加载
            if args.ckpt_path is not None:
                ckpt = torch.load(args.ckpt_path, map_location=train_params['device'])['state_dict']
                for k in list(ckpt.keys()):
                    if k == 'pos_embed.pe.weight':
                        del ckpt[k]
                        
                incompatible = model.load_state_dict(ckpt, strict=False)

                if incompatible.missing_keys:
                    print(f"\n[⚠️ 严重警告] 当前模型有 {len(incompatible.missing_keys)} 个参数未找到预训练权重 (保持随机初始化)!")
                    print(">>> 缺失 Keys 示例 (前 10 个):")
                    for k in incompatible.missing_keys[:10]: print(f"    - {k}")
                if incompatible.unexpected_keys:
                    print(f"\n[⚠️ 严重警告] 预训练权重中有 {len(incompatible.unexpected_keys)} 个参数未被使用 (已被丢弃)!")
                    print(">>> 废弃 Keys 示例 (前 10 个):")
                    for k in incompatible.unexpected_keys[:10]: print(f"    - {k}")
                if not incompatible.missing_keys and not incompatible.unexpected_keys:
                    print('\n[✅ 成功] 预训练权重完美对齐并加载完毕！')


            # train and valid
            print(f"train size: {len(train_dataset)}, valid size: {len(valid_dataset)}, test size: {len(test_dataset)}")
            best_valid_result, _ = model_train(model, train_dataset, valid_dataset, model_params, train_params, args.dataset, args.experiment_name,idx + 1)
            best_valid_csv = pd.DataFrame.from_dict({'smile': best_valid_result['smile'], 'actual': best_valid_result['label'], 'predict': best_valid_result['prediction']})
            best_valid_csv.to_csv(f'./Result/{args.dataset}/best_valid_result_{args.experiment_name}_{args.dataset}_fold_{idx + 1}.csv', sep=',', index=False, encoding='UTF-8')
            total_metrics['valid'].append(best_valid_result[train_params['metric']])

            # test
            print('=' * 20 + f' test on fold {idx + 1} ' + '=' * 20)
            checkpoint = torch.load(f'./Model/{args.dataset}/best_model_{args.experiment_name}_{args.dataset}_fold_{idx + 1}.pt', map_location=train_params['device'])
            test_result, test_embedding = model_test(checkpoint, test_dataset, model_params, train_params)
            test_csv = pd.DataFrame.from_dict({'smile': test_result['smile'], 'actual': test_result['label'], 'predict': test_result['prediction']})
            test_csv.to_csv(f'./Result/{args.dataset}/best_test_result_{args.experiment_name}_{args.dataset}_fold_{idx + 1}.csv', sep=',', index=False, encoding='UTF-8')
            total_metrics['test'].append(test_result[train_params['metric']])

            total_embedding = dict()
            for smile, embedding in zip(test_result['smile'], test_embedding):
                total_embedding[smile] = embedding

            with open(f'./Result/{args.dataset}/total_test_embedding_fold_{idx + 1}.pickle', 'wb') as fw:
                pkl.dump(total_embedding, fw)

         # --- Summary and Save CSV ---
         print('=' * 20 + ' summary ' + '=' * 20)
         print('Seed: {}'.format(args.seed))
         
         summary_data = []
         for idx in range(args.fold):
            print('fold {}, valid {} = {:.4f}, test {} = {:.4f}'
                  .format(idx + 1, train_params['metric'], total_metrics['valid'][idx], train_params['metric'], total_metrics['test'][idx]))
            
            summary_data.append({
                'Fold': idx + 1,
                f'Valid_{train_params["metric"]}': total_metrics['valid'][idx],
                f'Test_{train_params["metric"]}': total_metrics['test'][idx]
            })

         print('{} folds valid average {} = {:.4f} ± {:.4f}, test average {} = {:.4f} ± {:.4f}'
              .format(args.fold,
                      train_params['metric'], np.nanmean(total_metrics['valid']), np.nanstd(total_metrics['valid']),
                      train_params['metric'], np.nanmean(total_metrics['test']), np.nanstd(total_metrics['test']),
                      ))
         print('=' * 20 + " finished! " + '=' * 20)

         summary_df = pd.DataFrame(summary_data)
         summary_df.loc[len(summary_df)] = ['Mean', np.nanmean(total_metrics['valid']), np.nanmean(total_metrics['test'])]
         summary_df.loc[len(summary_df)] = ['Std', np.nanstd(total_metrics['valid']), np.nanstd(total_metrics['test'])]
         
         save_path = f'./Result/{args.dataset}/summary_{args.experiment_name}_{args.dataset}_scaffold.csv'
         summary_df.to_csv(save_path, index=False, encoding='UTF-8')
         print(f"Summary saved to: {save_path}")

    elif args.split == 'cv':
        import os
        os.makedirs(f'./Result/{args.dataset}', exist_ok=True)
        os.makedirs(f'./Model/{args.dataset}', exist_ok=True)    
        # k-fold
        kf = KFold(n_splits=args.fold, shuffle=True, random_state=args.seed)
        for idx, (train_index, valid_index) in enumerate(kf.split(X=dataset)):
            print('=' * 20 + f' train on fold {idx + 1} ' + '=' * 20)
            # get dataset
            train_dataset, valid_dataset = dataset[train_index], dataset[valid_index]

            # [Modified] Calculate Alpha Weight
            if train_params['task'] == 'classification':
                 dynamic_alpha = calculate_alpha_weight(train_dataset, data_label)
                 train_params['alpha'] = dynamic_alpha

            # calculate total warmup steps
            train_params['total_warmup_steps'] = \
                int(len(train_dataset) / train_params['batch_size']) * train_params['total_warmup_epochs']
            print('train warmup step is: {}'.format(train_params['total_warmup_steps']))

            if train_params['task'] == 'regression':
                train_params['mean'] = np.mean(np.array(data_label)[train_index])
                train_params['std'] = np.std(np.array(data_label)[train_index])
            else:
                train_params['mean'], train_params['std'] = 0, 1

            # define a model
            model = make_model(**model_params).to(train_params['device'])

            # [Modified] 预训练模型诊断式加载
            if args.ckpt_path is not None:
                ckpt = torch.load(args.ckpt_path, map_location=train_params['device'])['state_dict']
                for k in list(ckpt.keys()):
                    if k == 'pos_embed.pe.weight':
                        del ckpt[k]
                        
                incompatible = model.load_state_dict(ckpt, strict=False)

                if incompatible.missing_keys:
                    print(f"\n[⚠️ 严重警告] 当前模型有 {len(incompatible.missing_keys)} 个参数未找到预训练权重 (保持随机初始化)!")
                    print(">>> 缺失 Keys 示例 (前 10 个):")
                    for k in incompatible.missing_keys[:10]: print(f"    - {k}")
                if incompatible.unexpected_keys:
                    print(f"\n[⚠️ 严重警告] 预训练权重中有 {len(incompatible.unexpected_keys)} 个参数未被使用 (已被丢弃)!")
                    print(">>> 废弃 Keys 示例 (前 10 个):")
                    for k in incompatible.unexpected_keys[:10]: print(f"    - {k}")
                if not incompatible.missing_keys and not incompatible.unexpected_keys:
                    print('\n[✅ 成功] 预训练权重完美对齐并加载完毕！')

            # train and valid
            print(f"train size: {len(train_dataset)}, valid size: {len(valid_dataset)}")
            best_valid_result, best_valid_embedding = model_train(model, train_dataset, valid_dataset, model_params, train_params, args.dataset, args.experiment_name, idx + 1)
            best_valid_csv = pd.DataFrame.from_dict({'smile': best_valid_result['smile'], 'actual': best_valid_result['label'], 'predict': best_valid_result['prediction']})
            best_valid_csv.to_csv(f'./Result/{args.dataset}/best_valid_result_{args.dataset}_fold_{idx + 1}.csv', sep=',', index=False, encoding='UTF-8')
            total_metrics['cv'].append(best_valid_result[train_params['metric']])

            total_embedding = dict()
            for smile, embedding in zip(best_valid_result['smile'], best_valid_embedding):
                total_embedding[smile] = embedding

            with open(f'./Result/{args.dataset}/total_valid_embedding_fold_{idx + 1}.pickle', 'wb') as fw:
                pkl.dump(total_embedding, fw)


        # --- Summary and Save CSV ---
        print('=' * 20 + ' summary ' + '=' * 20)
        print('Seed: {}'.format(args.seed))
        
        summary_data = []
        for idx in range(args.fold):
            print('fold {} {} = {:.4f}'
                  .format(idx + 1, train_params['metric'], total_metrics['cv'][idx]))
            
            summary_data.append({
                'Fold': idx + 1,
                f'Valid_{train_params["metric"]}': total_metrics['cv'][idx]
            })

        print('{} folds {} = {:.4f} ± {:.4f}'
              .format(args.fold, train_params['metric'], np.nanmean(total_metrics['cv']), np.nanstd(total_metrics['cv'])))
        print('=' * 20 + " finished! " + '=' * 20)

        # 保存 CSV
        summary_df = pd.DataFrame(summary_data)
        summary_df.loc[len(summary_df)] = ['Mean', np.nanmean(total_metrics['cv'])]
        summary_df.loc[len(summary_df)] = ['Std', np.nanstd(total_metrics['cv'])]
        
        save_path = f'./Result/{args.dataset}/summary_{args.experiment_name}_{args.dataset}_cv.csv'
        summary_df.to_csv(save_path, index=False, encoding='UTF-8')
        print(f"Summary saved to: {save_path}")
    
    elif args.split == 'stratified':
        import os
        from sklearn.model_selection import StratifiedKFold

        os.makedirs(f'./Result/{args.dataset}', exist_ok=True)
        os.makedirs(f'./Model/{args.dataset}', exist_ok=True)

        labels = np.array(data_label)
        assert labels.ndim == 2 and labels.shape[1] == 1
        stratify_label = labels[:, 0]

        skf = StratifiedKFold(n_splits=args.fold, shuffle=True, random_state=args.seed)

        for idx, (train_valid_idx, test_idx) in enumerate(skf.split(np.zeros(len(stratify_label)), stratify_label)):
            print('=' * 20 + f' train on fold {idx + 1} ' + '=' * 20)

            np.random.seed(args.seed)
            torch.manual_seed(args.seed)
            if torch.cuda.is_available():
                train_params['device'] = torch.device(f'cuda:{args.gpu}')
                torch.cuda.manual_seed(args.seed)

            # 再从 train_valid 里划分 valid
            train_idx, valid_idx = train_test_split(
                train_valid_idx,
                test_size=len(test_idx),
                random_state=args.seed,
                stratify=stratify_label[train_valid_idx] # Add stratify for validation split too
            )

            train_dataset = dataset[train_idx]
            valid_dataset = dataset[valid_idx]
            test_dataset = dataset[test_idx]
            
            # [Modified] Calculate Alpha Weight
            if train_params['task'] == 'classification':
                 dynamic_alpha = calculate_alpha_weight(train_dataset, data_label)
                 train_params['alpha'] = dynamic_alpha

            print(f"train size: {len(train_dataset)}, valid size: {len(valid_dataset)}, test size: {len(test_dataset)}")

            train_params['total_warmup_steps'] = \
                int(len(train_dataset) / train_params['batch_size']) * train_params['total_warmup_epochs']

            if train_params['task'] == 'regression':
                train_params['mean'] = np.mean(labels[train_idx])
                train_params['std'] = np.std(labels[train_idx])
            else:
                train_params['mean'], train_params['std'] = 0, 1

            model = make_model(**model_params).to(train_params['device'])

            # [Modified] 预训练模型诊断式加载
            if args.ckpt_path is not None:
                ckpt = torch.load(args.ckpt_path, map_location=train_params['device'])['state_dict']
                for k in list(ckpt.keys()):
                    if k == 'pos_embed.pe.weight':
                        del ckpt[k]
                        
                incompatible = model.load_state_dict(ckpt, strict=False)

                if incompatible.missing_keys:
                    print(f"\n[⚠️ 严重警告] 当前模型有 {len(incompatible.missing_keys)} 个参数未找到预训练权重 (保持随机初始化)!")
                    print(">>> 缺失 Keys 示例 (前 10 个):")
                    for k in incompatible.missing_keys[:10]: print(f"    - {k}")
                if incompatible.unexpected_keys:
                    print(f"\n[⚠️ 严重警告] 预训练权重中有 {len(incompatible.unexpected_keys)} 个参数未被使用 (已被丢弃)!")
                    print(">>> 废弃 Keys 示例 (前 10 个):")
                    for k in incompatible.unexpected_keys[:10]: print(f"    - {k}")
                if not incompatible.missing_keys and not incompatible.unexpected_keys:
                    print('\n[✅ 成功] 预训练权重完美对齐并加载完毕！')

            best_valid_result, _ = model_train(model, train_dataset, valid_dataset, model_params, train_params, args.dataset, args.experiment_name, idx + 1)
            
            best_valid_csv = pd.DataFrame.from_dict({
                'smile': best_valid_result['smile'],
                'actual': best_valid_result['label'],
                'predict': best_valid_result['prediction']
            })
            best_valid_csv.to_csv(f'./Result/{args.dataset}/best_valid_result_{args.experiment_name}_{args.dataset}_fold_{idx + 1}.csv', sep=',', index=False, encoding='UTF-8')
            total_metrics['valid'].append(best_valid_result[train_params['metric']])

            checkpoint = torch.load(f'./Model/{args.dataset}/best_model_{args.experiment_name}_{args.dataset}_fold_{idx + 1}.pt', map_location=train_params['device'])
            test_result, _ = model_test(checkpoint, test_dataset, model_params, train_params)
            
            test_csv = pd.DataFrame.from_dict({
                'smile': test_result['smile'],
                'actual': test_result['label'],
                'predict': test_result['prediction']
            })
            test_csv.to_csv(f'./Result/{args.dataset}/best_test_result_{args.experiment_name}_{args.dataset}_fold_{idx + 1}.csv', sep=',', index=False, encoding='UTF-8')
            total_metrics['test'].append(test_result[train_params['metric']])

        # --- Summary and Save CSV ---
        print('=' * 20 + ' summary ' + '=' * 20)
        
        summary_data = []
        for i in range(args.fold):
            print(f'fold {i+1}, valid {total_metrics["valid"][i]:.4f}, test {total_metrics["test"][i]:.4f}')
            
            summary_data.append({
                'Fold': i + 1,
                f'Valid_{train_params["metric"]}': total_metrics['valid'][i],
                f'Test_{train_params["metric"]}': total_metrics['test'][i]
            })
            
        print('{} folds valid average {} = {:.4f} ± {:.4f}, test average {} = {:.4f} ± {:.4f}'
              .format(args.fold,
                      train_params['metric'], np.nanmean(total_metrics['valid']), np.nanstd(total_metrics['valid']),
                      train_params['metric'], np.nanmean(total_metrics['test']), np.nanstd(total_metrics['test']),
                      ))
        print('=' * 20 + " finished! " + '=' * 20)

        # 保存 CSV
        summary_df = pd.DataFrame(summary_data)
        summary_df.loc[len(summary_df)] = ['Mean', np.nanmean(total_metrics['valid']), np.nanmean(total_metrics['test'])]
        summary_df.loc[len(summary_df)] = ['Std', np.nanstd(total_metrics['valid']), np.nanstd(total_metrics['test'])]
        
        save_path = f'./Result/{args.dataset}/summary_{args.experiment_name}_{args.dataset}_stratified.csv'
        summary_df.to_csv(save_path, index=False, encoding='UTF-8')
        print(f"Summary saved to: {save_path}")
    
    else:
        raise Exception('We only support random split, scaffold split, cv and stratified!')
