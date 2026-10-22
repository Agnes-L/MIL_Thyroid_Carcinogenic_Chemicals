import math
import torch
from random import Random
from torch.nn.init import _calculate_fan_in_and_fan_out, _no_grad_normal_, _no_grad_uniform_
from sklearn import metrics
from collections import defaultdict
import numpy as np
import torch.nn.functional as F
from rdkit.Chem.Scaffolds import MurckoScaffold


def xavier_normal_small_init_(tensor, gain=1.):
    # type: (Tensor, float) -> Tensor
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    std = gain * math.sqrt(2.0 / float(fan_in + 4 * fan_out))

    return _no_grad_normal_(tensor, 0., std)


def xavier_uniform_small_init_(tensor, gain=1.):
    # type: (Tensor, float) -> Tensor
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    std = gain * math.sqrt(2.0 / float(fan_in + 4 * fan_out))
    a = math.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation

    return _no_grad_uniform_(tensor, -a, a)


# =====================================================================
# [新增] 严谨的 Focal Loss 实现
# =====================================================================
class BinaryFocalLoss(torch.nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='none'):
        super(BinaryFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        # 1. 计算基础的 BCE Loss (reduction='none' 确保返回和 batch size 一致的矩阵)
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        
        # 2. 计算预测概率 p_t
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1 - probs) * (1 - targets)
        
        # 3. 动态分配 alpha 权重
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        
        # 4. 组合 Focal 权重
        focal_weight = alpha_t * (1 - p_t) ** self.gamma
        loss = focal_weight * bce_loss
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


# [Modified] 升级 get_loss，支持 focal loss 和动态 alpha
def get_loss(loss_name, pos_weight=None, alpha=None):
    if loss_name == 'rmse':
        return torch.nn.MSELoss()
    elif loss_name == 'mae':
        return torch.nn.L1Loss()
    elif loss_name == 'bce':
        if pos_weight is not None:
            return torch.nn.BCEWithLogitsLoss(reduction='none', pos_weight=pos_weight)
        else:
            return torch.nn.BCEWithLogitsLoss(reduction='none')
    elif loss_name == 'focal':
        actual_alpha = alpha if alpha is not None else 0.25
        return BinaryFocalLoss(alpha=actual_alpha, gamma=2.0, reduction='none')
    elif loss_name == 'ce':
        return torch.nn.CrossEntropyLoss(ignore_index=-1)
    else:
        assert False, 'Loss function not supported !'


def cal_loss(y_true, y_pred, loss_name, criterion, data_mean, data_std, device):
    # y_true, y_pred.shape = (batch, num_tasks)
    if loss_name == 'rmse':
        y_true, y_pred = (y_true.flatten() - data_mean)/data_std, y_pred.flatten()
        loss = torch.sqrt(criterion(y_pred, y_true))
    elif loss_name == 'mae':
        y_true, y_pred = (y_true.flatten() - data_mean)/data_std, y_pred.flatten()
        loss = criterion(y_pred, y_true)
    
    # [Modified] 让 bce 和 focal 共享这套极其严谨的掩码处理机制
    elif loss_name in ['bce', 'focal']:
        # find all -1 in y_true (masking invalid labels)
        y_true = y_true.long()
        y_mask = torch.where(y_true == -1, torch.tensor([0]).to(device), torch.tensor([1]).to(device))
        y_cal_true = torch.where(y_true == -1, torch.tensor([0]).to(device), y_true).float()
        
        # element_loss 内部此时会自动应用 focal weight 或是 pos_weight
        element_loss = criterion(y_pred, y_cal_true)
        
        # 应用 mask，只计算有效标签的 loss
        loss = element_loss * y_mask
        loss = loss.sum() / (y_mask.sum() + 1e-8)
    else:
        loss = criterion(y_pred, y_true)
    return loss


def evaluate(y_true, y_pred, y_smile, requirement, data_mean, data_std, data_task):
    # y_true.shape = y_pred.shape = (samples, task_numbers)
    collect_result = {}
    if 'sample' in requirement:
        collect_result['smile'] = y_smile.tolist()
        if data_task == 'classification':
            collect_result['prediction'] = y_pred.tolist()
            collect_result['label'] = y_true.tolist()
        else:
            collect_result['prediction'] = (y_pred * data_std + data_mean).tolist()
            collect_result['label'] = y_true.tolist()
    if 'rmse' in requirement:
        y_true, y_pred = y_true.flatten(), (y_pred.flatten() * data_std + data_mean).tolist()
        collect_result['rmse'] = np.sqrt(F.mse_loss(torch.tensor(y_pred), torch.tensor(y_true), reduction='mean'))
    if 'mae' in requirement:
        y_true, y_pred = y_true.flatten(), (y_pred.flatten() * data_std + data_mean).tolist()
        collect_result['mae'] = F.l1_loss(torch.tensor(y_pred), torch.tensor(y_true), reduction='mean')
    
    # Evaluate 阶段只用 bce 看无偏的基础交叉熵，无需修改 focal
    if 'bce' in requirement or 'focal' in requirement:
        # find all -1 in y_true
        y_mask = np.where(y_true == -1, 0, 1)
        y_cal_true = np.where(y_true == -1, 0, y_true)
        loss = F.binary_cross_entropy_with_logits(torch.tensor(y_pred), torch.tensor(y_cal_true), reduction='none') * y_mask
        collect_result['bce'] = loss.sum() / y_mask.sum()
        
    if 'auc' in requirement:
        auc_score_list = []
        if y_true.shape[1] > 1:
            for label in range(y_true.shape[1]):
                true, pred = y_true[:, label], y_pred[:, label]
                # all 0's or all 1's
                if len(set(true)) == 1:
                    auc_score_list.append(float('nan'))
                else:
                    if len(set(true[np.where(true >= 0)])) !=1:
                        auc_score_list.append(metrics.roc_auc_score(true[np.where(true >= 0)], pred[np.where(true >= 0)]))
                    else:
                        auc_score_list.append(float('nan'))
            collect_result['auc'] = np.nanmean(auc_score_list)
        else:
            # 单任务处理
            valid_indices = np.where(y_true != -1)[0]
            if len(valid_indices) > 0:
                y_true_valid = y_true[valid_indices]
                y_pred_valid = y_pred[valid_indices]
                if len(set(y_true_valid.reshape(-1))) == 1:
                     collect_result['auc'] = float('nan')
                else:
                    collect_result['auc'] = metrics.roc_auc_score(y_true_valid, y_pred_valid)
            else:
                collect_result['auc'] = float('nan')
    return collect_result


def scaffold_split(mol_list, frac=None, balanced=False, include_chirality=False, ramdom_state=0):
    if frac is None:
        frac = [0.8, 0.1, 0.1]
    assert sum(frac) == 1

    n_total_valid = int(np.floor(frac[1] * len(mol_list)))
    n_total_test = int(np.floor(frac[2] * len(mol_list)))
    n_total_train = len(mol_list) - n_total_valid - n_total_test

    scaffolds_sets = defaultdict(list)
    for idx, mol in enumerate(mol_list):
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=include_chirality)
        scaffolds_sets[scaffold].append(idx)

    random = Random(ramdom_state)

    if balanced:
        index_sets = list(scaffolds_sets.values())
        big_index_sets, small_index_sets = list(), list()
        for index_set in index_sets:
            if len(index_set) > n_total_valid / 2 or len(index_set) > n_total_test / 2:
                big_index_sets.append(index_set)
            else:
                small_index_sets.append(index_set)

        random.seed(ramdom_state)
        random.shuffle(big_index_sets)
        random.shuffle(small_index_sets)
        index_sets = big_index_sets + small_index_sets
    else:  
        index_sets = sorted(list(scaffolds_sets.values()), key=lambda index_set: len(index_set), reverse=True)

    train_index, valid_index, test_index = list(), list(), list()
    for index_set in index_sets:
        if len(train_index) + len(index_set) <= n_total_train:
            train_index += index_set
        elif len(valid_index) + len(index_set) <= n_total_valid:
            valid_index += index_set
        else:
            test_index += index_set

    return train_index, valid_index, test_index


class ScheduledOptim:
    """ https://github.com/jadore801120/attention-is-all-you-need-pytorch/blob/master/transformer/Optim.py """

    def __init__(self, optimizer, factor, d_model, n_warmup_steps):
        self._optimizer = optimizer
        self.factor = factor
        self.d_model = d_model
        self.n_warmup_steps = n_warmup_steps
        self.n_steps = 0

    def step_and_update_lr(self):
        self._update_learning_rate()
        self._optimizer.step()

    def view_lr(self):
        return self._optimizer.param_groups[0]['lr']

    def zero_grad(self):
        self._optimizer.zero_grad()

    def _get_lr_scale(self):
        d_model = self.d_model
        n_steps, n_warmup_steps = self.n_steps, self.n_warmup_steps
        return (d_model ** -0.5) * min(n_steps ** (-0.5), n_steps * n_warmup_steps ** (-1.5))

    def _update_learning_rate(self):
        self.n_steps += 1
        lr = self.factor * self._get_lr_scale()
        for param_group in self._optimizer.param_groups:
            param_group['lr'] = lr


def get_options(dataset_name):
    # graph regression
    if dataset_name == 'esol':
        model_params = {
            'd_atom': 115,                          
            'd_edge': 13,                           
            'd_model': 256,                         
            'N': 3,                                 
            'h': 4,                                 
            'N_dense': 1,                           
            'n_generator_layers': 2,                
            'n_output': 1,
            'leaky_relu_slope': 0.1,
            'dense_output_nonlinearity': 'mish',
            'distance_matrix_kernel': 'softmax',
            'aggregation_type': 'gru',
            'scale_norm': True,
            'dropout': 0.0,
            'num_labels': 1,
        }
        train_params = {
            'total_epochs': 150,
            'batch_size': 32,
            'warmup_factor': 0.1,
            'total_warmup_epochs': 30,
            'loss_function': 'rmse',
            'metric': 'rmse',
            'task': 'regression'
        }

    # =====================================================================
    # [Modified] 多分类靶点的核心超参数配置，已默认激活 focal
    # =====================================================================
    elif dataset_name in ['tox21', 'DIO1', 'DIO2', 'NIS', 'TPO', 'TRHRago', 'TRHRant', 'TRago', 'TRant', 'TSHRago', 'TSHRant']:
        n_tasks = 1
        model_params = {
            'd_atom': 115,                          
            'd_edge': 13,                           
            'd_model': 256,                         
            'N': 8,                                 
            'h': 4,                                 
            'N_dense': 2,                           
            'n_generator_layers': 4,                
            'n_output': n_tasks,                    
            'leaky_relu_slope': 0.1,
            'dense_output_nonlinearity': 'mish',
            'distance_matrix_kernel': 'exp',
            'aggregation_type': 'gru',
            'scale_norm': True,
            'dropout': 0.0,
            'num_labels': n_tasks,
        }
        train_params = {
            'total_epochs': 150,
            'batch_size': 32,
            'warmup_factor': 0.1,
            'total_warmup_epochs': 30,
            'loss_function': 'focal',   # <--- 从 'bce' 正式切换为 'focal'
            'metric': 'auc',
            'task': 'classification'
        }
    
    else:
        raise ValueError(f"未识别的靶点名称: {dataset_name}，请检查超参数配置。")
        
    return model_params, train_params