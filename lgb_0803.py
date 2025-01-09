import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error

df_original = pd.read_csv("data/train_data.csv")
n_original = df_original.shape[0]
df_submit = pd.read_csv("data/sample_submission.csv")
df = pd.concat([df_original, df_submit], axis=0).reset_index(drop=True)


# siRNA 特征构建函数
def siRNA_feat_builder(s: pd.Series, anti: bool = False):
    name = "anti" if anti else "sense"
    df = s.to_frame()
    df[f"feat_siRNA_{name}_seq_len"] = s.str.len()
    for pos in [0, -1]:
        for c in list("AUGC"):
            df[f"feat_siRNA_{name}_seq_{c}_{'front' if pos == 0 else 'back'}"] = (s.str[pos] == c)
    df[f"feat_siRNA_{name}_seq_pattern_1"] = s.str.startswith("AA") & s.str.endswith("UU")
    df[f"feat_siRNA_{name}_seq_pattern_2"] = s.str.startswith("GA") & s.str.endswith("UU")
    df[f"feat_siRNA_{name}_seq_pattern_3"] = s.str.startswith("CA") & s.str.endswith("UU")
    df[f"feat_siRNA_{name}_seq_pattern_4"] = s.str.startswith("UA") & s.str.endswith("UU")
    df[f"feat_siRNA_{name}_seq_pattern_5"] = s.str.startswith("UU") & s.str.endswith("AA")
    df[f"feat_siRNA_{name}_seq_pattern_6"] = s.str.startswith("UU") & s.str.endswith("GA")
    df[f"feat_siRNA_{name}_seq_pattern_7"] = s.str.startswith("UU") & s.str.endswith("CA")
    df[f"feat_siRNA_{name}_seq_pattern_8"] = s.str.startswith("UU") & s.str.endswith("UA")
    df[f"feat_siRNA_{name}_seq_pattern_9"] = s.str[1] == "A"
    df[f"feat_siRNA_{name}_seq_pattern_10"] = s.str[-2] == "A"
    df[f"feat_siRNA_{name}_seq_pattern_GC_frac"] = (s.str.contains("G") + s.str.contains("C")) / s.str.len()
    return df.iloc[:, 1:]


# 进行特征工程
df_publication_id = pd.get_dummies(df.publication_id)
df_publication_id.columns = [f"feat_publication_id_{c}" for c in df_publication_id.columns]
df_gene_target_symbol_name = pd.get_dummies(df.gene_target_symbol_name)
df_gene_target_symbol_name.columns = [f"feat_gene_target_symbol_name_{c}" for c in df_gene_target_symbol_name.columns]
df_gene_target_ncbi_id = pd.get_dummies(df.gene_target_ncbi_id)
df_gene_target_ncbi_id.columns = [f"feat_gene_target_ncbi_id_{c}" for c in df_gene_target_ncbi_id.columns]
df_gene_target_species = pd.get_dummies(df.gene_target_species)
df_gene_target_species.columns = [f"feat_gene_target_species_{c}" for c in df_gene_target_species.columns]

# 处理 siRNA_duplex_id
siRNA_duplex_id_values = df.siRNA_duplex_id.str.split("-|\.").str[1].astype("int")
siRNA_duplex_id_values = (siRNA_duplex_id_values - siRNA_duplex_id_values.min()) / (
            siRNA_duplex_id_values.max() - siRNA_duplex_id_values.min())
df_siRNA_duplex_id = pd.DataFrame(siRNA_duplex_id_values)

# 处理 cell_line_donor
df_cell_line_donor = pd.get_dummies(df.cell_line_donor)
df_cell_line_donor.columns = [f"feat_cell_line_donor_{c}" for c in df_cell_line_donor.columns]
df_cell_line_donor["feat_cell_line_donor_hepatocytes"] = (df.cell_line_donor.str.contains("Hepatocytes")).fillna(
    False).astype("int")
df_cell_line_donor["feat_cell_line_donor_cells"] = df.cell_line_donor.str.contains("Cells").fillna(False).astype("int")

# 处理 siRNA_concentration
df_siRNA_concentration = df.siRNA_concentration.to_frame()

# 处理 Transfection_method
df_Transfection_method = pd.get_dummies(df.Transfection_method)
df_Transfection_method.columns = [f"feat_Transfection_method_{c}" for c in df_Transfection_method.columns]

# 处理 Duration_after_transfection_h
df_Duration_after_transfection_h = pd.get_dummies(df.Duration_after_transfection_h)
df_Duration_after_transfection_h.columns = [f"feat_Duration_after_transfection_h_{c}" for c in
                                            df_Duration_after_transfection_h.columns]

# 特征合并
features = pd.concat(
    [
        df_publication_id,
        df_gene_target_symbol_name,
        df_gene_target_ncbi_id,
        df_gene_target_species,
        df_siRNA_duplex_id,
        df_cell_line_donor,
        df_siRNA_concentration,
        df_Transfection_method,
        df_Duration_after_transfection_h,
        siRNA_feat_builder(df.siRNA_sense_seq, False),
        siRNA_feat_builder(df.siRNA_antisense_seq, True),
    ],
    axis=1,
)

# 确保 mRNA_remaining_pct 是最后一列
features['mRNA_remaining_pct'] = df['mRNA_remaining_pct']

# 计算样本权重
weight_ls = np.array(features['mRNA_remaining_pct'].apply(lambda x: 2 if ((x <= 30) and (x >= 0)) else 1))

# 将特征和标签分开
X = features.iloc[:, :-1]
y = features['mRNA_remaining_pct']


# 自定义评价函数
def calculate_metrics(preds, data, threshold=30):
    y_pred = preds
    y_true = data.get_label()
    mae = np.mean(np.abs(y_true - y_pred))

    y_true_binary = ((y_true <= threshold) & (y_true >= 0)).astype(int)
    y_pred_binary = ((y_pred <= threshold) & (y_pred >= 0)).astype(int)

    mask = (y_pred >= 0) & (y_pred <= threshold)
    range_mae = mean_absolute_error(y_true[mask], y_pred[mask]) if np.sum(mask) > 0 else 100

    precision = (np.array(y_pred_binary) & y_true_binary).sum() / np.sum(y_pred_binary) if np.sum(
        y_pred_binary) > 0 else 0
    recall = (np.array(y_pred_binary) & y_true_binary).sum() / np.sum(y_true_binary) if np.sum(y_true_binary) > 0 else 0

    if precision + recall == 0:
        f1 = 0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    score = (1 - mae / 100) * 0.5 + (1 - range_mae / 100) * f1 * 0.5
    return "custom_score", score, True


# 自适应学习率回调函数
def adaptive_learning_rate(decay_rate=0.8, patience=50):
    best_score = float("-inf")
    wait = 0

    def callback(env):
        nonlocal best_score, wait
        current_score = env.evaluation_result_list[-1][2]
        current_lr = env.model.params.get('learning_rate')

        if current_score > best_score:
            best_score = current_score
            wait = 0
        else:
            wait += 1

        if wait >= patience:
            new_lr = float(current_lr) * decay_rate
            wait = 0
            env.model.params['learning_rate'] = new_lr
            print(f"Learning rate adjusted to {env.model.params.get('learning_rate')}")

    return callback


# 训练函数
def train(features, n_original, weight_ls):
    n_splits = 10
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    gbms = []
    for fold, (train_idx, val_idx) in enumerate(kf.split(features.iloc[:n_original, :]), 1):
        print(f"Starting fold {fold}")
        X_train, X_val = features.iloc[train_idx, :-1], features.iloc[val_idx, :-1]
        y_train, y_val = features.iloc[train_idx, -1], features.iloc[val_idx, -1]
        w_train = weight_ls[train_idx]

        train_data = lgb.Dataset(X_train, label=y_train, weight=w_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

        params = {
            "boosting_type": "gbdt",
            "objective": "regression",
            "metric": "None",
            "max_depth": 8,
            "num_leaves": 63,
            "min_data_in_leaf": 2,
            "learning_rate": 0.05,
            "feature_fraction": 0.9,
            "lambda_l1": 0.1,
            "lambda_l2": 0.2,
            "verbose": -1,
            "num_threads": 8,
        }

        adaptive_lr = adaptive_learning_rate(decay_rate=0.9, patience=1000)
        gbm = lgb.train(
            params,
            train_data,
            num_boost_round=25000,
            valid_sets=[val_data],
            feval=calculate_metrics,
            callbacks=[
                adaptive_lr,
                lgb.log_evaluation(period=200, show_stdv=True),
                lgb.early_stopping(stopping_rounds=int(25000 * 0.1), first_metric_only=True, verbose=True,
                                   min_delta=0.00001)
            ]
        )
        valid_score = gbm.best_score["valid_0"]["custom_score"]
        print(f"Fold {fold} best valid score: {valid_score}")
        gbms.append(gbm)

    return gbms


# 进行模型训练
n_original = len(df)
trained_gbms = train(features, n_original, weight_ls)

# 预测并保存结果
y_pred = np.mean([gbm.predict(features.iloc[n_original:, :-1]) for gbm in trained_gbms], axis=0)
# %%
df_submit = pd.DataFrame({"mRNA_remaining_pct": y_pred})
df_submit.to_csv("submission.csv", index=False)