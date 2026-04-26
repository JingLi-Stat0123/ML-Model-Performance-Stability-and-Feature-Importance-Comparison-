import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from datetime import datetime
from xgboost import XGBClassifier
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer
import matplotlib
matplotlib.use('TkAgg')
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
import numpy as np

data = pd.read_csv(r"D:\HuaweiMoveData\Users\HUAWEI\Desktop\cs-training.csv",index_col=0)
print(f"前五行数据为：{data.iloc[0:10,:]}")

print(f"每列缺失值数量为：{data.isna().sum()}")
print(f"每列缺失值占比为：{data.isna().mean()}")
# 剔除缺失值所在行
data = data.dropna()
# 划分训练集和测试集
X,y=data.iloc[:,1:],data.iloc[:,0]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)
print(f"特征训练集的前10行数据为：{X_train.head(10)}")
print(f"特征训练集的前10行数据为：{y_train.head(10)}")

# 创建XGBoost模型_定义好搜索范围以及模型
t0 = datetime.now()
model_XGBoost = XGBClassifier(objective="binary:logistic",eval_metric="mlogloss",random_state=0)
search_spaces_XGBoost = {
    "n_estimators":Integer(100,300),
    "max_depth":Integer(3, 10),    # 树深度过于浅没啥意义，一般不搜索浪费时间
    "subsample":Real(0.6,1),
    "colsample_bytree":Real(0.5,1),
    "learning_rate":Real(0.01,0.3),
}
opt_XGBoost = BayesSearchCV(
    estimator=model_XGBoost,
    search_spaces=search_spaces_XGBoost,
    scoring='roc_auc',  # 这里要能熟练的辨认几个常见指标的使用范围
    n_iter=30,
    cv=3,
    n_jobs=-1,
    verbose=1,
    random_state=0,
)
# 创建随机森林模型_定义超参数搜索范围及模型
t1 = datetime.now()
model_RF = RandomForestClassifier(random_state=0)
search_spaces_RF = {
    "n_estimators":Integer(300,600),
    "max_depth":Integer(3, 10),    # 树深度过于浅没啥意义，一般不搜索浪费时间
    "max_samples":Real(0.6,1),
    "max_features": Categorical(['log2','sqrt']),
    "min_samples_split":Integer(2,30),    # Integer不支持步长，因为贝叶斯优化不是暴力穷举，而是智能优化
    "min_samples_leaf":Integer(5,30),     #
    "bootstrap":Categorical([True]),
}
# 创建逻辑回归模型_定义好超参数搜索范围及模型
model_LR = LogisticRegression(max_iter=1000,random_state=0)
grid_params = {
    "C":[0.01, 0.05, 0.1,1,3,5,10,20,50,100],
    "penalty":['l1', 'l2'],
    "solver":['liblinear'],
    "class_weight":[None,"balanced"]
}
# 嵌套交叉验证
models_config = [
    {
        "name": "随机森林模型",
        "estimator": model_RF,
        "searcher": "bayes",
        "search_space":search_spaces_RF
    },
    {
        "name": "逻辑回归模型",
        "estimator": Pipeline([
            ("scaler",StandardScaler()),
            ('clf', model_LR)]),
        "searcher": "params",
        "params":grid_params   # 使用网格搜索
    },
    {
        "name": "XGBoost模型",
        "estimator":model_XGBoost,
        "searcher": "bayes", # 标记使用贝叶斯搜索
        "search_space":search_spaces_XGBoost,
    }
]
inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=0)   #这种写法相较于直接写数字，可以指定随机数种子以及显示的设定shuffle=True
outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)


for cfg in models_config:
    inner_searcher=None
    if cfg["searcher"]=="bayes":
        inner_searcher=BayesSearchCV(
            estimator=cfg["estimator"],
            search_spaces=cfg["search_space"],
            scoring='roc_auc',
            n_iter=30,
            cv=inner_cv,
            n_jobs=-1,
            verbose=1,
            random_state=0,
        )
    else:
        inner_searcher=GridSearchCV(
            estimator=cfg["estimator"],
            param_grid=cfg["params"],
            scoring='roc_auc',
            cv=inner_cv,
            n_jobs=-1,
            verbose=1,
        )
    try:
         scores_acc = cross_val_score(inner_searcher, X, y, cv=outer_cv, scoring='accuracy', n_jobs=-1)
         scores_auc = cross_val_score(inner_searcher, X, y, cv=outer_cv, scoring='roc_auc_ovr', n_jobs=-1)
    except Exception as e:
        print(f"发生错误: {e}")
    print(f"{cfg['name']:<25} | Accuracy    | {np.mean(scores_acc):.3f}   | {np.std(scores_acc):.3f}")
    print(f"{cfg['name']:<25} | AUC         | {np.mean(scores_auc):.3f}   | {np.std(scores_auc):.3f}")