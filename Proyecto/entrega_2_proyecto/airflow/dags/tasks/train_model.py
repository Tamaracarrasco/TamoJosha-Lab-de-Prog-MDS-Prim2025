# En este scripts se busca que se definan
# las funciones para entrenar el modelo

from xgboost import XGBClassifier

def train_model(**context):

    n_neg = np.sum(y_train == 0)
    n_pos = np.sum(y_train == 1)

    scale_pos_weight = n_neg / n_pos

    params = {'learning_rate': 0.06017425142691644, 
    'n_estimators': 414, 
    'max_depth': 9, 
    'min_child_weight': 1, 
    'reg_alpha': 0.6131484039416987,
    'reg_lambda': 0.6134461818710661}

    pipeline_xgb = Pipeline(
        steps=[
        ("col_transformer", col_transformer),
        ("clasificador", XGBClassifier(**params,
        scale_pos_weight=scale_pos_weight))
    ]
)