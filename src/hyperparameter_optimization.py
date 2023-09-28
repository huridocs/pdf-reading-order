import optuna
import lightgbm as lgb
import pickle
import numpy as np
from os.path import exists
from functools import partial
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from benchmark_candidate_finder import loop_current_token_candidate_token_labels
from benchmark_reading_order import loop_current_token_candidate_token_labels as loop_reading_order_labels
from pdf_reading_order.ReadingOrderCandidatesTrainer import ReadingOrderCandidatesTrainer
from pdf_reading_order.ReadingOrderTrainer import ReadingOrderTrainer
from pdf_reading_order.load_labeled_data import load_labeled_data
from pdf_reading_order.config import PDF_LABELED_DATA_ROOT_PATH

CANDIDATES_DATA_PATH = "data/candidates_X.pickle"
CANDIDATES_LABEL_PATH = "data/candidates_y.pickle"
READING_ORDER_DATA_PATH = "data/reading_order_X.pickle"
READING_ORDER_LABEL_PATH = "data/reading_order_y.pickle"


def create_candidates_pickle():
    pdf_reading_order_tokens_list = load_labeled_data(PDF_LABELED_DATA_ROOT_PATH, filter_in="train")

    pdf_features_list = [pdf_reading_order_tokens.pdf_features for pdf_reading_order_tokens in pdf_reading_order_tokens_list]
    trainer = ReadingOrderCandidatesTrainer(pdf_features_list, None)
    x = trainer.get_model_input()
    y = []
    for label in loop_current_token_candidate_token_labels(trainer, pdf_reading_order_tokens_list):
        y.append(label)

    with open(CANDIDATES_DATA_PATH, "wb") as x_file:
        pickle.dump(x, x_file)
    with open(CANDIDATES_LABEL_PATH, "wb") as y_file:
        pickle.dump(y, y_file)


def create_reading_order_pickle():
    pdf_reading_order_tokens_list = load_labeled_data(PDF_LABELED_DATA_ROOT_PATH, filter_in="train")

    pdf_features_list = [pdf_reading_order_tokens.pdf_features for pdf_reading_order_tokens in pdf_reading_order_tokens_list]
    trainer = ReadingOrderTrainer(pdf_features_list, None)
    x = trainer.get_model_input()
    y = []
    for label in loop_reading_order_labels(trainer, pdf_reading_order_tokens_list):
        y.append(label)

    with open(READING_ORDER_DATA_PATH, "wb") as x_file:
        pickle.dump(x, x_file)
    with open(READING_ORDER_LABEL_PATH, "wb") as y_file:
        pickle.dump(y, y_file)


def create_pickle_files():
    if not exists(CANDIDATES_DATA_PATH):
        print("Getting candidates data")
        create_candidates_pickle()
    # if not exists(READING_ORDER_DATA_PATH):
    #     print("Getting reading order data")
    #     create_reading_order_pickle()


def get_data(data_path: str, label_path: str):
    print("Loading X from: ", data_path)
    print("Loading y from: ", label_path)
    with open(data_path, "rb") as f:
        x_train = pickle.load(f)
    with open(label_path, "rb") as f:
        y_train = pickle.load(f)
    return x_train, np.array(y_train)


def objective(trial: optuna.trial.Trial, task: str):
    if task == "candidates":
        X, y = get_data(CANDIDATES_DATA_PATH, CANDIDATES_LABEL_PATH)
    else:
        X, y = get_data(READING_ORDER_DATA_PATH, READING_ORDER_LABEL_PATH)
    n_splits = 5
    random_states = [42, 427, 311]
    roc_aucs = []

    for random_state in random_states:
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        for train_idx, val_idx in skf.split(X, y):
            x_train, x_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            train_data = lgb.Dataset(x_train, label=y_train)
            val_data = lgb.Dataset(x_val, label=y_val, reference=train_data)

            params = {
                "boosting_type": "gbdt",
                "objective": "multiclass",
                "metric": "multi_logloss",
                "learning_rate": 0.1,
                "num_class": 2,
                "num_leaves": trial.suggest_int("num_leaves", 10, 500),
                "bagging_fraction": trial.suggest_float("bagging_fraction", 0.1, 1.0),
                "bagging_freq": trial.suggest_int("bagging_freq", 1, 10),
                "feature_fraction": trial.suggest_float("feature_fraction", 0.1, 1.0),
                "lambda_l1": trial.suggest_float("lambda_l1", 1e-08, 10.0, log=True),
                "lambda_l2": trial.suggest_float("lambda_l2", 1e-08, 10.0, log=True),
                "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 5, 100),
                "feature_pre_filter": trial.suggest_categorical("feature_pre_filter", [False, True]),
                # 'num_boost_round': 100,
                "early_stopping_rounds": 10,
                "verbose": -1,
            }
            model = lgb.train(params, train_data, valid_sets=[train_data, val_data], num_boost_round=100)
            y_pred_scores = model.predict(x_val, num_iteration=model.best_iteration)
            roc_auc = roc_auc_score(y_val, y_pred_scores[:, 1], multi_class="ovr")
            roc_aucs.append(roc_auc)

    return sum(roc_aucs) / (n_splits * len(random_states))


def optuna_automatic_tuning(task: str):
    create_pickle_files()
    study = optuna.create_study(direction="maximize")
    objective_with_task = partial(objective, task=task)
    study.optimize(objective_with_task, n_trials=100)

    print("Number of finished trials: ", len(study.trials))
    print("Best trial: ")
    trial = study.best_trial

    print("Value: ", trial.value)
    print("Params: ")
    result_string: str = ""
    for key, value in trial.params.items():
        print(f"\t{key}: {value}")
        result_string += f'"{key}": {value},\n'
    result_string += f"-> Best trial value: {str(trial.value)}\n"

    result_string += "\n\n\n"

    with open(f"src/tuned_parameters/{task}.txt", "a") as result_file:
        result_file.write(result_string)


if __name__ == "__main__":
    optuna_automatic_tuning("candidates")
