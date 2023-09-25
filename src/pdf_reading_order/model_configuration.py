from pdf_tokens_type_trainer.ModelConfiguration import ModelConfiguration

candidate_config_json = {
    "boosting_type": "gbdt",
    "verbose": -1,
    "learning_rate": 0.1,
    "num_class": 2,
    "context_size": 18,
    "num_boost_round": 500,
    "num_leaves": 287,
    "bagging_fraction": 0.6785938652428409,
    "bagging_freq": 7,
    "feature_fraction": 0.8144210423840036,
    "lambda_l1": 4.620714189907504e-06,
    "lambda_l2": 0.29998129470422397,
    "min_data_in_leaf": 74,
    "feature_pre_filter": True,
    "seed": 22,
    "deterministic": True,
}


reading_order_config_json = {
    "boosting_type": "gbdt",
    "verbose": -1,
    "learning_rate": 0.1,
    "num_class": 2,
    "context_size": 18,
    "num_boost_round": 800,
    "num_leaves": 255,
    "bagging_fraction": 1.0,
    "bagging_freq": 0,
    "feature_fraction": 0.9,
    "lambda_l1": 0.03042483474,
    "lambda_l2": 0.42010028,
    "min_data_in_leaf": 20,
    "feature_pre_filter": False,
    "seed": 22,
    "deterministic": True,
}

CANDIDATE_MODEL_CONFIGURATION = ModelConfiguration(**candidate_config_json)
READING_ORDER_MODEL_CONFIGURATION = ModelConfiguration(**reading_order_config_json)


if __name__ == "__main__":
    print(CANDIDATE_MODEL_CONFIGURATION)
