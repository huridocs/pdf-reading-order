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
    "num_boost_round": 150,
    "num_leaves": 251,
    "bagging_fraction": 0.8688878683590495,
    "bagging_freq": 7,
    "feature_fraction": 0.3625550811452055,
    "lambda_l1": 1.6542447883818264e-06,
    "lambda_l2": 0.1213023968072871,
    "min_data_in_leaf": 73,
    "feature_pre_filter": True,
    "seed": 22,
    "deterministic": True,
}

CANDIDATE_MODEL_CONFIGURATION = ModelConfiguration(**candidate_config_json)
READING_ORDER_MODEL_CONFIGURATION = ModelConfiguration(**reading_order_config_json)


if __name__ == "__main__":
    print(CANDIDATE_MODEL_CONFIGURATION)
