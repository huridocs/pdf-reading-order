from pdf_tokens_type_trainer.ModelConfiguration import ModelConfiguration

candidate_config_json = {
    "boosting_type": "gbdt",
    "verbose": -1,
    "learning_rate": 0.1,
    "num_class": 2,
    "context_size": 18,
    "num_boost_round": 500,
    "num_leaves": 455,
    "bagging_fraction": 0.9582447218059061,
    "bagging_freq": 1,
    "feature_fraction": 0.7479496700086276,
    "lambda_l1": 0.00017789899076539243,
    "lambda_l2": 0.050461704863219915,
    "min_data_in_leaf": 98,
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
    "bagging_fraction": 0.8140916039821084,
    "bagging_freq": 9,
    "feature_fraction": 0.3526400612810575,
    "lambda_l1": 5.058643948078386e-08,
    "lambda_l2": 0.017293649765588552,
    "min_data_in_leaf": 34,
    "feature_pre_filter": False,
    "seed": 22,
    "deterministic": True,
}

CANDIDATE_MODEL_CONFIGURATION = ModelConfiguration(**candidate_config_json)
READING_ORDER_MODEL_CONFIGURATION = ModelConfiguration(**reading_order_config_json)


if __name__ == "__main__":
    print(CANDIDATE_MODEL_CONFIGURATION)
