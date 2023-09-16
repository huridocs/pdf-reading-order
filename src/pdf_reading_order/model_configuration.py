from pdf_tokens_type_trainer.ModelConfiguration import ModelConfiguration

candidate_config_json = {
    "boosting_type": "gbdt",
    "verbose": -1,
    "learning_rate": 0.1,
    "num_class": 2,
    "context_size": 17,
    "num_boost_round": 250,
    "num_leaves": 207,
    "bagging_fraction": 1.0,
    "bagging_freq": 0,
    "feature_fraction": 0.752,
    "lambda_l1": 0.00112,
    "lambda_l2": 0.28169,
    "min_data_in_leaf": 25,
    "feature_pre_filter": False,
    "seed": 22,
    "deterministic": True,
}

CANDIDATE_MODEL_CONFIGURATION = ModelConfiguration(**candidate_config_json)

if __name__ == "__main__":
    print(CANDIDATE_MODEL_CONFIGURATION)
