import os
import pickle
from os.path import join
import numpy as np
from time import time
from pdf_reading_order.config import ROOT_PATH, PDF_LABELED_DATA_ROOT_PATH
from pdf_reading_order.load_labeled_data import load_labeled_data
from pdf_reading_order.ReadingOrderCandidatesTrainer import ReadingOrderCandidatesTrainer
from pdf_reading_order.model_configuration import CANDIDATE_MODEL_CONFIGURATION
from TableFigureProcessor import TableFigureProcessor

BENCHMARK_MODEL_PATH = join(ROOT_PATH, "model", "candidate_selector_benchmark")
CANDIDATES_X_TRAIN_PATH = "data/candidates_X_train.pickle"
CANDIDATES_Y_TRAIN_PATH = "data/candidates_y_train.pickle"
CANDIDATES_X_TEST_PATH = "data/candidates_X_test.pickle"
CANDIDATES_Y_TEST_PATH = "data/candidates_y_test.pickle"
PDF_READING_ORDER_TOKENS_TRAIN_PATH = "data/pdf_reading_order_tokens_train.pickle"
PDF_READING_ORDER_TOKENS_TEST_PATH = "data/pdf_reading_order_tokens_test.pickle"


def prepare_features(dataset_type, x_path, y_path):
    pdf_reading_order_tokens_list = load_labeled_data(PDF_LABELED_DATA_ROOT_PATH, filter_in=dataset_type)
    for pdf_reading_order in pdf_reading_order_tokens_list:
        table_figure_processor = TableFigureProcessor(pdf_reading_order)
        table_figure_processor.process()
    trainer = ReadingOrderCandidatesTrainer(pdf_reading_order_tokens_list, None)
    x, y = trainer.get_training_data()
    with open(x_path, "wb") as x_file:
        pickle.dump(x, x_file)
    with open(y_path, "wb") as y_file:
        pickle.dump(y, y_file)
    return x, np.array(y)


def get_features(dataset_type: str = "train"):
    x_path = CANDIDATES_X_TRAIN_PATH if dataset_type == "train" else CANDIDATES_X_TEST_PATH
    y_path = CANDIDATES_Y_TRAIN_PATH if dataset_type == "train" else CANDIDATES_Y_TEST_PATH

    if os.path.exists(x_path) and os.path.exists(y_path):
        with open(x_path, "rb") as f:
            x_features = pickle.load(f)
        with open(y_path, "rb") as f:
            y_features = np.array(pickle.load(f))
        return x_features, np.array(y_features)

    return prepare_features(dataset_type, x_path, y_path)


def prepare_pdf_reading_order_tokens_list(dataset_type, file_path):
    pdf_reading_order_tokens_list = load_labeled_data(PDF_LABELED_DATA_ROOT_PATH, filter_in=dataset_type)
    for pdf_reading_order in pdf_reading_order_tokens_list:
        table_figure_processor = TableFigureProcessor(pdf_reading_order)
        table_figure_processor.process()
    with open(file_path, "wb") as pdf_reading_order_tokens_file:
        pickle.dump(pdf_reading_order_tokens_list, pdf_reading_order_tokens_file)
    return pdf_reading_order_tokens_list


def get_pdf_reading_order_tokens(dataset_type: str = "train"):
    file_path = PDF_READING_ORDER_TOKENS_TRAIN_PATH if dataset_type == "train" else PDF_READING_ORDER_TOKENS_TEST_PATH
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            return pickle.load(f)

    return prepare_pdf_reading_order_tokens_list(dataset_type, file_path)


def train_for_benchmark(model_path: str, include_test_set: bool = False):
    x_train, y_train = get_features("train")
    if include_test_set:
        x_test, y_test = get_features("test")
        x_train = np.concatenate((x_train, x_test), axis=0)
        y_train = np.append(y_train, y_test)
    trainer = ReadingOrderCandidatesTrainer([], CANDIDATE_MODEL_CONFIGURATION)
    trainer.train(model_path, x_train, y_train)


def test_for_benchmark():
    pdf_reading_order_tokens_list = get_pdf_reading_order_tokens("train")
    pdf_reading_order_tokens_list.extend(get_pdf_reading_order_tokens("test"))
    results = []
    start_time = time()
    for model_name in sorted(os.listdir(join(ROOT_PATH, "model"))):
        for candidate_count in [18]:
            if not model_name.startswith("candidate"):
                continue
            print(f"Testing: {model_name} with {candidate_count} candidate")
            model_path = join(join(ROOT_PATH, "model", model_name))
            trainer = ReadingOrderCandidatesTrainer(pdf_reading_order_tokens_list, CANDIDATE_MODEL_CONFIGURATION)
            mistake_count_for_model = trainer.predict(model_path, candidate_count)
            results.append([model_name, mistake_count_for_model, candidate_count])
    print(f"Elapsed time: {time() - start_time} seconds")
    results.sort(key=lambda result: result[1])
    for result in results:
        print(result)


def train_models_for_comparison():
    start_time = time()
    for num_boost_round in [500]:
        for num_leaves in [455]:
            CANDIDATE_MODEL_CONFIGURATION.num_boost_round = num_boost_round
            CANDIDATE_MODEL_CONFIGURATION.num_leaves = num_leaves
            model_path: str = BENCHMARK_MODEL_PATH + f"_nbr{num_boost_round}_nl{num_leaves}.model"
            train_for_benchmark(model_path)
    print(f"Elapsed time: {time() - start_time} seconds")


if __name__ == "__main__":
    train_models_for_comparison()
    test_for_benchmark()
