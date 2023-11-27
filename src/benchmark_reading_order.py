import os
import pickle
from os.path import join
import numpy as np
from time import time
from BenchmarkTable import BenchmarkTable
from pdf_reading_order.PdfReadingOrderTokens import PdfReadingOrderTokens
from pdf_reading_order.ReadingOrderTrainer import ReadingOrderTrainer
from pdf_reading_order.config import ROOT_PATH, PDF_LABELED_DATA_ROOT_PATH
from pdf_reading_order.load_labeled_data import load_labeled_data
from pdf_reading_order.model_configuration import READING_ORDER_MODEL_CONFIGURATION
from TableFigureProcessor import TableFigureProcessor
from pdf_reading_order.ReadingOrderTrainer import CANDIDATE_COUNT

BENCHMARK_MODEL_PATH = join(ROOT_PATH, "model", "reading_order_benchmark.model")
BENCHMARK_COMPARISON_MODEL_PATH = join(ROOT_PATH, "model", "reading_order_benchmark")
READING_ORDER_X_TRAIN_PATH = f"data/reading_order_{CANDIDATE_COUNT}_X_train.pickle"
READING_ORDER_Y_TRAIN_PATH = f"data/reading_order_{CANDIDATE_COUNT}_y_train.pickle"
READING_ORDER_X_TEST_PATH = f"data/reading_order_{CANDIDATE_COUNT}_X_test.pickle"
READING_ORDER_Y_TEST_PATH = f"data/reading_order_{CANDIDATE_COUNT}_y_test.pickle"
PDF_READING_ORDER_TOKENS_TRAIN_PATH = "data/pdf_reading_order_tokens_train.pickle"
PDF_READING_ORDER_TOKENS_TEST_PATH = "data/pdf_reading_order_tokens_test.pickle"


def prepare_features(dataset_type, x_path, y_path):
    pdf_reading_order_tokens_list = get_pdf_reading_order_tokens(dataset_type)
    trainer = ReadingOrderTrainer(pdf_reading_order_tokens_list, None)
    x, y = trainer.get_training_data()
    with open(x_path, "wb") as x_file:
        pickle.dump(x, x_file)
    with open(y_path, "wb") as y_file:
        pickle.dump(y, y_file)
    return x, np.array(y)


def get_features(dataset_type: str = "train"):
    x_path = READING_ORDER_X_TRAIN_PATH if dataset_type == "train" else READING_ORDER_X_TEST_PATH
    y_path = READING_ORDER_Y_TRAIN_PATH if dataset_type == "train" else READING_ORDER_Y_TEST_PATH
    if os.path.exists(x_path) and os.path.exists(y_path):
        with open(x_path, "rb") as f:
            x_features = pickle.load(f)
        with open(y_path, "rb") as f:
            y_features = np.array(pickle.load(f))
        return x_features, np.array(y_features)

    return prepare_features(dataset_type, x_path, y_path)


def loop_pages(pdf_reading_order_tokens_list: list[PdfReadingOrderTokens]):
    for pdf_reading_order_tokens in pdf_reading_order_tokens_list:
        for page in pdf_reading_order_tokens.pdf_features.pages:
            label_page = pdf_reading_order_tokens.labeled_page_by_raw_page[page]
            yield label_page, page


def find_mistake_count(pdf_reading_order_tokens_list: list[PdfReadingOrderTokens]):
    mistakes = 0
    for label_page, page in loop_pages(pdf_reading_order_tokens_list):
        for token_1, token_2 in zip(page.tokens, page.tokens[1:]):
            mistakes += 0 if label_page.is_next_token(token_1, token_2) else 1
    return mistakes


def prepare_pdf_reading_order_tokens_list(dataset_type, file_path):
    pdf_reading_order_tokens_list = load_labeled_data(PDF_LABELED_DATA_ROOT_PATH, filter_in=dataset_type)
    for pdf_reading_order in pdf_reading_order_tokens_list:
        table_figure_processor = TableFigureProcessor(pdf_reading_order)
        table_figure_processor.process()
    with open(file_path, "wb") as pdf_reading_order_tokens_file:
        pickle.dump(pdf_reading_order_tokens_list, pdf_reading_order_tokens_file)
    return pdf_reading_order_tokens_list


def get_pdf_reading_order_tokens(dataset_type: str = "train"):
    print(f"Loading {dataset_type} data...")
    file_path = PDF_READING_ORDER_TOKENS_TRAIN_PATH if dataset_type == "train" else PDF_READING_ORDER_TOKENS_TEST_PATH
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            return pickle.load(f)

    return prepare_pdf_reading_order_tokens_list(dataset_type, file_path)


def predict_for_benchmark(pdf_reading_order_tokens_list: list[PdfReadingOrderTokens], get_granular_scores: bool):
    trainer = ReadingOrderTrainer(pdf_reading_order_tokens_list, READING_ORDER_MODEL_CONFIGURATION)
    print(f"Model prediction started...")
    start_time = time()
    trainer.predict(BENCHMARK_MODEL_PATH)
    total_time = time() - start_time
    if get_granular_scores:
        benchmark_table = BenchmarkTable(pdf_reading_order_tokens_list, total_time)
        benchmark_table.prepare_benchmark_table()
    return total_time


def train_for_benchmark(include_test_set: bool = False):
    x_train, y_train = get_features("train")
    if include_test_set:
        x_test, y_test = get_features("test")
        x_train = np.concatenate((x_train, x_test), axis=0)
        y_train = np.append(y_train, y_test)
    trainer = ReadingOrderTrainer([], READING_ORDER_MODEL_CONFIGURATION)
    trainer.train(BENCHMARK_MODEL_PATH, x_train, y_train)


def benchmark(get_granular_scores: bool):
    pdf_reading_order_tokens_list = get_pdf_reading_order_tokens("test")
    total_time = predict_for_benchmark(pdf_reading_order_tokens_list, get_granular_scores)
    mistake_count_for_model = find_mistake_count(pdf_reading_order_tokens_list)
    print(f"{mistake_count_for_model} mistakes found. Total time: {total_time}")


if __name__ == "__main__":
    train_for_benchmark()
    benchmark(False)
