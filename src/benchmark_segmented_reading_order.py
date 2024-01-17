import os
import pickle
from os.path import join
from time import time
import numpy as np
import pdf_reading_order.ReadingOrderTrainer
from BenchmarkTable import BenchmarkTable
from benchmark_reading_order import find_mistake_count
from pdf_reading_order.PdfReadingOrderTokens import PdfReadingOrderTokens
from pdf_reading_order.ReadingOrderTrainer import ReadingOrderTrainer
from pdf_reading_order.config import PDF_LABELED_DATA_ROOT_PATH
from pdf_reading_order.load_labeled_data import load_labeled_data
from SegmentProcessor import SegmentProcessor
from pdf_reading_order.config import ROOT_PATH
from pdf_reading_order.model_configuration import SEGMENTED_READING_ORDER_MODEL_CONFIGURATION

BENCHMARK_MODEL_PATH = join(ROOT_PATH, "model", "segmented_reading_order_benchmark.model")
BENCHMARK_COMPARISON_MODEL_PATH = join(ROOT_PATH, "model", "segmented_reading_order_benchmark")
SEGMENTED_PDF_READING_ORDER_TOKENS_TRAIN_PATH = "data/segmented_pdf_reading_order_tokens_train.pickle"
SEGMENTED_PDF_READING_ORDER_TOKENS_TEST_PATH = "data/segmented_pdf_reading_order_tokens_test.pickle"
SEGMENTED_READING_ORDER_X_TRAIN_PATH = f"data/segmented_reading_order_X_train.pickle"
SEGMENTED_READING_ORDER_Y_TRAIN_PATH = f"data/segmented_reading_order_y_train.pickle"
SEGMENTED_READING_ORDER_X_TEST_PATH = f"data/segmented_reading_order_X_test.pickle"
SEGMENTED_READING_ORDER_Y_TEST_PATH = f"data/segmented_reading_order_y_test.pickle"


def prepare_features(dataset_type, x_path, y_path):
    pdf_reading_order_tokens_list = get_segmented_pdf_reading_order_tokens(dataset_type)
    trainer = ReadingOrderTrainer(pdf_reading_order_tokens_list, None)
    x, y = trainer.get_training_data()
    if not os.path.exists(join(ROOT_PATH, "src", "data")):
        os.makedirs(join(ROOT_PATH, "src", "data"))
    with open(x_path, "wb") as x_file:
        pickle.dump(x, x_file)
    with open(y_path, "wb") as y_file:
        pickle.dump(y, y_file)
    return x, np.array(y)


def get_features(dataset_type: str = "train"):
    x_path = SEGMENTED_READING_ORDER_X_TRAIN_PATH if dataset_type == "train" else SEGMENTED_READING_ORDER_X_TEST_PATH
    y_path = SEGMENTED_READING_ORDER_Y_TRAIN_PATH if dataset_type == "train" else SEGMENTED_READING_ORDER_Y_TEST_PATH
    if os.path.exists(x_path) and os.path.exists(y_path):
        with open(x_path, "rb") as f:
            x_features = pickle.load(f)
        with open(y_path, "rb") as f:
            y_features = np.array(pickle.load(f))
        return x_features, np.array(y_features)

    return prepare_features(dataset_type, x_path, y_path)


def prepare_segmented_pdf_reading_order_tokens_list(dataset_type, file_path):
    pdf_reading_order_tokens_list = load_labeled_data(PDF_LABELED_DATA_ROOT_PATH, filter_in=dataset_type)
    start_time = time()
    segment_processor = SegmentProcessor(pdf_reading_order_tokens_list)
    segment_processor.process()
    total_time = time() - start_time
    print(f"Segment processing took: {round(total_time, 2)} seconds.")
    with open(file_path, "wb") as pdf_reading_order_tokens_file:
        pickle.dump(pdf_reading_order_tokens_list, pdf_reading_order_tokens_file)
    return pdf_reading_order_tokens_list


def get_segmented_pdf_reading_order_tokens(dataset_type: str = "train"):
    print(f"Loading {dataset_type} data...")
    file_path = (
        SEGMENTED_PDF_READING_ORDER_TOKENS_TRAIN_PATH
        if dataset_type == "train"
        else SEGMENTED_PDF_READING_ORDER_TOKENS_TEST_PATH
    )
    if not os.path.exists(join(ROOT_PATH, "src", "data")):
        os.makedirs(join(ROOT_PATH, "src", "data"))
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            return pickle.load(f)

    return prepare_segmented_pdf_reading_order_tokens_list(dataset_type, file_path)


def train_for_benchmark(include_test_set: bool = False):
    pdf_reading_order.ReadingOrderTrainer.USE_CANDIDATES_MODEL = False
    x_train, y_train = get_features("train")
    if include_test_set:
        x_test, y_test = get_features("test")
        x_train = np.concatenate((x_train, x_test), axis=0)
        y_train = np.append(y_train, y_test)
    trainer = ReadingOrderTrainer([], SEGMENTED_READING_ORDER_MODEL_CONFIGURATION)
    trainer.train(BENCHMARK_MODEL_PATH, x_train, y_train)


def predict_for_benchmark(pdf_reading_order_tokens_list: list[PdfReadingOrderTokens], get_granular_scores: bool):
    trainer = ReadingOrderTrainer(pdf_reading_order_tokens_list, SEGMENTED_READING_ORDER_MODEL_CONFIGURATION)
    print(f"Model prediction started...")
    start_time = time()
    trainer.predict(BENCHMARK_MODEL_PATH)
    total_time = time() - start_time
    if get_granular_scores:
        table_name = f"_segmented_reading_order"
        benchmark_table = BenchmarkTable(pdf_reading_order_tokens_list, total_time, table_name)
        benchmark_table.prepare_benchmark_table()
    return total_time


def benchmark(get_granular_scores: bool):
    pdf_reading_order.ReadingOrderTrainer.USE_CANDIDATES_MODEL = False
    pdf_reading_order_tokens_list = get_segmented_pdf_reading_order_tokens("test")
    total_time = predict_for_benchmark(pdf_reading_order_tokens_list, get_granular_scores)
    mistake_count_for_model = find_mistake_count(pdf_reading_order_tokens_list)
    print(f"{mistake_count_for_model} mistakes found. Total time: {total_time}")


if __name__ == "__main__":
    train_for_benchmark()
    benchmark(True)
