import os
from os.path import join
from pathlib import Path
from time import time

import numpy as np
from pdf_features.PdfPage import PdfPage
from pdf_reading_order.PdfReadingOrderTokens import PdfReadingOrderTokens
from sklearn.metrics import f1_score, accuracy_score
from pdf_reading_order.config import ROOT_PATH, PDF_LABELED_DATA_ROOT_PATH
from pdf_reading_order.load_labeled_data import load_labeled_data
from pdf_reading_order.ReadingOrderTrainer import ReadingOrderTrainer
from pdf_reading_order.model_configuration import READING_ORDER_MODEL_CONFIGURATION

BENCHMARK_MODEL_PATH = Path(join(ROOT_PATH, "model", "reading_order_benchmark.model"))


def loop_pdf_reading_order_tokens(pdf_reading_order_tokens_list: list[PdfReadingOrderTokens], page: PdfPage):
    for pdf_reading_order_tokens in pdf_reading_order_tokens_list:
        if page not in pdf_reading_order_tokens.labeled_page_by_raw_page:
            continue
        yield pdf_reading_order_tokens


def loop_current_token_candidate_token_labels(trainer, pdf_reading_order_tokens_list: list[PdfReadingOrderTokens]):
    for _, candidate_token_1, candidate_token_2, _, page in trainer.loop_candidate_tokens():
        for pdf_reading_order_tokens in loop_pdf_reading_order_tokens(pdf_reading_order_tokens_list, page):
            label = pdf_reading_order_tokens.labeled_page_by_raw_page[page].is_coming_earlier(candidate_token_1, candidate_token_2)
            yield label


def loop_reading_order_labels(trainer, pdf_reading_order_tokens_list: list[PdfReadingOrderTokens]):
    for _, page in trainer.loop_token_features():
        for pdf_reading_order_tokens in loop_pdf_reading_order_tokens(pdf_reading_order_tokens_list, page):
            yield page, pdf_reading_order_tokens.labeled_page_by_raw_page[page].reading_order_by_token_id


def loop_next_id_by_current_ids(trainer, pdf_reading_order_tokens_list: list[PdfReadingOrderTokens]):
    token_count = 0
    for page, reading_order_by_token_id in loop_reading_order_labels(trainer, pdf_reading_order_tokens_list):
        token_count += len(page.tokens)
        tokens_ids_in_reading_order = [dict_item[0] for dict_item in sorted(reading_order_by_token_id.items(), key=lambda item: item[1])]
        tokens_ids_in_reading_order.remove("pad_token")
        next_id_by_current_id_labels = {current_id: next_id for current_id, next_id in zip(tokens_ids_in_reading_order, tokens_ids_in_reading_order[1:])}
        token_ids_by_prediction_order = [token.id for token in sorted(page.tokens, key=lambda t: t.prediction)]
        next_id_by_current_id_predictions = {current_id: next_id for current_id, next_id in zip(token_ids_by_prediction_order, token_ids_by_prediction_order[1:])}
        for current_id, next_id in next_id_by_current_id_labels.items():
            yield current_id, next_id, next_id_by_current_id_predictions
    print(f"There are {token_count} tokens in tested data")


def train_for_benchmark():
    pdf_reading_order_tokens_list = load_labeled_data(PDF_LABELED_DATA_ROOT_PATH, filter_in="train")

    pdf_features_list = [pdf_reading_order_tokens.pdf_features for pdf_reading_order_tokens in pdf_reading_order_tokens_list]
    trainer = ReadingOrderTrainer(pdf_features_list, READING_ORDER_MODEL_CONFIGURATION)
    # trainer.set_token_types()

    labels = []
    for label in loop_current_token_candidate_token_labels(trainer, pdf_reading_order_tokens_list):
        labels.append(label)

    os.makedirs(BENCHMARK_MODEL_PATH.parent, exist_ok=True)
    trainer.train(str(BENCHMARK_MODEL_PATH), labels)


def predict_for_benchmark():
    pdf_reading_order_tokens_list = load_labeled_data(PDF_LABELED_DATA_ROOT_PATH, filter_in="test")

    pdf_features_list = [pdf_reading_order_tokens.pdf_features for pdf_reading_order_tokens in pdf_reading_order_tokens_list]
    trainer = ReadingOrderTrainer(pdf_features_list, READING_ORDER_MODEL_CONFIGURATION)
    # trainer.set_token_types()

    truths = []
    for label in loop_current_token_candidate_token_labels(trainer, pdf_reading_order_tokens_list):
        truths.append(label)
    print("predicting")
    predictions = trainer.predict(BENCHMARK_MODEL_PATH)
    print(len(truths))
    print(len(predictions))
    return truths, predictions


def get_reading_orders_for_benchmark():
    pdf_reading_order_tokens_list = load_labeled_data(PDF_LABELED_DATA_ROOT_PATH, filter_in="test")
    pdf_features_list = [pdf_reading_order_tokens.pdf_features for pdf_reading_order_tokens in pdf_reading_order_tokens_list]
    trainer = ReadingOrderTrainer(pdf_features_list, READING_ORDER_MODEL_CONFIGURATION)
    print("predicting")
    trainer.get_reading_ordered_pages(BENCHMARK_MODEL_PATH)
    mistakes = 0
    for current_id, next_id, next_id_by_current_id_predictions in loop_next_id_by_current_ids(trainer, pdf_reading_order_tokens_list):
        if current_id not in next_id_by_current_id_predictions or next_id != next_id_by_current_id_predictions[current_id]:
            mistakes += 1
    print(f"There are {mistakes} sequential mistakes")


def benchmark():
    train_for_benchmark()
    truths, predictions = predict_for_benchmark()
    get_reading_orders_for_benchmark()
    print("non-zero count:", np.count_nonzero(truths))
    print("truths len:", len(truths))

    f1 = round(f1_score(truths, predictions, average="macro") * 100, 2)
    accuracy = round(accuracy_score(truths, predictions) * 100, 2)
    print(f"F1 score {f1}%")
    print(f"Accuracy score {accuracy}%")


if __name__ == "__main__":
    start = time()
    print("start")
    benchmark()
    print("finished in", int(time() - start), "seconds")
