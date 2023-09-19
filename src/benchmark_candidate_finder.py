import os
from os.path import join
from pathlib import Path
from time import time
from pdf_features.PdfPage import PdfPage
from pdf_reading_order.PdfReadingOrderTokens import PdfReadingOrderTokens
from sklearn.metrics import f1_score, accuracy_score
from pdf_reading_order.config import ROOT_PATH, PDF_LABELED_DATA_ROOT_PATH
from pdf_reading_order.load_labeled_data import load_labeled_data
from pdf_reading_order.ReadingOrderCandidatesTrainer import ReadingOrderCandidatesTrainer
from pdf_reading_order.model_configuration import CANDIDATE_MODEL_CONFIGURATION

BENCHMARK_MODEL_PATH = Path(join(ROOT_PATH, "model", "candidate_selector_benchmark.model"))


def loop_pdf_reading_order_tokens(pdf_reading_order_tokens_list: list[PdfReadingOrderTokens], page: PdfPage):
    for pdf_reading_order_tokens in pdf_reading_order_tokens_list:
        if page not in pdf_reading_order_tokens.labeled_page_by_raw_page:
            continue
        yield pdf_reading_order_tokens


def loop_current_token_candidate_token_labels(trainer, pdf_reading_order_tokens_list: list[PdfReadingOrderTokens]):
    for current_token, possible_candidate_token, page in trainer.loop_token_combinations():
        for pdf_reading_order_tokens in loop_pdf_reading_order_tokens(pdf_reading_order_tokens_list, page):
            label = pdf_reading_order_tokens.labeled_page_by_raw_page[page].is_next_token(
                current_token, possible_candidate_token
            )
            yield label


def train_for_benchmark():
    pdf_reading_order_tokens_list = load_labeled_data(PDF_LABELED_DATA_ROOT_PATH, filter_in="train")

    pdf_features_list = [pdf_reading_order_tokens.pdf_features for pdf_reading_order_tokens in pdf_reading_order_tokens_list]
    trainer = ReadingOrderCandidatesTrainer(pdf_features_list, CANDIDATE_MODEL_CONFIGURATION)

    labels = []
    for label in loop_current_token_candidate_token_labels(trainer, pdf_reading_order_tokens_list):
        labels.append(label)

    os.makedirs(BENCHMARK_MODEL_PATH.parent, exist_ok=True)
    trainer.train(str(BENCHMARK_MODEL_PATH), labels)


def predict_for_benchmark():
    pdf_reading_order_tokens_list = load_labeled_data(PDF_LABELED_DATA_ROOT_PATH, filter_in="test")

    pdf_features_list = [pdf_reading_order_tokens.pdf_features for pdf_reading_order_tokens in pdf_reading_order_tokens_list]
    trainer = ReadingOrderCandidatesTrainer(pdf_features_list, CANDIDATE_MODEL_CONFIGURATION)

    truths = []
    for label in loop_current_token_candidate_token_labels(trainer, pdf_reading_order_tokens_list):
        truths.append(label)
    print("predicting")
    predictions = trainer.predict(BENCHMARK_MODEL_PATH)
    print(len(truths))
    print(len(predictions))
    return truths, predictions


def benchmark():
    train_for_benchmark()
    truths, predictions = predict_for_benchmark()

    f1 = round(f1_score(truths, predictions, average="macro") * 100, 2)
    accuracy = round(accuracy_score(truths, predictions) * 100, 2)
    print(f"F1 score {f1}%")
    print(f"Accuracy score {accuracy}%")


if __name__ == "__main__":
    start = time()
    print("start")
    benchmark()
    print("finished in", int(time() - start), "seconds")
