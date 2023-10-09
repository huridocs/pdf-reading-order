import os
from os.path import join
from pathlib import Path
from time import time
from pdf_features.PdfPage import PdfPage
from CandidateScore import CandidateScore
from CandidatesEvaluator import CandidatesEvaluator
from PopplerEvaluator import PopplerEvaluator
from pdf_reading_order.PdfReadingOrderTokens import PdfReadingOrderTokens
from sklearn.metrics import f1_score, accuracy_score
from pdf_reading_order.config import ROOT_PATH, PDF_LABELED_DATA_ROOT_PATH
from pdf_reading_order.load_labeled_data import load_labeled_data
from pdf_reading_order.ReadingOrderCandidatesTrainer import ReadingOrderCandidatesTrainer
from pdf_reading_order.model_configuration import CANDIDATE_MODEL_CONFIGURATION

BENCHMARK_MODEL_PATH = Path(join(ROOT_PATH, "model", "candidate_selector_benchmark.model"))


def loop_pdf_reading_order_tokens(pdf_reading_order_list: list[PdfReadingOrderTokens], page: PdfPage):
    for pdf_reading_order_tokens in pdf_reading_order_list:
        if page not in pdf_reading_order_tokens.labeled_page_by_raw_page:
            continue
        yield pdf_reading_order_tokens


def loop_current_token_candidate_token_labels(trainer, pdf_reading_order_list: list[PdfReadingOrderTokens]):
    for current_token, possible_candidate_token, page in trainer.loop_token_combinations():
        for pdf_reading_order_tokens in loop_pdf_reading_order_tokens(pdf_reading_order_list, page):
            label = pdf_reading_order_tokens.labeled_page_by_raw_page[page].is_next_token(
                current_token, possible_candidate_token
            )
            yield label


def train_for_benchmark():
    pdf_reading_order_list = load_labeled_data(PDF_LABELED_DATA_ROOT_PATH, filter_in="train")

    pdf_features_list = [pdf_reading_order_tokens.pdf_features for pdf_reading_order_tokens in pdf_reading_order_list]
    trainer = ReadingOrderCandidatesTrainer(pdf_features_list, CANDIDATE_MODEL_CONFIGURATION)

    labels = []
    for label in loop_current_token_candidate_token_labels(trainer, pdf_reading_order_list):
        labels.append(label)
    os.makedirs(BENCHMARK_MODEL_PATH.parent, exist_ok=True)
    trainer.train(str(BENCHMARK_MODEL_PATH), labels)


def predict_for_benchmark():
    pdf_reading_order_list: list[PdfReadingOrderTokens] = load_labeled_data(
        PDF_LABELED_DATA_ROOT_PATH, filter_in="multi_column_test"
    )

    pdf_features_list = [pdf_reading_order_tokens.pdf_features for pdf_reading_order_tokens in pdf_reading_order_list]
    trainer = ReadingOrderCandidatesTrainer(pdf_features_list, CANDIDATE_MODEL_CONFIGURATION)

    truths = []
    for label in loop_current_token_candidate_token_labels(trainer, pdf_reading_order_list):
        truths.append(label)

    print("predicting")
    predictions = trainer.predict(BENCHMARK_MODEL_PATH)

    return trainer, pdf_reading_order_list, truths, predictions


def evaluate_contains_next_token(
    trainer: ReadingOrderCandidatesTrainer, pdf_reading_order_list: list[PdfReadingOrderTokens], predictions: list[float]
):
    candidates_scores: list[CandidateScore] = get_candidates_scores(trainer, predictions)

    for candidate_count in [49, 50, 51]:
        contains_next_token_list = list()
        for pdf_reading_order in pdf_reading_order_list:
            candidates_evaluator = CandidatesEvaluator(pdf_reading_order, candidates_scores, candidate_count)
            contains_next_token_list.extend(candidates_evaluator.contains_next_token())

        correct = [x for x in contains_next_token_list if x]
        accuracy = 100 * len(correct) / len(contains_next_token_list)
        print("For candidate count", candidate_count)
        print("Contains next token", round(accuracy, 2), "%")
        print("Contains next token mistakes", len(contains_next_token_list) - len(correct))
        print()


def evaluate_poppler_contains_next_token():
    pdf_reading_order_list: list[PdfReadingOrderTokens] = load_labeled_data(PDF_LABELED_DATA_ROOT_PATH, filter_in="train")
    token_count = 0
    for pdf_reading_order in pdf_reading_order_list:
        token_count += sum([len(page.tokens) for page in pdf_reading_order.pdf_features.pages])
    for candidate_count in [1, 10, 25, 50, 100, 150, 200]:
        missing_next_token_count = 0
        for pdf_reading_order in pdf_reading_order_list:
            poppler_evaluator = PopplerEvaluator(pdf_reading_order, candidate_count)
            missing_next_token_count += poppler_evaluator.get_missing_next_token_count()

        correct = token_count - missing_next_token_count
        accuracy = 100 * correct / token_count
        print("For candidate count: ", candidate_count)
        print("Contains next token: ", round(accuracy, 2), "%")
        print("Missing next token count: ", missing_next_token_count)
        print()


def get_candidates_scores(trainer: ReadingOrderCandidatesTrainer, predictions: list[float]) -> list[CandidateScore]:
    candidates_scores: list[CandidateScore] = list()
    for index, (current_token, possible_candidate_token, page) in enumerate(trainer.loop_token_combinations()):
        candidate_score = CandidateScore(
            current_token=current_token, candidate=possible_candidate_token, score=predictions[index]
        )
        candidates_scores.append(candidate_score)
    return candidates_scores


def benchmark():
    train_for_benchmark()
    trainer, pdf_reading_order_list, truths, predictions = predict_for_benchmark()
    evaluate_contains_next_token(trainer, pdf_reading_order_list, predictions)
    benchmark_scores(predictions, truths)
    evaluate_poppler_contains_next_token()


def benchmark_scores(predictions, truths):
    predictions_for_benchmark = [1 if prediction > 0.5 else 0 for prediction in predictions]
    f1 = round(f1_score(truths, predictions_for_benchmark, average="macro") * 100, 2)
    accuracy = round(accuracy_score(truths, predictions_for_benchmark) * 100, 2)
    print(f"F1 score {f1}%")
    print(f"Accuracy score {accuracy}%")


if __name__ == "__main__":
    print("start")
    start = time()
    benchmark()
    print("finished in", int(time() - start), "seconds")
