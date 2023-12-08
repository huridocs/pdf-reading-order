import os

import lightgbm as lgb
import numpy as np
from random import shuffle
from pathlib import Path
from os.path import join
from pdf_features.PdfPage import PdfPage
from pdf_features.PdfToken import PdfToken
from pdf_token_type_labels.TokenType import TokenType
from pdf_tokens_type_trainer.TokenFeatures import TokenFeatures
from pdf_reading_order.config import ROOT_PATH
from pdf_reading_order.ReadingOrderBase import ReadingOrderBase
from pdf_reading_order.ReadingOrderLabelPage import ReadingOrderLabelPage
from pdf_reading_order.ReadingOrderCandidatesTrainer import ReadingOrderCandidatesTrainer as CandidatesTrainer
from pdf_reading_order.download_models import reading_order_model
from pdf_reading_order.download_models import candidate_selector_model

CANDIDATE_COUNT = 18
candidate_selector_model_path = join(ROOT_PATH, "model", "candidate_selector_benchmark.model")
if not os.path.exists(candidate_selector_model_path):
    candidate_selector_model_path = candidate_selector_model
CANDIDATE_TOKEN_MODEL = lgb.Booster(model_file=candidate_selector_model_path)


class ReadingOrderTrainer(ReadingOrderBase):
    def get_candidate_tokens(self, current_token, remaining_tokens: list[PdfToken]):
        features = [CandidatesTrainer.get_features(current_token, remaining) for remaining in remaining_tokens]
        prediction_scores = CANDIDATE_TOKEN_MODEL.predict(self.features_rows_to_x(features))
        candidate_token_indexes = np.argsort([prediction_scores[:, 1]], axis=1)[:, -CANDIDATE_COUNT:]
        candidate_tokens = [remaining_tokens[i] for i in candidate_token_indexes[0]]
        return candidate_tokens

    @staticmethod
    def get_token_type_features(token: PdfToken) -> list[int]:
        return [1 if token_type == token.token_type else 0 for token_type in TokenType]

    def get_features(self, current_token, first_candidate, second_candidate, token_features, page_tokens):
        bounding_box = current_token.bounding_box
        features = [bounding_box.top, bounding_box.left, bounding_box.width, bounding_box.height]
        features += self.get_token_type_features(current_token)
        features += token_features.get_features(first_candidate, second_candidate, page_tokens)
        features += self.get_token_type_features(first_candidate)
        features += token_features.get_features(second_candidate, first_candidate, page_tokens)
        features += self.get_token_type_features(second_candidate)
        return features

    @staticmethod
    def get_next_token_label(reading_order_no, label_page: ReadingOrderLabelPage, remaining_tokens: list[PdfToken]):
        for remaining_token in remaining_tokens:
            if label_page.reading_order_by_token_id[remaining_token.id] == reading_order_no:
                return remaining_token

    def loop_candidates_for_each_token(self):
        for pdf_reading_order_tokens, token_features, page in self.loop_token_features():
            label_page = pdf_reading_order_tokens.labeled_page_by_raw_page[page]
            current_token = self.get_padding_token(-1, page.page_number)
            reading_order_no = 1
            remaining_tokens = page.tokens.copy()
            for i in range(len(page.tokens)):
                candidate_tokens = self.get_candidate_tokens(current_token, remaining_tokens)
                yield current_token, candidate_tokens, token_features, label_page, page
                current_token = self.get_next_token_label(reading_order_no, label_page, remaining_tokens)
                reading_order_no += 1
                remaining_tokens.remove(current_token)

    def get_training_data(self):
        features_rows = []
        labels = []
        for current_token, candidate_tokens, token_features, label_page, page in self.loop_candidates_for_each_token():
            shuffle(candidate_tokens)
            next_token = candidate_tokens[0]
            for candidate_token in candidate_tokens[1:]:
                feature_row = self.get_features(current_token, next_token, candidate_token, token_features, page.tokens)
                features_rows.append(feature_row)
                labels.append(label_page.is_coming_earlier(next_token, candidate_token))
                if label_page.is_coming_earlier(next_token, candidate_token):
                    next_token = candidate_token

        return self.features_rows_to_x(features_rows), labels

    def find_next_token(self, lightgbm_model, token_features, page_tokens, candidate_tokens, current_token):
        next_token = candidate_tokens[0]
        for candidate_token in candidate_tokens[1:]:
            features_rows = [self.get_features(current_token, next_token, candidate_token, token_features, page_tokens)]
            X = self.features_rows_to_x(features_rows)
            if int(np.argmax(lightgbm_model.predict(X))) == 1:
                next_token = candidate_token
        return next_token

    def get_reading_orders_for_page(self, lightgbm_model: lgb.Booster, token_features: TokenFeatures, page: PdfPage):
        current_token = self.get_padding_token(-1, page.page_number)
        remaining_tokens = page.tokens.copy()
        reading_order_by_token_id = {}
        current_reading_order_no = 1
        for i in range(len(page.tokens)):
            candidates = self.get_candidate_tokens(current_token, remaining_tokens)
            current_token = self.find_next_token(lightgbm_model, token_features, page.tokens, candidates, current_token)
            remaining_tokens.remove(current_token)
            reading_order_by_token_id[current_token.id] = current_reading_order_no
            current_reading_order_no += 1

        return reading_order_by_token_id

    @staticmethod
    def reorder_page_tokens(page: PdfPage, reading_order_by_token_id: dict[str, int]):
        for token in page.tokens:
            token.prediction = reading_order_by_token_id[token.id]
        page.tokens.sort(key=lambda _token: _token.prediction)

    def predict(self, model_path: str | Path = None):
        model_path = model_path if model_path else reading_order_model
        lightgbm_model = lgb.Booster(model_file=model_path)
        for pdf_reading_order_tokens, token_features, page in self.loop_token_features():
            reading_order_by_token_id = self.get_reading_orders_for_page(lightgbm_model, token_features, page)
            self.reorder_page_tokens(page, reading_order_by_token_id)
