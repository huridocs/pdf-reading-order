import lightgbm as lgb
import numpy as np
from pathlib import Path
from os.path import join
from pdf_features.PdfPage import PdfPage
from pdf_features.PdfToken import PdfToken
from pdf_token_type_labels.TokenType import TokenType
from pdf_tokens_type_trainer.TokenFeatures import TokenFeatures
from random import shuffle
from pdf_reading_order.ReadingOrderBase import ReadingOrderBase
from pdf_reading_order.ReadingOrderCandidatesTrainer import ReadingOrderCandidatesTrainer
from pdf_reading_order.ReadingOrderLabelPage import ReadingOrderLabelPage
from pdf_reading_order.config import ROOT_PATH

CANDIDATE_COUNT = 18
CANDIDATE_TOKEN_MODEL = lgb.Booster(model_file=Path(join(ROOT_PATH, "model", "candidate_selector_benchmark.model")))


class ReadingOrderTrainer(ReadingOrderBase):

    def get_candidate_tokens_for_current_token(self, current_token: PdfToken,
                                               possible_candidate_tokens: list[PdfToken]):
        features = [
            ReadingOrderCandidatesTrainer.get_candidate_token_features(current_token, possible_candidate_token)
            for possible_candidate_token in possible_candidate_tokens
        ]
        prediction_scores = CANDIDATE_TOKEN_MODEL.predict(self.features_rows_to_x(features))
        candidate_token_indexes = np.argsort([prediction_scores[:, 1]], axis=1)[:, -CANDIDATE_COUNT:]
        candidate_tokens = [possible_candidate_tokens[i] for i in candidate_token_indexes[0]]
        return candidate_tokens

    @staticmethod
    def get_token_type_features(token: PdfToken) -> list[int]:
        return [1 if token_type == token.token_type else 0 for token_type in TokenType]

    def get_reading_order_features(self, current_token: PdfToken, candidate_token_1: PdfToken,
                                   candidate_token_2: PdfToken, token_features: TokenFeatures,
                                   page_tokens: list[PdfToken], ):
        features = [
            current_token.bounding_box.top,
            current_token.bounding_box.left,
            current_token.bounding_box.width,
            current_token.bounding_box.height,
        ]
        features += self.get_token_type_features(current_token)
        features += token_features.get_features(candidate_token_1, candidate_token_2, page_tokens)
        features += self.get_token_type_features(candidate_token_1)
        features += token_features.get_features(candidate_token_2, candidate_token_1, page_tokens)
        features += self.get_token_type_features(candidate_token_2)
        return features

    @staticmethod
    def get_next_token_label(reading_order_no: int, label_page: ReadingOrderLabelPage,
                             possible_candidate_tokens: list[PdfToken]):
        for possible_candidate_token in possible_candidate_tokens:
            if label_page.reading_order_by_token_id[possible_candidate_token.id] == reading_order_no:
                return possible_candidate_token

    @staticmethod
    def add_next_token_in_poppler_order(next_token_in_poppler_order: PdfToken, candidate_tokens: list[PdfToken],
                                        possible_candidate_tokens: list[PdfToken]):
        if next_token_in_poppler_order.id in [candidate_token.id for candidate_token in candidate_tokens]:
            return
        if next_token_in_poppler_order.id not in [possible_candidate_token.id for possible_candidate_token in
                                                  possible_candidate_tokens]:
            return
        candidate_tokens.append(next_token_in_poppler_order)

    def loop_candidates_for_each_token(self):
        for pdf_reading_order_tokens, token_features, page in self.loop_token_features():
            label_page = pdf_reading_order_tokens.labeled_page_by_raw_page[page]
            current_token = self.get_padding_token(-1, page.page_number)
            reading_order_no = 1
            possible_candidate_tokens = page.tokens.copy()
            for i in range(len(page.tokens)):
                candidate_tokens = self.get_candidate_tokens_for_current_token(current_token, possible_candidate_tokens)
                next_token_in_poppler_order = page.tokens[i]
                self.add_next_token_in_poppler_order(next_token_in_poppler_order, candidate_tokens,
                                                     possible_candidate_tokens)
                yield current_token, candidate_tokens, token_features, label_page, page
                current_token = self.get_next_token_label(reading_order_no, label_page, possible_candidate_tokens)
                reading_order_no += 1
                possible_candidate_tokens.remove(current_token)

    def get_training_data(self):
        features_rows = []
        labels = []
        for current_token, candidate_tokens, token_features, label_page, page in self.loop_candidates_for_each_token():
            shuffle(candidate_tokens)
            possible_next_token = candidate_tokens[0]
            for candidate_token in candidate_tokens[1:]:
                features_rows.append(
                    self.get_reading_order_features(current_token, possible_next_token, candidate_token, token_features,
                                                    page.tokens))
                labels.append(label_page.is_coming_earlier(possible_next_token, candidate_token))
                if label_page.is_coming_earlier(possible_next_token, candidate_token):
                    possible_next_token = candidate_token

        return self.features_rows_to_x(features_rows), labels

    def find_next_token_from_candidates(self, lightgbm_model: lgb.Booster, token_features: TokenFeatures,
                                        page_tokens: list[PdfToken], candidate_tokens: list[PdfToken],
                                        current_token: PdfToken):
        next_token = candidate_tokens[0]
        for candidate_token in candidate_tokens[1:]:
            X = self.features_rows_to_x([self.get_reading_order_features(current_token, next_token, candidate_token,
                                                                         token_features, page_tokens)])
            if int(np.argmax(lightgbm_model.predict(X))) == 1:
                next_token = candidate_token
        return next_token

    def get_reading_orders_for_page(self, lightgbm_model: lgb.Booster, token_features: TokenFeatures, page: PdfPage):
        current_token = self.get_padding_token(-1, page.page_number)
        possible_candidate_tokens = page.tokens.copy()
        reading_order_by_token_id = {}
        current_reading_order_no = 1
        for i in range(len(page.tokens)):
            candidate_tokens = self.get_candidate_tokens_for_current_token(current_token, possible_candidate_tokens)
            current_token = self.find_next_token_from_candidates(lightgbm_model, token_features, page.tokens,
                                                                 candidate_tokens, current_token)
            possible_candidate_tokens.remove(current_token)
            reading_order_by_token_id[current_token.id] = current_reading_order_no
            current_reading_order_no += 1

        return reading_order_by_token_id

    @staticmethod
    def reorder_page_tokens(page: PdfPage, reading_order_by_token_id: dict[str, int]):
        for token in page.tokens:
            token.prediction = reading_order_by_token_id[token.id]
        page.tokens.sort(key=lambda _token: _token.prediction)

    def get_reading_ordered_pages(self, model_path: str | Path = None):
        lightgbm_model = lgb.Booster(model_file=model_path)
        for pdf_reading_order_tokens, token_features, page in self.loop_token_features():
            reading_order_by_token_id = self.get_reading_orders_for_page(lightgbm_model, token_features, page)
            self.reorder_page_tokens(page, reading_order_by_token_id)

    def predict(self, model_path: str | Path = None):
        self.get_reading_ordered_pages(model_path)
        mistakes = 0
        for pdf_reading_order_tokens, _, page in self.loop_token_features():
            label_page = pdf_reading_order_tokens.labeled_page_by_raw_page[page]
            for token_1, token_2 in zip(page.tokens, page.tokens[1:]):
                if not label_page.is_next_token(token_1, token_2):
                    mistakes += 1
        return mistakes
