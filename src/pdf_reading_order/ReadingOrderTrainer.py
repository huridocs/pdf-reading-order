import lightgbm as lgb
import numpy as np
from itertools import permutations
from pathlib import Path
from os.path import join
from pdf_features.PdfPage import PdfPage
from pdf_features.PdfToken import PdfToken
from pdf_token_type_labels.TokenType import TokenType
from pdf_tokens_type_trainer.TokenFeatures import TokenFeatures
from pdf_tokens_type_trainer.TokenTypeTrainer import TokenTypeTrainer
from pdf_reading_order.ReadingOrderCandidatesTrainer import ReadingOrderCandidatesTrainer
from pdf_reading_order.config import ROOT_PATH

CANDIDATE_COUNT = 18
CANDIDATE_TOKEN_MODEL = lgb.Booster(model_file=Path(join(ROOT_PATH, "model", "candidate_selector_benchmark.model")))


class ReadingOrderTrainer(TokenTypeTrainer):

    def get_candidate_tokens_for_current_token(self, current_token: PdfToken, possible_candidate_tokens: list[PdfToken]):
        features = [ReadingOrderCandidatesTrainer.get_candidate_token_features(current_token, candidate_token)
                    for candidate_token in possible_candidate_tokens]
        prediction_scores = CANDIDATE_TOKEN_MODEL.predict(self.features_rows_to_x(features))
        candidate_token_indexes = np.argsort(prediction_scores, axis=1)[:, -CANDIDATE_COUNT:]
        candidate_tokens = [possible_candidate_tokens[i] for i in candidate_token_indexes[0]]
        return candidate_tokens

    @staticmethod
    def get_token_type_features(token: PdfToken) -> list[int]:
        return [1 if token_type == token.token_type else 0 for token_type in TokenType]

    def get_reading_order_features(self, current_token: PdfToken, candidate_token_1: PdfToken,
                                   candidate_token_2: PdfToken, token_features: TokenFeatures, page_tokens: list[PdfToken]):
        features = [current_token.bounding_box.top, current_token.bounding_box.left,
                    current_token.bounding_box.width, current_token.bounding_box.height
                    ]
        features += self.get_token_type_features(current_token)
        features += token_features.get_features(candidate_token_1, candidate_token_2, page_tokens)
        features += self.get_token_type_features(candidate_token_1)
        features += token_features.get_features(candidate_token_2, candidate_token_1, page_tokens)
        features += self.get_token_type_features(candidate_token_2)
        return features

    def loop_candidate_tokens_in_page(self, page: PdfPage):
        page_tokens = [self.get_padding_token(-1, page.page_number)] + page.tokens
        for current_token in page_tokens:
            possible_candidate_tokens = [token for token in page_tokens if token != current_token and token.id != "pad_token"]
            if len(possible_candidate_tokens) < 2:
                possible_candidate_tokens.append(possible_candidate_tokens[0])
            candidate_tokens = self.get_candidate_tokens_for_current_token(current_token, possible_candidate_tokens)
            for candidate_token_1, candidate_token_2 in permutations(candidate_tokens, 2):
                yield current_token, candidate_token_1, candidate_token_2

    def loop_candidate_tokens(self):
        for token_features, page in self.loop_token_features():
            for current_token, candidate_token_1, candidate_token_2 in self.loop_candidate_tokens_in_page(page):
                yield current_token, candidate_token_1, candidate_token_2, token_features, page

    def get_model_input(self):
        features_rows = []
        for current_token, candidate_token_1, candidate_token_2, token_features, page in list(self.loop_candidate_tokens()):
            features_rows.append(self.get_reading_order_features(current_token, candidate_token_1, candidate_token_2, token_features, page.tokens))

        return self.features_rows_to_x(features_rows)

    def find_next_token_from_candidates(self, lightgbm_model: lgb.Booster, current_token: PdfToken, candidate_tokens: list[PdfToken], token_features: TokenFeatures, page_tokens: list[PdfToken]):
        next_token = candidate_tokens[0]
        for candidate_token in candidate_tokens[1:]:
            X = self.features_rows_to_x([self.get_reading_order_features(current_token, next_token, candidate_token, token_features, page_tokens)])
            if int(np.argmax(lightgbm_model.predict(X))) == 1:
                next_token = candidate_token
        return next_token

    def get_reading_orders_for_page(self, page: PdfPage, lightgbm_model: lgb.Booster, token_features: TokenFeatures):
        current_token = self.get_padding_token(-1, page.page_number)
        remaining_tokens = [current_token] + page.tokens
        reading_order_by_token_id = {}
        current_reading_order_no = 1
        for i in range(len(page.tokens)):
            possible_candidate_tokens = [token for token in remaining_tokens if token != current_token]
            if len(possible_candidate_tokens) < 2:
                possible_candidate_tokens.append(possible_candidate_tokens[0])
            candidate_tokens = self.get_candidate_tokens_for_current_token(current_token, possible_candidate_tokens)
            next_token = self.find_next_token_from_candidates(lightgbm_model, current_token, candidate_tokens, token_features, page.tokens)
            remaining_tokens.remove(next_token)
            reading_order_by_token_id[next_token.id] = current_reading_order_no
            current_reading_order_no += 1
        return reading_order_by_token_id

    @staticmethod
    def reorder_page_tokens(page: PdfPage, reading_order_by_token_id: dict[str, int]):
        for token in page.tokens:
            token.prediction = reading_order_by_token_id[token.id]
        page.tokens.sort(key=lambda _token: _token.prediction)

    def get_reading_ordered_pages(self, model_path: str | Path = None):
        token_type_trainer = TokenTypeTrainer(self.pdfs_features)
        token_type_trainer.set_token_types()
        lightgbm_model = lgb.Booster(model_file=model_path)
        for token_features, page in self.loop_token_features():
            reading_order_by_token_id = self.get_reading_orders_for_page(page, lightgbm_model, token_features)
            self.reorder_page_tokens(page, reading_order_by_token_id)

    def predict(self, model_path: str | Path = None):
        token_type_trainer = TokenTypeTrainer(self.pdfs_features)
        token_type_trainer.set_token_types()
        x = self.get_model_input()
        if not x.any():
            return self.pdfs_features
        lightgbm_model = lgb.Booster(model_file=model_path)
        prediction_scores = lightgbm_model.predict(x)
        predictions = []
        for prediction_score in prediction_scores:
            predictions.append(int(np.argmax(prediction_score)))
        return predictions
