import numpy as np
import lightgbm as lgb
from pathlib import Path
from pdf_features.PdfToken import PdfToken
from pdf_reading_order.ReadingOrderLabelPage import ReadingOrderLabelPage
from pdf_reading_order.ReadingOrderBase import ReadingOrderBase


class ReadingOrderCandidatesTrainer(ReadingOrderBase):
    @staticmethod
    def get_features(token_1: PdfToken, token_2: PdfToken):
        return [
            token_1.bounding_box.top,
            token_1.bounding_box.left,
            token_1.bounding_box.right,
            token_1.bounding_box.bottom,
            token_2.bounding_box.top,
            token_2.bounding_box.left,
            token_2.bounding_box.right,
            token_2.bounding_box.bottom,
            token_1.bounding_box.bottom - token_2.bounding_box.top,
        ]

    @staticmethod
    def get_next_token(reading_order_no: int, label_page: ReadingOrderLabelPage, remaining_tokens: list[PdfToken]):
        for remaining_token in remaining_tokens:
            if label_page.reading_order_by_token_id[remaining_token.id] == reading_order_no:
                return remaining_token

    def loop_token_combinations(self):
        for pdf_reading_order, page in self.loop_pages():
            label_page = pdf_reading_order.labeled_page_by_raw_page[page]
            current_token = self.get_padding_token(-1, page.page_number)
            reading_order_no = 1
            remaining_tokens = page.tokens.copy()
            for i in range(len(page.tokens)):
                yield current_token, remaining_tokens, label_page, reading_order_no
                current_token = self.get_next_token(reading_order_no, label_page, remaining_tokens)
                reading_order_no += 1
                remaining_tokens.remove(current_token)

    def get_training_data(self):
        features_rows = []
        labels = []
        for current_token, remaining_tokens, label_page, _ in self.loop_token_combinations():
            for remaining_token in remaining_tokens:
                features_rows.append(self.get_features(current_token, remaining_token))
                labels.append(int(label_page.is_next_token(current_token, remaining_token)))
        return self.features_rows_to_x(features_rows), labels

    def predict(self, model_path: str | Path = None, candidate_count: int = 18):
        mistake_count = 0
        model = lgb.Booster(model_file=model_path)
        for current_token, remaining_tokens, label_page, reading_order_no in self.loop_token_combinations():
            candidate_tokens = self.get_candidate_tokens(candidate_count, current_token, model, remaining_tokens)
            if self.get_next_token(reading_order_no, label_page, remaining_tokens) not in candidate_tokens:
                mistake_count += 1
        return mistake_count

    def get_candidate_tokens(self, candidate_count, current_token, model, remaining_tokens):
        features = [self.get_features(current_token, remaining_token) for remaining_token in remaining_tokens]
        prediction_scores = model.predict(self.features_rows_to_x(features))
        candidate_token_indexes = np.argsort([prediction_scores[:, 1]], axis=1)[:, -candidate_count:]
        candidate_tokens = [remaining_tokens[i] for i in candidate_token_indexes[0]]
        return candidate_tokens
