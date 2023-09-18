from pathlib import Path
import numpy as np
from pdf_features.PdfPage import PdfPage
from pdf_features.PdfToken import PdfToken
from pdf_tokens_type_trainer.PdfTrainer import PdfTrainer


class ReadingOrderCandidatesTrainer(PdfTrainer):

    @staticmethod
    def get_candidate_token_features(token_1: PdfToken, token_2: PdfToken):
        return [token_1.bounding_box.top, token_1.bounding_box.left, token_1.bounding_box.right, token_1.bounding_box.bottom,
                token_2.bounding_box.top, token_2.bounding_box.left, token_2.bounding_box.right, token_2.bounding_box.bottom,
                token_1.bounding_box.bottom - token_2.bounding_box.top
                ]

    def loop_pages(self):
        for pdf_features in self.pdfs_features:
            for page in pdf_features.pages:
                yield page

    def loop_token_combinations_in_page(self, page: PdfPage):
        sorted_page_tokens = [self.get_padding_token(-1, page.page_number)] + page.tokens
        for i, current_token in enumerate(sorted_page_tokens):
            for token_2 in sorted_page_tokens[i + 1:]:
                yield current_token, token_2

    def loop_token_combinations(self):
        for page in self.loop_pages():
            for current_token, token_2 in self.loop_token_combinations_in_page(page):
                yield current_token, token_2

    def get_model_input(self):
        features_rows = []
        for current_token, token_2 in list(self.loop_token_combinations()):
            features_rows.append(self.get_candidate_token_features(current_token, token_2))

        return self.features_rows_to_x(features_rows)

    def predict(self, model_path: str | Path = None):
        prediction_scores = super().predict(model_path)
        predictions = []
        for prediction_score in prediction_scores:
            predictions.append(int(np.argmax(prediction_score)))
        return predictions
