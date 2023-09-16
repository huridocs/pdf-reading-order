from pathlib import Path
import numpy as np
from pdf_features.PdfToken import PdfToken
from pdf_tokens_type_trainer.PdfTrainer import PdfTrainer


class ReadingOrderCandidatesTrainer(PdfTrainer):

    @staticmethod
    def get_candidate_token_features(token_1: PdfToken, token_2: PdfToken):
        return [token_1.bounding_box.top, token_1.bounding_box.left, token_1.bounding_box.right, token_1.bounding_box.bottom,
                token_2.bounding_box.top, token_2.bounding_box.left, token_2.bounding_box.right, token_2.bounding_box.bottom,
                token_1.bounding_box.bottom - token_2.bounding_box.top
                ]

    def loop_labels(self):
        for pdf_features in self.pdfs_features:
            for page in pdf_features.pages:
                if not page.tokens:
                    continue
                page_tokens = [self.get_padding_token(-1, page.page_number)]
                page_tokens += page.tokens
                for i, current_token in enumerate(page_tokens):
                    yield page_tokens[i+1:], current_token

    def get_model_input(self):
        features_rows = []
        for candidate_tokens, current_token in self.loop_labels():
            features_rows.extend([self.get_candidate_token_features(current_token, token_2) for token_2 in candidate_tokens])

        return self.features_rows_to_x(features_rows)

    def predict(self, model_path: str | Path = None):
        prediction_scores = super().predict(model_path)
        predictions = []
        for prediction_score in prediction_scores:
            predictions.append(int(np.argmax(prediction_score)))
        return predictions
