from pathlib import Path
import numpy as np
from pdf_features.PdfFeatures import PdfFeatures
from pdf_features.PdfToken import PdfToken
from pdf_tokens_type_trainer.PdfTrainer import PdfTrainer
from pdf_reading_order.PdfReadingOrderTokens import PdfReadingOrderTokens
from model_configuration import CANDIDATE_MODEL_CONFIGURATION


class ReadingOrderCandidatesTrainer(PdfTrainer):
    def __init__(self, pdfs_features: list[PdfFeatures], pdf_reading_order_tokens_list: list[PdfReadingOrderTokens]):
        super().__init__(pdfs_features, CANDIDATE_MODEL_CONFIGURATION)
        print(f"{CANDIDATE_MODEL_CONFIGURATION.lambda_l1=}")
        self.pdf_reading_order_tokens_list: list[PdfReadingOrderTokens] = pdf_reading_order_tokens_list

    @staticmethod
    def get_candidate_token_features(token_1: PdfToken, token_2: PdfToken):
        return [token_1.bounding_box.top, token_1.bounding_box.left, token_1.bounding_box.right, token_1.bounding_box.bottom,
                token_2.bounding_box.top, token_2.bounding_box.left, token_2.bounding_box.right, token_2.bounding_box.bottom,
                token_1.bounding_box.bottom - token_2.bounding_box.top
                ]

    def loop_labels(self):
        for pdf_reading_order_tokens in self.pdf_reading_order_tokens_list:
            for page in pdf_reading_order_tokens.pdf_features.pages:
                if not page.tokens:
                    continue
                page_tokens = [self.get_padding_token(-1, page.page_number)]
                page_tokens += sorted(page.tokens, key=lambda token: pdf_reading_order_tokens.reading_order_by_token[token])
                for i, current_token in enumerate(page_tokens):
                    yield i, page_tokens, current_token

    def get_model_input(self):
        features_rows = []
        for i, page_tokens, current_token in self.loop_labels():
            features_rows.extend([self.get_candidate_token_features(current_token, token_2) for token_2 in page_tokens[i+1:]])

        return self.features_rows_to_x(features_rows)

    def predict(self, model_path: str | Path = None):
        prediction_scores = super().predict(model_path)
        predictions = []
        for prediction_score in prediction_scores:
            predictions.append(int(np.argmax(prediction_score)))
        return predictions
