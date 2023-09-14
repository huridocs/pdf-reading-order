from pathlib import Path

import numpy as np
from pdf_features.PdfToken import PdfToken
from pdf_tokens_type_trainer.TokenFeatures import TokenFeatures
from pdf_tokens_type_trainer.TokenTypeTrainer import TokenTypeTrainer


class ReadingOrderCandidatesTrainer(TokenTypeTrainer):

    def get_model_input(self):
        features_rows = []
        y = np.array([])

        contex_size = self.model_configuration.context_size
        for token_features, page in self.loop_pages():
            page_tokens = [
                self.get_padding_token(segment_number=i - 999999, page_number=page.page_number) for i in range(contex_size)
            ]
            page_tokens += page.tokens
            page_tokens += [
                self.get_padding_token(segment_number=999999 + i, page_number=page.page_number) for i in range(contex_size)
            ]

            tokens_indexes = range(contex_size, len(page_tokens) - contex_size)
            page_features = [self.get_context_features(token_features, page_tokens, i) for i in tokens_indexes]
            features_rows.extend(page_features)

            y = np.append(y, [int(page_tokens[i].segment_no == page_tokens[i + 1].segment_no) for i in tokens_indexes])

        return self.features_rows_to_x(features_rows), y

    @staticmethod
    def get_labels(page_tokens: list[PdfToken], tokens_indexes: range):
        return [page_tokens[i].token_type.get_index() for i in tokens_indexes]

    def get_context_features(self, token_features: TokenFeatures, page_tokens: list[PdfToken], token_index: int):
        token_row_features = list()
        first_token_from_context = token_index - self.model_configuration.context_size
        for i in range(self.model_configuration.context_size * 2):
            first_token = page_tokens[first_token_from_context + i]
            second_token = page_tokens[first_token_from_context + i + 1]
            features = token_features.get_features(first_token, second_token, page_tokens)
            features += self.get_paragraph_extraction_features(first_token, second_token)
            token_row_features.extend(features)

        return token_row_features

    def predict(self, model_path: str | Path = None):
        token_type_trainer = TokenTypeTrainer(self.pdfs_features)
        token_type_trainer.set_token_types()
        super().predict(model_path)