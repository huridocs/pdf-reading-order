import numpy as np
import lightgbm as lgb
from pathlib import Path
from pdf_features.PdfFont import PdfFont
from pdf_features.PdfToken import PdfToken
from pdf_features.Rectangle import Rectangle
from pdf_token_type_labels.TokenType import TokenType
from pdf_tokens_type_trainer.ModelConfiguration import ModelConfiguration
from pdf_tokens_type_trainer.TokenFeatures import TokenFeatures
from pdf_reading_order.PdfReadingOrderTokens import PdfReadingOrderTokens


class ReadingOrderBase:
    def __init__(
        self, pdf_reading_order_tokens_list: list[PdfReadingOrderTokens], model_configuration: ModelConfiguration = None
    ):
        self.pdf_reading_order_tokens_list = pdf_reading_order_tokens_list
        self.model_configuration = model_configuration

    def loop_token_features(self):
        for pdf_reading_order_tokens in self.pdf_reading_order_tokens_list:
            token_features = TokenFeatures(pdf_reading_order_tokens.pdf_features)

            for page in pdf_reading_order_tokens.pdf_features.pages:
                if not page.tokens:
                    continue

                yield pdf_reading_order_tokens, token_features, page

    def loop_pages(self):
        for pdf_reading_order in self.pdf_reading_order_tokens_list:
            for page in pdf_reading_order.pdf_features.pages:
                yield pdf_reading_order, page

    @staticmethod
    def features_rows_to_x(features_rows):
        if not features_rows:
            return np.zeros((0, 0))

        x = np.zeros(((len(features_rows)), len(features_rows[0])))
        for i, v in enumerate(features_rows):
            x[i] = v
        return x

    @staticmethod
    def get_padding_token(segment_number: int, page_number: int):
        return PdfToken(
            page_number,
            "pad_token",
            "",
            PdfFont("pad_font_id", False, False, 0.0, "#000000"),
            segment_number,
            Rectangle(0, 0, 0, 0),
            TokenType.TEXT,
        )

    def train(self, model_path: str | Path, x_train_data: np.ndarray = None, labels: np.ndarray = None):
        x_train, y_train = self.get_training_data() if x_train_data is None else x_train_data, labels

        if not x_train.any():
            print("No data for training")
            return

        lgb_train = lgb.Dataset(x_train, y_train)
        print(f"Training: {model_path}")

        gbm = lgb.train(self.model_configuration.dict(), lgb_train)
        print(f"Saving")
        gbm.save_model(model_path, num_iteration=gbm.best_iteration)

    def get_training_data(self):
        pass
