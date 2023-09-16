from os.path import join
from pdf_token_type_labels.TokenTypeLabels import TokenTypeLabels
from pdf_features.PdfToken import PdfToken
from pdf_features.PdfFeatures import PdfFeatures
from pdf_reading_order.config import READING_ORDER_RELATIVE_PATH
from pdf_tokens_type_trainer.config import LABELS_FILE_NAME


class PdfReadingOrderTokens:
    def __init__(self, pdf_features: PdfFeatures, reading_order_by_token: dict[PdfToken, int]):
        self.pdf_features: PdfFeatures = pdf_features
        self.reading_order_by_token: dict[PdfToken, int] = reading_order_by_token

    @staticmethod
    def loop_labels(reading_order_labels):
        for page in reading_order_labels.pages:
            for label in sorted(page.labels, key=lambda _label: _label.area()):
                yield label, page.number

    @staticmethod
    def from_labeled_data(pdf_labeled_data_root_path, dataset, pdf_name):
        pdf_features = PdfFeatures.from_labeled_data(pdf_labeled_data_root_path, dataset, pdf_name)
        reading_order_labeled_data_path = join(pdf_labeled_data_root_path, READING_ORDER_RELATIVE_PATH)
        reading_order_labels_path = join(reading_order_labeled_data_path, dataset, pdf_name, LABELS_FILE_NAME)
        reading_order_labels = PdfFeatures.load_token_type_labels(reading_order_labels_path)
        return PdfReadingOrderTokens.set_reading_orders(pdf_features, reading_order_labels)

    @staticmethod
    def set_reading_orders(pdf_features: PdfFeatures, reading_order_labels: TokenTypeLabels):
        reading_order_by_token: dict[PdfToken, int] = {}
        for page, token in pdf_features.loop_tokens():
            for label, label_page_number in PdfReadingOrderTokens.loop_labels(reading_order_labels):
                if page.page_number != label_page_number:
                    continue
                if token.inside_label(label):
                    reading_order_by_token[token] = label.token_type
                    break

        return PdfReadingOrderTokens(pdf_features, reading_order_by_token)
