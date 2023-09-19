from os.path import join

from pdf_features.PdfPage import PdfPage
from pdf_token_type_labels.TokenTypeLabels import TokenTypeLabels
from pdf_features.PdfFeatures import PdfFeatures
from pdf_reading_order.ReadingOrderLabelPage import ReadingOrderLabelPage
from pdf_reading_order.config import READING_ORDER_RELATIVE_PATH
from pdf_tokens_type_trainer.config import LABELS_FILE_NAME


class PdfReadingOrderTokens:
    def __init__(self, pdf_features: PdfFeatures, labeled_page_by_raw_page: dict[PdfPage, ReadingOrderLabelPage]):
        self.pdf_features: PdfFeatures = pdf_features
        self.labeled_page_by_raw_page: dict[PdfPage, ReadingOrderLabelPage] = labeled_page_by_raw_page

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
        labeled_page_by_raw_page: dict[PdfPage, ReadingOrderLabelPage] = {}
        last_page = None
        for page, token in pdf_features.loop_tokens():
            if page != last_page:
                labeled_page_by_raw_page[page] = ReadingOrderLabelPage()
                last_page = page
            for label, label_page_number in PdfReadingOrderTokens.loop_labels(reading_order_labels):
                if page.page_number != label_page_number:
                    continue
                if token.inside_label(label):
                    labeled_page_by_raw_page[page].reading_order_by_token_id[token.id] = label.token_type
                    break

        return PdfReadingOrderTokens(pdf_features, labeled_page_by_raw_page)
