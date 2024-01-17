from os.path import join
from pdf_features.PdfPage import PdfPage
from pdf_features.Rectangle import Rectangle
from pdf_token_type_labels.PdfLabels import PdfLabels
from pdf_features.PdfFeatures import PdfFeatures
from pdf_reading_order.ReadingOrderLabelPage import ReadingOrderLabelPage
from pdf_reading_order.config import READING_ORDER_RELATIVE_PATH
from pdf_tokens_type_trainer.config import LABELS_FILE_NAME


class PdfReadingOrderTokens:
    def __init__(self, pdf_features: PdfFeatures, labeled_page_by_raw_page: dict[PdfPage, ReadingOrderLabelPage]):
        self.pdf_features: PdfFeatures = pdf_features
        self.labeled_page_by_raw_page: dict[PdfPage, ReadingOrderLabelPage] = labeled_page_by_raw_page

    @staticmethod
    def loop_labels(reading_order_labels: PdfLabels):
        for page in reading_order_labels.pages:
            for label in sorted(page.labels, key=lambda _label: _label.area()):
                yield label, page.number

    @staticmethod
    def loop_tokens_sorted_by_area(pdf_features: PdfFeatures):
        for page in pdf_features.pages:
            for token in sorted(page.tokens, key=lambda t: Rectangle.area(t.bounding_box)):
                yield page, token

    @staticmethod
    def from_labeled_data(pdf_labeled_data_root_path, dataset, pdf_name):
        pdf_features = PdfFeatures.from_labeled_data(pdf_labeled_data_root_path, dataset, pdf_name)
        reading_order_labeled_data_path = join(pdf_labeled_data_root_path, READING_ORDER_RELATIVE_PATH)
        reading_order_labels_path = join(reading_order_labeled_data_path, dataset, pdf_name, LABELS_FILE_NAME)
        reading_order_labels = PdfFeatures.load_labels(reading_order_labels_path)
        return PdfReadingOrderTokens.set_reading_orders(pdf_features, reading_order_labels)

    @staticmethod
    def set_reading_orders(pdf_features: PdfFeatures, reading_order_labels: PdfLabels):
        labeled_page_by_raw_page: dict[PdfPage, ReadingOrderLabelPage] = {}
        last_page = None
        used_labels = []
        for page, token in PdfReadingOrderTokens.loop_tokens_sorted_by_area(pdf_features):
            if page != last_page:
                labeled_page_by_raw_page[page] = ReadingOrderLabelPage()
                last_page = page
                used_labels = []
            for label, label_page_number in PdfReadingOrderTokens.loop_labels(reading_order_labels):
                if page.page_number != label_page_number:
                    continue
                if label in used_labels:
                    continue
                if label.intersection_percentage(token.bounding_box) > 99.9:
                    used_labels.append(label)
                    labeled_page_by_raw_page[page].reading_order_by_token_id[token.id] = label.label_type
                    break

        return PdfReadingOrderTokens(pdf_features, labeled_page_by_raw_page)
