from pdf_features.PdfPage import PdfPage
from pdf_features.PdfToken import PdfToken
from pdf_reading_order.PdfReadingOrderTokens import PdfReadingOrderTokens


class PredictionInfo:
    def __init__(self, pdf_reading_order_tokens: PdfReadingOrderTokens):
        self.pdf_reading_order_tokens = pdf_reading_order_tokens
        self.file_name = pdf_reading_order_tokens.pdf_features.file_name
        self.file_type = pdf_reading_order_tokens.pdf_features.file_type
        self.label_count = 0
        self.mistake_count = 0
        self.actual_reading_orders_by_page: dict[PdfPage, list[PdfToken]] = {}
        self.predicted_reading_orders_by_page: dict[PdfPage, list[PdfToken]] = {}
        self.get_actual_and_predicted_orders()

    def get_actual_and_predicted_orders(self):
        for page in self.pdf_reading_order_tokens.pdf_features.pages:
            page_reading_orders = self.pdf_reading_order_tokens.labeled_page_by_raw_page[page].reading_order_by_token_id
            actual_order = sorted([token for token in page.tokens], key=lambda t: page_reading_orders[t.id])
            predicted_order = sorted([token for token in page.tokens], key=lambda t: t.prediction)
            self.actual_reading_orders_by_page[page] = actual_order
            self.predicted_reading_orders_by_page[page] = predicted_order
