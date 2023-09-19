from pdf_features.PdfToken import PdfToken


class ReadingOrderLabelPage:
    def __init__(self):
        self.reading_order_by_token_id: dict[str, int] = {"pad_token": 0}

    def is_next_token(self, current_token: PdfToken, candidate_token: PdfToken):
        return self.reading_order_by_token_id[candidate_token.id] == self.reading_order_by_token_id[current_token.id] + 1

    def is_coming_earlier(self, token_1: PdfToken, token_2: PdfToken):
        return self.reading_order_by_token_id[token_2.id] < self.reading_order_by_token_id[token_1.id]
