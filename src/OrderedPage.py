from pdf_features.PdfToken import PdfToken
from ReadingOrderToken import ReadingOrderToken


class OrderedPage:
    def __init__(self, pdf_name: str, page_number: int, reading_order_tokens: list[ReadingOrderToken]):
        self.pdf_name = pdf_name
        self.page_number = page_number
        self.reading_order_tokens = reading_order_tokens

    @staticmethod
    def from_pdf_tokens(pdf_name: str, page_number: int, pdf_tokens: list[PdfToken]):
        reading_order_tokens = [
            ReadingOrderToken(token.bounding_box, token.content, token.token_type, reading_order_no)
            for reading_order_no, token in enumerate(pdf_tokens)
        ]

        return OrderedPage(pdf_name, page_number, reading_order_tokens)

    def to_dict(self):
        return {
            "pdf_name": self.pdf_name,
            "page_number": self.page_number,
            "tokens": [reading_order_token.to_dict() for reading_order_token in self.reading_order_tokens],
        }
