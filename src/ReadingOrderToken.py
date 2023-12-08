from pdf_features.Rectangle import Rectangle
from pdf_token_type_labels.TokenType import TokenType


class ReadingOrderToken:
    def __init__(self, bounding_box: Rectangle, content: str, token_type: TokenType, reading_order_no: int):
        self.bounding_box = bounding_box
        self.content = content
        self.token_type = token_type
        self.reading_order_no = reading_order_no

    def to_dict(self):
        return {
            "bounding_box": self.bounding_box.to_dict(),
            "content": self.content,
            "token_type": self.token_type.value,
            "reading_order_no": self.reading_order_no
        }
