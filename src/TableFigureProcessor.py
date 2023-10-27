from pathlib import Path
from statistics import mode
from paragraph_extraction_trainer.Paragraph import Paragraph
from pdf_features.PdfPage import PdfPage
from pdf_features.Rectangle import Rectangle
from pdf_token_type_labels.TokenType import TokenType
from pdf_features.PdfToken import PdfToken
from pdf_reading_order.PdfReadingOrderTokens import PdfReadingOrderTokens
from paragraph_extraction_trainer.ParagraphExtractorTrainer import ParagraphExtractorTrainer
from paragraph_extraction_trainer.download_models import paragraph_extraction_model_path
from paragraph_extraction_trainer.model_configuration import MODEL_CONFIGURATION as PARAGRAPH_EXTRACTOR_CONFIGURATION

from pdf_reading_order.ReadingOrderLabelPage import ReadingOrderLabelPage


class TableFigureProcessor:
    def __init__(self, pdf_reading_order_tokens: PdfReadingOrderTokens, model_path: str | Path = None):
        self.pdf_reading_order_tokens = pdf_reading_order_tokens
        self.paragraph_extractor = ParagraphExtractorTrainer([self.pdf_reading_order_tokens.pdf_features],
                                                             PARAGRAPH_EXTRACTOR_CONFIGURATION)
        self.model_path = paragraph_extraction_model_path if model_path is None else model_path

    @staticmethod
    def get_processed_token_from_paragraph(paragraph_tokens: list[PdfToken], label_page: ReadingOrderLabelPage):
        page_number = paragraph_tokens[0].page_number
        tokens_in_poppler_order = sorted([token for token in paragraph_tokens], key=lambda t: int(t.id.split('_t')[-1]))
        token_id_number = tokens_in_poppler_order[0].id.split('_t')[-1]
        token_id = f"p{page_number}_m{token_id_number}"
        content = " ".join([token.content for token in paragraph_tokens])
        pdf_font = mode([token.font for token in paragraph_tokens])
        reading_order = min([label_page.reading_order_by_token_id[token.id] for token in paragraph_tokens])
        label_page.reading_order_by_token_id[token_id] = reading_order
        bounding_box = Rectangle.merge_rectangles([token.bounding_box for token in paragraph_tokens])
        token_type = mode([token.token_type for token in paragraph_tokens])
        return PdfToken(page_number, token_id, content, pdf_font, reading_order, bounding_box, token_type)

    @staticmethod
    def remove_paragraph_tokens_from_labels(paragraph_tokens: list[PdfToken], label_page: ReadingOrderLabelPage):
        for paragraph_token in paragraph_tokens:
            del label_page.reading_order_by_token_id[paragraph_token.id]

    @staticmethod
    def remove_paragraph_tokens_from_page(paragraph_tokens: list[PdfToken], page: PdfPage):
        for paragraph_token in paragraph_tokens:
            page.tokens.remove(paragraph_token)

    @staticmethod
    def add_processed_token_to_page(figure_table_token: PdfToken, paragraph_tokens: list[PdfToken], page: PdfPage):
        insert_index = sorted([page.tokens.index(token) for token in paragraph_tokens])[0]
        page.tokens.insert(insert_index, figure_table_token)

    @staticmethod
    def reassign_labels(label_page):
        reading_order = 0
        for token_id, _ in sorted(label_page.reading_order_by_token_id.items(), key=lambda item: item[1]):
            label_page.reading_order_by_token_id[token_id] = reading_order
            reading_order += 1

    def process(self):
        paragraphs: list[Paragraph] = self.paragraph_extractor.get_paragraphs(self.model_path)
        for paragraph in paragraphs:
            segment_type = mode([token.token_type for token in paragraph.tokens])
            if not (segment_type == TokenType.FIGURE or segment_type == TokenType.TABLE):
                continue
            page = self.pdf_reading_order_tokens.pdf_features.pages[paragraph.tokens[0].page_number - 1]
            label_page = self.pdf_reading_order_tokens.labeled_page_by_raw_page[page]
            figure_table_token = self.get_processed_token_from_paragraph(paragraph.tokens, label_page)
            self.add_processed_token_to_page(figure_table_token, paragraph.tokens, page)
            self.remove_paragraph_tokens_from_labels(paragraph.tokens, label_page)
            self.remove_paragraph_tokens_from_page(paragraph.tokens, page)
            self.reassign_labels(label_page)
