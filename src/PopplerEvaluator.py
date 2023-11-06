from pdf_features.PdfPage import PdfPage
from pdf_features.PdfToken import PdfToken
from pdf_reading_order.PdfReadingOrderTokens import PdfReadingOrderTokens


class PopplerEvaluator:
    def __init__(self, pdf_reading_order: PdfReadingOrderTokens, candidates_count: int):
        self.pdf_reading_order = pdf_reading_order
        self.candidates_count = candidates_count

    def iterate_with_neighbors(self, page: PdfPage, current_token_index: int) -> list[PdfToken]:
        previous_tokens = page.tokens[max(0, current_token_index - self.candidates_count) : current_token_index]
        next_tokens = page.tokens[
            current_token_index + 1 : min(len(page.tokens), current_token_index + 1 + self.candidates_count)
        ]
        return previous_tokens + next_tokens

    def contains_next_token(self, page: PdfPage, current_token: PdfToken, candidate_tokens: list[PdfToken]) -> bool:
        if self.pdf_reading_order.labeled_page_by_raw_page[page].reading_order_by_token_id[current_token.id] == len(
            page.tokens
        ):
            return True
        for candidate_token in candidate_tokens:
            if self.pdf_reading_order.labeled_page_by_raw_page[page].is_next_token(current_token, candidate_token):
                return True
        return False

    def get_missing_next_token_count(self) -> int:
        missing_next_token_count: int = 0
        last_page = None
        current_token_index = 0
        for page, token in self.pdf_reading_order.pdf_features.loop_tokens():
            if page != last_page:
                current_token_index = 0
                last_page = page
            candidate_tokens = self.iterate_with_neighbors(page, current_token_index)
            if not self.contains_next_token(page, token, candidate_tokens):
                missing_next_token_count += 1
            current_token_index += 1
        return missing_next_token_count
