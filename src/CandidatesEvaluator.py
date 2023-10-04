from CandidateScore import CandidateScore
from pdf_reading_order.PdfReadingOrderTokens import PdfReadingOrderTokens


class CandidatesEvaluator:
    def __init__(
        self, pdf_reading_order: PdfReadingOrderTokens, candidates_scores: list[CandidateScore], candidates_count: int
    ):
        self.pdf_reading_order = pdf_reading_order
        self.candidates_scores = candidates_scores
        self.candidates_count = candidates_count

    def contains_next_token(self):
        contains_next_token = list()
        already_selected_tokens = list()
        last_page = None
        for page, token in self.pdf_reading_order.pdf_features.loop_tokens():
            if last_page != page:
                already_selected_tokens = []
                last_page = page
            next_token = self.get_next_token(token, page)
            if next_token is None:
                continue
            sorted_candidates_scores = [candidate_score for candidate_score in self.get_sorted_candidates_scores(token)
                                        if candidate_score.candidate not in already_selected_tokens]

            possible_candidates = [
                candidate_score.candidate for candidate_score in sorted_candidates_scores[: self.candidates_count]
            ]
            already_selected_tokens.append(next_token)
            contains_next_token.append(next_token in possible_candidates)

        return contains_next_token

    def get_next_token(self, current_token, current_page):
        for page, token in self.pdf_reading_order.pdf_features.loop_tokens():
            if current_page != page:
                continue

            if self.pdf_reading_order.labeled_page_by_raw_page[page].is_next_token(current_token, token):
                return token

    def get_sorted_candidates_scores(self, token):
        list_of_candidates_scores: list[CandidateScore] = list()
        for candidate_score in self.candidates_scores:
            if candidate_score.current_token != token:
                continue

            if candidate_score.current_token == token:
                list_of_candidates_scores.append(candidate_score)
        list_of_candidates_scores = sorted(list_of_candidates_scores, key=lambda x: x.score, reverse=True)
        return list_of_candidates_scores
