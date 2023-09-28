from dataclasses import dataclass

from pdf_features.PdfToken import PdfToken


@dataclass
class CandidateScore:
    current_token: PdfToken
    candidate: PdfToken
    score: float
