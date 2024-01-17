import typer
from pdf_token_type_labels.TokenType import TokenType
from pdf_tokens_type_trainer.TokenTypeTrainer import TokenTypeTrainer
from OrderedPage import OrderedPage
from pdf_features.PdfFeatures import PdfFeatures
from SegmentProcessor import SegmentProcessor
from pdf_reading_order.PdfReadingOrderTokens import PdfReadingOrderTokens
from pdf_tokens_type_trainer.ModelConfiguration import ModelConfiguration
from pdf_reading_order.ReadingOrderTrainer import ReadingOrderTrainer


def predict(pdf_path: str, extract_figures_and_tables: bool = False, model_path: str = None):
    pdf_features = PdfFeatures.from_pdf_path(pdf_path)
    pdf_reading_order_tokens = PdfReadingOrderTokens(pdf_features, {})
    token_type_trainer = TokenTypeTrainer([pdf_features])
    token_type_trainer.set_token_types()
    if extract_figures_and_tables:
        table_figure_processor = SegmentProcessor([pdf_reading_order_tokens], [TokenType.FIGURE, TokenType.TABLE])
        table_figure_processor.process()
    trainer = ReadingOrderTrainer([pdf_reading_order_tokens], ModelConfiguration())
    trainer.predict(model_path)

    predictions: list[OrderedPage] = list()

    for page in pdf_reading_order_tokens.pdf_features.pages:
        predictions.append(OrderedPage.from_pdf_tokens(pdf_features.file_name, page.page_number, page.tokens))

    print([prediction.to_dict() for prediction in predictions])


if __name__ == "__main__":
    typer.run(predict)
