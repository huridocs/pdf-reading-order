import typer
from OrderedPage import OrderedPage
from pdf_features.PdfFeatures import PdfFeatures
from pdf_reading_order.PdfReadingOrderTokens import PdfReadingOrderTokens
from pdf_tokens_type_trainer.ModelConfiguration import ModelConfiguration
from pdf_reading_order.ReadingOrderTrainer import ReadingOrderTrainer


def predict(pdf_path: str, model_path: str = None):
    pdf_features = PdfFeatures.from_pdf_path(pdf_path)
    pdf_reading_order_tokens = PdfReadingOrderTokens(pdf_features, {})
    trainer = ReadingOrderTrainer([pdf_reading_order_tokens], ModelConfiguration())
    trainer.predict(model_path)

    predictions: list[OrderedPage] = list()

    for page in pdf_reading_order_tokens.pdf_features.pages:
        predictions.append(OrderedPage.from_pdf_tokens(pdf_features.file_name, page.page_number, page.tokens))

    print([prediction.to_dict() for prediction in predictions])


if __name__ == "__main__":
    typer.run(predict)