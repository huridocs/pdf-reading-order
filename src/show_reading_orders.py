import shutil
import pdf_reading_order.ReadingOrderTrainer
from pdf_token_type_labels.TaskMistakes import TaskMistakes
from benchmark_segmented_reading_order import get_segmented_pdf_reading_order_tokens
from pdf_reading_order.PdfReadingOrderTokens import PdfReadingOrderTokens
from pdf_reading_order.ReadingOrderTrainer import ReadingOrderTrainer
from pdf_reading_order.model_configuration import SEGMENTED_READING_ORDER_MODEL_CONFIGURATION

PDF_LABELED_DATA_ROOT_PATH = "/path/to/pdf-labeled-data"
MISTAKES_NAME = "segmented_reading_order"


def get_reading_order_predictions(model_path: str, pdf_reading_order_tokens_list: list[PdfReadingOrderTokens]):
    pdf_reading_order.ReadingOrderTrainer.USE_CANDIDATES_MODEL = False
    trainer = ReadingOrderTrainer(pdf_reading_order_tokens_list, SEGMENTED_READING_ORDER_MODEL_CONFIGURATION)
    print(f"Model prediction started...")
    trainer.predict(model_path)


def show_mistakes(pdf_reading_order_tokens_list: list[PdfReadingOrderTokens]):
    shutil.rmtree("/path/to/pdf-labeled-data/labeled_data/task_mistakes/segmented_reading_order", ignore_errors=True)

    for pdf_reading_order_tokens in pdf_reading_order_tokens_list:
        task_mistakes = TaskMistakes(
            PDF_LABELED_DATA_ROOT_PATH, MISTAKES_NAME, pdf_reading_order_tokens.pdf_features.file_name
        )
        for page in pdf_reading_order_tokens.pdf_features.pages:
            labeled_page = pdf_reading_order_tokens.labeled_page_by_raw_page[page]
            for segment_index, segment in enumerate(page.tokens):
                if segment.prediction != labeled_page.reading_order_by_token_id[segment.id]:
                    task_mistakes.add(page.page_number, segment.bounding_box, 1, 0, str(segment_index))
                    continue
                task_mistakes.add(page.page_number, segment.bounding_box, 1, 1, str(segment_index))

        task_mistakes.save()


def run():
    model_path = "/path/to/pdf-reading-order/model/segmented_reading_order_benchmark.model"
    pdf_reading_order_tokens_list = get_segmented_pdf_reading_order_tokens("test")
    get_reading_order_predictions(model_path, pdf_reading_order_tokens_list)
    show_mistakes(pdf_reading_order_tokens_list)


if __name__ == "__main__":
    run()
