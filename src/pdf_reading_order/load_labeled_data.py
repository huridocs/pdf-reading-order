from os import listdir
from os.path import join, isdir
from pdf_reading_order.config import READING_ORDER_RELATIVE_PATH
from pdf_reading_order.PdfReadingOrderTokens import PdfReadingOrderTokens


def loop_datasets(reading_order_labeled_data_path: str, filter_in: str):
    print(reading_order_labeled_data_path)
    for dataset_name in listdir(reading_order_labeled_data_path):
        if filter_in and filter_in not in dataset_name:
            continue

        dataset_path = join(reading_order_labeled_data_path, dataset_name)

        if not isdir(dataset_path):
            continue

        yield dataset_name, dataset_path


def load_labeled_data(pdf_labeled_data_root_path: str, filter_in: str = None) -> list[PdfReadingOrderTokens]:
    if filter_in:
        print(f"Loading only datasets with the key word: {filter_in}")
        print()

    pdf_paragraph_tokens_list: list[PdfReadingOrderTokens] = list()
    reading_order_labeled_data_path: str = join(pdf_labeled_data_root_path, READING_ORDER_RELATIVE_PATH)

    for dataset_name, dataset_path in loop_datasets(reading_order_labeled_data_path, filter_in):
        print(f"loading {dataset_name} from {dataset_path}")

        dataset_pdf_name = [(dataset_name, pdf_name) for pdf_name in listdir(dataset_path)]
        for dataset, pdf_name in dataset_pdf_name:
            pdf_paragraph_tokens = PdfReadingOrderTokens.from_labeled_data(pdf_labeled_data_root_path, dataset, pdf_name)
            pdf_paragraph_tokens_list.append(pdf_paragraph_tokens)

    return pdf_paragraph_tokens_list
