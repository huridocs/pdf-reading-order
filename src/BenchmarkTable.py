from os.path import exists
from os import makedirs
from tabulate import tabulate
from PredictionInfo import PredictionInfo
from pdf_reading_order.PdfReadingOrderTokens import PdfReadingOrderTokens


class BenchmarkTable:
    def __init__(self, pdf_reading_order_tokens_list: list[PdfReadingOrderTokens], total_time: float, table_name=""):
        self.pdf_paragraphs_tokens_list: list[PdfReadingOrderTokens] = pdf_reading_order_tokens_list
        self.total_time = total_time
        self.prediction_info_list = [
            PredictionInfo(pdf_reading_order_tokens) for pdf_reading_order_tokens in pdf_reading_order_tokens_list
        ]
        self.table_name = table_name

    @staticmethod
    def get_mistakes_for_file(predictions_for_file: PredictionInfo):
        labels_for_file = 0
        mistakes_for_file = 0
        for page in predictions_for_file.pdf_reading_order_tokens.pdf_features.pages:
            actual_orders = predictions_for_file.actual_reading_orders_by_page[page]
            labels_for_file += len(actual_orders)
            predicted_orders = predictions_for_file.predicted_reading_orders_by_page[page]
            labels_map = {prev_token: next_token for prev_token, next_token in zip(actual_orders, actual_orders[1:])}
            mistakes_for_file += sum(
                1
                for prev_token, next_token in zip(predicted_orders, predicted_orders[1:])
                if prev_token not in labels_map or labels_map[prev_token] != next_token
            )
        return labels_for_file, mistakes_for_file

    def get_mistakes_for_file_type(self, predictions_for_file_type: list[PredictionInfo]):
        labels_for_file_type = 0
        mistakes_for_file_type = 0
        for predictions_for_file in predictions_for_file_type:
            labels_for_file, mistakes_for_file = self.get_mistakes_for_file(predictions_for_file)
            labels_for_file_type += labels_for_file
            mistakes_for_file_type += mistakes_for_file
        return labels_for_file_type, mistakes_for_file_type

    def get_benchmark_table_rows(self):
        benchmark_table_rows: list[list[str]] = []
        file_types = set(info.file_type for info in self.prediction_info_list)
        total_label_count = 0
        total_mistake_count = 0
        for file_type in file_types:
            predictions_for_file_type = [info for info in self.prediction_info_list if info.file_type == file_type]
            labels_for_file_type, mistakes_for_file_type = self.get_mistakes_for_file_type(predictions_for_file_type)
            total_label_count += labels_for_file_type
            total_mistake_count += mistakes_for_file_type
            accuracy = round(100 - (100 * mistakes_for_file_type / labels_for_file_type), 2)
            benchmark_table_rows.append([file_type, f"{mistakes_for_file_type}/{labels_for_file_type} ({accuracy}%)"])

        return benchmark_table_rows, total_label_count, total_mistake_count

    def prepare_benchmark_table(self):
        table_headers = ["File Type", "Mistakes"]
        table_rows, total_label_count, total_mistake_count = self.get_benchmark_table_rows()
        average_accuracy = round(100 - (100 * total_mistake_count / total_label_count), 2)
        if not exists("benchmark_tables"):
            makedirs("benchmark_tables")
        table_path = f"benchmark_tables/benchmark_table{self.table_name}.txt"
        with open(table_path, "w") as benchmark_file:
            benchmark_table = (
                tabulate(tabular_data=table_rows, headers=table_headers)
                + "\n\n"
                + f"Average Accuracy: {total_mistake_count} Mistakes/{total_label_count} Labels ({average_accuracy}%)"
                + "\n"
                + f"Total Time: {round(self.total_time, 2)}"
            )
            benchmark_file.write(benchmark_table)
