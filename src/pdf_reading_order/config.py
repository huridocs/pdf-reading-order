from os.path import join
from pathlib import Path

ROOT_PATH = Path(__file__).parent.parent.parent.absolute()

PDF_LABELED_DATA_ROOT_PATH = Path(join(ROOT_PATH.parent, "pdf-labeled-data"))
READING_ORDER_RELATIVE_PATH = join("labeled_data", "reading_order")
