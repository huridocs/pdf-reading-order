from os.path import join
from os import makedirs
from pathlib import Path
import numpy as np
from benchmark_candidate_finder import get_features
from pdf_reading_order.ReadingOrderCandidatesTrainer import ReadingOrderCandidatesTrainer
from pdf_reading_order.model_configuration import CANDIDATE_MODEL_CONFIGURATION
from pdf_reading_order.config import ROOT_PATH


MODEL_PATH = Path(join(ROOT_PATH, "model", "candidate_selector_model.model"))


def train_model():
    x_train, y_train = get_features("train")
    x_test, y_test = get_features("test")
    x_train = np.concatenate((x_train, x_test), axis=0)
    y_train = np.append(y_train, y_test)
    trainer = ReadingOrderCandidatesTrainer([], CANDIDATE_MODEL_CONFIGURATION)
    makedirs(MODEL_PATH.parent, exist_ok=True)
    trainer.train(MODEL_PATH, x_train, y_train)


if __name__ == "__main__":
    train_model()
