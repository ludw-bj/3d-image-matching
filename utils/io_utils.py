import numpy as np
import os
import dataclasses
import pandas as pd

@dataclasses.dataclass
class Prediction:
    image_id: str | None  # A unique identifier for the row -- unused otherwise. Used only on the hidden test set.
    dataset: str
    filename: str
    cluster_index: int | None = None
    rotation: np.ndarray | None = None
    translation: np.ndarray | None = None

def load_predictions(data_dir, is_train):
    if is_train:
        sample_submission_csv = os.path.join(data_dir, 'train_labels.csv')
    else:
        sample_submission_csv = os.path.join(data_dir, 'sample_submission.csv')

    samples = {}
    competition_data = pd.read_csv(sample_submission_csv)
    for _, row in competition_data.iterrows():
        # Note: For the test data, the "scene" column has no meaning, and the rotation_matrix and translation_vector columns are random.
        if row.dataset not in samples:
            samples[row.dataset] = []
        samples[row.dataset].append(
            Prediction(
                image_id=None if is_train else row.image_id,
                dataset=row.dataset,
                filename=row.image
            )
        )

    for dataset in samples:
        print(f'Dataset "{dataset}" -> num_images={len(samples[dataset])}')
    
    return samples

