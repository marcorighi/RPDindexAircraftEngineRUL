# Input data

The input data are not redistributed in this repository.

The analysis uses the Kaggle dataset:

    Predictive Maintenance: Aircraft Engine
    Author: Mohammed Hadani
    Dataset handle:
    mhadani/predictive-maintenance-aircraft-engine

Source:

    https://www.kaggle.com/datasets/mhadani/predictive-maintenance-aircraft-engine

Required files:

    PM_train.csv
    PM_test.csv
    PM_truth.csv

To retrieve the files automatically from the repository root:

    python3 -m pip install kagglehub
    python3 download_data.py

`DATA_SHA256.txt` records SHA-256 hashes of the exact files used for the
analysis reported in the manuscript. The download script verifies the
retrieved files against these hashes.
