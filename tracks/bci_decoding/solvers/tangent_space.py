"""Riemannian baseline: covariance -> tangent space -> ridge classifier.

The classic motor-imagery pipeline, trained on all labeled windows:

1. band-pass each window to 8-30 Hz (4th-order Butterworth, zero-phase);
2. one spatial covariance matrix per window (OAS shrinkage);
3. project the matrices to the tangent space at their Riemannian mean;
4. standardize and fit a ridge classifier, its regularization chosen by
   built-in leave-one-out cross-validation.

The fitted pipeline ships as ``weights.joblib``. Output folder:
``TangentSpace-Ridge-scratch-shotall-seed<seed>``.
"""

import joblib
import numpy as np
from scipy.signal import butter, sosfiltfilt
from sklearn.linear_model import RidgeClassifierCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from benchmark_utils.base_solver import CompetSolver
from benchmark_utils.data import to_numpy


def band_pass(X, sfreq, low=8.0, high=30.0, order=4):
    sos = butter(order, [low, high], btype="bandpass", fs=sfreq,
                 output="sos")
    return sosfiltfilt(sos, np.asarray(X, dtype=np.float64), axis=-1)


def make_pipeline_():
    # Only sklearn / pyriemann objects, so the joblib dump unpickles inside
    # any module name (the platform loads this file as ``submission.py``).
    from pyriemann.estimation import Covariances
    from pyriemann.tangentspace import TangentSpace
    return make_pipeline(
        Covariances(estimator="oas"),
        TangentSpace(metric="riemann"),
        StandardScaler(),
        RidgeClassifierCV(alphas=np.logspace(-3, 3, 13)),
    )


class TangentSpaceModel:

    def __init__(self, pipeline, sfreq):
        self.pipeline = pipeline
        self.sfreq = sfreq

    def fit(self, X, y):
        self.pipeline.fit(band_pass(X, self.sfreq), y)

    def predict(self, X):
        return self.pipeline.predict(band_pass(to_numpy(X), self.sfreq))


class Solver(CompetSolver):

    name = "TangentSpace-Ridge"

    requirements = ["pip::pyriemann"]

    def set_objective(self, train_loader, **meta):
        self.name = self.run_label("TangentSpace-Ridge", "scratch", "shotall")
        super().set_objective(train_loader, **meta)

    def load_model(self, meta):
        weights = meta["submission_dir"] / "weights.joblib"
        if self.train_loader is None:
            if not weights.is_file():
                raise FileNotFoundError(
                    "TangentSpace-Ridge inference needs weights.joblib"
                )
            print(f"[loading] {weights} into {self.name}")
            return TangentSpaceModel(joblib.load(weights), float(meta["sfreq"]))
        return TangentSpaceModel(make_pipeline_(), float(meta["sfreq"]))

    def fit(self, model, train_loader):
        signals, labels = [], []
        for X, y, _info in train_loader:
            signals.append(to_numpy(X))
            labels.append(to_numpy(y).reshape(-1))
        model.fit(np.concatenate(signals), np.concatenate(labels))

    def save_model(self, model, path):
        joblib.dump(model.pipeline, path / "weights.joblib")
