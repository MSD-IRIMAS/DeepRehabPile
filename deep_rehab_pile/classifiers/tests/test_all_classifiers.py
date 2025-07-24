"""Unit tests for classifiers deep learning functionality."""

import inspect
import tempfile
import time

import numpy as np
import pytest

from deep_rehab_pile import classifiers

_deep_cls_classes = [
    member[1] for member in inspect.getmembers(classifiers, inspect.isclass)
]


@pytest.mark.parametrize("deep_cls", _deep_cls_classes)
def test_all_classifiers(deep_cls):
    """Test Deep Classifiers."""
    with tempfile.TemporaryDirectory() as tmp:
        if deep_cls.__name__ not in [
            "BASE_CLASSIFIER",
        ]:
            if tmp[-1] != "/":
                tmp = tmp + "/"
            curr_time = str(time.time_ns())
            best_file_name = curr_time + "best"
            init_file_name = curr_time + "init"

            X = np.random.normal(size=(5, 1, 10))
            y = np.array([0, 0, 0, 1, 1])

            if (
                deep_cls.__name__ != "ConvTran_CLASSIFIER"
                and deep_cls.__name__ != "GRU_CLASSIFIER"
                and deep_cls.__name__ != "ConvLSTM_CLASSIFIER"
                and deep_cls.__name__ != "DisjointCNN_CLASSIFIER"
                and deep_cls.__name__ != "LITE_MV_CLASSIFIER"
                and deep_cls.__name__ != "STGCN_CLASSIFIER"
                and deep_cls.__name__ != "VanTran_CLASSIFIER"
            ):
                _deep_cls = deep_cls(
                    output_dir=tmp,
                    best_file_name=best_file_name,
                    init_file_name=init_file_name,
                    epochs=2,
                    length_TS=10,
                    n_joints=1,
                    n_dim=1,
                )
            else:
                if deep_cls.__name__ == "ConvLSTM_CLASSIFIER":
                    _deep_cls = deep_cls(
                        output_dir=tmp,
                        length_TS=10,
                        n_joints=1,
                        n_dim=1,
                        best_file_name=best_file_name,
                        init_file_name=init_file_name,
                        epochs=2,
                        kernel_size=2,
                    )
                elif deep_cls.__name__ == "DisjointCNN_CLASSIFIER":
                    _deep_cls = deep_cls(
                        output_dir=tmp,
                        length_TS=10,
                        n_joints=1,
                        n_dim=1,
                        best_file_name=best_file_name,
                        init_file_name=init_file_name,
                        epochs=2,
                        kernel_size=[2, 2, 2, 2],
                        pool_size=2,
                    )
                elif deep_cls.__name__ == "STGCN_CLASSIFIER":
                    _deep_cls = deep_cls(
                        output_dir=tmp,
                        length_TS=10,
                        n_joints=1,
                        n_dim=1,
                        best_file_name=best_file_name,
                        init_file_name=init_file_name,
                        epochs=2,
                        kinematic_tree=[[0]],
                        n_temporal_convolution_layers=1,
                        n_temporal_filters=4,
                        temporal_kernel_size=[2],
                        n_bottleneck_filters=2,
                        n_lstm_layers=1,
                        n_lstm_units=[2],
                    )
                elif deep_cls.__name__ == "VanTran_CLASSIFIER":
                    _deep_cls = deep_cls(
                        output_dir=tmp,
                        length_TS=10,
                        n_joints=1,
                        n_dim=1,
                        best_file_name=best_file_name,
                        init_file_name=init_file_name,
                        epochs=2,
                        emb_size=2,
                        n_layers=1,
                        num_heads=1,
                        dimm_ff=2,
                    )
                else:
                    _deep_cls = deep_cls(
                        output_dir=tmp,
                        length_TS=10,
                        n_joints=1,
                        n_dim=1,
                        best_file_name=best_file_name,
                        init_file_name=init_file_name,
                        epochs=2,
                    )

            training_time = _deep_cls.fit(X, y)

            assert isinstance(training_time, float)
            assert training_time >= 0.0

            ypred_probas = _deep_cls.predict_proba(X)
            assert int(ypred_probas.shape[1]) == 2

            ypred, inference_time = _deep_cls.predict(X)

            assert isinstance(inference_time, float)
            assert len(ypred.shape) == 1
            assert len(ypred) == len(y)

            score = _deep_cls.score(X, y)
            assert score is not None
            assert isinstance(score, float)
