"""Unit tests for classifiers deep learning functionality."""

import inspect
import tempfile
import time

import numpy as np
import pytest

from deep_rehab_pile import regressors

_deep_rgs_classes = [
    member[1] for member in inspect.getmembers(regressors, inspect.isclass)
]


@pytest.mark.parametrize("deep_rgs", _deep_rgs_classes)
def test_all_classifiers(deep_rgs):
    """Test Deep Classifiers."""
    with tempfile.TemporaryDirectory() as tmp:
        if deep_rgs.__name__ not in [
            "BASE_REGRESSOR",
        ]:
            if tmp[-1] != "/":
                tmp = tmp + "/"
            curr_time = str(time.time_ns())
            best_file_name = curr_time + "best"
            init_file_name = curr_time + "init"

            X = np.random.normal(size=(5, 1, 10))
            y = np.array([0, 0, 1, 1, 1])

            if (
                deep_rgs.__name__ != "ConvTran_REGRESSOR"
                and deep_rgs.__name__ != "GRU_REGRESSOR"
                and deep_rgs.__name__ != "ConvLSTM_REGRESSOR"
                and deep_rgs.__name__ != "DisjointCNN_REGRESSOR"
                and deep_rgs.__name__ != "LITE_MV_REGRESSOR"
                and deep_rgs.__name__ != "STGCN_REGRESSOR"
                and deep_rgs.__name__ != "VanTran_REGRESSOR"
            ):
                _deep_rgs = deep_rgs(
                    output_dir=tmp,
                    best_file_name=best_file_name,
                    init_file_name=init_file_name,
                    epochs=2,
                    length_TS=10,
                    n_joints=1,
                    n_dim=1,
                )
            else:
                if deep_rgs.__name__ == "ConvLSTM_REGRESSOR":
                    _deep_rgs = deep_rgs(
                        output_dir=tmp,
                        length_TS=10,
                        n_joints=1,
                        n_dim=1,
                        best_file_name=best_file_name,
                        init_file_name=init_file_name,
                        epochs=2,
                        kernel_size=2,
                    )
                elif deep_rgs.__name__ == "DisjointCNN_REGRESSOR":
                    _deep_rgs = deep_rgs(
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
                elif deep_rgs.__name__ == "STGCN_REGRESSOR":
                    _deep_rgs = deep_rgs(
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
                elif deep_rgs.__name__ == "VanTran_REGRESSOR":
                    _deep_rgs = deep_rgs(
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
                    _deep_rgs = deep_rgs(
                        output_dir=tmp,
                        length_TS=10,
                        n_joints=1,
                        n_dim=1,
                        best_file_name=best_file_name,
                        init_file_name=init_file_name,
                        epochs=2,
                    )

            training_time = _deep_rgs.fit(X, y)

            assert isinstance(training_time, float)
            assert training_time >= 0.0

            ypred, inference_time = _deep_rgs.predict(X)

            assert isinstance(inference_time, float)
            assert inference_time >= 0.0
            assert len(ypred.shape) == 1
            assert len(ypred) == len(y)

            score = _deep_rgs.score(X, y, max_score_value=1.0)
            assert score is not None
            assert isinstance(score, float)
