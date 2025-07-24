"""Deep Classifiers."""

__all__ = [
    "BASE_CLASSIFIER",
    "ConvTran_CLASSIFIER",
    "FCN_CLASSIFIER",
    "GRU_CLASSIFIER",
    "H_Inception_CLASSIFIER",
    "LITE_MV_CLASSIFIER",
    "ConvLSTM_CLASSIFIER",
    "DisjointCNN_CLASSIFIER",
    "STGCN_CLASSIFIER",
    "VanTran_CLASSIFIER",
]

from deep_rehab_pile.classifiers._convlstm import ConvLSTM_CLASSIFIER
from deep_rehab_pile.classifiers._convtran import ConvTran_CLASSIFIER
from deep_rehab_pile.classifiers._discnn import DisjointCNN_CLASSIFIER
from deep_rehab_pile.classifiers._fcn import FCN_CLASSIFIER
from deep_rehab_pile.classifiers._gru import GRU_CLASSIFIER
from deep_rehab_pile.classifiers._hinception import H_Inception_CLASSIFIER
from deep_rehab_pile.classifiers._lite_mv import LITE_MV_CLASSIFIER
from deep_rehab_pile.classifiers._stgcn import STGCN_CLASSIFIER
from deep_rehab_pile.classifiers._van_tran import VanTran_CLASSIFIER
from deep_rehab_pile.classifiers.base import BASE_CLASSIFIER
