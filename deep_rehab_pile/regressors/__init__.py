"""Deep Regressors."""

__all__ = [
    "BASE_REGRESSOR",
    "ConvTran_REGRESSOR",
    "FCN_REGRESSOR",
    "GRU_REGRESSOR",
    "H_Inception_REGRESSOR",
    "LITE_MV_REGRESSOR",
    "ConvLSTM_REGRESSOR",
    "DisjointCNN_REGRESSOR",
    "STGCN_REGRESSOR",
    "VanTran_REGRESSOR",
]

from deep_rehab_pile.regressors._convlstm import ConvLSTM_REGRESSOR
from deep_rehab_pile.regressors._convtran import ConvTran_REGRESSOR
from deep_rehab_pile.regressors._discnn import DisjointCNN_REGRESSOR
from deep_rehab_pile.regressors._fcn import FCN_REGRESSOR
from deep_rehab_pile.regressors._gru import GRU_REGRESSOR
from deep_rehab_pile.regressors._hinception import H_Inception_REGRESSOR
from deep_rehab_pile.regressors._lite_mv import LITE_MV_REGRESSOR
from deep_rehab_pile.regressors._stgcn import STGCN_REGRESSOR
from deep_rehab_pile.regressors._van_tran import VanTran_REGRESSOR
from deep_rehab_pile.regressors.base import BASE_REGRESSOR
