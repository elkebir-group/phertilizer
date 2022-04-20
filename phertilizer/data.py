from dataclasses import dataclass
import numpy as np
import pandas as pd

@dataclass
class Data:
    cell_lookup : pd.Series
    mut_lookup : pd.Series
    var : np.array
    total : np.array
    like0 : np.array
    like1_marg : np.array
    like1_dict : dict = None
    copy_distance : np.array = None
    cna_hmm : CNA_HMM = None


