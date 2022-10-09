from dataclasses import dataclass


@dataclass
class Params:

    starts : int = 5
    iterations : int = 30
    radius : float = 0.975
    minobs: int = 4113
    seed: int = 1026
    use_copy_kernel: bool = True
    post_process: bool = True
    low_cmb: float= 0.05
    high_cmb: float=0.15
    nobs_per_cluster: int = 4

    def __post_init__(self):
        if self.radius > 1:
            self.radius = 1
        if self.low_cmb >= 1:
            self.low_cmb = 0.05
        if self.high_cmb >= 1:
            self.high_cmb = 0.15
        

