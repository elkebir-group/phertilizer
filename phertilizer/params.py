from dataclasses import dataclass


@dataclass
class Params:
    starts : int = 5
    iterations : int = 10 
    radius : float = 1
    minobs: int = 4113
    seed: int = 1026
    post_process: bool = True
    use_copy_kernel: bool = False
    low_cmb: float = 0.05
    high_cmb: float = 0.15
    nobs_per_cluster: float = 3
    def __post_init__(self):
        if self.radius > 1:
            self.radius = 1
        
