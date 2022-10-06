from dataclasses import dataclass


@dataclass
class Params:

    starts : int = 5
    iterations : int = 10 
    radius : float = 0.975
    minobs: int = 4113
    seed: int = 1026
    use_copy_kernel: bool = False
    post_process: bool = True
    def __post_init__(self):
        if self.radius > 1:
            self.radius = 1
        

