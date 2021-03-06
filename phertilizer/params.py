from dataclasses import dataclass


@dataclass
class Params:
    lamb : int = 200
    tau : int = 200
    starts : int = 5
    iterations : int = 10 
    spectral_gap : float = 0.05
    jump_percentage : float = 0.075
    radius : float = 0.5
    npass : int = 1
    def __post_init__(self):
        if self.radius > 1:
            self.radius = 1
        
        if self.jump_percentage > 1:
            self.jump_percentage = 0.075
        
        if self.npass  > 2:
            self.npass = 2
