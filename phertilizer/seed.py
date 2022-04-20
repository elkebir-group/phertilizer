from dataclasses import dataclass
import numpy as np



@dataclass
class Seed:

    cells : np.array 
    muts : np.array
    ancestral_muts : np.array = np.empty(shape=0, dtype=int)
    key : int = None

    def __str__(self):

        outstring = f"Cells: {len(self.cells)} Muts: {len(self.muts)} Ancestral Muts: {len(self.ancestral_muts)} "
        return outstring
    
    def __eq__(self, object):
        
        ancestral_muts_same = np.array_equal(np.sort(self.ancestral_muts), np.sort(object.ancestral_muts))
   
        if type(object) is type(self):
            return np.array_equal(self.cells, object.cells) \
                 and np.array_equal(self.muts, object.muts) \
                 and ancestral_muts_same
        else:
            return False
    

    def set_key(self, key):
        self.key = key


            

