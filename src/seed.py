import numpy as np




class Seed:
    def __init__(self, cells, muts, ancestral_muts= None, key=None):
        
        
        self.cells = cells 
        self.muts = muts
        if ancestral_muts is None:
            self.ancestral_muts = np.empty(shape=0, dtype=int)
        else:
            self.ancestral_muts = ancestral_muts
        self.key = key
        self.valid = None

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

    def explored(self):
        return self.valid is not None
    
    def set_valid(self,bool):
        self.valid = bool
    

    def set_key(self, key):
        self.key = key


            

