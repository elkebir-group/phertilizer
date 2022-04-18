

import numpy as np
from clonal_tree import LinearTree
from clonal_tree_list import ClonalTreeList
from utils import normalizedMinCut, check_stats, snv_kernel_width, cnv_kernel_width, impute_mut_features
import pandas as pd
import logging
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt



class Linear_split():
    """
    A class to perform a linear tree operation on a given seed and input data

    ...

    Attributes
    ----------
    like0: : np.array
        precomputed likelihood given y=0
    like1_marg : np.array
        precomputed likelihood given y=1 and latent vafs marginalized over all possibilities
    like1_dict: dict
        a dictionary indexed by state s of precomputed likelihoods given y=1 and z=s
    var: np.array
        an n x m array containing the variant read counts for each cell and SNV
    copy_distance : np.array
        a precomputed n x n array of RDR distances for each cell i in [n]
    total: np.array
        an n x m array containing the  total read counts for each cell and SNV
    cnn_hmm : CNA_HMM object
        a prefit hidden Markov model for CNA genotypes
    snv_bin_mapping: pandas series
        a mapping of each SNV to bin
    states : tuple
        the names of allowable CNA genotype states ("gain", "loss", "neutral")
    cells : np.array
        the cell indices to be included in the linear tree operation
    muts : np.array
       the SNV indices to be included in the linear tree operation
    rng ; random number generator
        random number generator to use for initialization
    lamb : int 
        minimum number of cells in a leaf in order to perform a tree operation
    tau : int 
        minimum number of SNVs in a leaf in order to perform a tree operation
    iterations : int
        maximum number of iterations for tree operations if convergence is not met
    starts : int
        number restarts for each tree operations 
    radius : float
        the quantile of the RDR distance matrix for which similarity of RDR values 
        should be consered (default: 0.5 [median])
    npass : int
        the number of heuristics that a cell clustering must pass in order for a tree 
        operation to be performed on a leaf node (default: 1)
    debug : boolean
        turn on debugging mode (not used)
    spectral_gap : float
        parameter for the minimum value of the spectral gap between k=2 and k=1
    jump_percentage: float
        parameter for the minimum jump percentage to conclude that k > 1

    

    Methods
    -------

    random_init_array(array, p)
        returns input array split into two arrays where entries have probability p of being in the first array
    
    random_weighted_array(cells, muts,  p)
        returns input muts array split into two arrays where the probability of being in the first array
        is weighted by the binary variant allele frequency for the given set of input cells 
 
 
    mut_assignment( cellsA, muts)
        returns two arrays mutsA, mutsB based on the maximum likelihood with the node assignment of each SNV for 
        the fixed input assignment of cellsA
            
    create_affinity(cells, lamb= 0.5, mutsB=None)
        returns the edge weights w_ii' for the edge weights for the input graph to normalized min cut
    
    cluster_cells( cells, mutsB, weight=0.5)
        clusters an input set of cells into cellsA and cellsB using normalized min cut on a graph with edge
        weights mixing SNV and CNA features
    
    assign_cna_events(cellsA, cellsB)
        use the precomputed cnn_hmm to compute the most likely CNA genotypes for the given cell clusters cellsA, cellsB
    
    run( cells, muts, weight, p)
        a helper method to assist with finding recursively finding the maximum likelihood linear tree
    
    best_norm_like(tree_list, norm_like_list)
        for a given candidate ClonalTreeList of linear trees, identify and return the tree with maximum normalized likelihood
    

    
    compute_norm_likelihood(cellsA, cellsB, mutsA, mutsB):
        for a given cell and mutation partition, compute the normalized likelihood by excluding the mutsB, cellsB partitions
        and normalizing by the total observations
    

    sprout( seed, infer_cna_events=True):
        main flow control to obtain the maximum likelihood linear tree from a given input seed


    """

    def __init__(self,
                like0, 
                like1,
                var,  
                total, 
                cells, 
                muts, 
                rng, 
                starts=7, 
                iterations=20,
                weights=None,
                spectral_gap= 0.05,
                jump_percentage = 0.03,
                lamb = 50,
                tau = 50,
                copy_distance_matrix=None, 
                radius =0.5,
                npass= 1,
                debug=False, 
                ):
        self.rng = rng
        self.like0 = like0
    
        self.like1 = like1
            
        self.var= var
        self.total = total


        
        self.cells = cells
        self.muts =muts


        self.radius = radius
        self.npass = npass 

        self.iterations = iterations      
        self.starts = starts

        if weights is not None:
            self.weights = weights
        else:
            self.weights = [0.5]

        self.lamb = lamb
        self.tau = tau 
        self.spectral_gap = spectral_gap
        self.jump_percentage = jump_percentage

    
        self.copy_distance_matrix = copy_distance_matrix          

        self.debug = debug

       
        

    def random_init_array(self, array, p):
        """Randomly initialize two arrays

        :return: two numpy arrays created from the original array with the first having success probability p and the second (1-p)
        """
        rand_vals =self.rng.random(len(array))
        arrA = array[rand_vals <=p]
        arrB = array[rand_vals > p]
        return arrA, arrB
       
    

    def random_weighted_init_array(self, cells, muts,  p):
        """Randomly initialize two arrays

        :return: two numpy arrays created from the original array with the first having success probability p and the second (1-p)
        """
        binary_counts = np.count_nonzero(self.var[np.ix_(cells, muts)],axis=0)
        binary_tcounts = np.count_nonzero(self.total[np.ix_(cells, muts)],axis=0)
        
        vaf = binary_counts/binary_tcounts
        vaf_prob = vaf/np.sum(vaf)
        num_mutsA = int(np.floor(p*muts.shape[0]))
        if num_mutsA < np.count_nonzero(vaf_prob) and not np.any(np.isnan(vaf_prob)):
            mutsA = self.rng.choice(muts, size= num_mutsA, p=vaf_prob, replace=False)
            mutsB = np.setdiff1d(muts, mutsA)
        else:
            mutsA, mutsB = self.random_init_array(muts, p)


        return mutsA, mutsB



    




    def cell_assignment(self, mutsB):
        """ Updates the assignment of cells to cellsA or cellsB by comparing 
        the likelihood based off the mutations currently assigned to mutsB.

        :param number the minimum number of cells to assign to Ca
        :param var_val the maximum number of observed variant reads in Mb in order to assign
                        a cells to Ca if that cells is not the top number of cells with likelihood y_ij =0
        :return: a numpy arrays containing the cell ids in cellsA and cellsB
        """
     
        like0_array =self.like0[np.ix_(self.cells, mutsB)].sum(axis=1).reshape(-1)

        like1_array= self.like1[np.ix_(self.cells, mutsB)].sum(axis=1).reshape(-1)

        cellsA = self.cells[like0_array > like1_array]

        cellsB = np.setdiff1d(self.cells, cellsA)

        return cellsA, cellsB
    

    def mut_assignment(self, cellsA, muts):
        """ Updates the assignment of mutations to mutsA or mutsB by comparing 
        the likelihood based off the cells currently assigned to cellsA/

        :param cellsA the set of Ca cells to use to find Mb
        :return: numpy arrays contain the mutation ids in mutsA and mutsB
        """
      
        like0_array =self.like0[np.ix_(cellsA, muts)].sum(axis=0)
        like1_array =self.like1[np.ix_(cellsA, muts)].sum(axis=0)

        mask_pred = np.array(like0_array > like1_array)
        mutsB = muts[(mask_pred)]
        mutsA = np.setdiff1d(muts, mutsB)


        return mutsA, mutsB

    


         
    def create_affinity(self, cells, lamb= 0.5, mutsB=None):
        
        if lamb > 0 and mutsB is not None:
            mb_mut_count = np.count_nonzero(self.var[np.ix_(cells, mutsB)],axis=1).reshape(-1)
            mb_tot_count = np.count_nonzero(self.total[np.ix_(cells, mutsB)],axis=1).reshape(-1)
            
            cdist = self.copy_distance_matrix.copy()
            tmb = mb_mut_count/mb_tot_count
            tmb = tmb.reshape(-1,1)
            if np.any(np.isnan(tmb)):
                tmb = impute_mut_features(cdist, tmb, cells)
          
            d_mat = squareform(pdist(tmb, metric="euclidean"))
            
            kernel = np.exp(-1*d_mat/snv_kernel_width(d_mat))
        if lamb < 1:
            c_mat = self.copy_distance_matrix[np.ix_(cells, cells)]

            r = np.quantile(c_mat,self.radius)
            if np.max(c_mat) > 1:
          
                copy_kernel = np.exp(-1*c_mat/cnv_kernel_width(c_mat))
                copy_kernel[c_mat > r] =0

            else:
                print("Using SNV kernel only")
                lamb = 1

        if lamb ==1 and len(mutsB) ==0:
            return None, None
        if lamb == 0:
            kernel = copy_kernel
            mb_mut_count = None
        elif lamb==1:
            kernel = kernel
        else:
            kernel = np.multiply(kernel, copy_kernel)
    
        mb_mut_count_series = pd.Series(mb_mut_count.reshape(-1), index= cells)
            
    
        return kernel, mb_mut_count_series

    def cluster_cells(self, cells, mutsB, weight=0.5):
   
        W, mb_count_series = self.create_affinity(cells, weight, mutsB)
        if W is None:
            return cells, np.empty(shape=0, dtype=int), None
        
        if np.any(np.isnan(W)):
            print("Terminating cell clustering due to NANs in affinity matrix")
            return cells, np.empty(shape=0, dtype=int), None

       
        clusters, y_vals, labels, stats = normalizedMinCut(W, cells)
        cells1, cells2 = clusters

        norm_mut_count1 = mb_count_series[cells1].mean()
        norm_mut_count2 = mb_count_series[cells2].mean()
        if np.isnan(norm_mut_count1) or np.isnan(norm_mut_count2):
                print("warning mut count mean is Nan")
        if norm_mut_count1 < norm_mut_count2:
          
            cellsA, cellsB = cells1, cells2
            logging.info(f"Mb Count a: {norm_mut_count1} Mb Count b: {norm_mut_count2}")
        else:
            cellsA, cellsB =  cells2, cells1
            logging.info(f"Mb Count a: {norm_mut_count2} Mb Count b: {norm_mut_count1}")




        return cellsA, cellsB, stats

  

    def assign_cna_events(self, cellsA, cellsB):
       
        eA = self.cnn_hmm.run(cellsA)


        eB  = self.cnn_hmm.run(cellsB)
      
        
        return eA, eB


  

    def run(self, cells, muts, weight, p):
        if len(cells)  > 2*self.lamb:
            internal_tree_list = ClonalTreeList()
            norm_list = []
            
            for s in range(self.starts):
              
                mutsA, mutsB = self.random_weighted_init_array(cells, muts, p)
                oldCellsA = cells
                for j in range(self.iterations):
                    cellsA, cellsB, stats = self.cluster_cells(cells, mutsB, weight)
                    if len(cellsB) ==0 or np.array_equal(np.sort(cellsA), np.sort(oldCellsA)):
                        break    
                    else:
                        oldCellsA = cellsA
               
                    mutsA, mutsB = self.mut_assignment(cellsA, muts)
                
                    print(f"start:{s} iteration:{j}")
                    print(f"CellsA: {len(cellsA)} CellsB: {len(cellsB)}")
                    print(f"MutsA: {len(mutsA)} MutsB: {len(mutsB)}")
               

                if len(cellsA) > self.lamb and len(mutsA) > self.tau and len(cellsB) > 5: 
                    if check_stats(stats, self.jump_percentage, self.spectral_gap, self.npass):

        
                            cellsB_tree = np.setdiff1d(self.cells, cellsA)
                          
                            mutsA, mutsB_tree = self.mut_assignment(cellsA, muts)

                        
                            lt = LinearTree(cellsA, cellsB_tree, mutsA, mutsB_tree)
                            print("adding tree to internal list:")
                            print(lt)
                            internal_tree_list.insert(lt)

                            norm_list.append(self.compute_norm_likelihood(cellsA, cellsB, mutsA, mutsB_tree))
            
            best_tree = self.best_norm_like(internal_tree_list, norm_list)
            if best_tree is not None:
                print("adding tree to candidate list:")
                print(best_tree)
                self.cand_splits.insert(best_tree)
                like_norm = self.compute_norm_likelihood(best_tree.get_tip_cells(0),
                                                                best_tree.get_tip_cells(1),
                                                                best_tree.get_tip_muts(0),
                                                                best_tree.get_tip_muts(1))
                
                print(f"Normalized Likelihood: {like_norm}")
                self.norm_like_list.append(like_norm)
                self.run_helper_no_cna(best_tree.get_tip_cells(0), muts, weight, p)

            

    
    @staticmethod        
    def best_norm_like(tree_list, norm_like_list):
        if tree_list.size() ==0:
            return None
        
        best_tree_index =np.argmax(np.array(norm_like_list))
    
        best_tree_like = tree_list.index_tree(best_tree_index)

        return best_tree_like



    

    def compute_norm_likelihood(self,cellsA, cellsB, mutsA, mutsB):

        total_like = self.like0[np.ix_(cellsA, mutsB)].sum() + self.like1[np.ix_(self.cells, mutsA)].sum()
        total = np.count_nonzero(self.like0[np.ix_(cellsA, mutsB)]) + np.count_nonzero(self.like1[np.ix_(self.cells, mutsA)])
        
        norm_like  = total_like/total

        return norm_like 

    @staticmethod
    def save(cellsA, cellsB, mutsA, mutsB, name):
            a_series =pd.Series(data= np.zeros_like(mutsA, dtype=int), index=mutsA)
            b_series = pd.Series( np.full_like(mutsB, fill_value=1), index=mutsB)
            mut_series = pd.concat([a_series, b_series])
            mut_series.to_csv(f"test/{name}_mut_series.csv")

            a_series =pd.Series(np.zeros_like(cellsA, dtype=int), index=cellsA)
            b_series = pd.Series( np.full_like(cellsB, fill_value=1), index=cellsB)
            cell_series = pd.concat([a_series, b_series])
            cell_series.to_csv(f"test/{name}_cell_series.csv")

    

    def sprout(self, seed):
    
        self.norm_like_list = []
 

        self.cand_splits = ClonalTreeList()


  
        muts = seed.muts
        cells = seed.cells
        self.run(cells, muts, 0.5, p=0.5)


 
        if self.cand_splits.size() == 0:
            return None
        
        best_tree_index =np.argmax(np.array(self.norm_like_list))

        for tree in self.cand_splits.get_all_trees():
            print(tree)
            print(f"Normalized Likelihood: {self.norm_like_list[tree.key]}")
            


        best_tree_like = self.cand_splits.index_tree(best_tree_index)

        print("linear split norm likelihood best tree")
        print(best_tree_like)


        return best_tree_like

        