

import numpy as np
import pandas as pd
import logging
from scipy.spatial.distance import pdist, squareform

from clonal_tree import IdentityTree, LinearTree, LossTree
from clonal_tree_list import ClonalTreeList
from utils import normalizedMinCut, check_stats, snv_kernel_width, cnv_kernel_width, impute_mut_features


# from phertilizer.clonal_tree import LinearTree
# from phertilizer.clonal_tree_list import ClonalTreeList
# from phertilizer.utils import normalizedMinCut, check_stats, snv_kernel_width, cnv_kernel_width, impute_mut_features


class Loss_split():
    """
    A class to perform a linear tree operation on a given seed and input data

    ...

    Attributes
    ----------
    data : Data
        an object of class Data that contains preprocessed input data for Phertilizer
    states : tuple
        the names of allowable CNA genotype states ("gain", "loss", "neutral")
    cells : np.array
        the cell indices to be included in the tree operation
    muts : np.array
       the SNV indices to be included in the tree operation
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

    mut_assignment_with_cna( cellsA, muts)
        returns two arrays mutsA, mutsB based on the maximum likelihood with the node assignment, considering the current
        CNA genotype of each SNV for the fixed input assignment of cellsA


    create_affinity(cells,  mutsB=None)
        returns the edge weights w_ii' for the edge weights for the input graph to normalized min cut

    cluster_cells( cells, mutsB)
        clusters an input set of cells into cellsA and cellsB using normalized min cut on a graph with edge
        weights mixing SNV and CNA features

    assign_cna_events(cellsA, cellsB)
        use the precomputed cnn_hmm to compute the most likely CNA genotypes for the given cell clusters cellsA, cellsB

    run(cells, muts, p)
        a helper method to assist with recursively finding the maximum likelihood linear tree

    best_norm_like(tree_list, norm_like_list)
        for a given candidate ClonalTreeList of linear trees, identify and return the tree with maximum normalized likelihood

    compute_norm_likelihood(cellsA, cellsB, mutsA, mutsB):
        for a given cell and mutation partition, compute the normalized likelihood by excluding the mutsB, cellsB partitions
        and normalizing by the total observations

    sprout():
        main flow control to obtain the maximum likelihood linear tree from a given input seed


    """

    def __init__(self,
                 data,
                 cna_genotype_mode,
                 seed,
                 rng,
                 params
                 ):

        self.cna_genotype_mode = cna_genotype_mode
        self.rng = rng
        self.like0 = data.like0
        self.like1_dict = data.like1_dict
        self.like1 = data.like1_marg

        self.var = data.var
        self.total = data.total

        self.states = ("gain", "loss", "neutral")

        self.cells = seed.cells
        self.mutsB = seed.muts
        self.mutsA = seed.ancestral_muts
        self.snv_bin_mapping = data.snv_bin_mapping

        self.radius = params.radius
        self.npass = params.npass

        self.iterations = params.iterations
        self.starts = params.starts

       # stopping parameters
        self.lamb = params.lamb
        self.tau = params.tau
        self.spectral_gap = params.spectral_gap
        self.jump_percentage = params.jump_percentage
        self.bayes_factor_threshold = params.bayes_factor

        # self.copy_distance_matrix = data.copy_distance

        # self.cnn_hmm = data.cna_hmm

    def random_init_array(self, array, p):
        ''' uniformly at random splits a given array into two parts (A,B)

        Parameters
        ----------
        arr : np.array
            the array to be partitioned into three parts
        p : float
            the probability of being in the "A" array


        Returns
        -------
        arrA : np.array
            a partition of the input array
        arrB : np.array
            a partition of the input array

        '''

        rand_vals = self.rng.random(len(array))
        arrA = array[rand_vals <= p]
        arrB = array[rand_vals > p]
        return arrA, arrB



    def mut_assignment_with_cna_events(self, cellsA, muts, eA):
        """ Updates the assignment of mutations to mutsA or mutsB by comparing 
        the likelihood based off the cells currently assigned to cellsA/

        :param cellsA the set of Ca cells to use to find Mb
        :return: numpy arrays contain the mutation ids in mutsA and mutsB
        """

        like0_array = self.like0[np.ix_(cellsA, muts)].sum(axis=0)
        like0_series = pd.Series(like0_array, index=muts).sort_index()
        all_snvs = []
        like1_arrays = []
        for s in self.states:
            bins = eA[s]  # find mutations in bins assigned to state s
            snvs = self.snv_bin_mapping[self.snv_bin_mapping.isin(
                bins)].index.to_numpy()
            snvs = np.intersect1d(muts, snvs)
            all_snvs.append(snvs)
            like1 = self.like1_dict[s]
            like1_array = like1[np.ix_(cellsA, snvs)].sum(axis=0)
            like1_arrays.append(like1_array)

        # find SNVs mapped to excluded bins
        all_snvs = np.concatenate(all_snvs)
        missing_snvs = np.setdiff1d(muts, all_snvs)
        like1_array = self.like1_dict["neutral"][np.ix_(
            cellsA, missing_snvs)].sum(axis=0)
        like1_arrays.append(like1_array)
        all_snvs = np.concatenate([all_snvs, missing_snvs])

        like1_series = pd.Series(np.concatenate(
            like1_arrays), all_snvs).sort_index()

        mutsB = like0_series[like0_series > like1_series].index.to_numpy()
        mutsA = np.setdiff1d(muts, mutsB)

        return mutsA, mutsB


    def cell_assignment(self):
        """ Updates the assignment of mutations to mutsA or mutsB by comparing 
        the likelihood based off the cells currently assigned to cellsA/

        :param cellsA the set of Ca cells to use to find Mb
        :return: numpy arrays contain the mutation ids in mutsA and mutsB
        """
      
    
        like0_array =self.like0[np.ix_(self.cells, self.mutsB)].sum(axis=1)
        like1_array =self.like1[np.ix_(self.cells, self.mutsB)].sum(axis=1)

        mask_pred = np.array(like0_array > like1_array)
        cellsA = self.cells[(mask_pred)]
        cellsB = np.setdiff1d(self.cells,cellsA)

        return cellsA, cellsB

    def mut_assignment(self):
        """ Updates the assignment of mutations to mutsA or mutsB by comparing 
        the likelihood based off the cells currently assigned to cellsA/

        :param cellsA the set of Ca cells to use to find Mb
        :return: numpy arrays contain the mutation ids in mutsA and mutsB
        """
      
    
        like0_array =self.like0[np.ix_(self.cells, self.mutsA)].sum(axis=0)
        like1_array =self.like1[np.ix_(self.cells, self.mutsA)].sum(axis=0)

        mask_pred = np.array(like0_array > like1_array)
        ma_minus = self.mutsA[(mask_pred)]
        # ma_minus_series = pd.Series(like0_array[ma_minus], index=ma_minus).sort_values(ascending=False)
        logging.info(f"Ma minus: {len(ma_minus)}")
       

        
        ma_plus= np.setdiff1d(self.mutsA, ma_minus)
        logging.info(f"Ma plus: {len(ma_plus)} Ma minus: {len(ma_minus)}")


    
        return ma_plus, ma_minus
  

    def assign_cna_events(self, cellsA, cellsB):
        '''    use the precomputed hmm to compute the most likely CNA genotypes for the 
        given cell clusters cellsA, cellsB


        Parameters
        ----------
        cellsA : np.array
            the cell indices in the first cluster
        cellsB : np.array
            the cell indices in the second cluster


        Returns
        -------
        eventsA : dict
            a dictionary mapping a CNA genotype state to a set of bins for the cells in first cluster
        eventsB : np.array
            a dictionary mapping a CNA genotype state to a set of bins for the cells in second cluster


        '''
    
        eA, eB = None, None
        if self.cna_genotype_mode:
            eA = self.cnn_hmm.run(cellsA)

            eB = self.cnn_hmm.run(cellsB)

        return eA, eB



    def compute_likelihood(self, cellsA, cellsB, ma_plus, ma_minus):
        '''    calculated the normalized variant log likelihood for the 
                given partition

        
        Parameters
        ----------
        cellsA : np.array
            the cell indices in the first cluster
        cellsB : np.array
            the cell indices in the second cluster
        mutsA : np.array
            the SNV indices in the first cluster
        mutsB : np.array
            the SNV indices in the second cluster


        Returns
        -------
        norm_like : float
           the normalized likelihood of the partitions, excluding the cellsB/mutsB 
           partition


        '''

        ca_like = self.like1[np.ix_(cellsA, ma_plus)].sum() + self.like0[np.ix_(cellsA, np.union1d(ma_minus, self.mutsB))].sum()
        cb_like = self.like1[np.ix_(cellsA, np.union1d(ma_plus, self.mutsB))].sum() + self.like0[np.ix_(cellsA, ma_minus)].sum()
      

        likelihood = ca_like + cb_like

        return likelihood

    def sprout(self):
        """ main flow control to obtain the maximum likelihood linear tree
        
        Returns
        -------
        best_tree_like : LinearTree
            a LinearTree with maximum normalized likelihood among all restarts
      
        """

        if len(self.mutsA) ==0:
            return None
        
        # self.snv_bin_mapping = self.snv_bin_mapping.filter(items=self.mutsA)

        best_like = np.NINF
        #find model1

        cellsA, cellsB = self.cell_assignment()


        ma_plus, ma_minus = self.mut_assignment()

        if len(ma_minus) ==0:
            return None

        likelihood = self.compute_likelihood(cellsA, cellsB, ma_plus, ma_minus)
        identity_like =  self.like1[np.ix_(self.cells, np.union1d(self.mutsA, self.mutsB))].sum()

        bayes_factor = likelihood/identity_like

        if bayes_factor >= self.bayes_factor_threshold:
            eA, eB = self.assign_cna_events(cellsA, cellsB)
            lt =LossTree(cellsA, cellsB, ma_plus, ma_minus, self.mutsB, eA, eB)
            return lt
        else:
            return None






        #compare against model2 
        # events = CNA_Events(self.cells,self.bins, self.rdr,chrom_bin_mapping=self.chrom_bin_mapping).run()
      
           
        if not include_cna:
            ma_plus, ma_minus = self.mut_assignment()
        else:
            events = self.cna_hmm.run(self.cells)
            ma_plus, ma_minus = self.mut_assignment_with_cna(events)

        if len(ma_minus) <= self.min_snvs:
            return None

        if not include_cna:
            cellsA, cellsB = self.cell_assignment()
       
        else:
           cellsA, cellsB = self.cell_assignment_with_cna(events)
        
        # ma_plus, ma_minus,   = self.check_genomic_locations(ma_minus)
        

        if self.debug:
            pd.Series(ma_minus).to_csv("test/ma_minus.csv", index=False)
        #check to to see if just a loss node should be placed
    

        logging.info(f"CellsA: {len(cellsA)} Cells B: {len(cellsB)}")
        # like, _ = self.compute_likelihood(cellsA, cellsB, ma_plus, ma_minus)
        #check one
        # pval = self.permutation_test(cellsA, cellsB, ma_plus, ma_minus, like)

       
       
          
        eB = self.cna_hmm.run(cellsB)

        if len(cellsA) > 0:
            eA = self.cna_hmm.run(cellsA)
        else:
            eA = None

   
        loss_tree = LossTree(cellsA, cellsB, ma_minus, self.mutsB, eA, eB)



       
        return loss_tree



        # self.cand_splits = ClonalTreeList()

        # self.run(self.cells, self.muts, p=0.5)

        # if self.cand_splits.size() == 0:
        #     return None

        # best_tree_index = np.argmax(np.array(self.norm_like_list))

        # best_tree_like = self.cand_splits.index_tree(best_tree_index)

        # return best_tree_like
