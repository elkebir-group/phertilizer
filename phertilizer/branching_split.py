
import numpy as np
from scipy.spatial.distance import pdist, squareform
import pandas as pd



from phertilizer.clonal_tree_list import ClonalTreeList
from phertilizer.utils import normalizedMinCut, snv_kernel_width, cnv_kernel_width, impute_mut_features
from phertilizer.clonal_tree import BranchingTree
from phertilizer.clonal_tree_list import ClonalTreeList

np.seterr(invalid='ignore')

class Branching_split():
    """
    A class to perform a branching tree operation on a given seed and input data

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

    random_init(array)
        splits the given array into three parts uniformly at random

    calc_tmb(cells, muts)
        calculate the binary tumor mutational burden for each of the input cells over the set of input SNVs

    mut_assignment( cellsA, muts)
        returns three arrays (mutsA, mutsB, mutsC) based on the maximum likelihood with the node assignment of each SNV for 
        the fixed input assignment of cellsA

    mut_assignment_with_cna( cellsA, muts)
        returns two arrays mutsA, mutsB based on the maximum likelihood with the node assignment, considering the current
        CNA genotype of each SNV for the fixed input assignment of cellsA

    create_affinity(mutsA, mutsB, cells)
        returns the edge weights w_ii' for the edge weights for the input graph to normalized min cut

    cluster_cells( cells, mutsB, weight=0.5)
        clusters an input set of cells into cellsA and cellsB using normalized cut on a graph with edge
        weights mixing SNV and CNA features

    assign_cna_events(cellsA, cellsB)
        use the precomputed cnn_hmm to compute the most likely CNA genotypes for the given cell clusters cellsA, cellsB

    run()
         a helper method to add candidate branching for each restart

    sprout()
        main flow control to obtain the maximum likelihood branching tree


    """

    def __init__(self,
                 data,
                 seed,
                 rng,
                 params
                 ):



        self.data = data
        self.total = data.total
        self.like0 = data.like0
        self.like1_dict = data.like1_dict
        self.like1 = data.like1_marg


        self.cells = seed.cells
        self.muts = seed.muts

        self.rng = rng
        self.np_rng = np.random.RandomState(params.seed)
        self.starts = params.starts
        self.iterations = params.iterations
        self.use_copy_kernel = params.use_copy_kernel
        self.radius = params.radius
        self.params =params 
    
    low_cmb: float= 0.075
    high_cmb: float=0.15


    def random_init(self, arr):
        ''' uniformly at random splits a given array into three parts (A,B,C)

        Parameters
        ----------
        arr : np.array
            the array to be partitioned into three parts


        Returns
        -------
        arrA : np.array
            a partition of the input array
        arrB : np.array
            a partition of the input array
        arrC: np.array
            a partition of the input array

        '''

        rand_vals = self.rng.random(arr.shape[0])
        arrA = arr[rand_vals < 1/3]
        arrB = arr[rand_vals > 2/3]
        arrC = np.setdiff1d(arr, np.concatenate([arrA, arrB]))
        return arrA, arrB, arrC

    def calc_tmb(self, cells, muts):
        ''' calculates the binary tumor mutational burden (tmb) of the given SNVs for each cell

        Parameters
        ----------
        cells : np.array
            the array of cell indices for which the tmb needs to be calcualted
        muts : np.array
            the SNVs to include in the tmb calculations


        Returns
        -------
        tmb : np.array
            the binary tumor mutational burden for each cell 
 

        '''

        mut_var_count_by_cell = np.count_nonzero(
            self.data.var[np.ix_(cells, muts)], axis=1).reshape(-1, 1)
        mut_total_reads_by_cell = np.count_nonzero(
            self.data.total[np.ix_(cells, muts)], axis=1).reshape(-1, 1)

        tmb = mut_var_count_by_cell/mut_total_reads_by_cell

        return tmb

    def create_affinity(self,  mutsA, mutsB, cells):
        '''   computes the edge weights w_ii' for the edge weights for the input graph to normalized min cut

        Parameters
        ----------
        mutsA : np.array
            the current set of mutsA to include in the SNV features
        mutsB : np.array
            the current set of mutsB to include in the SNV features
        cells : np.array
            the set of cell indices to represent nodes of the graph


        Returns
        -------
        kernel : np.array
            a cell x cell matrix containing the edge weights of the graph
        mut_features_df : pd.Dataframe
            a pandas dataframe containing the calculated mutsA and mutsB features for each cell
 

        '''
      

        if len(mutsA) > 0 and len(mutsB) > 0:
            tmb_mb = self.calc_tmb(cells, mutsB)

            tmb_ma = self.calc_tmb(cells, mutsA)

            mut_features = np.hstack([tmb_ma, tmb_mb])

            copy_dist_mat = self.data.copy_distance.copy()

            mut_features = impute_mut_features(
                copy_dist_mat, mut_features, cells)

            # if we aren't able to impute mut features for cells, then we can't cluster
            if np.any(np.isnan(mut_features)):
                return None, None

            mut_features_df = pd.DataFrame(mut_features, cells)

            d_mat = squareform(pdist(mut_features, metric="euclidean"))

            kw = snv_kernel_width(d_mat)
            kernel = np.exp(-1*d_mat/kw)

            c_mat = self.data.copy_distance[np.ix_(cells, cells)]

            r = np.quantile(c_mat, self.radius)
  
            if self.use_copy_kernel:
                copy_kernel = np.exp(-1*c_mat/cnv_kernel_width(c_mat))
                copy_kernel[c_mat >= r] = 0
                kernel = np.multiply(copy_kernel, kernel)


            return kernel, mut_features_df
        else:
            return None, None


    def cluster_cells(self, mutsA, mutsB, cells):
        '''   clusters an input set of cells into cellsA and cellsB using normalized min cut on a graph with edge
        weights mixing SNV and CNA features

        Parameters
        ----------
        mutsA : np.array
            the current set of mutsA to include in the SNV features
        mutsB : np.array
            the current set of mutsB to include in the SNV features
        cells : np.array
            the set of cell indices to cluster into two partitions


        Returns
        -------
        cellsA : np.array
            the cell indices of the first cluster
        cellsB : n.array
            the cell indices of the second cluster
        stats : dict
            a dictionary containing the values of the stopping heuristics

        '''


        W, mut_features = self.create_affinity(mutsA, mutsB, cells)
        if W is None:
            return cells, np.empty(shape=0, dtype=int)

        if np.any(np.isnan(W)):
            print("Terminating cell clustering due to NANs in affinity matrix")
            return cells, np.empty(shape=0, dtype=int)

        clusters, y_vals, labels, stats = normalizedMinCut(W, cells, self.np_rng)

        cells1, cells2 = clusters

        if mut_features is not None:
            avg_muts_ma = mut_features[mut_features.index.isin(
                cells1)].mean(axis=0)
            avg_muts_mb  = mut_features[mut_features.index.isin(
                cells2)].mean(axis=0)

            if avg_muts_ma[0] > avg_muts_ma[1]:
                cellsA = cells1
                cellsB = cells2
            else:
                cellsA = cells2
                cellsB = cells1
        else:
            cellsA = cells1
            cellsB = cells2
   
        return cellsA, cellsB

    def mut_assignment(self, cellsA, cellsB):
        '''    returns three arrays (mutsA, mutsB, mutsC) based on the maximum likelihood with the node assignment of each SNV for 
        the fixed input assignment of cellsA

        Parameters
        ----------
        cellsA : np.array
            the cell indices in the first cluster
       cellsB : np.array
            the cell indices in the second cluster

        Returns
        -------
        mutsA : np.array
            the SNV indices in the left child
        mutsB : np.array
            the SNV indices in the right child
        mutsC : np.array
             the SNV indices in the parent

        '''

        #check to make sure we have enough observations to make a good assignment
        num_obsA = np.count_nonzero(self.total[np.ix_(cellsA, self.muts)],axis=0)
        num_obsB = np.count_nonzero(self.total[np.ix_(cellsB, self.muts)],axis=0)
        obsA = np.quantile(num_obsA, 0.75)
        obsB = np.quantile(num_obsB, 0.75)
        if obsA < self.params.nobs_per_cluster or obsB < self.params.nobs_per_cluster:
            mutsA = np.empty(shape=0, dtype=int)
            mutsB = np.empty(shape=0, dtype=int)
            mutsC = self.muts
            return mutsA, mutsB, mutsC
    
        like1 = self.data.like1_marg

        mutsA = []
        mutsB = []
        mutsC = []

        mutsA_cand = like1[np.ix_(cellsA, self.muts)].sum(
            axis=0) + self.data.like0[np.ix_(cellsB, self.muts)].sum(axis=0)
        mutsB_cand = self.data.like0[np.ix_(cellsA, self.muts)].sum(
            axis=0) + like1[np.ix_(cellsB, self.muts)].sum(axis=0)
        mutsC_cand = like1[np.ix_(cellsA, self.muts)].sum(
            axis=0) + like1[np.ix_(cellsB, self.muts)].sum(axis=0)
        
   
        for idx, m in enumerate(self.muts):
            if mutsA_cand[idx] == max(mutsA_cand[idx], mutsB_cand[idx], mutsC_cand[idx]):
                mutsA.append(m)
            elif mutsB_cand[idx] == max(mutsA_cand[idx], mutsB_cand[idx], mutsC_cand[idx]):
                mutsB.append(m)
            else:
                mutsC.append(m)

        return np.array(mutsA), np.array(mutsB), np.array(mutsC)


    def run(self):
        '''   a helper method to add candidate branching for each restart

        '''



        cells = self.cells
        muts = self.muts

        mutsA, mutsB, mutsC = self.random_init(muts)
        oldA = np.empty(shape=0, dtype=int)
        for j in range(self.iterations):
            if mutsB.shape[0] ==0 or mutsA.shape[0]==0:
                break
            cellsA, cellsB = self.cluster_cells(mutsA, mutsB, cells)

            if len(cellsB) == 0 or len(cellsA) == 0 or \
                np.array_equal(np.sort(cellsA), np.sort(oldA)):
                break
            else:
                oldA = cellsA


            mutsA, mutsB, mutsC = self.mut_assignment(cellsA, cellsB)

        if len(cellsA) > 0 and len(cellsB) > 0 and len(mutsA) >0 and len(mutsB) > 0:
            norm_like = self.compute_norm_likelihood(cellsA, cellsB, mutsA, mutsB, mutsC)
            cand_tree= BranchingTree(
                            cellsA, cellsB, mutsA, mutsB, mutsC)
    
            f1, f2, f3, f4  = self.check_metrics(cellsA, cellsB, mutsA, mutsB, mutsC)
            if f1 <= self.params.low_cmb and f2 <= self.params.low_cmb and \
                f3 >= self.params.high_cmb and f4:
                
                if norm_like > self.best_norm_like:
                    self.best_norm_like = norm_like
                    self.best_tree = cand_tree
            
                 
       
    
    def check_metrics(self, ca,cb, ma, mb, mc):
      
        feat1_var = np.count_nonzero(
            self.data.var[np.ix_(ca, mb)], axis=1)
        feat1_total = np.count_nonzero(
            self.data.total[np.ix_(ca, mb)], axis=1)
        #should be low  <= 0.05
        feat1 =np.nanmedian((feat1_var/feat1_total))

    
        #should be low  <= 0.05
        feat2 =np.nanmedian((feat1_var/feat1_total))

        if len(mc) > 0:
            mc = mc.astype(int)
            var =  np.count_nonzero(
                self.data.var[np.ix_(self.cells, mc)], axis=1)
            tot = np.count_nonzero(
                self.data.total[np.ix_(self.cells, mc)], axis=1)
            
        #should be high  >= 0.15
            feat3 =np.nanmedian((var/tot))
        else:
            feat3 = 1

        feats = []

        for a in [0,1]:
            for c in [ca, cb]:
                for m in [ma, mb]:
                    feats.append( self.check_reads(c,m,a))
        feat3 = all(feats)

        return feat1, feat2, feat3, feats #feat4, feat5

    def check_reads(self, cells, muts, axis):
        nobs= np.count_nonzero(self.data.total[np.ix_(cells, muts)], axis=axis)
        obs = np.median(nobs)
        return obs >= self.params.nobs_per_cluster


    def compute_norm_likelihood(self, cellsA, cellsB, mutsA, mutsB, mutsC):
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
        mutsC = mutsC.astype(int)
        total_like = self.like1[np.ix_(self.cells, mutsC)].sum(
        ) + self.like0[np.ix_(cellsA, mutsB)].sum() + self.like0[np.ix_(cellsB, mutsA)].sum()
        total = np.count_nonzero(self.like0[np.ix_(
            self.cells, mutsC)]) + np.count_nonzero(self.like0[np.ix_(cellsA, mutsB)]) + np.count_nonzero(self.like0[np.ix_(cellsB, mutsA)])

        # total = len(cellsA)*len(self.muts) + len(cellsB)*len(mutsA)
        norm_like = total_like/total

        return norm_like

    def sprout(self):
        """ main flow control to obtain the maximum likelihood branching tree
        
        Returns
        -------
        best_branching_tree : BranchingTree
            a BranchingTree with maximum likelihood among all restarts
      
        """
        self.cand_trees = ClonalTreeList()
        self.best_norm_like = np.NINF
        self.best_tree = None
        for i in range(self.starts):
      
            self.run()
            self.use_copy_kernel = not self.use_copy_kernel
        return self.best_tree
   
        
