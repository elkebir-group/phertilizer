import numpy as np
from scipy.spatial.distance import pdist, squareform
import pandas as pd


# from phertilizer.clonal_tree_list import ClonalTreeList
# from phertilizer.utils import normalizedMinCut, check_stats, snv_kernel_width, cnv_kernel_width, impute_mut_features, find_cells_with_no_reads, find_muts_with_no_reads
# from phertilizer.clonal_tree import BranchingTree
# from phertilizer.clonal_tree_list import ClonalTreeList

from clonal_tree_list import ClonalTreeList
from utils import normalizedMinCut, check_stats, snv_kernel_width, cnv_kernel_width, impute_mut_features, check_obs
from clonal_tree import BranchingTree
from clonal_tree_list import ClonalTreeList


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
                 cna_genotype_mode,
                 seed,
                 rng,
                 params
                 ):



        self.data = data
        self.total = data.total

        self.cna_genotype_mode = cna_genotype_mode

        self.cells = seed.cells
        self.muts = seed.muts

        self.lamb = params.lamb
        self.tau = params.tau

        self.npass = params.npass
        self.rng = rng
        self.np_rng = np.random.RandomState(params.seed)

        self.starts = params.starts
        self.iterations = params.iterations

        self.spectral_gap = params.spectral_gap
        self.jump_percentage = params.jump_percentage

        self.use_copy_kernel = params.use_copy_kernel

        self.radius = params.radius

        self.states = ["gain", "loss", "neutral"]
        if self.data.snv_bin_mapping is not None:
            self.bin_mapping = {
                m: self.data.snv_bin_mapping.loc[m] for m in self.data.snv_bin_mapping.index}

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
            # if np.max(c_mat) > 4:
            # if np.max(c_mat) > 1:
            if self.use_copy_kernel:
                copy_kernel = np.exp(-1*c_mat/cnv_kernel_width(c_mat))
                copy_kernel[c_mat >= r] = 0
                kernel = np.multiply(copy_kernel, kernel)

            # else:
            #     print("Max copy distance < 1, using SNV kernel only")

            return kernel, mut_features_df
        else:
            return None, None

    def mut_assignment_with_cna(self,  curr_tree):
        m = self.data.like0.shape[1]
        cellsA = curr_tree.get_tip_cells(1)
        cellsB = curr_tree.get_tip_cells(2)
        all_cells = curr_tree.get_all_cells()
        zA = curr_tree.event_by_node(m, cellsA, 1, self.data.snv_bin_mapping)
        zB = curr_tree.event_by_node(m, cellsB, 2, self.data.snv_bin_mapping)
        z = curr_tree.event_matrix(m,  self.data.snv_bin_mapping)

        for s in self.states:
            mutsA = self.data.like0[cellsB, :].sum(axis=0)
            mutsB = self.data.like0[cellsA, :].sum(axis=0)
            mutsC = np.zeros(shape=m, dtype=float)
            for s in self.states:
                like1_mat = self.data.like1_dict[s]
                mutsA += np.ma.array(like1_mat[cellsA, :],
                                     mask=zA != s).sum(axis=0)
                mutsB += np.ma.array(like1_mat[cellsB, :],
                                     mask=zB != s).sum(axis=0)
                mutsC += np.ma.array(like1_mat[all_cells, :],
                                     mask=z != s).sum(axis=0)

        combined_matrix = np.vstack([mutsC,mutsA, mutsB ])
        mut_assignments = np.argmax(combined_matrix, axis=0).reshape(-1)
        mut_assignments = mut_assignments[self.muts]

        mutsA = np.array(self.muts[mut_assignments == 1])
        mutsB = np.array(self.muts[mut_assignments == 2])
        mutsC = np.array(self.muts[mut_assignments == 0])

        return mutsA, mutsB, mutsC

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
            return cells, np.empty(shape=0, dtype=int), None

        if np.any(np.isnan(W)):
            print("Terminating cell clustering due to NANs in affinity matrix")
            return cells, np.empty(shape=0, dtype=int), None

        clusters, y_vals, labels, stats = normalizedMinCut(W, cells, self.np_rng)

        cells1, cells2 = clusters

        if mut_features is not None:
            avg_muts_ma = mut_features[mut_features.index.isin(
                cells1)].mean(axis=0)

            if avg_muts_ma[0] > avg_muts_ma[1]:
                cellsA = cells1
                cellsB = cells2
            else:
                cellsA = cells2
                cellsB = cells1
        else:
            cellsA = cells1
            cellsB = cells2
        stats['min_avg_ma'] = min( avg_muts_ma[0], avg_muts_ma[1])
        stats['max_avg_ma'] = max( avg_muts_ma[0] ,avg_muts_ma[1])
        return cellsA, cellsB, stats

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
        
        # nobs = np.count_nonzero(self.total[:,self.muts], axis=0)
        # a_obs = self.total[np.ix_(cellsA,self.muts)]
        # a_obs = np.count_nonzero(a_obs,axis=0)
        # b_obs = self.total[np.ix_(cellsB,self.muts)]
        # b_obs =np.count_nonzero(b_obs, axis=0)
        # a_weight=a_obs/nobs
        # b_weight=b_obs/nobs
        # mutsA_cand = like1[np.ix_(cellsA, self.muts)].sum(
        #     axis=0)*a_weight + self.data.like0[np.ix_(cellsB, self.muts)].sum(axis=0)*b_weight
        # mutsB_cand = a_weight*self.data.like0[np.ix_(cellsA, self.muts)].sum(
        #     axis=0) + b_weight*like1[np.ix_(cellsB, self.muts)].sum(axis=0)
        # mutsC_cand = a_weight*like1[np.ix_(cellsA, self.muts)].sum(
        #     axis=0) + b_weight*like1[np.ix_(cellsB, self.muts)].sum(axis=0)

        # for idx, m in enumerate(self.muts):
        for idx, m in enumerate(self.muts):
            if mutsC_cand[idx] == max(mutsA_cand[idx], mutsB_cand[idx], mutsC_cand[idx]):
                mutsC.append(m)
            elif mutsB_cand[idx] == max(mutsA_cand[idx], mutsB_cand[idx], mutsC_cand[idx]):
                mutsB.append(m)
            else:
                mutsA.append(m)

        return np.array(mutsA), np.array(mutsB), np.array(mutsC)

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

        eventsA, eventsB = None, None
        if self.cna_genotype_mode:
            eventsA = self.data.cna_hmm.run(cellsA)

            eventsB = self.data.cna_hmm.run(cellsB)

        return eventsA, eventsB

    def run(self):
        '''   a helper method to add candidate branching for each restart

        '''



        cells = self.cells
        muts = self.muts

        mutsA, mutsB, mutsC = self.random_init(muts)
        oldA = np.empty(shape=0, dtype=int)
        for j in range(self.iterations):
            cellsA, cellsB, stats = self.cluster_cells(mutsA, mutsB, cells)

            if len(cellsB) == 0 or len(cellsA) == 0 or np.array_equal(np.sort(cellsA), np.sort(oldA)):
                break
            else:
                oldA = cellsA

            eA, eB = self.assign_cna_events(cellsA, cellsB)

            mutsA, mutsB, mutsC = self.mut_assignment(cellsA, cellsB)

        # if (len(cellsA) > self.lamb and len(cellsB) > 1) or (len(cellsB) > self.lamb and len(cellsA) > 1):
        if len(cellsA) > 0 and len(cellsB) > 0:
            if  stats['min_avg_ma'] >= 0.05 or stats['max_avg_ma'] <= 0.15:
                print("here")
            # if check_stats(stats, self.jump_percentage, self.spectral_gap, self.npass):
            # if stats['abs_avg_ma_diff'] > 2:
                cand_tree = BranchingTree(
                    cellsA, cellsB, mutsA, mutsB, mutsC, eA, eB, eC=None)
                self.cand_trees.insert(cand_tree)

    def sprout(self):
        """ main flow control to obtain the maximum likelihood branching tree
        
        Returns
        -------
        best_branching_tree : BranchingTree
            a BranchingTree with maximum likelihood among all restarts
      
        """

        # var_counts_by_snv= self.data.var[np.ix_(self.cells, self.muts)].sum(axis=0)
        # bad_snvs = self.muts[var_counts_by_snv==0]
        # self.muts = np.setdiff1d(self.muts, bad_snvs)
        
        # var_counts_by_cells = self.data.var[np.ix_(self.cells,self.muts)].sum(axis=1)
        # bad_cells = self.cells[var_counts_by_cells ==0]
        # self.cells = np.setdiff1d(self.cells, bad_cells)

        # cell_mean_val = check_obs(self.cells, self.muts, self.total, axis=1)
        # print(cell_mean_val)
        # snv_mean_val = check_obs(self.cells, self.muts, self.total, axis=0)
        # print(snv_mean_val)

        # seed_obs = check_obs(self.cells, self.muts, self.total)
        # print(f"Number of Observations in Seed: {seed_obs}")
        # if seed_obs < self.params.minobs:
        #     return None


        # if cell_mean_val <=  12 or snv_mean_val <= 12:
        #     return None
        

        self.cand_trees = ClonalTreeList()

        for i in range(self.starts):

            self.run()
            self.use_copy_kernel = not self.use_copy_kernel

        if self.cand_trees.has_trees():
            best_branching_tree, _ = self.cand_trees.find_best_tree(self.data)

          
            # cellsA = best_branching_tree.cell_mapping[0][0]
            # best_branching_tree.cell_mapping[0][0]=np.union1d(bad_cells, cellsA)
            # mutsA =  best_branching_tree.mut_mapping[0]
            # best_branching_tree.mut_mapping[0] = np.union1d(bad_snvs, mutsA)

            return best_branching_tree
        else:
            return None
