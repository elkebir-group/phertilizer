

import numpy as np
import pandas as pd
import logging
from scipy.spatial.distance import pdist, squareform

from clonal_tree import LinearTree
from clonal_tree_list import ClonalTreeList
from utils import normalizedMinCut, check_stats, snv_kernel_width, cnv_kernel_width, impute_mut_features, check_obs
from scipy.stats import multivariate_normal


# from phertilizer.clonal_tree import LinearTree
# from phertilizer.clonal_tree_list import ClonalTreeList
# from phertilizer.utils import normalizedMinCut, check_stats, snv_kernel_width, cnv_kernel_width, impute_mut_features


class Linear_split():
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
        self.np_rng= np.random.RandomState(params.seed)
        self.like0 = data.like0
        self.like1_dict = data.like1_dict
        self.like1 = data.like1_marg

        self.data = data
        self.var = data.var
        self.total = data.total

        self.states = ("gain", "loss", "neutral")

        self.cells = seed.cells
        self.muts = seed.muts
        self.snv_bin_mapping = data.snv_bin_mapping

        self.radius = params.radius
        self.npass = params.npass
        self.minobs = params.minobs

        self.use_copy_kernel = params.use_copy_kernel
    

        self.iterations = params.iterations
        self.starts = params.starts

       # stopping parameters
        self.lamb = params.lamb
        self.tau = params.tau
        self.spectral_gap = params.spectral_gap
        self.jump_percentage = params.jump_percentage

        self.copy_distance_matrix = data.copy_distance

        self.cnn_hmm = data.cna_hmm

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

    def random_weighted_init_array(self, cells, muts,  p):
        ''' weighted random splits a given array into two parts (A,B)

        Parameters
        ----------
        cells : np.array
            the cell indices to be used in calculating the weight
        muts : np.array
            the SNV indices to be partitioned
        p : float
            the probability of being in the "A" array


        Returns
        -------
        arrA : np.array
            a partition of the input array
        arrB : np.array
            a partition of the input array

        '''


        binary_counts = np.count_nonzero(self.var[np.ix_(cells, muts)], axis=0)
        binary_tcounts = np.count_nonzero(
            self.total[np.ix_(cells, muts)], axis=0)

        vaf = binary_counts/binary_tcounts
        vaf_prob = vaf/np.sum(vaf)

        num_mutsA = int(np.floor(p*muts.shape[0]))
        if num_mutsA < np.count_nonzero(vaf_prob) and not np.any(np.isnan(vaf_prob)):
            mutsA = self.rng.choice(
                muts, size=num_mutsA, p=vaf_prob, replace=False)
            mutsB = np.setdiff1d(muts, mutsA)
        else:
            mutsA, mutsB = self.random_init_array(muts, p)

        return mutsA, mutsB

    def mut_assignment(self, cellsA, muts):
        '''    returns two arrays (mutsA, mutsB) based on the maximum likelihood with the node assignment of each SNV for 
        the fixed input assignment of cellsA

        Parameters
        ----------
        cellsA : np.array
            the cell indices in the first cluster
        muts : np.array
            the SNV indicies to be partitioned

        Returns
        -------
        mutsA : np.array
            the SNV indices in the parent
        mutsB : np.array
            the SNV indices in the child


        '''


        like0_array = self.like0[np.ix_(cellsA, muts)].sum(axis=0)
        like1_array = self.like1[np.ix_(cellsA, muts)].sum(axis=0)

        mask_pred = np.array(like0_array > like1_array)
        mutsB = muts[(mask_pred)]
        mutsA = np.setdiff1d(muts, mutsB)

        return mutsA, mutsB

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

    def create_affinity(self, cells, mutsB=None):
        '''   computes the edge weights w_ii' for the edge weights for the input graph to normalized min cut

        Parameters
        ----------
        cells : np.array
            the set of cell indices to represent nodes of the graph
        mutsB : np.array
            the current set of mutsB to include in the SNV features



        Returns
        -------
        kernel : np.array
            a cell x cell matrix containing the edge weights of the graph
        mb_mut_count_series : pd.Series
            a pandas series containing the calculated  mutsB features for each cell
 

        '''

        if mutsB is not None:

            mb_mut_count = np.count_nonzero(
                self.var[np.ix_(cells, mutsB)], axis=1).reshape(-1)
            mb_tot_count = np.count_nonzero(
                self.total[np.ix_(cells, mutsB)], axis=1).reshape(-1)

            cdist = self.copy_distance_matrix.copy()
            tmb = mb_mut_count/mb_tot_count
            tmb = tmb.reshape(-1, 1)
            if np.any(np.isnan(tmb)):
                tmb = impute_mut_features(cdist, tmb, cells)

            d_mat = squareform(pdist(tmb, metric="euclidean"))

            kernel = np.exp(-1*d_mat/snv_kernel_width(d_mat))
            # kw = snv_kernel_width(d_mat)
            # kernel = np.exp(-1*np.power(d_mat/kw,2))
            mb_mut_count_series = pd.Series(
                mb_mut_count.reshape(-1), index=cells)

            c_mat = self.copy_distance_matrix[np.ix_(cells, cells)]

            r = np.quantile(c_mat, self.radius)
            if self.use_copy_kernel:

                copy_kernel = np.exp(-1*c_mat/cnv_kernel_width(c_mat))
                # copy_kernel = np.exp(-1*np.power(c_mat/cnv_kernel_width(c_mat),2))
                copy_kernel[c_mat > r] = 0
                kernel = np.multiply(kernel, copy_kernel)

            # else:
            #     print("Using SNV kernel only")

            return kernel, mb_mut_count_series

        else:
            return None, None

    def cluster_cells(self, cells, mutsB):
        '''   clusters an input set of cells into cellsA and cellsB using normalized min cut on a graph with edge
        weights mixing SNV and CNA features

        Parameters
        ----------
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

        W, mb_count_series = self.create_affinity(cells, mutsB)
        if W is None:
            return cells, np.empty(shape=0, dtype=int), None

        if np.any(np.isnan(W)):
            print("Terminating cell clustering due to NANs in affinity matrix")
            return cells, np.empty(shape=0, dtype=int), None

        clusters, y_vals, labels, stats = normalizedMinCut(W, cells, self.np_rng)

        cells1, cells2 = clusters

        norm_mut_count1 = mb_count_series[cells1].mean()
        norm_mut_count2 = mb_count_series[cells2].mean()
        if np.isnan(norm_mut_count1) or np.isnan(norm_mut_count2):
            print("warning mut count mean is Nan")
        if norm_mut_count1 < norm_mut_count2:

            cellsA, cellsB = cells1, cells2
            logging.info(
                f"Mb Count a: {norm_mut_count1} Mb Count b: {norm_mut_count2}")
        else:
            cellsA, cellsB = cells2, cells1
            logging.info(
                f"Mb Count a: {norm_mut_count2} Mb Count b: {norm_mut_count1}")
        stats["norm_count_diff"] = abs(norm_mut_count1- norm_mut_count2)
        stats["ca_mb_count"] = min(norm_mut_count1, norm_mut_count2)
        return cellsA, cellsB, stats

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

    def run(self, cells, muts, p, parent_norm = np.NINF):
        '''     a helper method to add candidate linear tree for each restart


        Parameters
        ----------
        cells : np.array
            the cell indices to be partitioned
        cellsB : np.array
            the SNV indices  to be partioned
        p : float
            the probability parameter to use for random intiialization


        '''

        if check_obs(cells, muts, self.total) < self.minobs:
            print("not enough observations, terminating")
            return 
        
        internal_tree_list = ClonalTreeList()
        norm_list = []

        for s in range(self.starts):

          

            mutsA, mutsB = self.random_weighted_init_array(cells, muts, p)
            # mutsA, mutsB = self.random_init_array( muts, p)
            oldCellsA = cells
            for j in range(self.iterations):
                cellsA, cellsB, stats = self.cluster_cells(cells, mutsB)
                if len(cellsB) == 0 or np.array_equal(np.sort(cellsA), np.sort(oldCellsA)):
                    break
                else:
                    oldCellsA = cellsA

                eA, eB = self.assign_cna_events(cellsA, cellsB)

                mutsA, mutsB = self.mut_assignment(cellsA, muts)

            print(f"number of iterations: {j}")
            if len(cellsA) >0 and len(cellsB) > 0:
            # if len(cellsA) > self.lamb and len(mutsA) > self.tau and len(cellsB) > 5:
                if stats['norm_count_diff'] > 2:#   and stats['ca_mb_count'] < 2:
                #check_stats(stats, self.jump_percentage, self.spectral_gap, self.npass) and :

                    cellsB_tree = np.setdiff1d(self.cells, cellsA)
                    # eA, eB = self.assign_cna_events(cellsA, cellsB_tree)

                    # mutsA, mutsB_tree = self.mut_assignment(cellsA, self.muts)
                    mutsB_tree = np.setdiff1d(self.muts, mutsA)
                    lt = LinearTree(cellsA, cellsB_tree,
                                    mutsA, mutsB_tree, eA, eB)
                    # print(lt)
                    internal_tree_list.insert(lt)

                    norm_list.append(self.compute_norm_likelihood(
                        cellsA, cellsB, mutsA, mutsB_tree))
            # self.use_copy_kernel = not self.use_copy_kernel
        best_tree = self.best_norm_like(internal_tree_list, norm_list)
        print(best_tree)
        if best_tree is not None:

       
            like_norm = self.compute_norm_likelihood(best_tree.get_tip_cells(0),
                                                        best_tree.get_tip_cells(
                                                            1),
                                                        best_tree.get_tip_muts(
                                                            0),
                                                        best_tree.get_tip_muts(1))
       
            print(f"like norm {like_norm}: parent norm {parent_norm}")
            if like_norm > parent_norm:
                self.cand_splits.insert(best_tree)
                # cb =np.setdiff1d(cells, best_tree.get_tip_cells(0))
                self.norm_like_list.append(like_norm)
       
                self.run(best_tree.get_tip_cells(0), best_tree.get_tip_muts(0), p, parent_norm=like_norm)
                # self.run(cb, self.muts, p, parent_norm= like_norm)

    @staticmethod
    def best_norm_like(tree_list, norm_like_list):
        if tree_list.size() == 0:
            return None

        best_tree_index = np.argmax(np.array(norm_like_list))

        best_tree_like = tree_list.index_tree(best_tree_index)

        return best_tree_like

    def compute_norm_likelihood(self, cellsA, cellsB, mutsA, mutsB):
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

        total_like = self.like0[np.ix_(cellsA, mutsB)].sum(
        ) + self.like1[np.ix_(self.cells, mutsA)].sum()
        total = np.count_nonzero(self.like0[np.ix_(
            cellsA, mutsB)]) + np.count_nonzero(self.like1[np.ix_(self.cells, mutsA)])
        # total = len(cellsA)*len(self.muts) + len(cellsB)*len(mutsA)
        norm_like = total_like/total

        return norm_like
    
    # def compute_norm_likelihood(self,cellsA, cellsB, mutsA, mutsB):
    #     var_like = self.compute_norm_var_likelihood(cellsA, cellsB, mutsA, mutsB)
    #     rd_like =self.read_depth_likelihood_by_node(cellsA, cellsB)
    #     total_like = var_like + rd_like
    #     like = total_like.sum()
    #     return like
    def compute_norm_var_likelihood(self, cellsA, cellsB, mutsA, mutsB):
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
        ca_like = self.like0[np.ix_(cellsA, mutsB)].sum(axis=1) + self.like1[np.ix_(cellsA, mutsA)].sum(axis=1)
        total = np.count_nonzero(self.like0[np.ix_(cellsA, self.muts)],axis=1)
        ca_like = ca_like/total
        ca_like_series = pd.Series(ca_like, cellsA)

        cb_like =self.like1[np.ix_(cellsB, mutsA)].sum(axis=1)
        cb_total =np.count_nonzero(self.like0[np.ix_(cellsB, mutsA)], axis=1)

        cb_like_series = pd.Series(cb_like/cb_total, index=cellsB)

        var_like =pd.concat([ca_like_series, cb_like_series])

        return var_like

    def read_depth_likelihood_by_node(self, cellsA, cellsB):
        likes = []
        for cells in [cellsA, cellsB]:
            cluster_data = self.data.read_depth[cells,:]
            bin_means = cluster_data.mean(axis=0)
            bin_variance = np.diag(np.var(cluster_data, axis=0))
            mv_normal = multivariate_normal(bin_means, bin_variance,allow_singular=True)
            cell_likelihood =mv_normal.logpdf(cluster_data)
            cell_like_series = pd.Series(cell_likelihood, index=cells)
            likes.append(cell_like_series)
        cell_like_series = pd.concat(likes)

        return cell_like_series

    def sprout(self):
        """ main flow control to obtain the maximum likelihood linear tree
        
        Returns
        -------
        best_tree_like : LinearTree
            a LinearTree with maximum normalized likelihood among all restarts
      
        """

        # var_counts_by_snv= self.var[np.ix_(self.cells, self.muts)].sum(axis=0)
        # bad_snvs = self.muts[var_counts_by_snv==0]
        # self.muts = np.setdiff1d(self.muts, bad_snvs)
        
        # var_counts_by_cells = self.var[np.ix_(self.cells,self.muts)].sum(axis=1)
        # bad_cells = self.cells[var_counts_by_cells ==0]
        # self.cells = np.setdiff1d(self.cells, bad_cells)

        #check the average number of observations per cell and snv
        # cell_mean_val = check_obs(self.cells, self.muts, self.total, axis=1)
        # print(cell_mean_val)
        # snv_mean_val = check_obs(self.cells, self.muts, self.total, axis=0)
        # print(snv_mean_val)

        # if cell_mean_val <=  12 or snv_mean_val <= 12:
        #     return None
        
        # seed_obs = check_obs(self.cells, self.muts, self.total)
        # if seed_obs < self.params.minobs:
        #     return None


        self.norm_like_list = []

        self.cand_splits = ClonalTreeList()

        self.run(self.cells, self.muts, p=0.5)
        # if self.cand_splits.has_trees():
        #     best_tree, _ = self.cand_splits.find_best_tree(self.data)

        if self.cand_splits.size() == 0:
            return self.cand_splits, None

        best_tree_index = np.argmax(np.array(self.norm_like_list))

        best_tree_like = self.cand_splits.index_tree(best_tree_index)
        # cellsA = best_tree_like.cell_mapping[0][0]
        # best_tree_like.cell_mapping[0][0]=np.union1d(bad_cells, cellsA)
        # mutsA =  best_tree_like.mut_mapping[0]
        # best_tree_like.mut_mapping[0] = np.union1d(bad_snvs, mutsA)

        # return best_tree_like
        return self.cand_splits, best_tree_like
