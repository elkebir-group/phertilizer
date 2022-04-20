import numpy as np
from clonal_tree_list import ClonalTreeList
from utils import normalizedMinCut, check_stats, snv_kernel_width, cnv_kernel_width, impute_mut_features, find_cells_with_no_reads, find_muts_with_no_reads
from scipy.spatial.distance import pdist, squareform
import pandas as pd
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

    random_init(array, p)
        returns input array split into two arrays where entries have probability p of being in the first array

    calc_tmb(cells, muts)
        calculate the binary tumor mutational burden for each of the input cells over the set of input SNVs

    mut_assignment( cellsA, muts)
        returns two arrays mutsA, mutsB based on the maximum likelihood with the node assignment of each SNV for 
        the fixed input assignment of cellsA

    mut_assignment_with_cna( cellsA, muts)
        returns two arrays mutsA, mutsB based on the maximum likelihood with the node assignment, considering the current
        CNA genotype of each SNV for the fixed input assignment of cellsA


    create_affinity(cells, lamb= 0.5, mutsB=None)
        returns the edge weights w_ii' for the edge weights for the input graph to normalized min cut

    cluster_cells( cells, mutsB, weight=0.5)
        clusters an input set of cells into cellsA and cellsB using normalized min cut on a graph with edge
        weights mixing SNV and CNA features

    assign_cna_events(cellsA, cellsB)
        use the precomputed cnn_hmm to compute the most likely CNA genotypes for the given cell clusters cellsA, cellsB

    run_helper_with_cna( cells, muts, weight, p)
        a helper method to assist with finding recursively finding the maximum likelihood branching tree

    best_norm_like(tree_list, norm_like_list)
        for a given candidate ClonalTreeList of linear trees, identify and return the tree with maximum normalized likelihood


    compute_norm_likelihood(cellsA, cellsB, mutsA, mutsB):
        for a given cell and mutation partition, compute the normalized likelihood by excluding the mutsB, cellsB partitions
        and normalizing by the total observations


    sprout( seed, include_cna=True):
        main flow control to obtain the maximum likelihood branching tree from a given input seed


    """

    def __init__(self,
                 data,
                 cna_genotype_mode,
                 seed,
                 rng,
                 params
                 ):


        self.data = data

        self.cna_genotype_mode = cna_genotype_mode

        self.cells = seed.cells
        self.muts = seed.muts

        self.lamb = params.lamb
        self.tau = params.tau

        self.npass = params.npass
        self.rng = rng

        self.starts = params.starts
        self.iterations = params.iterations

        self.spectral_gap = params.spectral_gap
        self.jump_percentage = params.jump_percentage

        self.radius = params.radius

        self.states = ["gain", "loss", "neutral"]
        if self.data.snv_bin_mapping is not None:
            self.bin_mapping = {
                m: self.data.snv_bin_mapping.loc[m] for m in self.data.snv_bin_mapping.index}

    def random_init(self, arr):
        """Randomly splits an input array

        :return: three numpy arrays containing the disjoint partition of the input array
        """

        rand_vals = self.rng[0].random(arr.shape[0])
        arrA = arr[rand_vals < 1/3]
        arrB = arr[rand_vals > 2/3]
        arrC = np.setdiff1d(arr, np.concatenate([arrA, arrB]))
        return arrA, arrB, arrC

    def calc_tmb(self, cells, muts):

        mut_var_count_by_cell = np.count_nonzero(
            self.data.var[np.ix_(cells, muts)], axis=1).reshape(-1, 1)
        mut_total_reads_by_cell = np.count_nonzero(
            self.data.total[np.ix_(cells, muts)], axis=1).reshape(-1, 1)

        tmb = mut_var_count_by_cell/mut_total_reads_by_cell

        return tmb

    def create_affinity(self,  mutsA, mutsB, cells):

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
            if np.max(c_mat) > 1:
                copy_kernel = np.exp(-1*c_mat/cnv_kernel_width(c_mat))
                copy_kernel[c_mat >= r] = 0
                kernel = np.multiply(copy_kernel, kernel)

            else:
                print("Max copy distance < 1, using SNV kernel only")

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

        combined_matrix = np.vstack([mutsA, mutsB, mutsC])
        mut_assignments = np.argmax(combined_matrix, axis=0).reshape(-1)
        mut_assignments = mut_assignments[self.muts]

        mutsA = np.array(self.muts[mut_assignments == 0])
        mutsB = np.array(self.muts[mut_assignments == 1])
        mutsC = np.array(self.muts[mut_assignments == 2])

        return mutsA, mutsB, mutsC

    def cluster_cells(self, mutsA, mutsB, cells):

        W, mut_features = self.create_affinity(mutsA, mutsB, cells)
        if W is None:
            return cells, np.empty(shape=0, dtype=int), None

        if np.any(np.isnan(W)):
            print("Terminating cell clustering due to NANs in affinity matrix")
            return cells, np.empty(shape=0, dtype=int), None

        clusters, y_vals, labels, stats = normalizedMinCut(W, cells)

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

        return cellsA, cellsB, stats

    def mut_assignment(self, cellsA, cellsB):
        """ Updates the assignment of mutations to mutsA or mutsB by comparing 
        the likelihood based off the cells currently assigned to cellsA and cellsB.

        :return: a numpy arrays contain the mutation ids in mutsA and mutsB
        """

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

    def assign_cna_events(self, cellsA, cellsB):
        eventsA, eventsB = None, None
        if self.cna_genotype_mode:
            eventsA = self.data.cna_hmm.run(cellsA)

            eventsB = self.data.cna_hmm.run(cellsB)

        return eventsA, eventsB

    def run(self):

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

        if (len(cellsA) > self.lamb and len(cellsB) > 1) or (len(cellsB) > self.lamb and len(cellsA) > 1):

            if check_stats(stats, self.jump_percentage, self.spectral_gap, self.npass):

                cand_tree = BranchingTree(
                    cellsA, cellsB, mutsA, mutsB, mutsC, eA, eB, eC=None)
                print(cand_tree)
                self.cand_trees.insert(cand_tree)

    def sprout(self):
        """ Driver function for performing a branched bipartition by looping over multiple
        restarts, multiple iterations and alternating between cell and mutation assignments
        to find the split that has the highest normalized likelihood.


        :returns a dictionary of cell assignments, a dictionary of mutation assignments
        and the normalized likelihood. 
        """

        self.cand_trees = ClonalTreeList()

        for i in range(self.starts):

            self.run()

        if self.cand_trees.has_trees():
            best_branching_tree, _ = self.cand_trees.find_best_tree(self.data)

            return best_branching_tree
        else:
            return None
