import numpy as np
from clonal_tree_list import ClonalTreeList
from utils import normalizedMinCut, check_stats, snv_kernel_width, cnv_kernel_width, impute_mut_features, find_cells_with_no_reads, find_muts_with_no_reads
from scipy.spatial.distance import pdist, squareform
import numba as nb
import pandas as pd
from clonal_tree import BranchingTree
from clonal_tree_list import ClonalTreeList


class Branching_split():
    """
    A class to perform a branching tree operation on a given seed and input data

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

    create_affinity(cells, lamb= 0.5, mutsB=None)
        returns the edge weights w_ii' for the edge weights for the input graph to normalized min cut

    cluster_cells( cells, mutsB, weight=0.5)
        clusters an input set of cells into cellsA and cellsB using normalized min cut on a graph with edge
        weights mixing SNV and CNA features

    run()
        a helper method to assist with finding the maximum likelihood branching tree

    sprout( seed):
        main flow control to obtain the maximum likelihood branching tree from a given input seed


    """

    def __init__(self,
                 like0,
                 like1,
                 var,
                 total,
                 cells,
                 muts,
                 rng,
                 lamb=100,
                 tau=100,
                 starts=7,
                 iterations=20,
                 weights=[0.5],
                 spectral_gap=0.05,
                 jump_percentage=0.03,
                 copy_matrix=None,
                 radius=0.5,
                 npass=1,
                 debug=False
                 ):

        # set data
        self.like0 = like0
        self.like1 = like1

        self.var = var
        self.total = total

        self.cells = cells
        self.muts = muts

        self.lamb = lamb
        self.tau = tau

        self.rng = rng,

        self.starts = starts
        self.iterations = iterations
        self.weights = weights

        self.npass = npass

        self.spectral_gap = spectral_gap
        self.jump_percentage = jump_percentage

        self.radius = radius
        self.debug = debug

        self.copy_distance_matrix = copy_matrix

    def random_init(self, arr):
        """Randomly splits an input array

        :return: two numpy arrays containing the split of the input array
        """

        rand_vals = self.rng[0].random(arr.shape[0])
        arrA = arr[rand_vals < 1/3]
        arrB = arr[rand_vals > 2/3]
        arrC = np.setdiff1d(arr, np.concatenate([arrA, arrB]))
        return arrA, arrB, arrC

    def calc_tmb(self, cells, muts):

        mut_var_count_by_cell = np.count_nonzero(
            self.var[np.ix_(cells, muts)], axis=1).reshape(-1, 1)
        mut_total_reads_by_cell = np.count_nonzero(
            self.total[np.ix_(cells, muts)], axis=1).reshape(-1, 1)
        tmb = mut_var_count_by_cell/mut_total_reads_by_cell

        return tmb

    def create_affinity(self,  mutsA, mutsB, mutsC, cells, weight=0.5):

        if len(mutsA) > 0 and len(mutsB) > 0:
            tmb_mb = self.calc_tmb(cells, mutsB)

            tmb_ma = self.calc_tmb(cells, mutsA)

            mut_features = np.hstack([tmb_ma, tmb_mb])

            copy_dist_mat = self.copy_distance_matrix.copy()

            mut_features = impute_mut_features(
                copy_dist_mat, mut_features, cells)

            # if we aren't able to impute mut features for cells, then we can't cluster
            if np.any(np.isnan(mut_features)):
                return None, None

            mut_features_df = pd.DataFrame(mut_features, cells)

            d_mat = squareform(pdist(mut_features, metric="euclidean"))

            kw = snv_kernel_width(d_mat)
            kernel = np.exp(-1*d_mat/kw)

            if self.copy_distance_matrix is not None:
                c_mat = self.copy_distance_matrix[np.ix_(cells, cells)]

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

    def cluster_cells(self, mutsA, mutsB, mutsC, cells):

        W, mut_features = self.create_affinity(mutsA, mutsB, mutsC, cells)
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

        like1 = self.like1

        mutsA = []
        mutsB = []
        mutsC = []

        mutsA_cand = like1[np.ix_(cellsA, self.muts)].sum(
            axis=0) + self.like0[np.ix_(cellsB, self.muts)].sum(axis=0)
        mutsB_cand = self.like0[np.ix_(cellsA, self.muts)].sum(
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
        cells = self.cells
        muts = self.muts
        mutsA, mutsB, mutsC = self.random_init(muts)
        oldA = np.empty(shape=0, dtype=int)
        for j in range(self.iterations):
            cellsA, cellsB, stats = self.cluster_cells(
                mutsA, mutsB, mutsC, cells)

            if len(cellsB) == 0 or len(cellsA) == 0 or np.array_equal(np.sort(cellsA), np.sort(oldA)):
                break
            else:
                oldA = cellsA
            print(f"CellsA: {len(cellsA)} CellsB: {len(cellsB)}")

            mutsA, mutsB, mutsC = self.mut_assignment(cellsA, cellsB)
            print(f"MutsA: {len(mutsA)} mutsB: {len(mutsB)}")

        if (len(cellsA) > self.lamb and len(cellsB) > 1) or (len(cellsB) > self.lamb and len(cellsA) > 1):

            if check_stats(stats, self.jump_percentage, self.spectral_gap, 0):

                cand_tree = BranchingTree(cellsA, cellsB, mutsA, mutsB, mutsC)
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
            best_branching_tree, _ = self.cand_trees.find_best_tree(
                self.like0, self.like1)

            return best_branching_tree
        else:
            return None
