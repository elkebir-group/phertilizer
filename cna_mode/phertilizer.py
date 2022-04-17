from numba.core.dispatcher import LiftedLoop
from copy import deepcopy
import numpy as np
import pandas as pd
from cna_events import CNA_HMM
import scipy.special
from scipy.spatial.distance import pdist, squareform
import numba
import branching_split as bs
import linear_split as ls
from seed import Seed
from clonal_tree import IdentityTree
from collections import deque
from clonal_tree_list import ClonalTreeList


@numba.jit(nopython=True)
def binomial(n,k):
    return 1 if k==0 else (0 if n==0 else binomial(n-1, k) + binomial(n-1, k-1))

@numba.jit(nopython=True)
def binom_pdf(k,n,p, coeff):
    """Computes the probability mass function of the binomial distribution

    :param k: the number of successes
    :param n: the the number of trials
    :param p: the probability of success
    :return: number. P(X=k), where X~Bin(n,p)
    """
    #coeff= factorial(n)/(factorial(k)*factorial(n-k))
    # coeff = binomial(n,k)
    prob = coeff * (p**k) *(1-p)**(n-k)
    return prob

@numba.jit(nopython=True)
def factorial(n):
    """Computes the factorial of a number

    :param n: a number

    :return: number. n!
    """
    return(np.prod(np.arange(1,n+1,1)))

@numba.jit(nopython=True)
def likelihood1_numba(x, n, vafs, vaf_prob, coeff):
    """Computes the probability mass function of the binomial distribution

    :param k: the number of successes
    :param n: the the number of trials
    :param p: the probability of success
    :return: number. P(X=k), where X~Bin(n,p)
    """
    prob = 0
   
    for i  in range(len(vafs)):
        prob += binom_pdf(x,n, vafs[i], coeff)*vaf_prob[i]
    

    return prob


@numba.jit(nopython=True)
def apply_like1_numba(var, total, min_copies, max_copies,  alpha, coeff):
    """Helper function that loops through two numpy arrays and computes the 
    the likelihood that y=1 (variant present) for each paired enty.

    :param var: a numpy array of variant counts
    :param total: a numpy array of total read counts
    :param max_copies: an integer representing the max number of copies at each locus
    :param alpha: a float that represents the per base false positive read error rate
    :return: a numpy array containing the likelihood a variant is present for each locus
    """

    n = len(var)
    vafs_all = np.array([i/c for c in np.arange(min_copies, max_copies+1, 1) for i in np.arange(1, c+1, 1 )])

    vafs = np.unique(vafs_all)
 

    vaf_probs = np.zeros_like(vafs)
    vafs_prime =np.zeros_like(vafs)

    for i in range(len(vafs)):
        vaf_probs[i] = np.count_nonzero(vafs_all ==vafs[i])/len(vafs_all)
        vafs_prime[i] = vafs[i]*(1- alpha) + (1-vafs[i])*(alpha/3)

    
    result = np.empty(n, dtype=np.float64)
    assert len(var) == len(total) == n
    for i in range(n):
        result[i] = likelihood1_numba(var[i], total[i], vafs_prime, vaf_probs, coeff[i])
    return result

@numba.jit(nopython=True)
def apply_like0_numba(var, total, alpha, coeff):
    """Wrapper function that loops through two numpy arrays and computes the 
    the likelihood that y=0 (variant no present) for each paired enty.

    :param var: a numpy array of variant counts
    :param total: a numpy array of total read counts
    :param alpha: a float that represents the per base false positive read error rate
    :return: a numpy array containing the likelihood a variant is not present for each locus
    """
    n = len(var)
    result = np.empty(n, dtype=np.float64)
    assert len(var) == len(total) == n

    for i in range(n):
        # coeff = binomial(total[i], var[i])
        result[i] = binom_pdf(var[i], total[i], alpha, coeff[i])
    return result

def compute_like0(df, alpha, coeff):
    """Wrapper function that computes the likelihood no variant is present for each row in the dataframe.

    :param df: a pandas dataframe that at least contains two columns named "var" and "total"
    :param alpha: a float that represents the per base false positive read error rate
    :return: a pandas series for the likelihood a variant is not present for each locus
    """

    result = apply_like0_numba(
        df["var"].to_numpy(), df["total"].to_numpy(), alpha, coeff
    )

    return pd.Series(np.log(result), index=df.index, name="like0")




def compute_like1(df, min_copies, max_copies,  alpha, coeff):
    """Wrapper function that computes the likelihood a variant is present for each row in the dataframe.

    :param df: a pandas dataframe that at least contains two columns named "var" and "total"
    :param alpha: a float that represents the per base false positive read error rate
    :return: a pandas series for the likelihood a variant is present for each locus
    """

    result = apply_like1_numba(
            df["var"].to_numpy(), df["total"].to_numpy(), min_copies, max_copies,  alpha, coeff
        )
     

    return pd.Series(np.log(result), index=df.index, name="like1")

def sparse_matrix(series: pd.Series):
    """Creates a numpy atrix from a pandas series with cells as rows and mutations as columns

    :param a pandas series: series with multi level index cell and mutation
    :return: an n x m numpy array
    """

    sparse_mat = series.unstack(level=1, fill_value=0).to_numpy()
    return sparse_mat

class Phertilizer:
    """
    A class to grow a clonal tree with SNV/CNA genotypes and cell attachments from ultra-low coverage sequencing data

    ...

    Attributes
    ----------
    states : tuple
        the names of allowable CNA genotype states ("gain", "loss", "neutral")
    n : int
        the number of cells
    m : int
        the number of SNVs
    data : dict
        a dictionary containing the processed input data 
    lamb ; int 
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
    neutral_mean : float
        mean hyperparameter for fitting the emission distributions for the CNA genotype 
        HMM. (default 1.0)
    neutral_eps : float
        starndard deviation hyperparameter for fitting the emission distributions for the CNA genotype 
        HMM. (default 1.0)
    debug : boolean
        turn on debugging mode (not used)
    include_cna : boolean
        indicates if CNA genotypes should be inferred (always TRUE in this mode)
    

    Methods
    -------

    get_id_mappings()
        returns the mapping of the internal cell and SNV index to the input labels
 
    till(max_copies, alpha, seed =10262022, dim_reduction = False, neutral_mean=1.0, neutral_eps=0.15)
        preprocesses the input data to make the likelihood computation more efficient and 
        prepares intial seed
    
        
    planting(seed_list)
        from a seed_list with a single seed containing all cells and mutations, 
        performs linear, branching operations on all seeds, storing the 
        elementary clonal trees for enumeration

    grow(max_trees=np.Inf)
         memoization is used to enumerate all clonal trees from the list of explored seeds
    phertilize(alpha, max_copies, min_cell, min_snvs, max_iterations, starts, 
                neutral_mean, neutral_eps, seed=10261982, dim_reduction=False, radius=0.5, npass=1)
        main flow control of class that handles directs tilling,planting and growing and 
        returns the maximum likelihood clonal tree, along with a ClonalTreeList of all grown trees

    """


    def __init__(self, variant_count_data, 
                      bin_count_data = None,
                      bin_count_normal = None,
                      snv_bin_mapping = None,
                      debug= False,
                      ):
   

        self.data = {}


        self.lamb = None 
        self.tau = None 
        self.iterations = None
        self.starts = None 
        self.radius = None 
        self.npass = None 
        self.neutral_mean = None
        self.neutral_eps = None 
        self.debug = debug

        #TODO: add as input parameters
        self.spectral_gap = 0.05
        self.jump_percentage = 0.075

    
        self.states = ("gain", "loss", "neutral")

        self.include_cna = True
        self.use_cna_info = True

        self.mapping_list = None
        self.explored_seeds = None

        self.min_loss =0

        #wrangle input data into the correct format
        variant_count_data['chr_mutation'] = variant_count_data['chr'].astype('str') + "_" + variant_count_data['mutation_id'].astype(str)
        self.cell_labels = np.sort(variant_count_data['cell_id'].unique())
        self.mut_labels = np.sort(variant_count_data['chr_mutation'].unique())

    
        self.n = self.cell_labels.shape[0]
        self.m = self.mut_labels.shape[0]
        self.cells = np.arange(self.n,dtype="int")
        self.muts = np.arange(self.m, dtype="int")

        self.cell_series = pd.Series(data=self.cells, index= self.cell_labels)
        self.cell_lookup = pd.Series(data=self.cell_labels, index=self.cells)        
    
        self.mut_series = pd.Series(data=self.muts, index =self.mut_labels )

        self.data["mutation_mapping"] = self.mut_series
        self.data["cell_mapping"] = self.cell_series

        self.mut_lookup = pd.Series(data=self.mut_labels, index=self.muts)

        variant_count_data['mutation'] = self.mut_series[variant_count_data['chr_mutation']].values
        variant_count_data['cell'] = self.cell_series[variant_count_data['cell_id']].values
        self.variant_count_data = variant_count_data

        self.variant_count_data= self.variant_count_data.set_index(["cell", "mutation"])

        self.snv_bin_mapping = snv_bin_mapping
     

        self.snv_bin_mapping['chr_mutation'] = self.snv_bin_mapping['chr'].astype('str') + "_" + self.snv_bin_mapping['mutation_id'].astype(str)
        
        self.snv_bin_mapping['mutation'] = self.mut_series.loc[self.snv_bin_mapping['chr_mutation']].values
        
        self.snv_bin_mapping = self.snv_bin_mapping.set_index("mutation").sort_index()
        self.snv_bin_mapping = self.snv_bin_mapping['bin']

        self.data['snv_bin_mapping'] = self.snv_bin_mapping
      

   
        if self.use_cna_info:
            bin_count_data['cell_index'] = self.cell_series[bin_count_data['cell']].values
            bin_count_data.sort_values(by=['cell_index'])

            
            bin_count_data.drop(['cell', 'cell_index'], inplace=True, axis=1)
            bin_count_raw =   bin_count_data.div(bin_count_data.sum(axis=1), axis=0)
            self.bin_count = bin_count_data.to_numpy()
            self.bin_count_raw = bin_count_raw.to_numpy()
          
            
            bin_count_normal = bin_count_normal.drop(['cell'], axis=1)
            bin_count_normal=   bin_count_normal.div(bin_count_normal.sum(axis=1), axis=0)
                
            self.bin_normal = bin_count_normal.median()

            self.bins = np.arange(0, self.bin_normal.shape[0])
            self.bin_mapping = pd.Series(data=self.bin_normal.index, index=self.bins)

            bin_locs = self.bin_mapping.to_list()
            chromosomes = [c.split(".")[0] for c in bin_locs]
            chromosomes = [int(c[3:]) for c in chromosomes]
            self.chrom_series = pd.Series(chromosomes, self.bin_mapping.index)
            self.data['chrom_bin_mapping'] = self.chrom_series
         
      
           
            self.bin_normal = pd.Series(data=self.bin_normal.values, index= self.bins)
         

            self.baseline = self.bin_normal.values
          
            self.rdr = self.bin_count_raw/self.baseline[ None, :]
            self.data['RDR'] = self.rdr  

   
    def get_id_mappings(self):
        return self.cell_lookup, self.mut_lookup
        


    def till(self, max_copies, alpha, seed  =10262022, neutral_mean=1.0, neutral_eps=0.15):
   

   

        #intialize the random number generator with the given seed
        self.rng = np.random.default_rng(seed)

        #compute distance matrix for binned read count data
        if  self.use_cna_info:
      
    
                self.copy_distance = squareform(pdist(self.rdr, metric="euclidean"))


     
        #precompute local likelihoods 
        coeff_array = scipy.special.comb(self.variant_count_data['total'].to_numpy(),self.variant_count_data['var'].to_numpy())
        self.likelihood_dict = {}
        self.like0 = sparse_matrix(compute_like0(self.variant_count_data, alpha, coeff_array))

        #fully marginalized array        
        like1_series= compute_like1(self.variant_count_data, 1, max_copies,  alpha, coeff_array)
        self.like1_marg = sparse_matrix(like1_series)
        if self.include_cna:
            self.like1 = {}

            for s in self.states:
                
                if s == "gain":
                    min_copies = 3
                    max_copies = max_copies
                elif s== "loss":
                    min_copies = 1
                    max_copies = 1
                else:
                    min_copies = 2
                    max_copies = 2

                self.like1[s] = sparse_matrix(compute_like1(self.variant_count_data, min_copies, max_copies,  alpha, coeff_array))
        else:
                self.like1 = sparse_matrix(like1_series)
        

        var= pd.Series(self.variant_count_data['var'].to_numpy(), index=self.variant_count_data.index)
        self.var = sparse_matrix(var)

        self.data['variant_counts'] = self.var


        total= pd.Series(self.variant_count_data['total'].to_numpy(), index=self.variant_count_data.index)
        self.total = sparse_matrix(total)

        self.data['total_counts'] = self.var

    
        seed_list = deque()

        self.init_seed = Seed(self.cells, self.muts, key=0)
        seed_list.append(self.init_seed)
     
        #initialize  and fit CNA HMM
        if self.include_cna:
            self.neutral_mean = neutral_mean 
            self.neutral_eps =neutral_eps
            self.cnn_hmm = CNA_HMM(self.bins, self.rdr, self.chrom_series, 
                                    neutral_mean=self.neutral_mean, neutral_eps=self.neutral_eps,
                                    debug= self.debug )
    
        data = {}
        data["like1_marg"] =self.like1_marg
        data["like1_dict"] = self.like1
        data["like0"] = self.like0 
        data["cna_hmm"] = self.cnn_hmm
        data["cell_lookup"] = self.cell_lookup
        data["mut_lookup"] = self.mut_lookup
        data["snv_bin_mapping"] = self.snv_bin_mapping


        return seed_list, data



    def phertilize(self, 
                    alpha, 
                    max_copies,
                    min_cell, 
                    min_snvs,
                    max_iterations, 
                    starts,
                    neutral_mean, 
                    neutral_eps, 
                    seed=10261982, 
                    radius=0.5, 
                    npass=1):


        """Recursively builds a clonal tree with SNV/CNA genotypes and cells attached to nodes

        :return: a clonal tree with maximum likelihood, a ClonalTreeList of all enumerated trees
        """
        n = len(self.cells)
        if type(min_cell) == float:
          

            self.lamb = int(min_cell*n)
          
        else:
            self.lamb = min_cell
        
        m= len(self.muts)
        if type(min_snvs) == float:
          
            self.tau = int(min_snvs*m)
          
        else:
            self.tau = min_snvs


        self.iterations = max_iterations
    
        self.starts = starts
 
     

        self.radius = radius 
        self.npass =npass
  
        self.neutral_mean = neutral_mean
        self.neutral_eps = neutral_eps 

        #TODO: add as input parameters
        self.spectral_gap = 0.05
        self.jump_percentage = 0.075
       
      
        ###tilling phase
        print("\nStarting tilling phase....")
        seed_list, data = self.till(max_copies, alpha,seed,self.neutral_mean, self.neutral_eps)
        print("Tilling phase complete!")
     
        print(f"Phertilizing a tree with n:{len(self.cells)} cells and m: {len(self.muts)} SNVs")

        #planting phase
        print("\nStarting planting phase....")
        self.planting(seed_list)
        print("Planting phase complete!")


        #growing phase
        print("\nStarting grow phase....")
        pre_proc_trees = self.grow()
        print("Growing phase complete!")

   
        print("\nIdentifying the maximum likelihood k-clonal tree....")
       
        optimal_tree, loglikelihoods = pre_proc_trees.find_best_tree(self.like0, self.like1, self.snv_bin_mapping, self.cnn_hmm)
        
        return optimal_tree, pre_proc_trees

    


      
    def lookup_seed_key(self, seed):

        for s in self.explored_seeds:
            if self.explored_seeds[s] == seed:
                return s


    def grow(self, max_trees=np.Inf):


        pre_proc_list = ClonalTreeList()
        cand_trees = ClonalTreeList()
        trees = self.mapping_list[0]
        for t in trees:
            
                pre_proc_list.insert(t)
                if type(t) is not IdentityTree:
                    cand_trees.insert(t)
                while cand_trees.has_trees():

                    if pre_proc_list.size() > max_trees:
                        break
                   
                    curr_tree = cand_trees.pop_tree()
                    leaf_nodes = curr_tree.get_leaves()
                    for l in leaf_nodes:
                        anc_muts = curr_tree.get_ancestral_muts(l)
                        seed = curr_tree.get_seed_by_node(l, self.lamb, self.tau, anc_muts)
                        if seed is None:
                            continue
                        seed_key = self.lookup_seed_key(seed)
                        if seed_key is not None:
                            seeded_trees = self.mapping_list[seed_key]
                        
                        #merge the two trees together 
                            for st in seeded_trees:
                          
                                if type(st) is not IdentityTree:
                                    merged_tree = deepcopy(curr_tree)
                                    merged_tree.merge(st, l)
                                    print(merged_tree)

                                    if not pre_proc_list.contains(merged_tree):
                                        pre_proc_list.insert(merged_tree)
                                    if not cand_trees.contains(merged_tree):    
                                        cand_trees.insert(merged_tree)
                                    
                              
                             
                                        
        
        return pre_proc_list

        

    
    def planting(self, seed_list):

        
        key = 0
        self.mapping_list = {}
        self.explored_seeds = {}
  
        # loop till stack is empty
        while len(seed_list) > 0:
           
            curr_seed = seed_list.pop()
            curr_seed.set_key(key)

      

            if len(curr_seed.cells) < 2*self.lamb or len(curr_seed.muts) < 2*self.tau:
                continue
            print("Starting k-clonal tree inference for seed:")
            print(curr_seed)
            self.explored_seeds[key] =curr_seed
    
        
            self.mapping_list[key] = []

                   
            sprout_list = [ self.sprout_branching, self.sprout_linear]


   
            for sprout in sprout_list:

                print("Sprouting seed:")
                print(curr_seed)

                #infer an elementary k-clonal tree from a given seed
                tree = sprout(curr_seed)

              
                print("Sprouted tree:")
                print(tree)
                if tree is not None:
                    
                    if tree.is_valid(self.lamb, self.tau, self.min_loss):
             
                        self.mapping_list[key].append(tree)
                        new_seeds = tree.get_seeds(self.lamb, self.tau, curr_seed.ancestral_muts)
                        for seed in new_seeds:
                            print(seed)
                            seed_list.append(seed)
                    else:
                        print(f"{sprout} Not Valid!")
                        
                else:
                    print(f"{sprout} invalid")
                    
           
            print(f"Number of subproblems remaining: {len(seed_list)}")
            key += 1
        
    
    def sprout_branching(self,seed):
        """performs a branching operation from a given input seed
        @param: seed a Seed object containing cells, and SNVs

        :return: an elementary branched clonal tree
         """
        print("Sprouting a branching tree..")  
        
        #perform a branching split
        br_split = bs.Branching_split(self.like0, 
                                      self.like1_marg,
                                      self.like1, 
                                      self.var,  
                                      self.total,
                                      seed.cells, 
                                      seed.muts, 
                                      self.rng,  
                                      self.lamb,
                                      self.tau, 
                                      self.starts, 
                                      self.iterations,
                                      weights= [0.5],
                                      spectral_gap = self.spectral_gap,
                                      jump_percentage = self.jump_percentage,
                                      copy_matrix=self.copy_distance,
                                      cnn_hmm = self.cnn_hmm,
                                      snv_bin_mapping = self.snv_bin_mapping,
                                      radius = self.radius,
                                      npass =self.npass,
                                      debug=self.debug,
                                 
                
                                    )
        
        branching_tree = br_split.sprout(self.include_cna)

        return branching_tree

        
    def sprout_linear(self, seed):
        """performs a linear operation from a given input seed
        @param: seed a Seed object containing cells, and SNVs

        :return: an elementary linear clonal tree
         """

        cells = seed.cells
        muts = seed.muts
        print("Sprouting a linear tree from seed")    
        lin_split =ls.Linear_split(
                                        self.like0, 
                                        self.like1_marg, 
                                        self.like1,
                                        self.var, 
                                        self.total, 
                                        cells,
                                        muts,
                                        self.rng,  
                                        starts=self.starts,
                                        iterations=self.iterations, 
                                        weights = [0.5],
                                        spectral_gap= self.spectral_gap,
                                        jump_percentage= self.jump_percentage,
                                        lamb = self.lamb,
                                        tau = self.tau,
                                        copy_distance_matrix=self.copy_distance,
                                        cnn_hmm= self.cnn_hmm,
                                        snv_bin_mapping = self.snv_bin_mapping,
                                        radius = self.radius,
                                        npass =self.npass,
                                        debug=self.debug,
                                    )
        
        best_tree = lin_split.sprout(seed, self.include_cna)


        return best_tree
       

    def sprout_identity(self, seed):
        """performs an identity operation from a given input seed
        @param: seed a Seed object containing cells, and SNVs

        :return: an elementary identity clonal tree
         """
       
        events= self.cnn_hmm.run(seed.cells)


        return IdentityTree(seed.cells, seed.muts, events)
      

        