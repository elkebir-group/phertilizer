from copy import deepcopy
import numpy as np
import pandas as pd
import scipy.special
from scipy.spatial.distance import pdist, squareform
import numba
import umap

from utils import pickle_save, check_obs, power_calc, pickle_load

# from phertilizer.cna_events import CNA_HMM
# import phertilizer.branching_split as bs
# import phertilizer.linear_split as ls
# from phertilizer.seed import Seed
# from phertilizer.data import Data
# from phertilizer.params import Params
# from phertilizer.clonal_tree import IdentityTree
# from collections import deque
# from phertilizer.clonal_tree_list import ClonalTreeList

from cna_events import CNA_HMM
import branching_split as bs
import linear_split as ls

from data import Data
from params import Params
from clonal_tree import IdentityTree, LinearTree, Seed
from collections import deque
from clonal_tree_list import ClonalTreeList
import matplotlib.pyplot as plt



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
    data : Data
        a object of class Data containing all transformed input data for Phertilizer
    cells : np.array
        the cell indices to be clustered
    muts : np.array
        the SNV indices to be clustered
    params : Params
        an object of class Params containing all needed parameters 
    cna_genotype_mode : boolean
        indicates if CNA genotypes should be inferred
    

    Methods
    -------

    get_id_mappings()
        returns the mapping of the internal cell and SNV index to the input labels
    
    planting(seed_list)
        from a seed_list with a single seed containing all cells and mutations, 
        performs linear, branching operations on all seeds, storing the 
        elementary clonal trees for enumeration

    grow(max_trees=np.Inf)
         memoization is used to enumerate all clonal trees from the list of explored seeds
    
    phertilize( min_cell, min_snvs, max_iterations, starts, 
                neutral_mean, neutral_eps, seed=10261982, radius=0.5, npass=1)
        main flow control of class that handles directs tilling,planting and growing and 
        returns the maximum likelihood clonal tree, along with a ClonalTreeList of all grown trees
    
    sprout_branching(seed)
        performs a branching operation on a given input seed
    
    sprout_linear(seed)
        performs a linear operation on a given input seed

    sprout_identity(seed)
         performs an identity operation on a given input seed

    """


    def __init__(self, variant_count_data, 
                      bin_count_data,
                      alpha = 0.001,
                      max_copies = 5,
                      mode  = "var",
                      dim_reduce = True
                      ):
   
        if mode == "var":
            use_read_depth =False
        else:
            use_read_depth = True
        
        
        self.mapping_list = None
        self.explored_seeds = None


        #wrangle input data into the correct format
        variant_count_data['chr_mutation'] = variant_count_data['chr'].astype('str') + "_" + \
             variant_count_data['mutation_id'].astype(str)
        cell_labels = np.sort(variant_count_data['cell_id'].unique())
        mut_labels = np.sort(variant_count_data['chr_mutation'].unique())

       
        self.cells = np.arange(cell_labels.shape[0],dtype="int")
        self.muts = np.arange(mut_labels.shape[0], dtype="int")

        cell_series = pd.Series(data=self.cells, index= cell_labels)
        cell_lookup = pd.Series(data=cell_labels, index=self.cells)        
    
        mut_series = pd.Series(data=self.muts, index =mut_labels )
        mut_lookup = pd.Series(data=mut_labels, index=self.muts)

        variant_count_data['mutation'] = mut_series[variant_count_data['chr_mutation']].values
        variant_count_data['cell'] = cell_series[variant_count_data['cell_id']].values
        
        variant_count_data= variant_count_data.set_index(["cell", "mutation"])

   


    
          
       
        bin_count_data['cell_index'] = cell_series[bin_count_data['cell']].values
        bin_count_data = bin_count_data.sort_values(by=['cell_index'])
        bin_count_data.drop(['cell', 'cell_index'], inplace=True, axis=1)
        bin_count_data = bin_count_data.to_numpy()
        bin_count_data = bin_count_data/(bin_count_data.sum(axis=1).reshape(-1,1))

        if dim_reduce:
      
            reducer = umap.UMAP(n_neighbors=40,
                                min_dist=0,
                                n_components=2,
                                spread=1,
                                metric="manhattan",
                                random_state=55)
            embedding = reducer.fit_transform(bin_count_data)
  

          

            # plt.scatter(
            #                 embedding[:, 0],
            #                 embedding[:, 1])
            # plt.gca().set_aspect('equal', 'datalim')
            # plt.title('UMAP projection of Read Depth', fontsize=24)
        else:
            embedding= bin_count_data

        copy_distance = squareform(pdist(embedding, metric="euclidean"))
        cnn_hmm = None
        
     
     
        #precompute local likelihoods 
        coeff_array = scipy.special.comb(variant_count_data['total'].to_numpy(),variant_count_data['var'].to_numpy())
        self.likelihood_dict = {}
        like0 = sparse_matrix(compute_like0(variant_count_data, alpha, coeff_array))

        #fully marginalized array        
        like1_series= compute_like1(variant_count_data, 1, max_copies,  alpha, coeff_array)
        like1_marg = sparse_matrix(like1_series)
        like1_dict = None

        var= sparse_matrix(pd.Series(variant_count_data['var'].to_numpy(), index=variant_count_data.index))
    
        total= sparse_matrix(pd.Series(variant_count_data['total'].to_numpy(), index=variant_count_data.index))
       


        
        self.data = Data(cell_lookup, 
                        mut_lookup,
                        var,
                        total,
                        like0,
                        like1_marg,
                        embedding,
                        like1_dict,
                        copy_distance,
                        cnn_hmm,
                        use_read_depth,
                        )
    

    def save_embedding(self, fname):
        '''saves the coordinates of the umap embedding to a files

        Parameters
        -------
        fname : str
         the filename where the embedding should be saves

        '''
        emb_dat = self.data.read_depth

        cells = self.data.cell_lookup.sort_index().values.reshape(-1,1)
  
        emb_dat = np.hstack([cells, emb_dat])
        df = pd.DataFrame(emb_dat, columns=["cell", "V1", "V2"])
        df.to_csv(fname, index=False)

    def get_id_mappings(self):
        '''Gets the mapping of the internal cell and SNV indices to the supplied labels

        Returns
        -------
        cell_lookup
            a pandas series with internal cell indices as the index and given cell labels as data
        mut_lookup
            a pandas series with internal SNV indices as the index and given SNV labels as data

        '''
        return self.data.cell_lookup, self.data.mut_lookup
        




    def phertilize(self, 
                    max_iterations =10, 
                    starts =3,
                    seed=1026, 
                    radius=0.975, 
                    gamma=0.95,
                    d=7,
                    use_copy_kernel=False,
                    post_process=False):
        '''Recursively enumerates all clonal trees on the given input data and
        identifies the clonal tree with maximum likelihood 

        Parameters
        ----------
        
        max_iterations : int
            the maximum number of iterations for each tree operation if convergence is not met
            (default : 10)
        
        starts : int
            the number of restarts to conduct within in each tree operation  (default : 3)
        
        seed : int
            a seed to initialize the random number generator (default: 1026)
        
        radius : float
            a parameter for determining similarity of rdr(bin_count) data between cells (default 0.5)
        

        Returns
        -------
        optimal_tree
            a Clonal Tree object with maximum likelihood
        
        pre_proc_trees 
            a ClonalTreeList object containing the enumeration of all clonal trees
        
        loglikelihoods
            a pandas Series with the tree keys as index and the loglikelihood of each tree

        '''

        Nmax = check_obs(self.cells, self.muts, self.data.total)
        minobs = power_calc(d, gamma, 6, Nmax)
        self.params = Params()
        self.rng = np.random.default_rng(seed)
  


        self.params = Params(starts, 
                            max_iterations, 
                            radius, 
                            minobs,
                            seed,
                            use_copy_kernel,
                            post_process)
   

        seed_list = deque()

        self.init_seed = Seed(self.cells, self.muts, key=0)
        seed_list.append(self.init_seed)

        self.mapping_list = {}
        self.explored_seeds = {}
             
        print(f"\nPhertilizing a tree with n: {len(self.cells)} cells and m: {len(self.muts)} SNVs")

        #planting phase
        print("\nStarting planting phase....")
        self.planting(seed_list)
        print("\nPlanting phase complete!")


        #growing phase
        print("\nStarting grow phase....")
        pre_proc_trees = self.grow()
        print("\nGrowing phase complete!")

   
        print("\nIdentifying the maximum likelihood clonal tree....")
       
        optimal_tree, loglikelihoods = pre_proc_trees.find_best_tree(self.data)

        if optimal_tree is None:
            optimal_tree =IdentityTree(self.cells, self.muts)
  
        if post_process:
            print("Post-processing...")
            loglike = optimal_tree.post_process(self.data)# 0.1, 20)
        
     

        return optimal_tree, pre_proc_trees, loglikelihoods

    
    def planting(self, seed_list):
        '''Performs all possible elementary tree operations on every possible seed (leaf) node
        and memoizes the results for enumerating the candidate set of clonal trees 

        Parameters
        ----------
        seed_list : dequeue
            a dequeue containing all initial seeds
        

        '''
        
        key = 0
    
        while len(seed_list) > 0:
            # if key > 0:
            #     break
           
            curr_seed = seed_list.pop()
            curr_seed.set_key(key)
            print(curr_seed)
           
            self.mapping_list[key] = []
            
            self.explored_seeds[key] = deepcopy(curr_seed)
         
            if curr_seed.has_linear():
                sprout_list = [self.sprout_branching] 
 
                self.mapping_list[key].append(curr_seed.linear_tree)

            elif curr_seed.has_branching():
                sprout_list = [self.sprout_linear]
                self.mapping_list[key].append(curr_seed.branching_tree)
            
            else:
                sprout_list =[ self.sprout_linear, self.sprout_branching]

      
            curr_seed.strip(self.data.var)
            nobs = curr_seed.count_obs(self.data.total)
           
            if nobs < self.params.minobs:
             
                print("Insufficient number of observations, seed is terminal")
                continue
            
            print("Starting k-clonal tree inference for seed:")
      
    
           
            for sprout in sprout_list:

                print("Sprouting seed:")
                print(curr_seed)

                #perform an elementary tree operation from the given seed
                tree, new_seeds = sprout(curr_seed)

                if tree is not None:
                    
        
                    print("\nSprouted tree:")
                    print(tree)
                    self.mapping_list[key].append(tree)
                    seed_list += new_seeds
                else:
                    print("No inferred elementary tree.")
                 
           
            print(f"Number of subproblems remaining: {len(seed_list)}")
            key += 1
 

      
    def lookup_seed_key(self, seed):
        '''finds a seed in the list of all explored seeds 

        Parameters
        ----------
        seed : a Seed object to find in the list of explored seeds
           
        Returns
        -------
        s
            a matching seed in the list of explored seeds
        '''
  
        for s in self.explored_seeds:
            if self.explored_seeds[s] == seed:
                return s


    def grow(self, max_trees=np.Inf):
        '''Performs all possible elementary tree operations on every possible seed (leaf) node
        and memoizes the results for enumerating the candidate set of clonal trees 

        Parameters
        ----------
        max_trees : the maximum number of tree to construct
           
        Returns
        -------
        pre_proc_trees
            a ClonalTreeList object containing all candidate clonal trees
        '''

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
                        seed = curr_tree.get_seed_by_node(l)
                    
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
                            

                                    if not pre_proc_list.contains(merged_tree):
                                        pre_proc_list.insert(merged_tree)
                                    if not cand_trees.contains(merged_tree):    
                                        cand_trees.insert(merged_tree)
                                             
        
        return pre_proc_list

        


    def sprout_branching(self,seed):
        '''Performs a branching operation on a given seed

        Parameters
        ----------
        seed : Seed
            a leaf node for which the operation should be performed
           
        Returns
        -------
        best_tree
            a BranchingTree with the highest likelihood or None is no valid BranchingTree is found
        '''

        print("\nSprouting a branching tree..")  
        
        #perform a branching split
        br_split = bs.Branching_split(self.data,
                                      seed,
                                      self.rng,  
                                      self.params,
                                    )

        
        branching_tree = br_split.sprout()
        if branching_tree is not None:
            seed_list = branching_tree.get_seeds(np.empty(shape=0, dtype=int))
        else:
            seed_list = []

        return branching_tree, seed_list

        
    def sprout_linear(self, seed):
        '''Performs a linear operation on a given seed

        Parameters
        ----------
        seed : Seed
            a leaf node for which the operation should be performed
           
        Returns
        -------
        best_tree
            a LinearTree with the highest likelihood or None is no valid LinearTree is found
        '''
        print("\nSprouting a linear tree...")    
        lin_split =ls.Linear_split(     self.data,
                                        seed,
                                        self.rng,  
                                        self.params,
                                    )

        tree_list, best_tree = lin_split.sprout()

        num_trees = tree_list.size()
        if best_tree is not None:
            seed_list =best_tree.get_seeds()
            seed.set_linear(best_tree)
            # print(best_tree)
            # print(num_trees)
            if num_trees >= 2:
                start = num_trees -1
                # root = tree_list.index_tree(start)
                cA = best_tree.get_tip_cells(0)
                mA= best_tree.get_tip_muts(0)
                # cB = np.setdiff1d(seed.cells, cA)
                # mB  = np.setdiff1d(seed.muts, mA)
                # prev_tree = LinearTree(cA, cB, mA, mB)
                ca_cells = cA 
                ma_muts = mA
                start = start - 1
                curr_seed = seed_list[0]
            
                while start >= 0:
                    next_tree= tree_list.index_tree(start)
                    cA = next_tree.get_tip_cells(0)
                    mA = next_tree.get_tip_muts(0)
    
                    ca_cells = np.setdiff1d(  cA, ca_cells)
                    ma_muts = np.setdiff1d( mA, ma_muts)
            
                    next_tree.cell_mapping[0] = {0: ca_cells }
                    next_tree.mut_mapping[0] = ma_muts
                    # print(curr_seed)
                    curr_seed.set_linear(next_tree)
                    ca_cells = cA
                    ma_muts = mA
                    # print(next_tree)
                    next_seeds = next_tree.get_seeds()
                    seed_list += next_seeds
                    
                    curr_seed = next_seeds[0]
                    # print(curr_seed)
            
                    start = start -1
        else:
            seed_list = []

            ##need to add last seed



        return best_tree, seed_list
       

    def sprout_identity(self, seed):
        '''Performs an identity operation on a given seed

        Parameters
        ----------
        seed : Seed
            a leaf node for which the operation should be performed
           
        Returns
        -------
        an identity tree from the given seed
        '''
        events = None
        if self.cna_genotype_mode:
            events= self.cnn_hmm.run(seed.cells)
    


        return IdentityTree(seed.cells, seed.muts, events)
      

        