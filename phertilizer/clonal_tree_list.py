from collections import deque
import numpy as np
from pandas import Series, DataFrame


class ClonalTreeList:
    """
    A class to store and operate on a list of ClonalTrees 

    ...

    Attributes
    ----------
    stack : deque
        the stack of ClonalTrees in the list 
    counter : int
        the total number of ClonalTrees added to the list
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

    insert(tree)
        inserts a clonal tree to the list and increments the counter

    index_tree(index)
        looks up returns a clonal tree by its index in the list 

    pop_tree()
        pops a clonal tree from the list

    get_tree(key)
      looks up returns a clonal tree by its key

    has_trees()
        returns a boolean indicating if the list has any trees

    get_total_count()
        returns the current value of the counter

    size()
        return the current length of the list

    get_all_trees()
        returns the stack attribute

    contains(object)
        returns a boolean indicating if object is in the list

    find_best_tree(data)
        computes the loglikelihood of the data given each tree in the list and returns the 
        clonal tree with the maximum loglikelihood

    save_the_trees(path)
        saves the pngs of all clonal trees in the list to the directory at the specified path

    find_elbow(epsilon)
        a regularizer that uses the elbow method to identify the output clonal tree


    """

    def __init__(self):
        self.counter = 0
        self.stack = deque()

    def insert(self, tree):
        '''inserts a clonal tree into the stack

        Parameters
        ----------
        tree : ClonalTree
            the clonal tree to be added to the stack
        
        '''
 
        tree.set_key(self.counter)
        self.counter += 1
        self.stack.append(tree)

    def index_tree(self, index):
        '''gets a clonal tree from the stack by its index 

        Parameters
        ----------
        index : int
            the index for the tree to be obtained
        
        Returns
        -------
        a clonal tree at the specified index or None if the index does not exist

        '''
        if index < self.size():
            return self.stack[index]
        return None

    def pop_tree(self):
        '''pops a clonal tree from the stack
        
        Returns
        -------
        a clonal tree at the left most position in the stack

        '''
        return self.stack.popleft()

    def get_tree(self, key):
        '''gets a clonal tree from the stack by its internal key
        
        Returns
        -------
        a clonal tree with the specified key

        '''
        for n in self.stack:
            if key == n.key:
                return n

    def has_trees(self):
        '''checks if the current stack has any clonal trees
        
        Returns
        -------
        a boolean specifiying if the stack has clonal trees

        '''        
        return len(self.stack) > 0

    def get_total_count(self):
        '''obtains the value of the counter, which counts the total number of 
        trees added to the stack
        
        Returns
        -------
        counter : int
            the value of the stacks internal counter

        '''         
        return self.counter

    def size(self):
        '''obtains the current length of the stack. This differs from the internal
        counter, which counts all added trees and doesn't decrement the counter
        when trees are popped.
       
        
        Returns
        -------
        the current length of the stack

        '''         

        return len(self.stack)

    def get_all_trees(self):
        ''' gets the internal stack of all clonal trees
       
        
        Returns
        -------
        the internal stack

        ''' 
        return self.stack

    def contains(self, object):
        ''' returns a boolean indicating if the specified object is in the stack
       
        Parameters
        ----------
        object : ClonalTree
            the clonal tree object to be checked if it is contained in the current stack


        Returns
        -------
        a boolean indicating if the given object is in the stack

        ''' 
        for node in self.stack:
            if node == object:
                return True
        return False

    def find_best_tree(self, data):
        ''' traverses the stack of clonal trees and identifies the clonal tree with maximum likelihood
       
        Parameters
        ----------
        data : Data
            a Phertilizer data object for which the likelihood of each clonal tree should be computed


        Returns
        -------
        best_tree : ClonalTree
            the clonal tree with maximum likelihood for the given data
        
        log_likelihood_series : pandas Series
            a series with the tree keys as index and the loglikelihood of each tree as the data

        ''' 
        n = self.size()
        print(f"Finding the maximum likelihood tree out of {n} trees....")
        log_likelihood_array = np.zeros(shape=n)
        best_like = np.NINF
        best_tree = None
        keys = []
        for i, tree in enumerate(self.stack):
            print(tree)
            keys.append(tree.key)
            log_likelihood = tree.compute_likelihood(data)
            log_likelihood = tree.norm_loglikelihood 
            print(f"Log Likelihood: {log_likelihood}\n")

            log_likelihood_array[i] = log_likelihood

            if log_likelihood > best_like:
                best_like = log_likelihood
                best_tree = tree

        log_likelihood_series = Series(log_likelihood_array, index=keys)
        return best_tree, log_likelihood_series

    def save_the_trees(self, path):
        ''' saves all the trees in the stack as pngs in the directory at the specified path
       
        Parameters
        ----------
        path : str
            path there tree pngs should be saved

        ''' 

        for tree in self.stack:
            loglike = np.round(tree.loglikelihood)
            if loglike >= 0:
                loglike = "+" + str(loglike)
            else:
                loglike = "-" + str(loglike)

            fname = f"{path}/tree{tree.key}_{loglike}.png"
            tree.tree_png(fname)

    def find_elbow(self, epsilon=0.025):
        ''' a regularizer that uses the elbow method to identify the output clonal tree
       
        Parameters
        ----------
        epsilon : float
            the value to use for adding pseudolikelihoods for the max number of nodes + 1


        Returns
        -------
        regularized_tree : ClonalTree
            the clonal tree selected via the elbow method
        
        elbow_df : pandas Dataframe
            a dataframe containing the pertinent information used to determine the elbow
        
        final_num_nodes : int
            the number of nodes in the regularized tree

        ''' 

        data = []
        max_node = 0
        for ct in self.stack:
            curr_nodes = len(list(ct.tree.nodes()))
            if curr_nodes > max_node:
                max_node = curr_nodes

            loglike, _, _ = ct.get_loglikelihood()

            data.append([ct.key, curr_nodes, -1*loglike])

        df = DataFrame(data, columns=['key', 'num_nodes', 'loglikelihood'])
        print(df)
        df = df.set_index('key')
        max_like_by_node = df.groupby(
            'num_nodes')['loglikelihood'].min().sort_index()

        last_log_like = max_like_by_node.iloc[-1]

        max_like_by_node.loc[max_node+1] = last_log_like*(1-epsilon)

        elbow_df = DataFrame({'likelihood': max_like_by_node})
        elbow_df['lead'] = elbow_df['likelihood'].shift(-1)
        elbow_df['lag'] = elbow_df['likelihood'].shift(1)
        elbow_df['delta1'] = (
            elbow_df['lag'] - elbow_df['likelihood'])/elbow_df['lag']
        elbow_df['delta2'] = (elbow_df['likelihood'] -
                              elbow_df['lead'])/elbow_df['likelihood']
        elbow_df['f_n'] = elbow_df['delta1'] - elbow_df['delta2']

        max_series = elbow_df.idxmax()

        final_num_nodes = max_series.loc['f_n']

        like = elbow_df['likelihood'].loc[final_num_nodes]

        elbow_df.reset_index(inplace=True)

        tree_row = df[(df["num_nodes"] == final_num_nodes)
                      & (df["loglikelihood"] == like)]
        key = tree_row.index[0]
        regularized_tree = self.get_tree(key)
        return regularized_tree, elbow_df, final_num_nodes
