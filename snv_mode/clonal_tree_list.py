from collections import deque
import numpy as np
from pandas import Series, DataFrame


class ClonalTreeList:
    def __init__(self):
        self.counter = 0
        self.stack = deque()

    def insert(self, tree_node):
        tree_node.set_key(self.counter)
        self.counter += 1
        self.stack.append(tree_node)

    def index_tree(self, index):
        if index < self.size():
            return self.stack[index]
        return None

    def set_stack(self, stack):
        self.stack = stack

    def pop_tree(self):
        return self.stack.popleft()

    def get_tree(self, key):
        for n in self.stack:
            if key == n.key:
                return n

    def has_trees(self):
        return len(self.stack) > 0

    def get_total_count(self):
        return self.counter

    def size(self):
        return len(self.stack)

    def get_all_trees(self):
        return self.stack

    def contains(self, object):
        for node in self.stack:
            if node == object:
                return True
        return False

    def slice_backwards(self, n):
        end = self.size()
        start = end - n
        self.stack = deque([self.index_tree(i) for i in range(start, end)])

    def filter(self, loglikelihoods, n):
        loglike_order = np.argsort(loglikelihoods)
        loglike_order = loglike_order[::-1]
        loglike_order = loglike_order[0:n]
        self.stack = deque([self.index_tree(i) for i in loglike_order])

    def sort_by_likelihood(self):
        likelihoods = np.zeros(shape=self.size())
        keys = []
        for i, tree in enumerate(self.stack):
            keys.append(tree.key)
            likelihoods[i], var, bin = tree.get_loglikelihood()

        like_series = Series(likelihoods, keys).sort_values()
        return like_series.index.to_numpy(), like_series

    def find_best_tree(self, like0, like1, snv_bin_mapping=None, cnn_hmm=None, force_loss=False):

        n = self.size()
        print(f"Finding the maximum likelihood tree out of {n} trees....")
        log_likelihood_array = np.zeros(shape=n)
        best_like = np.NINF
        best_tree = None
        for i, tree in enumerate(self.stack):
            print(tree)
            tree.compute_likelihood(like0, like1, snv_bin_mapping, cnn_hmm)
            log_likelihood, varlike, binlike = tree.get_loglikelihood()

            log_likelihood_array[i] = log_likelihood

            if True:
                if log_likelihood > best_like:
                    best_like = log_likelihood
                    best_tree = tree

        return best_tree, log_likelihood_array

    def save_the_trees(self, path):
        for tree in self.stack:
            loglike = np.abs(round(tree.loglikelihood))

            fname = f"{path}/tree{tree.key}_{loglike}.png"
            tree.tree_png(fname)

    def find_elbow(self, epsilon=0.025):
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
        # pandas series with the node as index and loglike as value
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
