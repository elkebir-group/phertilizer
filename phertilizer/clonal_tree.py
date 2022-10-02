from copy import deepcopy
import numpy as np
import pandas as pd
import networkx as nx
from collections import Counter
from itertools import product, chain, combinations
from scipy.stats import multivariate_normal
# from phertilizer.seed import Seed
# from phertilizer.draw_clonal_tree import DrawClonalTree
# from phertilizer.utils import get_next_label, dict_to_series, pickle_save, generate_cell_dataframe, generate_mut_dataframe, dict_to_dataframe


# from seed import Seed
from draw_clonal_tree import DrawClonalTree
from utils import get_next_label, dict_to_series, pickle_save, generate_cell_dataframe, generate_mut_dataframe, dict_to_dataframe


from dataclasses import dataclass
import numpy as np
# from clonal_tree import LinearTree, BranchingTree



class ClonalTree:
    """
    A class to model a clonal tree with associated SNV/CNA genotypes and cell clustering

    ...

    Attributes
    ----------
    key : int
        the unique key for the clonal tree
    
    states : tuple
        the names of allowable CNA genotype states ("gain", "loss", "neutral")
    
    tree : networkx DiGraph
        the clonal tree graph
    
    cell_mapping : dict
        a dictionary with node ids as keys and np.arrays with the indices of cells attached to that node
    
    mut_mapping : dict
        a dictionary with node ids as keys and np.arrays with the indices of gained SNVs assigned to the 
        incoming edge of that node
    
    mut_loss_mapping : dict 
        a dictionary with node ids as keys and np.arrays with the indices of lost SNVs assigned to the 
        incoming edge of that node
    
    event_mapping : dict 
        a dictionary of dictionaries with node ids as keys for dictionaries with state containing the bins 
        assigned to each state
    
    loglikelihood : float
        the loglikelihood of the clonal tree for a given set of data
    
    variant_likelihood : float 
        the loglikelihood of the clonal tree for the given variant read count data
    
    bin_count_likelihood : float 
        the loglikelihood of the clonal tree for the given binned read count data
    
    cna_genotype_mode : boolean
        indicates if CNA genotypes should be inferred
    
   

    Methods
    -------

    get_tip_cells(t)
        returns the set of cells in node t

    get_tip_muts(t)
        returns the set of SNVs on the incoming edge of node t

    set_key(key)
        updates the key identifier for the clonal tree

    has_loss()
        returns a boolean if any SNVs are lost

    get_seed_by_node(node, lamb, tau, anc_muts)
        returns a Seed for the given node provided it meets criteria specified by lamb, tau

    inorder_traversal(root)
        performs an inorder traversal of the tree for relabeling purpuses

    relabel()
        relabels the nodes consecutively from an inorder traversal and updates the mapping dictionaries

    merge(tree, l)
        merges the clonal tree with a given tree by replacing the the leaf node l of the clonal tree
        with the root node of the given tree

    snv_genotypes()
        returns a dictionary with nodes as keys containing the snv genotype vectors

    presence_by_node(m, cells, node)
        returns an expanded snv_genotype matrix (cells x m) for a given set of cells for a specific node

    event_by_node(m, cells, node, bin_mapping)
        returns an expanded cna_genotype matrix (cells x m) for a given set of cells for a specific node

    cna_genotype( n,  nbins)
        returns a vector of length nbins containing the CNA genotype vector for node n

    save(path)
        a helper method to save a clonal tree to disk via pickle

    rdr_likelihood_by_node(n, cnn_hmm, cells=None)
        computes the likelihood of the bin count data for a specified node
    
    tree_png(fname, chrom_map_file: str=None)
        saves the png drawing of the clonal tree to disk
    
    tree_dot(fname, chrom_map_file: str=None)
        saves the dot string for the clonal tree to disk
    
    get_leaves()
        returns the node ids of a all leaf nodes
    
    compute_likelihood(data)
        computes the likelihood of the given data for the current clonal tree, genotypes and cell clustering
    
    compute_likelihood_by_node(n, data)
        computes the likelihood of the given data for a specified node  
        given the current clonal tree, genotypes and cell clustering
    
    get_loglikelihood()
        accesses the current values of the loglikelihood, variannt read count loglikelihood and binned 
        read count loglikelihood
    
    generate_results(cell_lookup, mut_lookup)
        converts all the internal mapping dictionaries to output dataframes 
    
    compute_variant_likelihood_by_node_without_events(node, like0, like1, bin_mapping=None)
        computes the loglikelihood of the variant read count data for the specified node of the clonal tree 
        without making use of the CNA genotypes 
    
    compute_variant_likelihood_by_node_with_events(node, like0, like1, bin_mapping=None)
        computes the loglikelihood of the variant read count data for the specified node of the clonal tree 
        making use of the CNA genotypes 

    find_root()
        returns the node id of the root of the clonal tree
    
    get_all_cells()
        returns an np.array of all cell indices currently assigned in the clonal tree
    
    get_all_muts()
        returns an np.array of all SNV indices currently assigned in the clonal tree


    """

    def __init__(self, tree, cell_mapping, mut_mapping, mut_loss_mapping=None, event_mapping=None, key=0):
        self.tree: nx.DiGraph = tree

        self.cell_mapping = cell_mapping

        self.mut_mapping = mut_mapping

        if mut_loss_mapping is None:
            self.mut_loss_mapping = {}
        else:
            self.mut_loss_mapping = mut_loss_mapping

        if event_mapping is None or len(event_mapping)==0:
            self.cna_genotype_mode = False
            self.event_mapping = {}
        else:
            self.cna_genotype_mode = True
            self.event_mapping = event_mapping

        self.key = key

        if self.cna_genotype_mode:
            self.compute_variant_likelihood = self.compute_variant_likelihood_by_node_with_events
        else:
            self.compute_variant_likelihood = self.compute_variant_likelihood_by_node_without_events
        self.loglikelihood, self.variant_likelihood, self.bin_count_likelihood = None, None, None
        self.states = ["gain", "loss", "neutral"]

        self.use_rd = True

    def __eq__(self, obj):

        is_isomorphic = nx.is_isomorphic(
            self.tree, obj.tree, node_match=self.cell_size)

        return is_isomorphic

    def __str__(self):
        outstring = f"Clonal Tree {self.key} \n"
        outstring += f"{list(self.tree.edges())} \n"
        outstring += f"Loss Nodes:{len(self.mut_loss_mapping)}\n"
        for n in self.tree.nodes():
            ncells = len(self.get_tip_cells(n))
            nmuts = len(self.get_tip_muts(n))
            if n in self.mut_loss_mapping:
                lost_muts = len(self.mut_loss_mapping[n])
            else:
                lost_muts = 0
            outstring += f"Node: {n} Cells: {ncells} Muts: {nmuts} LostMuts: {lost_muts}\n"
        outstring += f"Total Cells: {len(self.get_all_cells())} Total Muts: {len(self.get_all_muts())}"
        return outstring

    def get_tip_cells(self, t):
        if t in self.cell_mapping:
            return np.concatenate([self.cell_mapping[t][k] for k in self.cell_mapping[t]])
        else:
            return np.empty(shape=0, dtype=int)

    def set_key(self, key):
        self.key = key

    def has_loss(self):
        return len(self.mut_loss_mapping) > 0

    @staticmethod
    def cell_size(n1, n2):

        if n1["ncells"] == n2['ncells']:
            return True
        return False

    def get_seed_by_node(self, node, lamb, tau, anc_muts):

        if node in list(self.tree.nodes()):
            cells = self.get_tip_cells(node)
            muts = self.get_tip_muts(node)
            # if len(cells) > lamb and len(muts) > tau:
            return Seed(cells, muts, anc_muts)

        return None

    def inorder_traversal(self, root):
        self.labels.append(root)
        for child in self.tree.successors(root):
            self.inorder_traversal(child)

    def relabel(self):
        self.labels = []

        root = self.find_root()
        self.inorder_traversal(root)
        mapping = {self.labels[i]: i for i in range(len(self.labels))}
        self.tree = nx.relabel_nodes(self.tree, mapping, copy=True)

        cm, mm, ml, em = {}, {}, {}, {}

        for i in range(len(self.labels)):
            if self.labels[i] in self.cell_mapping:
                cm[i] = self.cell_mapping[self.labels[i]]
            if self.labels[i] in self.mut_mapping:
                mm[i] = self.mut_mapping[self.labels[i]]
            if self.labels[i] in self.mut_loss_mapping:
                ml[i] = self.mut_loss_mapping[self.labels[i]]
            if self.labels[i] in self.event_mapping:
                em[i] = self.event_mapping[self.labels[i]]

        self.cell_mapping = cm
        self.mut_mapping = mm
        self.mut_loss_mapping = ml
        self.event_mapping = em

    def merge(self, tree, l):

        if type(tree) is not IdentityTree:
            # get parent of leaf node
            parent = list(self.tree.predecessors(l))[0]
            self.tree.remove_node(l)
            # remove dictionary entries of leave done in growing tree
            self.cell_mapping.pop(l)
            self.mut_mapping.pop(l)
            if l in self.mut_loss_mapping:
                self.mut_loss_mapping.pop(l)

            if self.cna_genotype_mode and l in self.event_mapping:
                self.event_mapping.pop(l)

            # connect the parent of the leaf being removed to the root of the input tree
            # update the dictionaries
            new_label = get_next_label(self.tree)
            self.tree.add_node(new_label, ncells=len(tree.get_tip_cells(0)))
            self.tree.add_edge(parent, new_label)
            self.mut_mapping[new_label] = tree.mut_mapping[0]
            self.cell_mapping[new_label] = tree.cell_mapping[0]
            if 0 in tree.mut_loss_mapping:
                self.mut_loss_mapping[new_label] = tree.mut_loss_mapping[0]
            if 0 in tree.event_mapping:
                self.event_mapping[new_label] = tree.event_mapping[0]

            # add remaining nodes to tree and update the remaining data
            for child in tree.tree.successors(0):
                child_label = new_label + child
                num_cells = len(tree.get_tip_cells(child))
                self.tree.add_node(child_label, ncells=num_cells)
                self.tree.add_edge(new_label, child_label)
                self.mut_mapping[child_label] = tree.mut_mapping[child]
                self.cell_mapping[child_label] = tree.cell_mapping[child]
                if child in tree.mut_loss_mapping:
                    self.mut_loss_mapping[child_label] = tree.mut_mut_loss_mapping[child]
                if child in tree.event_mapping:
                    self.event_mapping[child_label] = tree.event_mapping[child]

        self.relabel()

    def snv_genotypes(self, m=None):
        if m is None:
            m = len(self.get_all_muts())
        
        y_dict = {}
        for node in self.mut_mapping:
            y = np.zeros(m, dtype=int)
            ancestral_present_muts = self.get_ancestral_muts(node).astype(int)
            present_muts = np.concatenate(
                [ancestral_present_muts, self.mut_mapping[node]])
            if node in self.mut_loss_mapping:
                present_muts = np.setdiff1d(
                    present_muts, self.mut_loss_mapping[node])
            y[present_muts.astype(int)] = 1
            y_dict[node] = y

        return y_dict
    
    def cell_cluster_genotypes(self, n=None):
        if n is None:
            n = len(self.get_all_cells())
        c_dict = {}
        for node in self.cell_mapping:
            c = np.zeros(n, dtype=int)
            present_clade = list(nx.dfs_preorder_nodes(self.tree, node))
            present_cells = np.concatenate([self.cell_mapping[i][0] for i in present_clade])
   
            c[present_cells] = 1
            c_dict[node] = c

        return c_dict

    def presence_by_node(self, m, cells, node):

        presence = np.zeros(shape=(len(cells), m), dtype=int)
        if len(cells) == 0:
            return presence

        ancestral_present_muts = self.get_ancestral_muts(node).astype(int)
        present_muts = np.concatenate(
            [ancestral_present_muts, self.mut_mapping[node]])
        if node in self.mut_loss_mapping:
            present_muts = np.setdiff1d(
                present_muts, self.mut_loss_mapping[node])

        presence[:, present_muts] = 1

        return presence  # n x m binary matrix cell y_ij=1 if mutation j  is harbored in cell i

    def event_by_node(self, m, cells, node, bin_mapping):

        # TODO: return events of parents
        labeled_events = np.full(
            shape=(len(cells), m), fill_value="neutral", dtype=object)
        if len(cells) == 0:
            return labeled_events

        if node in self.event_mapping:
            events = self.event_mapping[node]
        bin_dict = {}
        for s in events:
            for b in events[s]:
                bin_dict[b] = s

        for p in bin_mapping.index:
            bin = bin_mapping.loc[p]
            if bin in bin_dict:
                labeled_events[:, p] = bin_dict[bin]

        return labeled_events

    def rdr_likelihood_by_node(self, n, cnn_hmm, cells=None):
        if cells is None:
            cells = self.get_tip_cells(n)
        if len(cells) == 0:
            return None, None
        # instantiate the hmm
        bins = cnn_hmm.get_bins()
        cna_genotype = self.cna_genotype(n, bins.shape[0])
        cna_genotype_series = pd.Series(cna_genotype, index=bins)
        cell_like_series = cnn_hmm.compute_likelihood(
            cells, cna_genotype_series)

        return cell_like_series.sum(), cell_like_series
    def read_depth_likelihood_by_node(self,n, read_depth, cells=None):
        if cells is None:
            cells = self.get_tip_cells(n)
        if len(cells) == 0:
            return None, None
        cluster_data = read_depth[cells,:]
        bin_means = cluster_data.mean(axis=0)
        bin_variance = np.diag(np.var(cluster_data, axis=0))
        mv_normal = multivariate_normal(bin_means, bin_variance,allow_singular=True)
        cell_likelihood =mv_normal.logpdf(cluster_data)
        cell_like_series = pd.Series(cell_likelihood, index=cells)

        return cell_likelihood.sum(), cell_like_series
    def cna_genotype(self, n,  nbins):

        genotype = np.full(shape=nbins, fill_value="neutral")
        if n in self.event_mapping:
            events = self.event_mapping[n]
            for s in self.states:

                genotype[events[s]] = s

        return genotype  # ncells by nbins matrix with z_{ij} = s

    def save(self, path):
        pickle_save(self, path)
    

    def save_text(self, path):
        
        
        leafs = [n for n in self.tree.nodes if len(list(self.tree.successors(n))) ==0]
      
                    
        with open(path, "w+") as file:
            file.write(f"{len(list(self.tree.edges))} #edges\n")
            for u,v in list(self.tree.edges):
                file.write(f"{u} {v}\n")
            file.write(f"{len(leafs)} #leaves\n")
            for l in leafs:
                file.write(f"{l}\n")
            
     
            

    def tree_png(self, fname, chrom_map_file: str = None):
        #self.node_events = self.relabel()
        self.node_events = None
        bin2chrom = None
        if chrom_map_file is not None:
            bin2chrom = pd.read_csv(chrom_map_file, names=[
                                    "chrom", "arm", "start", "end"])
        dt = DrawClonalTree(self, bin2chrom)
        dt.savePNG(fname)

    def tree_dot(self, fname, chrom_map_file: str = None):
        bin2chrom = None
        if chrom_map_file is not None:
            bin2chrom = pd.read_csv(chrom_map_file, names=[
                                    "chrom", "arm", "start", "end"])
        dt = DrawClonalTree(self, bin2chrom)
        dt.saveDOT(fname)

    def get_tip_muts(self, t):
        if t in self.mut_mapping:
            return self.mut_mapping[t]
        return np.empty(shape=0, dtype=int)


    def get_leaves(self):
        leaves = [l for l in list(self.tree.nodes())
                  if self.tree.out_degree(l) == 0]
        return leaves

    def get_all_cells(self):
        cells = np.concatenate([self.cell_mapping[i][k]
                               for i in self.cell_mapping for k in self.cell_mapping[i]])
        return cells

    def get_cell_clusters(self):
        n_cell = len(self.get_all_cells())
        clusters = np.zeros(n_cell, dtype=int)
        for cluster, cells in self.cell_mapping.items():
            if len(cells[0]) > 0:
                clusters[cells[0]] = cluster
        return clusters

    def get_all_muts(self):
        muts = np.concatenate([self.mut_mapping[i] for i in self.mut_mapping])
        muts = np.sort(muts)
        return muts

    def get_mut_clusters(self, n_mut=None):
        if n_mut is None:
            n_mut = len(self.get_all_muts())
            clusters = np.zeros(n_mut, dtype=int)
        else:
            clusters  = np.full(shape=n_mut, fill_value=-1)
        for cluster, muts in self.mut_mapping.items():
            if len(muts) > 0:
                clusters[muts] = cluster
        return clusters

    def label_events_by_pair(self, parent, child):
        parent_events = dict_to_series(self.event_mapping[parent])
        child_events = dict_to_series(self.event_mapping[child])
        edge_events = {s: [] for s in self.states}
        for b in child_events.index:
            if child_events.loc[b] != parent_events.loc[b]:
                edge_events[child_events.loc[b]].append(b)

        return edge_events

    def get_root_events(self):
        root = self.find_root()

        root_events = deepcopy(self.event_mapping[root])
        root_events.pop("neutral")

        return root_events

    def label_events_helper(self, parent):
        for child in self.tree.successors(parent):
            self.node_events[child] = self.label_events_by_pair(parent, child)
            self.label_events_helper(child)

    def label_events(self):
        if self.cna_genotype_mode:
            self.node_events = {}

            root = self.find_root()
            self.node_events[root] = self.get_root_events()

        self.label_events_helper(root)

        return self.node_events


    def find_bad_cells(self, data, node, q):
     
        clade = list(nx.dfs_preorder_nodes(self.tree, node))
     
        cells = np.concatenate([self.cell_mapping[c][0] for c in clade])
        muts = self.mut_mapping[node]
        tot_count = np.count_nonzero(
                data.total[np.ix_(cells, muts)], axis=1).reshape(-1)
        na_cells = cells[tot_count ==0]
        cells = np.setdiff1d(cells, na_cells)
        mut_count = np.count_nonzero(
                data.var[np.ix_(cells, muts)], axis=1).reshape(-1)
        tot_count = np.count_nonzero(
                data.total[np.ix_(cells, muts)], axis=1).reshape(-1)
        cmb = mut_count/tot_count
        if len(cmb) > 0:
            cutoff= np.quantile(cmb, q)
            bad_cells = cells[cmb <= cutoff]
        else:
            bad_cells = np.empty(shape=0)
      
        return np.union1d(bad_cells, na_cells)

    # def find_bad_snvs(self, data, node, q):
    #     succ_nodes = list(nx.dfs_preorder_nodes(self.tree, node))
    #     clade = [n for n in self.tree.nodes if n not in succ_nodes]
    #     cells = np.concatenate([self.cell_mapping[c][0] for c in clade])
    #     muts = self.mut_mapping[node]
    #     mut_count = np.count_nonzero(
    #             data.var[np.ix_(cells, muts)], axis=0).reshape(-1)
    #     tot_count = np.count_nonzero(
    #             data.total[np.ix_(cells, muts)], axis=0).reshape(-1)
    #     binary_vaf = mut_count/tot_count
    #     cutoff = np.quantile(binary_vaf, 1-q)
    #     bad_snvs =muts[binary_vaf >= cutoff]
    #     return bad_snvs


    def find_bad_snvs(self, data, node, q):
        clade = list(nx.dfs_preorder_nodes(self.tree, node))
   
        cells = np.concatenate([self.cell_mapping[c][0] for c in clade])
        muts = self.mut_mapping[node]
        tot_count = np.count_nonzero(
                data.total[np.ix_(cells, muts)], axis=0).reshape(-1)

        na_snvs = muts[tot_count ==0]
        muts = np.setdiff1d(muts, na_snvs)

        tot_count = np.count_nonzero(
                data.total[np.ix_(cells, muts)], axis=0).reshape(-1)
        mut_count = np.count_nonzero(
                data.var[np.ix_(cells, muts)], axis=0).reshape(-1)

        binary_vaf = mut_count/tot_count
        if len(binary_vaf) >0:
            cutoff = np.quantile(binary_vaf, q)
            bad_snvs =muts[binary_vaf <= cutoff]
        else:
            bad_snvs = np.empty(shape=0,dtype=int)
    
        return np.union1d(bad_snvs,na_snvs)

    def get_cell_cluster(self, c):
        for node in self.cell_mapping:
            if c in self.cell_mapping[node][0]:
                return node

    def get_snv_cluster(self, s):
        for node in self.mut_mapping:
            if s in self.mut_mapping[node]:
                return node  

    def reassign_cell(self, c, y_dict, data):
        clust = self.get_cell_cluster(c)
        like0 = data.like0[c,:]
        like1 = data.like1_marg[c,:]
        if clust in y_dict:
            y = y_dict[clust]
     
            best_like =np.dot(y,like1) + np.dot(1-y, like0)

            original_like = best_like
            best_clust = clust 
        else:
            clust = -1
            original_like = np.NINF
            best_clust = None
            best_like = np.NINF
   
      
  
        for node, y in y_dict.items():
           node_like =np.dot(y,like1) + np.dot(1-y, like0)
           if node_like > best_like:
                best_like = node_like
                best_clust = node
        
        if best_clust != clust:
            if clust >= 0:
                self.cell_mapping[clust][0] = np.setdiff1d(self.cell_mapping[clust][0],np.array(c))
            self.cell_mapping[best_clust][0] = np.union1d(self.cell_mapping[best_clust][0],np.array(c))
        

        if original_like > best_like:
            print(f"cell: {c} clust: {clust} original like {original_like} best like {best_like}")

    def reassign_snv(self, s, c_dict, data):
        clust = self.get_snv_cluster(s)
        
        
        like0 = data.like0[:,s]
        like1 = data.like1_marg[:,s]
        if clust is None:
            clust = -1
    
        best_like = np.NINF
        for node, y in c_dict.items():
           node_like =np.dot(y,like1) + np.dot(1-y, like0)
           if node_like > best_like:
                best_like = node_like
                best_clust = node
        
        if best_clust != clust:
            if clust >=0:
                self.mut_mapping[clust] = np.setdiff1d(self.mut_mapping[clust],np.array(s))
            self.mut_mapping[best_clust]= np.union1d(self.mut_mapping[best_clust],np.array(s))

    def find_missing_cells(self, data):
        data_cells = data.cell_lookup.index.to_numpy()
        tree_cells = self.get_all_cells()
        missing_cells = np.setdiff1d(data_cells, tree_cells)
        return missing_cells 
    
    def find_missing_snvs(self, data):
        data_snvs = data.mut_lookup.index.to_numpy()
        tree_snvs = self.get_all_muts()
        missing_snvs = np.setdiff1d(data_snvs, tree_snvs)
        return missing_snvs

    def post_process(self, data, q=0.1, iterations=1, place_missing=True):
        ncells  = data.cell_lookup.shape[0]
        nmuts = data.mut_lookup.shape[0]
        nodes = list(nx.dfs_postorder_nodes(self.tree, source=self.find_root()))
        loglikelihood = self.compute_likelihood(data)
        print(f'loglikelihood {loglikelihood}')
        prev_loglikelihood = loglikelihood
        if place_missing:
            missing_snvs = self.find_missing_snvs(data)
            missing_cells = self.find_missing_cells(data)
            y_dict = self.snv_genotypes(m=nmuts)
            for c in missing_cells:
                self.reassign_cell(c, y_dict,data)
            for s in missing_snvs:
                c_dict = self.cell_cluster_genotypes(n=ncells)
                self.reassign_snv(s, c_dict, data)

        # for i in range(iterations):
            
        #     c_dict = self.cell_cluster_genotypes(n=ncells)
        #     for n in nodes:
        #         if n != self.find_root():
        #             muts = self.find_bad_snvs(data, n,q)
                
        #             for s in muts:
        #                 self.reassign_snv(s, c_dict, data)
        #             # loglikelihood = self.compute_likelihood(data)
        #             # print(f'iteration {i} node: {n} variant {self.variant_likelihood} loglike: {loglikelihood}')
        #             # cells = self.find_bad_cells(data, n,q)
        #             # y_dict = self.snv_genotypes(m=nmuts)
        #             # for c in cells:
        #             #     self.reassign_cell(c, y_dict,data)

        #         loglikelihood = self.compute_likelihood(data)
        #         print(f'iteration {i} node: {n} variant {self.variant_likelihood} loglike: {loglikelihood}')
            
        #     #check for termination if no improvements are made 
        #     if prev_loglikelihood == loglikelihood:
        #         break
        #     else:
        #         prev_loglikelihood = loglikelihood
        
        return loglikelihood

    def get_cell_ancestor_pairs(self) -> Counter:
        pairs = Counter()
        for node in self.tree.nodes:
            for children in nx.dfs_successors(self.tree, source=node).values():
                for child in children:
                    for cell1 in product(self.cell_mapping[node][0], [1]):
                        for cell2 in product(self.cell_mapping[child][0], [1]):
                            pairs[(cell1, cell2)] += 1
        return pairs

    def get_cell_cluster_pairs(self) -> Counter:
        pairs = Counter()
        for node in self.tree.nodes:
            for cell1, cell2 in combinations(product(self.cell_mapping[node][0], [1]), 2):
                if cell1 < cell2:
                    pairs[(cell1, cell2)] += 1
                else:
                    pairs[(cell2, cell1)] += 1
        return pairs

    def get_cell_incomparable_pairs(self) -> Counter:
        pairs = Counter()
        for u, v in combinations(self.tree.nodes, 2):
            if self.is_incomparable(self.tree, u, v):
                for cell1 in product(self.cell_mapping[u][0], [1]):
                    for cell2 in product(self.cell_mapping[v][0], [1]):
                        if cell1 < cell2:
                            pairs[(cell1, cell2)] += 1
                        else:
                            pairs[(cell2, cell1)] += 1
        return pairs

    def get_ancestor_pairs(self, include_loss: bool=True) -> Counter:
        pairs = Counter()
        for node in self.tree.nodes:
            for children in nx.dfs_successors(self.tree, source=node).values():
                for child in children:
                    if include_loss and node in self.mut_loss_mapping:
                        node_loss = self.mut_loss_mapping[node]
                    else:
                        node_loss = []

                    for mut1 in chain(
                        product(self.mut_mapping[node], [1]),
                        product(node_loss, [0])
                    ):
                        if include_loss and child in self.mut_loss_mapping:
                            child_loss = self.mut_loss_mapping[child]
                        else:
                            child_loss = []
                        for mut2 in chain(
                            product(self.mut_mapping[child], [1]),
                            product(child_loss, [0])
                        ):
                            pairs[(mut1, mut2)] += 1
        return pairs

    def get_cluster_pairs(self, include_loss: bool=True) -> Counter:
        pairs = Counter()
        for node in self.tree.nodes:
            if include_loss and node in self.mut_loss_mapping:
                node_loss = self.mut_loss_mapping[node]
            else:
                node_loss = []
            for mut1, mut2 in combinations(
                chain(
                    product(self.mut_mapping[node], [1]),
                    product(node_loss, [0])
                ),
                2
            ):
                if mut1 < mut2:
                    pairs[(mut1, mut2)] += 1
                else:
                    pairs[(mut2, mut1)] += 1
        return pairs

    @staticmethod
    def is_incomparable(graph: nx.DiGraph, u, v) -> bool:
        for path in nx.all_simple_paths(graph, source=0, target=v):
            if u in path:
                return False
        for path in nx.all_simple_paths(graph, source=0, target=u):
            if v in path:
                return False
        return True

    def get_incomparable_pairs(self, include_loss: bool=True) -> Counter:
        pairs = Counter()
        for u, v in combinations(self.tree.nodes, 2):
            if include_loss and u in self.mut_loss_mapping:
                u_loss = self.mut_loss_mapping[u]
            else:
                u_loss = []
            if include_loss and v in self.mut_loss_mapping:
                v_loss = self.mut_loss_mapping[v]
            else:
                v_loss = []

            if self.is_incomparable(self.tree, u, v):
                for mut1 in chain(
                    product(self.mut_mapping[u], [1]),
                    product(u_loss, [0])
                ):
                    for mut2 in chain(
                        product(self.mut_mapping[v], [1]),
                        product(v_loss, [0])
                    ):
                        if mut1 < mut2:
                            pairs[(mut1, mut2)] += 1
                        else:
                            pairs[(mut2, mut1)] += 1
        return pairs

    def compute_likelihood(self, data):

        self.use_rd = data.use_read_depth
      
        self.loglikelihood_dict = {"total": 0, "variant": 0, "bin": 0}
        self.node_likelihood = {}
        for n in self.cell_mapping:
            if len(self.get_tip_cells(n)) > 0:
                self.node_likelihood[n] = self.compute_likelihood_by_node(n, data)

        for n in self.node_likelihood:
            for key in self.loglikelihood_dict:
                self.loglikelihood_dict[key] += self.node_likelihood[n][key]

        self.loglikelihood = self.loglikelihood_dict['total']
        self.variant_likelihood = self.loglikelihood_dict['variant']
        self.bin_count_likelihood = self.loglikelihood_dict['bin']
        n= len(self.get_all_cells())
        m= len(self.get_all_muts())
        self.norm_loglikelihood = self.loglikelihood/(n*m)

        return self.loglikelihood

    def compute_likelihood_by_node(self, node, data):

        like0 = data.like0
        if self.cna_genotype_mode:
            like1 = data.like1_dict 
            
        else:
            like1 = data.like1_marg
      
        snv_bin_mapping = data.snv_bin_mapping
        node_like_dict = {}
        node_likelihood = self.compute_variant_likelihood(
            node, like0, like1, snv_bin_mapping)
        node_like_dict["variant"] = node_likelihood
        node_like_dict["total"] = node_likelihood

        if self.cna_genotype_mode:
            cnn_hmm = data.cna_hmm
            
            bin_node_likelihood, _ = self.rdr_likelihood_by_node(node, cnn_hmm)
            node_like_dict["bin"] = bin_node_likelihood
            node_like_dict['total'] += bin_node_likelihood
        else:
            node_like_dict["bin"] =0
            if self.use_rd:
                rd_node_likelihood, _ = self.read_depth_likelihood_by_node(node, data.read_depth)
                node_like_dict["bin"] = rd_node_likelihood
            node_like_dict['total'] += node_like_dict["bin"]

        return node_like_dict

    def get_loglikelihood(self):
        return self.loglikelihood, self.variant_likelihood, self.bin_count_likelihood

    def generate_results(self, cell_lookup, mut_lookup):
        pcell = generate_cell_dataframe(self.cell_mapping, cell_lookup)
        pmut = generate_mut_dataframe(self.mut_mapping, mut_lookup)
        ploss = generate_mut_dataframe(self.mut_loss_mapping, mut_lookup)
        
        pevents = dict_to_dataframe(self.event_mapping)

        return pcell, pmut, ploss, pevents

    def save_results(self, cell_lookup, mut_lookup, pcell_fname, pmut_fname, ploss_fname, pevents_fname):
        pcell, pmut, ploss, pevents = self.generate_results(
            cell_lookup, mut_lookup)
        pcell.to_csv(pcell_fname, index=False)
        pmut.to_csv(pmut_fname, index=False)
        ploss.to_csv(ploss_fname, index=False)
        pevents.to_csv(pevents_fname)

    def compute_variant_likelihood_by_node_without_events(self, node, like0, like1, bin_mapping=None):

        m = like0.shape[1]
        cells = self.get_tip_cells(node)
        like0 = like0[cells, :]
        like1 = like1[cells, :]

        y = self.presence_by_node(m, cells, node)     

        loglikelihood = np.multiply(
            (1-y), like0).sum() + np.multiply(y, like1).sum()

        return loglikelihood

    def compute_variant_likelihood_by_node_with_events(self, node, like0, like1, bin_mapping):

        m = like0.shape[1]
        cells = self.get_tip_cells(node)
        like0 = like0[cells, :]

        y = self.presence_by_node(m, cells, node)

        assert y.shape == like0.shape
        z = self.event_by_node(m, cells, node, bin_mapping)

        loglikelihood = np.multiply((1-y), like0).sum()

        states = np.unique(z)

        for s in states:
            like1_event = like1[s][cells, :]
            mask = np.logical_or(y != 1, z != s)
            if np.all(mask):
                continue
            like1_event_like = np.ma.array(like1_event, mask=mask).sum()

            loglikelihood += like1_event_like

        return loglikelihood



    def find_root(self):
        for n in list(self.tree.nodes()):
            if self.tree.in_degree(n) == 0:
                return n

    def get_ancestral_muts(self, node):
        root = self.find_root()
        path = list(nx.shortest_simple_paths(self.tree, root, node))[0]
        path = path[:-1]
        if len(path) > 0:
            present_muts = np.concatenate([self.mut_mapping[p] for p in path])
            lost_muts = [self.mut_loss_mapping[p]
                         for p in path if p in self.mut_loss_mapping]
            if len(lost_muts) > 0:
                lost_muts = np.concatenate(lost_muts)
                present_muts = np.setdiff1d(present_muts, lost_muts)
        else:
            present_muts = np.empty(shape=0, dtype=int)
        return present_muts

    def get_ancestral_events(self, node, event):
        root = self.find_root()
        path = list(nx.shortest_simple_paths(self.tree, root, node))[0]
        path = path[:-1]
        if len(path) > 0:
            present_events = [self.event_mapping[p][event]
                              for p in path if p in self.event_mapping]

            if len(present_events) > 0:
                present_events = np.concatenate(present_events)
            else:
                present_events = np.empty(shape=0, dtype=int)
        else:
            present_events = np.empty(shape=0, dtype=int)
        return present_events


class LinearTree(ClonalTree):
    
    def __init__(self, cellsA, cellsB, mutsA, mutsB, eA=None, eB=None, key=None):
        t = nx.DiGraph()
        t.add_node(0, ncells=len(cellsA))
        t.add_node(1, ncells=len(cellsB))
        t.add_edge(0, 1)
        cm, mm, ml, em = {}, {}, {}, {}
        cm[0] = {0: cellsA}
        cm[1] = {0: cellsB}

        mm[0] = mutsA
        mm[1] = mutsB
        if eA is not None:
            em[0] = eA
        if eB is not None:
            em[1] = eB
        super().__init__(t, cm, mm, ml, em)

    def is_valid(self, lamb, tau):
        return True
        #return len(self.get_tip_cells(0)) > lamb and len(self.mut_mapping[0]) > tau and len(self.get_tip_cells(1)) > lamb #and len(self.get_tip_muts(1)) > tau

    def get_seeds(self,  ancestral_muts):
        seed_list = []

        cellsB = self.get_tip_cells(1)
        mutsB = self.get_tip_muts(1)
        # if len(cellsB) > lamb and len(mutsB) > tau:
        anc_muts = np.sort(np.union1d(ancestral_muts, self.mut_mapping[0]))
        anc_muts = np.empty(shape=0, dtype=int)
        seed_list.append(Seed(cellsB, mutsB, anc_muts))

        return seed_list


class BranchingTree(ClonalTree):
    def __init__(self, cellsA, cellsB, mutsA, mutsB, mutsC, eA=None, eB=None, eC=None, key=None):
        t = nx.DiGraph()
        t.add_node(0, ncells=0)
        t.add_node(1, ncells=len(cellsA))
        t.add_node(2, ncells=len(cellsB))
        t.add_edge(0, 1)
        t.add_edge(0, 2)
        cm, mm, ml, em = {}, {}, {}, {}
        cm[0] = {0: np.empty(shape=0, dtype=int)}
        cm[1] = {0: cellsA}
        cm[2] = {0: cellsB}

        mm[0] = mutsC
        mm[1] = mutsA
        mm[2] = mutsB
        if eA is not None and eB is not None:
            em[1] = eA
            em[2] = eB

        ml = {}
        super().__init__(t, cm, mm, ml, em)

    def is_valid(self, lamb, tau):
        cells_valid = len(self.get_tip_cells(1)) > lamb and len(self.get_tip_cells(2)) > lamb

        muts_valid =len(self.mut_mapping[1]) > tau or len(self.mut_mapping[2]) > tau 
        muts_valid = True
        cells_valid = True
        return muts_valid and cells_valid

        # if (len(self.get_tip_cells(1)) > lamb and len(self.mut_mapping[1]) > tau) or (len(self.get_tip_cells(2)) > lamb and len(self.mut_mapping[2]) > tau):
        #     return True
        # else:
        #     return False

    def get_seeds(self, ancestral_muts):

        seed_list = []
        leaves = [1, 2]
        ancestral_muts = np.sort(np.union1d(
            ancestral_muts, self.mut_mapping[0]))
        ancestral_muts = np.empty(shape=0, dtype=int)
        for l in leaves:

            cells = self.get_tip_cells(l)
            muts = self.mut_mapping[l]
            # if len(cells) > lamb and len(muts) > tau:
          
            seed_list.append(Seed(cells, muts, ancestral_muts))

        return seed_list


class IdentityTree(ClonalTree):
    def __init__(self, cells, muts, events=None):
        t = nx.DiGraph()
        t.add_node(0, ncells=len(cells))

        cm, mm, ml, em = {}, {}, {}, {}
        cm[0] = {0: cells}
        mm[0] = muts
        if events is not None:
            em[0] = events
        super().__init__(t, cm, mm, ml, em)

    def is_valid(self, lamb, tau):
        return True

    def get_seeds(self,  ancestral_muts=None):
        return []



@dataclass
class Seed:

    cells: np.array
    muts: np.array
    ancestral_muts: np.array = np.empty(shape=0, dtype=int)
    key: int = None
    linear_tree: LinearTree = None
    branching_tree :BranchingTree = None
    def __post_init__(self):
        self.tree_list = []
     
    def __str__(self):

        outstring = f"Cells: {len(self.cells)} Muts: {len(self.muts)}" # Ancestral Muts: {len(self.ancestral_muts)} "
        return outstring

    def __eq__(self, object):

        # ancestral_muts_same = np.array_equal(
        #     np.sort(self.ancestral_muts), np.sort(object.ancestral_muts))

        if type(object) is type(self):
            return np.array_equal(self.cells, object.cells) \
                and np.array_equal(self.muts, object.muts) #\
        #         and ancestral_muts_same
        else:
            return False

    def set_key(self, key):
        self.key = key
    
    def get_key(self):
        return self.key

    def strip(self, var):
        var_counts_by_snv= var[np.ix_(self.cells, self.muts)].sum(axis=0)
        bad_snvs = self.muts[var_counts_by_snv==0]
        self.muts = np.setdiff1d(self.muts, bad_snvs)
        
        var_counts_by_cells = var[np.ix_(self.cells,self.muts)].sum(axis=1)
        bad_cells = self.cells[var_counts_by_cells ==0]
        self.cells = np.setdiff1d(self.cells, bad_cells)

    def count_obs(self,total):
        nobs =np.count_nonzero(total[np.ix_(self.cells,self.muts)])
        return nobs
    
    def set_linear(self, ct):
        self.tree_list.append(ct)
        self.linear_tree = ct

    def has_linear(self):
        return self.linear_tree is not None 

    def has_branching(self):
        return self.branching_tree is not None 
    def set_branching(self,ct):
        self.tree_list.append(ct)
        self.branching_tree = ct