from copy import deepcopy
import numpy as np
import pandas as pd
import networkx as nx
from seed import Seed
from draw_clonal_tree import DrawClonalTree
from utils import get_next_label, dict_to_series, pickle_save, generate_cell_dataframe, generate_mut_dataframe, dict_to_dataframe
import logging
from collections import Counter
from itertools import product, chain, combinations


class ClonalTree:
    def __init__(self, tree, cell_mapping, mut_mapping, mut_loss_mapping=None, event_mapping=None, key = None):
        self.tree: nx.DiGraph = tree
    
        self.cell_mapping = cell_mapping

        self.mut_mapping = mut_mapping
        if mut_loss_mapping is None:
            self.mut_loss_mapping = {}
        else:
            self.mut_loss_mapping = mut_loss_mapping

        if event_mapping is None:
            self.include_cna = False
            self.event_mapping = {}
        else:
            self.include_cna = True
            self.event_mapping = event_mapping 

        if key is None:
            self.key = 0
    
        
        if self.include_cna:
            self.compute_variant_likelihood = self.compute_variant_likelihood_by_node_with_events
        else:
            self.compute_variant_likelihood = self.compute_variant_likelihood_by_node_without_events
        self.connection_point = None
        self.loglikelihood, self.variant_likelihood, self.bin_count_likelihood = None, None, None
        self.states = ["gain", "loss", "neutral"]

    def __eq__(self, obj):
   
        is_isomorphic = nx.is_isomorphic(self.tree,obj.tree, node_match=self.cell_size)

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
                lost_muts =0
            outstring += f"Node: {n} Cells: {ncells} Muts: {nmuts} LostMuts: {lost_muts}\n"
        outstring += f"Total Cells: {len(self.get_all_cells())} Total Muts: {len(self.get_all_muts())}"
        return outstring

    def get_tip_cells(self, t):
        
            return np.concatenate([self.cell_mapping[t][k] for k in self.cell_mapping[t]])

    def set_key(self, key):
        self.key = key
    
     

    def has_loss(self):
        return len(self.mut_loss_mapping) > 0

  
    def get_identity_seq(self):
        seq = list(nx.dfs_preorder_nodes(self.tree, source=self.find_root()))
        self.identity_seq = [self.tree.out_degree(i) for i in seq]

        self.snv_seq = [i not in self.mut_loss_mapping for i in seq]
        self.cell_seq = [len(self.get_tip_cells(i)) for i in seq]

        return self.identity_seq, self.snv_seq, self.cell_seq

    @staticmethod
    def list_compare(l1, l2):
        if len(l1) != len(l2):
            return False 
        for x,y in zip(l1, l2):
            if x != y:
                return False 
        return True

    @staticmethod
    def cell_size(n1, n2):
  
        if n1["ncells"] ==n2['ncells']:
            return True
        return False



    
    def get_seed_by_node(self, node, lamb, tau, anc_muts):
        
        if node in list(self.tree.nodes()):
            cells = self.get_tip_cells(node)
            muts = self.get_tip_muts(node)
            if len(cells) > lamb and len(muts) > tau:
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
            cm[i] = self.cell_mapping[self.labels[i]]
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
        print(self)
        print(tree)
        print(len(self.get_all_muts()))
     
        if type(tree) is not IdentityTree:
        #get parent of leaf node     
            parent = list(self.tree.predecessors(l))[0]
            self.tree.remove_node(l)
                    #remove dictionary entries of leave done in growing tree
            self.cell_mapping.pop(l)
            self.mut_mapping.pop(l)
            if l in self.mut_loss_mapping:
                self.mut_loss_mapping.pop(l)
            
            if self.include_cna and l in self.event_mapping:
                self.event_mapping.pop(l)


            #connect the parent of the leaf being removed to the root of the input tree
            #update the dictionaries
            new_label = get_next_label(self.tree)
            self.tree.add_node(new_label, ncells =len(tree.get_tip_cells(0)))
            self.tree.add_edge(parent, new_label)
            self.mut_mapping[new_label] = tree.mut_mapping[0]
            self.cell_mapping[new_label] = tree.cell_mapping[0]
            if 0 in tree.mut_loss_mapping:
                self.mut_loss_mapping[new_label]  = tree.mut_loss_mapping[0]
            if 0 in tree.event_mapping:
                self.event_mapping[new_label] = tree.event_mapping[0]
            
            #add remaining nodes to tree and update the remaining data
            for child in tree.tree.successors(0):
                child_label = new_label + child
                num_cells = len(tree.get_tip_cells(child))
                self.tree.add_node(child_label, ncells=num_cells)
                self.tree.add_edge(new_label, child_label)
                self.mut_mapping[child_label] = tree.mut_mapping[child]
                self.cell_mapping[child_label] = tree.cell_mapping[child]
                if child in tree.mut_loss_mapping:
                    self.mut_loss_mapping[child_label]  = tree.mut_mut_loss_mapping[child]
                if child in tree.event_mapping:
                    self.event_mapping[child_label] = tree.event_mapping[child]

            mts = self.get_all_muts()
            print(len(mts))
            print(len(np.unique(mts)))

            print(self)
           
            

        # self.relabel()

    def snv_genotypes(self):
        m = len(self.get_all_muts())
        y_dict = {}
        for node in self.mut_mapping:
            y= np.zeros(m, dtype=int)
            ancestral_present_muts = self.get_ancestral_muts(node).astype(int)
            present_muts = np.concatenate([ancestral_present_muts, self.mut_mapping[node]])
            if node in self.mut_loss_mapping:
                present_muts = np.setdiff1d(present_muts, self.mut_loss_mapping[node])
            y[present_muts] = 1
            y_dict[node] = y 
        
        return y_dict
            
    
        
    def presence_by_node(self, m, cells, node):
        
   
  
        presence = np.zeros(shape=(len(cells),m), dtype=int)
        if len(cells) == 0:
            return presence
 
        ancestral_present_muts = self.get_ancestral_muts(node).astype(int)
        present_muts = np.concatenate([ancestral_present_muts, self.mut_mapping[node]])
        if node in self.mut_loss_mapping:
            present_muts = np.setdiff1d(present_muts, self.mut_loss_mapping[node])
        
        presence[:,present_muts] = 1
 

        return presence #n x m binary matrix cell y_ij=1 if mutation j  is harbored in cell i 
    



    def event_by_node(self, m, cells, node, bin_mapping):
   
        #TODO: return events of parents
        labeled_events = np.full(shape=(len(cells),m), fill_value="neutral", dtype=object)
        if len(cells) == 0:
            return labeled_events
        
        if node in self.event_mapping:
            events= self.event_mapping[node]
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
        if len(cells) ==0:
            return None, None
        #instantiate the hmm 
        bins = cnn_hmm.get_bins()
        cna_genotype = self.cna_genotype(n, bins.shape[0])
        cna_genotype_series = pd.Series(cna_genotype, index= bins)
        cell_like_series = cnn_hmm.compute_likelihood(cells, cna_genotype_series)

        return cell_like_series.sum(), cell_like_series

                
        
    def cna_genotype(self, n,  nbins):
        #TODO: give parent genotype
        genotype = np.full(shape= nbins, fill_value="neutral")
        if n in self.event_mapping:
            events = self.event_mapping[n]
            for s in self.states:
                
                genotype[events[s]] = s
        

        return genotype   #ncells by nbins matrix with z_{ij} = s




    def rdr_likelihood(self, cnn_hmm):
        cell_series_list = []
        total = 0
        for n in self.tree.nodes():
            cells = self.get_tip_cells(n)
            if len(cells) > 0:
                node_total, cell_series = self.rdr_likelihood_by_node(n, cnn_hmm)
                total += node_total
                cell_series_list.append(cell_series)
        
        cell_series = pd.concat(cell_series_list).sort_index()

        return total, cell_series
        
       

    def get_parent_cells(self, node):
        root = self.find_root()
        
        while True:
            if node == root:
                return None
            parent = list(self.tree.predecessors(node))[0]
            parent_cells = np.concatenate([self.cell_mapping[parent][k] for k in self.cell_mapping[parent]])
            if len(parent_cells) > 0:
                return parent_cells
            else:
                node = parent


    def save(self, path):
        pickle_save(self, path)
    
    def tree_png(self, fname, chrom_map_file: str=None):
        #self.node_events = self.relabel()
        self.node_events = None
        bin2chrom = None
        if chrom_map_file is not None:
            bin2chrom = pd.read_csv(chrom_map_file, names=["chrom", "arm", "start", "end"])
        dt = DrawClonalTree(self, bin2chrom)
        dt.savePNG(fname)

    
    def tree_dot(self, fname, chrom_map_file: str=None):
        bin2chrom = None
        if chrom_map_file is not None:
            bin2chrom = pd.read_csv(chrom_map_file, names=["chrom", "arm", "start", "end"])
        dt = DrawClonalTree(self, bin2chrom)
        dt.saveDOT(fname)
    
    

        
    def get_tip_muts(self, t):
        return self.mut_mapping[t]
    
    def get_leaves(self):
        leaves = [l for l in list(self.tree.nodes()) if self.tree.out_degree(l)==0]
        return leaves
    

    def get_all_cells(self):
        cells = np.concatenate([self.cell_mapping[i][k] for i in self.cell_mapping for k in self.cell_mapping[i]])
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
    
    def get_mut_clusters(self):
        n_mut = len(self.get_all_muts())
        clusters = np.zeros(n_mut, dtype=int)
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
        if self.include_cna:
            self.node_events = {}

            root = self.find_root()
            self.node_events[root] = self.get_root_events()
     
        self.label_events_helper(root)
        
        return self.node_events


    def presence_matrix(self, m):
   
    
        y_list = []

        for n in self.cell_mapping:
            cells = self.get_tip_cells(n)
            y_list.append(self.presence_by_node(m,cells, n))
        
        y = np.vstack(y_list)
    

        return y #n x m binary matrix cell y_ij=1 if mutation j was is harbored in cell i  
    
    def get_ancestor_pairs(self) -> Counter:
        pairs = Counter()
        for node in self.tree.nodes:
            for children in nx.dfs_successors(self.tree, source=node).values():
                for child in children:
                    if node in self.mut_loss_mapping:
                        node_loss = self.mut_loss_mapping[node]
                    else:
                        node_loss = []

                    for mut1 in chain(
                        product(self.mut_mapping[node], [1]),
                        product(node_loss, [0])
                    ):
                        if child in self.mut_loss_mapping:
                            child_loss = self.mut_loss_mapping[child]
                        else:
                            child_loss = []
                        for mut2 in chain(
                            product(self.mut_mapping[child], [1]),
                            product(child_loss, [0])
                        ):
                            pairs[(mut1, mut2)] += 1
        return pairs

    def get_cluster_pairs(self) -> Counter:
        pairs = Counter()
        for node in self.tree.nodes:
            if node in self.mut_loss_mapping:
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

    def get_incomparable_pairs(self) -> Counter:
        pairs = Counter()
        for u, v in combinations(self.tree.nodes, 2):
            if u in self.mut_loss_mapping:
                u_loss = self.mut_loss_mapping[u]
            else:
                u_loss = []
            if v in self.mut_loss_mapping:
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


    def event_matrix(self, m, bin_mapping):
        z_list = []
        for n in self.cell_mapping:

            cells = self.get_tip_cells(n)
            z_list.append(self.event_by_node(m, cells, n, bin_mapping))
        
        z = np.vstack(z_list)

        return z #n x m character matrix z_ij = state where state in {+, -, 0}

    
    def compute_likelihood(self, like0, like1,snv_bin_mapping=None, cnn_hmm=None):

        self.loglikelihood_dict = {"total":0, "variant" :0, "bin":0}
        self.node_likelihood = {}
        for n in self.cell_mapping:
            if len(self.get_tip_cells(n)) > 0:
                self.node_likelihood[n] = self.compute_likelihood_by_node(n,like0, like1,
                                                                          snv_bin_mapping, 
                                                                          cnn_hmm )
        
        for n in self.node_likelihood:
            for key in self.loglikelihood_dict:
                self.loglikelihood_dict[key] += self.node_likelihood[n][key]
        
           
        self.loglikelihood = self.loglikelihood_dict['total']
        self.variant_likelihood = self.loglikelihood_dict['variant']
        self.bin_count_likelihood =self.loglikelihood_dict['bin']

        return self.loglikelihood


    def compute_likelihood_by_node(self, node, like0, like1,snv_bin_mapping=None,cnn_hmm=None):

        node_like_dict = {}
        node_likelihood = self.compute_variant_likelihood(node, like0, like1, snv_bin_mapping)
        node_like_dict["variant"] = node_likelihood
        node_like_dict["total"] = node_likelihood
        

        if self.include_cna:
            bin_node_likelihood, _ = self.rdr_likelihood_by_node(node, cnn_hmm)
            node_like_dict["bin"] = bin_node_likelihood
            node_like_dict['total'] += bin_node_likelihood
        
    
        return node_like_dict


    def get_loglikelihood(self):
        return self.loglikelihood, self.variant_likelihood, self.bin_count_likelihood



    def generate_results(self, cell_lookup, mut_lookup):
        pcell = generate_cell_dataframe(self.cell_mapping, cell_lookup)
        pmut =  generate_mut_dataframe(self.mut_mapping, mut_lookup)
        ploss = generate_mut_dataframe(self.mut_loss_mapping, mut_lookup)
        pevents = dict_to_dataframe(self.event_mapping)

        
        
        return pcell, pmut, ploss, pevents
              

    def save_results(self, cell_lookup, mut_lookup, pcell_fname, pmut_fname, ploss_fname, pevents_fname):
        pcell, pmut, ploss, pevents = self.generate_results(cell_lookup, mut_lookup)
        pcell.to_csv(pcell_fname, index=False)
        pmut.to_csv(pmut_fname, index=False)
        ploss.to_csv(ploss_fname, index=False)
        pevents.to_csv(pevents_fname)


    

    def compute_variant_likelihood_by_node_without_events(self, node, like0, like1, bin_mapping=None):
        
        m = like0.shape[1]
        cells = self.get_tip_cells(node)
        muts = self.get_tip_muts(node)
        like0 = like0[np.ix_(cells, muts)]
        like1= like1[np.ix_(cells, muts)]
        

        y = self.presence_by_node (m, cells,node)
  
        y = y[:,muts]

        loglikelihood  = np.multiply((1-y), like0).sum() + np.multiply(y, like1).sum()
        
        return loglikelihood
        
         


    def compute_variant_likelihood_by_node_with_events(self, node, like0, like1, bin_mapping):
        
        m = like0.shape[1]
        cells = self.get_tip_cells(node)
        # ancestral_muts = self.get_ancestral_muts(node)
        # muts = self.get_tip_muts(node)
        # all_muts =
        like0 = like0[cells, :]
        
        y = self.presence_by_node(m, cells,node)

        assert y.shape == like0.shape
        z = self.event_by_node(m, cells, node, bin_mapping)
    

        loglikelihood  = np.multiply((1-y), like0).sum()

        states = np.unique(z)
        
        for s in states:
            # if np.any(z==s):
                # event_mask = z == s
                like1_event = like1[s][cells,:]
                mask = np.logical_or(y !=1, z !=s)
                if np.all(mask):
                    continue
                # mask = np.logical_not(np.logical_and(presence_mask, event_mask))
                like1_event_like = np.ma.array(like1_event, mask=mask).sum()
        
                loglikelihood += like1_event_like


        return loglikelihood


    def mut_likelihood_by_node_with_events(self, node, like0, like1, bin_mapping, ax):
        
        m = like0.shape[1]
        cells = self.get_tip_cells(node)
        # ancestral_muts = self.get_ancestral_muts(node)
        # muts = self.get_tip_muts(node)
        # all_muts =
        like0 = like0[cells, :]
        
        y = self.presence_by_node(m, cells,node)

        assert y.shape == like0.shape
        z = self.event_by_node(m, cells, node, bin_mapping)
    

        loglikelihood_vec  = np.multiply((1-y), like0).sum(axis=ax)

        
        states= np.unique(z)
        for s in states:
            # if np.any(z==s):
                # event_mask = z == s
                like1_event = like1[s][cells,:]
                mask = np.logical_or(y !=1, z !=s)
                if np.all(mask):
                    continue
                # mask = np.logical_not(np.logical_and(presence_mask, event_mask))
                like1_event_like = np.ma.array(like1_event, mask=mask).sum(axis=ax)
        
                loglikelihood_vec += like1_event_like


        return loglikelihood_vec
    
    def cell_genotype_vector(self, ncells, node, desc_dict):
        cell_geno = np.zeros(shape=ncells, dtype=float)
       
        cells = self.get_tip_cells(node)
        if len(desc_dict[node]) > 0:
                desc_cells = np.concatenate([self.get_tip_cells(child) for child in desc_dict[node]])

                cells = np.concatenate([cells, desc_cells])
  
        cell_geno[cells] = 1
        return cell_geno



    def get_cell_genotypes(self, ncells):
        desc_dict =  {n : nx.descendants(self.tree, n) for n in list(self.tree.nodes())}
        
        cell_geno_dict = {}
        for n in self.cell_mapping:
            cell_geno_dict[n] = self.cell_genotype_vector(ncells, n, desc_dict)
        
        return cell_geno_dict
        
      

    def find_root(self):
        for n in list(self.tree.nodes()):
            if self.tree.in_degree(n) ==0:
                return n

    def get_ancestral_muts(self, node):
        root = self.find_root()
        path = list(nx.shortest_simple_paths(self.tree, root, node))[0]
        path =  path[:-1]
        if len(path) > 0:
            present_muts = np.concatenate([self.mut_mapping[p] for p in path])
            lost_muts= [self.mut_loss_mapping[p] for p in path if p in self.mut_loss_mapping]
            if len(lost_muts) > 0:
                lost_muts = np.concatenate(lost_muts)
                present_muts = np.setdiff1d(present_muts, lost_muts)
        else:
            present_muts = np.empty(shape=0, dtype=int)
        return present_muts
    
    def get_ancestral_events(self, node, event):
        root = self.find_root()
        path = list(nx.shortest_simple_paths(self.tree, root, node))[0]
        path =  path[:-1]
        if len(path) > 0:
            present_events = [self.event_mapping[p][event] for p in path if p in self.event_mapping]
            
            if len(present_events) > 0:
                present_events = np.concatenate(present_events)
            else:
                present_events = np.empty(shape=0, dtype=int)
        else:
            present_events = np.empty(shape=0, dtype=int)
        return present_events

    def find_low_likelihood_cells(self, like0, like1, snv_bin_mapping, perc, cna_hmm=None):
        n = like0.shape[0]
        num_to_select = int(n*perc)
        cell_like_list = []

        all_cells = []
        for n in self.cell_mapping:
            cells = self.get_tip_cells(n)
            if len(cells) > 0:
                cell_like_list.append(self.mut_likelihood_by_node_with_events(n, like0, like1, snv_bin_mapping, ax=1))
                all_cells.append(cells)
        cell_total_like = pd.Series(np.concatenate(cell_like_list), index= np.concatenate(all_cells)).sort_index()
        # if cna_hmm is not None:
        #     _, rdr_like_by_cell = self.rdr_likelihood(cna_hmm)
        #     cell_total_like = cell_total_like.add(rdr_like_by_cell)
        

        #sort pandas series ascending
        cell_total_like = cell_total_like.sort_values()
        print(cell_total_like.head(num_to_select+5))

        lowest_ranking_series = cell_total_like.iloc[:num_to_select]
        return lowest_ranking_series.index.to_numpy()

        
   
    def find_low_likelihood_muts(self, like0, like1, snv_bin_mapping, perc):
        m = like0.shape[1]
        num_to_select = int(m*perc)
        mut_like_vec = np.zeros(m)
        for n in self.cell_mapping:
            if len(self.get_tip_cells(n)) > 0:
                mut_like_vec +=self.mut_likelihood_by_node_with_events(n, like0, like1, snv_bin_mapping, ax=0)
      
        likelihood_ranking = mut_like_vec.argsort()

        lowest_ranking  = likelihood_ranking[:num_to_select]
        # print(lowest_ranking)

        return lowest_ranking

        # for n in self.cell_mapping:
        #    node_ self.compute_likelihood_by_node(n, like0, like1,snv_bin_mapping )

        

    def post_process(self,like0, like1, snv_bin_mapping,  lamb, tau, min_loss, cna_hmm, perc, iterations):
        cells = self.find_low_likelihood_cells(like0, like1, snv_bin_mapping, perc, cna_hmm)
        muts = self.find_low_likelihood_muts(like0,like1, snv_bin_mapping, perc)
        # logging.info(f"Post Process: Cells: {len(cells)} Muts: {len(muts)}")
     
                
        if not self.include_cna:
            self.post_process_mapping(cells, muts,  lamb, tau, min_loss,like0, like1, iterations)
        else:
            self.post_process_mapping_with_cna(cells, muts, lamb, tau,min_loss, like0, like1, snv_bin_mapping, cna_hmm, iterations)

    def eliminate_illegal_linear_nodes(self):
        count = 0
        for n in self.cell_mapping:
            if len(self.get_tip_cells(n))==0 and self.tree.out_degree(n)==1 and len(self.get_tip_muts(n)) > 0:
                count += 1
                parents = list(self.tree.predecessors(n))
                if len(parents) > 0:

                    parent =parents[0]
                    child = list(self.tree.successors(n))[0]
                  
                    self.tree.add_edge(parent, child)
                else:
                    parent = list(self.tree.successors(n))[0]
                self.tree.remove_node(n)
                muts = self.get_tip_muts(n)
                del self.mut_mapping[n]
                self.mut_mapping[parent] = np.union1d(self.mut_mapping[parent], muts)
        return count

    def post_proc(self, lamb, tau,min_loss, like0, like1, snv_bin_mapping, cna_hmm,perc, iterations):
        print(self)
        illegal_nodes, pruned_cells, prune_muts = self.prune_nodes(lamb, tau, min_loss)
        post_cells = self.find_low_likelihood_cells( like0, like1, snv_bin_mapping, perc, cna_hmm)
        post_muts = self.find_low_likelihood_muts( like0, like1, snv_bin_mapping, perc)
        if pruned_cells is not None:
            post_cells = np.union1d(pruned_cells, post_cells)
        if prune_muts is not None:
            post_muts = np.union1d(prune_muts, post_muts)
        post_muts = post_muts.astype(int)
        post_cell = post_cells.astype(int)
        print(f"Removing {len(illegal_nodes)} illegal nodes")
        print("Tree after removing illegal nodes")
        if len(post_cells) > 0 or len(post_muts) > 0:
            self.post_process_mapping_with_cna(post_cells, post_muts, lamb, tau, min_loss,
                                                 like0, like1, snv_bin_mapping, cna_hmm, iterations=iterations)

        num_elim = self.eliminate_illegal_linear_nodes()
        if num_elim > 0:
            self.compute_likelihood(like0, like1, snv_bin_mapping, cna_hmm)
     
    def prune_and_reassign(self, lamb, tau,min_loss, like0, like1, snv_bin_mapping, cna_hmm):
        illegal_nodes, pruned_cells, prune_muts = self.prune_nodes(lamb, tau, min_loss)
        print(f"Removing {len(illegal_nodes)} illegal nodes")
        print("Tree after removing illegal nodes")

        if len(illegal_nodes) and (len(pruned_cells) > 0 or len(prune_muts) > 0):
            self.post_process_mapping_with_cna(pruned_cells, prune_muts, lamb, tau, min_loss, like0, like1, snv_bin_mapping, cna_hmm, iterations=1)

        num_elim = self.eliminate_illegal_linear_nodes()
        if num_elim > 0:
            self.compute_likelihood(like0, like1, snv_bin_mapping, cna_hmm)


    def post_process_mapping_with_cna(self,cells, muts, lamb, tau,min_loss, like0, like1, snv_bin_mapping, cna_hmm, iterations):
        
        """Updates the cell and mutation assignments using the inferred tree by 
        selecting the maximum likelihood node.
        
        :return: dataframe for new cell and mutation labels and likelihoods of all nodes
        """

        muts = muts.astype(int)
        cells = cells.astype(int)
        black_list_nodes = [x for x in list(self.tree.nodes()) if len(np.concatenate([self.cell_mapping[x][k] for k in self.cell_mapping[x]])) ==0 and self.tree.out_degree(x) > 0]
        pre_like = self.loglikelihood
      
        print("Tree before post-processing")
  
        for i in range(iterations):
            
            if len(list(self.tree.nodes()))==1:
                break
                
            logging.info(f"Starting Post-Processing iteration {i}:{pre_like}")
            print(self)
            # illegal_nodes = self.prune_nodes(lamb, tau, min_loss)
            # print(f"Removing {len(illegal_nodes)} illegal nodes")
            # print("Tree after removing illegal nodes")
            # print(self)
            if len(muts) > 0:
                self.mut_post_processing_with_cna(muts, like0, like1, snv_bin_mapping)
            
            if len(cells) > 0:
                self.cell_post_processing_with_cna(cells, like0, like1, snv_bin_mapping, cna_hmm, black_list_nodes)
               
                self.event_post_processing(cna_hmm)

       
            
            post_like = self.compute_likelihood(like0, like1, snv_bin_mapping, cna_hmm)
            pre_like = post_like
            
        
            # logging.info(f"Likelihood After Post-Processing iteration {i}:{self.loglikelihood}")
            print(f"Ending post-processing iteration {i}: {post_like}")
            print(self)
            print("###########")
            # illegal_nodes = self.find_illegal_nodes(lamb, tau, min_loss)
            # if (i > iterations and len(illegal_nodes) ==0) or (len(illegal_nodes)==0 and pre_like >= self.loglikelihood):
            #     print("Post-processing completed....")
            #     break

            # i+=1

    def cell_post_processing_with_cna(self, cells,  like0, like1,snv_bin_mapping, cna_hmm, black_list_nodes=[] ):
        
        #TODO: Fix black list node
        m = like0.shape[1]

        like0 = like0[cells,:]

        # like1 = like1[cells,:]
   
        like_list = []   
        nodes_with_cells = [n for n in self.cell_mapping if len(self.get_tip_cells(n) > 0)]

        nodes_evaluated = np.setdiff1d(nodes_with_cells, black_list_nodes)
     
      
        for n in nodes_evaluated:
            
            y = self.presence_by_node(m,cells, n)   

            
            z = self.event_by_node(m, cells,n, snv_bin_mapping)
            states = np.unique(z)
            like0_vals = np.multiply((1-y), like0).sum(axis=1)
            like1_vals =np.zeros_like(cells, dtype=float)
            for s in states:
                like1_events = like1[s][cells,:]
                
                mask = np.logical_or(y != 1, z !=s)
                if np.all(mask):
                    continue
                like1_events = np.ma.array(like1_events, mask= mask).sum(axis=1)
                like1_vals += like1_events
            
            total_like = like0_vals + like1_vals
            like_by_cell = pd.Series(total_like, index=cells)
    
    
            _, rdr_cell_like  =self.rdr_likelihood_by_node(n, cna_hmm, cells)
            like_by_cell = like_by_cell.add(rdr_cell_like.sort_index())

            like_list.append(like_by_cell.to_numpy().reshape(-1,1))
        

        all_cell_likelihoods = np.hstack(like_list)
        if len(nodes_evaluated) > 1:
            cell_assignment = np.argmax(all_cell_likelihoods, axis=1)
            # total = np.max(all_cell_likelihoods, axis=1).sum()
        else:
            cell_assignment = np.full(len(cells), 0, dtype=int)
           
        new_cell = {}
        for i,n in enumerate(nodes_evaluated):
            new_cell[n] = np.array(cells[np.argwhere(cell_assignment==i)]).reshape(-1)
        
        self.update_cell_dict(cells, new_cell)

    def event_post_processing(self, cna_hmm):
        for n in self.cell_mapping:
            cells = self.get_tip_cells(n)
            if len(cells) > 0:
                self.event_mapping[n]= cna_hmm.run(cells)

    def mut_post_processing_with_cna(self, muts, like0, like1, snv_bin_mapping):
        m = like0.shape[1]

        cells = self.get_all_cells()
      
        mut_mapping = {} 
        
        #identify any lost muts, reassign them to the same node they were introduced on and remove them
        # from the set of muts to be post_processed 
        if len(self.mut_loss_mapping) > 0:
            all_lost_muts = np.concatenate([self.mut_loss_mapping[k] for k in self.mut_loss_mapping])
            muts = np.setdiff1d(muts, all_lost_muts)

        #same order as self.get_all_cells()
    
        ncells = like0.shape[0]
        cell_geno_dict = self.get_cell_genotypes(ncells)
        like0 = like0[cells,:]
        like1 = like1["neutral"][cells,:]
        like_list = []
        nodes_evaluated = []
        for n in cell_geno_dict:
            nodes_evaluated.append(n)

            #cg needs to be reordered by cells to match the order of the likelihood matrices
            cg = cell_geno_dict[n][cells].reshape(-1,1)
            mut_likelihoods= np.multiply(1-cg, like0).sum(axis=0) + np.multiply(cg, like1).sum(axis=0)
            like_list.append(mut_likelihoods[muts])

        all_mut_likelihoods = np.vstack(like_list)
        if len(nodes_evaluated) > 1:
            mut_assignment = np.argmax(all_mut_likelihoods, axis=0)
  
        else:
            mut_assignment = np.full(len(muts), 0, dtype=int)
           
        
        for i,n in enumerate(nodes_evaluated):
            mut_mapping[n] = np.array(muts[np.argwhere(mut_assignment==i)]).reshape(-1).astype(int)
        

        self.update_mut_dict(muts, mut_mapping)   

    def update_mut_dict(self, muts, new_assign):
        for n in self.mut_mapping:
            self.mut_mapping[n] = np.setdiff1d(self.mut_mapping[n], muts)
            self.mut_mapping[n] = np.concatenate([self.mut_mapping[n], new_assign[n]]).astype(int)

    def update_cell_dict(self, cells, new_assign):
        for n in self.cell_mapping:
            remaining_cells= np.setdiff1d( self.get_tip_cells(n), cells)
            self.cell_mapping[n] = {0:remaining_cells}
            if n in new_assign:
                self.cell_mapping[n] = {0: np.concatenate([remaining_cells, new_assign[n]]).astype(int)}


    def prune_nodes(self,lamb, tau, min_loss):
            pruned_cells = []
            prune_muts =[]
            illegal_nodes = self.find_illegal_nodes(lamb, tau, min_loss)
            for n in illegal_nodes:
                parent = list(self.tree.predecessors(n))
                children = list(self.tree.successors(n))
                self.tree.remove_node(n)
                if len(parent)  > 0:
                    for c in children:
                        self.tree.add_edge(parent[0], c)
                pruned_cells.append(self.get_tip_cells(n))
                prune_muts.append(self.get_tip_muts(n))
                del self.cell_mapping[n]
                del self.mut_mapping[n]
                if n in self.mut_loss_mapping:
                    del self.mut_loss_mapping[n]
                if self.event_mapping is not None:
                    if n in self.event_mapping:
                        del self.event_mapping[n]
            
            if len(pruned_cells) > 0:
                pruned_cells = np.concatenate(pruned_cells)
            else:
                pruned_cells= np.array(pruned_cells, dtype=int)
            if len(prune_muts) > 0:
                prune_muts = np.concatenate(prune_muts)
            else:
                prune_muts = np.array(prune_muts, dtype=int)
            return illegal_nodes, pruned_cells, prune_muts
            
            
    
    def find_illegal_nodes(self, lamb, tau, min_loss):
    
            empty_nodes = []
            illegal_leaves = []
            for x in list(self.tree.nodes()):
                if x in self.mut_loss_mapping:
                    num_muts = len(self.mut_loss_mapping[x])
                    invalid_muts = num_muts <= min_loss
                else:
                    num_muts = len(self.get_tip_muts(x))
                    invalid_muts = num_muts <=tau
          
                num_cells = len(self.get_tip_cells(x))
                invalid_cells = num_cells <= lamb
                if num_muts ==0 and num_cells == 0:
                    empty_nodes.append(x)
                else:
                    if (invalid_cells or invalid_muts) and self.tree.out_degree(x) ==0:
                        illegal_leaves.append(x)


            illegal_leaves.extend(empty_nodes)
            return illegal_leaves


class LinearTree(ClonalTree):
    def __init__(self, cellsA, cellsB, mutsA, mutsB, eA, eB, key = None):    
        t = nx.DiGraph()
        t.add_node(0, ncells=len(cellsA))
        t.add_node(1, ncells=len(cellsB))
        t.add_edge(0,1)
        cm, mm, ml, em = {}, {}, {}, {}
        cm[0] = {0 : cellsA }
        cm[1] = {0: cellsB}
        
        mm[0] = mutsA
        mm[1] =mutsB
        em[0] = eA
        em[1] = eB
        super().__init__(t, cm, mm, ml, em )

    def is_valid(self, lamb, tau):
       return len(self.get_tip_cells(0)) > lamb and len(self.mut_mapping[0]) > tau and len(self.get_tip_cells(1)) > lamb and len(self.get_tip_muts(1)) > tau
          
    
    def get_seeds(self, lamb, tau, ancestral_muts):
        seed_list = []
        
        cellsB = self.get_tip_cells(1)
        mutsB = self.get_tip_muts(1)
        if len(cellsB) > lamb and len(mutsB) > tau:
            anc_muts  = np.sort(np.union1d(ancestral_muts, self.mut_mapping[0]))
            seed_list.append(Seed(cellsB, mutsB, anc_muts))
        
        return seed_list


class BranchingTree(ClonalTree):
    def __init__(self, cellsA, cellsB, mutsA, mutsB,mutsC, eA, eB,eC=None, key = None):    
        t = nx.DiGraph()
        t.add_node(0, ncells=0)
        t.add_node(1, ncells= len(cellsA))
        t.add_node(2, ncells= len(cellsB))
        t.add_edge(0,1)
        t.add_edge(0,2)
        cm, mm, ml, em = {}, {}, {}, {}
        cm[0] = {0 : np.empty(shape=0, dtype=int)}
        cm[1] = {0: cellsA}
        cm[2] = {0: cellsB}
        
        mm[0] = mutsC
        mm[1] =mutsA
        mm[2] = mutsB
        if eA is not None and eB is not None:
            em[1] = eA
            em[2] = eB
        
        ml = {}
        super().__init__(t, cm, mm, ml, em )

    def is_valid(self, lamb, tau):
        if (len(self.get_tip_cells(1)) > lamb and len(self.mut_mapping[1]) > tau) or  (len(self.get_tip_cells(2)) > lamb and len(self.mut_mapping[2]) > tau):
            return True
        else:
            return False



    def get_seeds(self, lamb, tau, ancestral_muts):

        seed_list = []
        leaves = [1,2]
        ancestral_muts = np.sort(np.union1d(ancestral_muts, self.mut_mapping[0]))
        for l in leaves:

            cells = self.get_tip_cells(l)
            muts = self.mut_mapping[l]
            if len(cells) > lamb and len(muts) > tau:
                seed_list.append(Seed(cells, muts, ancestral_muts))
            
        return seed_list
 


class IdentityTree(ClonalTree):
    def __init__(self, cells, muts, events):    
        t = nx.DiGraph()
        t.add_node(0, ncells=len(cells))

        cm, mm, ml, em = {}, {}, {}, {}
        cm[0] = {0: cells}
        mm[0] = muts
        em[0] = events
        super().__init__(t, cm, mm, ml, em )


    def is_valid(self, lamb, tau):
        return True
    
    def get_seeds(self, lamb, tau, ancestral_muts=None):
        return []


class LossTree(ClonalTree):
    def __init__(self, cellsA, cellsB, muts_loss, mutsB, eA, eB, key = None):    
        t = nx.DiGraph()
        t.add_node(0, ncells=len(cellsA))
        t.add_node(1, ncells= len(cellsB))
        t.add_edge(0,1)
        cm, mm, ml, em = {}, {}, {}, {}
        cm[0] = {0 : cellsA }
        cm[1] = {0: cellsB}
        
        mm[0] = np.empty(shape=0, dtype=int)
        mm[1] =mutsB
        ml[0] = muts_loss
        if eA is not None:
           em[0] = eA
        
        em[1] = eB
     
        super().__init__(t, cm, mm, ml, em, key )

    def is_valid(self, lamb, tau, min_loss=0):
          return len(self.mut_loss_mapping[0]) > min_loss


    
    def get_seeds(self, lamb, tau, ancestral_muts):
        seed_list = []
        cellsB = self.get_tip_cells(1)
        mutsB = self.mut_mapping[1]
        anc_muts  = np.sort(np.setdiff1d(ancestral_muts, self.mut_loss_mapping[0]))
        seed_list.append(Seed(cellsB, mutsB, anc_muts))
    
        return seed_list

  
 


