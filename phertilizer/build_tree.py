from typing import Tuple
from numpy import NaN, dtype
import argparse
import networkx as nx
import pandas as pd 
import numpy as np 
from clonal_tree import ClonalTree
# from phertilizer import Phertilizer
import pygraphviz as pgv
import pickle
import re


class BuildTree():
    def __init__(self, data=None):

        self.clonal_tree= None
        self.data = data 


#like0, like1,snv_bin_mapping=None, cnn_hmm=None
    def build(self,gt_cell, gt_mut,edge_list, gt_mut_loss=None, gt_events=None):
        
        cell_mapping = self.construct_mapping(gt_cell, label="cell")
        
        tree,cell_mapping = self.construct_tree(edge_list, cell_mapping)

        mut_mapping = self.construct_mapping(gt_mut, label="mutation")
        if gt_events is not None:
            event_mapping = self.construct_event_mapping(gt_events)
        
        if gt_mut_loss is not None:
            mut_loss_mapping = self.construct_mapping(gt_mut_loss, "mutation")
    
        mut_loss_mapping = None
        if gt_mut_loss is not None:
            mut_loss_mapping = self.construct_mut_loss_mapping(gt_mut_loss)
        event_mapping = None
        if gt_events is not None:
            event_mapping = self.construct_event_mapping(gt_events)
        self.clonal_tree = ClonalTree(tree, cell_mapping, mut_mapping, mut_loss_mapping, event_mapping)
        if self.data is not None:
            self.clonal_tree.compute_likelihood(self.data)

        return self.clonal_tree

    def build_phertilizer(self, cell, mut, edge_pickle: str, mut_loss=None, events=None):
        
        cell_mapping = self.construct_mapping(cell, label="cell")
        
        edge_list = self.read_edge_pickle(edge_pickle)
        tree,cell_mapping = self.construct_tree(edge_list, cell_mapping)


        # mutation
        mut_mapping = {}
        for clust, group in mut.groupby('cluster'):
            mutations = np.sort(group['mutation'].str.split('_').str[1].to_numpy(dtype=int))
            mut_mapping[clust] = mutations
        
        for n in tree.nodes():
            if n not in mut_mapping:
                mut_mapping[n] = np.empty(shape=0, dtype=int)
        # mut_mapping = self.construct_mapping(mut, label="mutation")

        mut_loss_mapping = None
        if mut_loss is not None:
            mut_loss_mapping = {}
            for clust, group in mut_loss.groupby('cluster'):
                mutations = np.sort(group['mutation'].str.split('_').str[1].to_numpy(dtype=int))
                mut_loss_mapping[clust] = mutations

        # CNA events
        event_mapping = None
        if events is not None:
            if "bin" in events.columns:
                events.set_index("bin", inplace=True)
            nodes = [int(x.split("_")[1]) for x in events.columns]
            event_mapping = {
                node: {
                    "gain": [],
                    "neutral": [],
                    "loss": [],
                } for node in nodes
            }
            for idx_bin, row in events.iterrows():
                for i, node in enumerate(nodes):
                    event_mapping[node][row.iloc[i]].append(idx_bin)

        self.clonal_tree = ClonalTree(tree, cell_mapping, mut_mapping, mut_loss_mapping, event_mapping)
        if self.data is not None:
            self.clonal_tree.compute_likelihood(
                self.data["like0"],
                self.data["like1_dict"],
                self.data["snv_bin_mapping"],
                self.data["cna_hmm"])

        return self.clonal_tree

    @staticmethod
    def get_sbmclone_clustering(clustering) -> Tuple[np.ndarray, np.ndarray]:
        with open(clustering) as ifile:
            cell_cluster = np.array(next(ifile).strip().split(',')).astype(int)
            mut_cluster = np.array(next(ifile).strip().split(',')).astype(int)
            return cell_cluster, mut_cluster

    def build_sphyr(self, clustering, tree_graphviz: str):
        cell_cluster, mut_cluster = self.get_sbmclone_clustering(clustering)
        
        cell_mapping = {clust:
            {0: (cell_cluster==clust).nonzero()[0]}
            for clust in np.unique(cell_cluster)}
        mut_mapping = {clust:
            (mut_cluster==clust).nonzero()[0]
            for clust in np.unique(mut_cluster)}
        
        edge_list, cell_mapping, mut_mapping, mut_loss_mapping = self.sphyr_graphviz(tree_graphviz)
        tree, cell_mapping = self.construct_tree(edge_list, cell_mapping)
    
        
    

        self.clonal_tree = ClonalTree(tree, cell_mapping, mut_mapping, mut_loss_mapping)
        if self.data is not None:
            self.clonal_tree.compute_likelihood(
                self.data["like0"],
                self.data["like1_dict"],
                self.data["snv_bin_mapping"],
                self.data["cna_hmm"])

        return self.clonal_tree, mut_cluster.shape[0]
    
    @staticmethod
    def construct_tree(edges, cell_mapping):

        tree = nx.DiGraph()
        nodes = []
        for edge in edges:
            for n in edge:
                if n not in nodes:
                    nodes.append(n)
                    if n in cell_mapping:
                        tree.add_node(n, ncells=len(cell_mapping[n][0]))
                    else:
                        cell_mapping[n] = {0: np.empty(shape=0, dtype=int)}
                        tree.add_node(n, ncells=0)
            tree.add_edge(edge[0], edge[1])

        return tree,cell_mapping, 

    
    @staticmethod
    def construct_tree(edges, cell_mapping):

        tree = nx.DiGraph()
        nodes = []
        for edge in edges:
            for n in edge:
                if n not in nodes:
                    nodes.append(n)
                    if n in cell_mapping:
                        tree.add_node(n, ncells=len(cell_mapping[n][0]))
                    else:
                        cell_mapping[n] = {0: np.empty(shape=0, dtype=int)}
                        tree.add_node(n, ncells=0)
            tree.add_edge(edge[0], edge[1])

        return tree,cell_mapping, 

    @staticmethod
    def construct_mapping(gt, label):
        clones = gt['cluster'].unique()
        mapping = {}

        for c in clones:
            
            df = gt[gt['cluster'] ==c]
            cells = np.sort(df[label].to_numpy())
            if label == "cell":
                mapping[c] = {0: cells}
            else:
                mapping[c] = cells 

        return mapping
    
    @staticmethod
    def construct_mut_loss_mapping(gt_mut_loss):
        groups = gt_mut_loss.groupby('cluster')
        return {c: group["mutation"].sort_values().values for c, group in groups}

    @staticmethod
    def construct_event_mapping(gt_event):
        gt_event = gt_event.set_index("genotype")
        event_mapping = {}
        node = 0
        for index, genotype in gt_event.iterrows():
            event_mapping[node] = {s : [] for s in ["gain", "loss", "neutral"]}
            
            for bin_name, value in genotype.items():
                bin_num = int(bin_name[3:])
                if value == 2:
                    event_mapping[node]["neutral"].append(bin_num)
                elif value > 2:
                    event_mapping[node]["gain"].append(bin_num)
                else:
                    event_mapping[node]["loss"].append(bin_num)
        
            event_mapping[node] = {s : np.array(event_mapping[node][s], dtype=int) for s in ["gain", "loss", "neutral"]}
            node +=1

        return event_mapping



    # def compute_likelihood(self):
    #     self.clonal_tree
    #     pass 
        #unpack data
        # self.clonal_tree.compute_likelihood()

    @staticmethod
    def read_edge_list(tree_file: str) -> list:
        """Read edge list from tree file

        Args:
            tree_file (str): tree file

        Returns:
            list: edge list
        """
        edge_list = []
        with open(tree_file, 'r') as tree_file:
            input_tree = tree_file.readlines()
        for line in input_tree:
            if "#leaves" in line:
                break
            elif "#edges" in line:
                continue
            else:
                edge = line.rstrip().split(" ")
                edge = tuple([int(e) for e in edge])
                edge_list.append(edge)
        return edge_list

    @staticmethod
    def read_edge_pickle(edge_pickle: str) -> list:
        """Read edge list from phertilizer pickle

        Args:
            edge_pickle (str): pickle file

        Returns:
            list: edge list
        """
        edge_list = []
        # TODO: fix relabel
        with open(edge_pickle, 'rb') as tree:
            
            input_tree = pickle.load(tree)
            for x, y in input_tree.tree.edges:
                edge_list.append((int(x), int(y)))
            return edge_list

    @staticmethod
    def sphyr_graphviz(tree_graphviz: str, cell_cluster: str=None, mut_cluster: str=None) -> list:
        """Parse SPhyR GraphViz dot file

        Args:
            tree_graphviz (str): graphviz tree file

        Returns:
            list: edge list
        """
        tree = pgv.AGraph(tree_graphviz)
        nodes = tree.nodes()
        edges = tree.edges()
        edge_list = []
        mut_mapping = {}
        mut_loss_mapping = {}
        for edge in edges:
            snvs = edge.attr["label"]
            if len(snvs) > 0:
                snvs = snvs.split("\\n")
                add_snvs = [snv for snv in snvs if snv.startswith("S") ]
                loss_snvs = [snv for snv in snvs if snv.startswith("-")]
                mut_mapping[int(edge[1])] = np.array([snv[3:] for snv in add_snvs]).astype(int)
                # starts with '--SNV'
                mut_loss_mapping[int(edge[1])] = np.array([snv[5:] for snv in loss_snvs]).astype(int)
            else:
                mut_mapping[int(edge[1])] = np.array([], dtype=int)
            edge_list.append((int(edge[0]), int(edge[1])))
        cell_mapping = {}
        for node in nodes:
            cells = node.attr["label"]
            if len(cells) > 0:
                cells = re.split(r"\\n| ", cells)
                cell_mapping[int(node)] = {0: np.array([cell[4:] for cell in cells]).astype(int)}
            else:
                cell_mapping[int(node)] = {0: np.array([], dtype=int)}
            if int(node) not in mut_mapping:
                mut_mapping[int(node)] = np.array([], dtype=int)
        return edge_list, cell_mapping, mut_mapping, mut_loss_mapping


def main(args):
    # with open("/scratch/data/chuanyi/phertilizer/test/grow_tree.pickle", "rb") as ifile:
    #     tree = pickle.load(ifile)
    #     pairs = tree.get_incomparable_pairs()

    gt_cell = pd.read_csv(args.cell_clust)
    gt_mut =  pd.read_csv(args.mut_clust)
    gt_loss = None
    if args.mut_loss is not None:
        gt_loss = pd.read_csv(args.mut_loss)
    events = None
    if args.events is not None:
        events = pd.read_csv(args.events)

    edge_list = BuildTree.read_edge_list(args.tree)

    clonal_tree = BuildTree().build(gt_cell, gt_mut, edge_list, gt_loss, events)
    # ancestral = clonal_tree.get_ancestor_pairs()
    # print(f"ancestral: {len(ancestral)}")
    # clustered = clonal_tree.get_cluster_pairs()
    # print(f"clustered: {len(clustered)}")
    # incomparable = clonal_tree.get_incomparable_pairs()
    # print(f"incomparable: {len(incomparable)}")
    clonal_tree.tree_png(args.png, "/scratch/data/chuanyi/phertilizer/simulations/phert_results/chrom_map.csv")
    
    
#     # data = pickle_load(args.data)
#     # if args.likelihood is not None:
#     #     like_df = read_csv(args.likelihood)
#     #     like = like_df["loglikelihood"].to_numpy()
#     #     varlike = like_df["variant_data_likelihood"].to_numpy()[0]
#     #     binlike =like_df["bin_count_likelihood"].to_numpy()[0]
#     #     like= like[0]
#     # else:
#     #     like = None
#     #     varlike= None
#     #     binlike = None



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--cell_clust", 
        help="ground truth cell clusters")
    parser.add_argument("-m", "--mut_clust", 
        help="ground truth_mut_clusters")
    parser.add_argument("-j", "--mut_loss", 
        help="ground truth_mutation loss clusters")
    parser.add_argument("-e", "--events", 
        help="ground truth of events")
    parser.add_argument("-t", "--tree", 
        help="ground truth tree file")
    parser.add_argument("-p", "--png", 
        help="output file for png")
    

    # args = parser.parse_args()
    # folder = "n5000_m2500"
    # seed = 13
    # prob = "0.01"
    # dcl = 2
    # dsnv = 2
    # dcnv = 2
    # cna = 1
    # loss = 0
    # loh = 0
    # clones =5
    
    # starts = 2
    # iterations = 5
    # lamb = 150
    # tau = 150
    # min_loss = 100

    # prune = 50
    # reg = 0.15
    # nbindist= 10
    # vafthresh = 0.033
    # tmb = 0.0
    # neutral_mean = 1.0
    # neutral_eps = 0.15
    # output= f"test/starts{starts}_iterations{iterations}_lamb{lamb}_tau{tau}_minloss{min_loss}_prune{prune}_reg{reg}_nbindist{nbindist}_vf{vafthresh}_tmb{tmb}"

    # prefix = f"s{seed}_{folder}_c{clones}_p{prob}_cna1_l{loss}_loh{loh}_dcl{dcl}_dsnv{dsnv}_dcnv{dcnv}"
    # prefix = "s12_n2500_m5000_c5_p0.01_cna1_l2_loh2_dcl2_dsnv2_dcnv2"

    # path = "/scratch/data/leah/phertilizer/simulations/input"
   
    # args = parser.parse_args([
    #     "-c", f"{path}/{prefix}_cellclust_gt.csv",
    #     "-m", f"{path}/{prefix}_mutclust_gt.csv",
    #     "-t", f"{path}/{prefix}_tree.txt",
    #     "-j", f"{path}/{prefix}_mut_loss_clust_gt.csv",
    #     "-e", f"{path}/{prefix}_copy_number_profiles.csv",
    #     "-p", f"simulations/test/{prefix}_true_tree.png"
    # ])
    
    main(args)