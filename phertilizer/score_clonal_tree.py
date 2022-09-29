from cmath import inf
from dataclasses import dataclass
from numpy import NaN
import argparse
import networkx as nx
import pandas as pd 
from clonal_tree import ClonalTree
from sklearn.metrics.cluster import adjusted_rand_score
from utils import pickle_load, pickle_save
from typing import Dict, Tuple
from build_tree import BuildTree
import numpy as np

PHERTILIZER = "phertilizer"
SPHYR = "sphyr"
SCG = "SCG"
BASELINE = "baseline"
SCITE= "SCITE"
METHODS = [PHERTILIZER, SPHYR, SCG, BASELINE, SCITE ]

@dataclass
class TreeEval():
    gt_tree: ClonalTree
    inferred_tree: ClonalTree
    n_mut:  int
    cell_series = None
    mut_series = None
  

    def __post_init__(self):
        if self.cell_series is not None:
            #fix the indices of inf_df 
            pass 
        
        if self.mut_series is not None:
            #fix the indices of inf_df 
            pass 
      




    @staticmethod
    def ari(v1, v2) -> float:
        return adjusted_rand_score(v1, v2)

    def ancestor_pair_recall(self, include_loss: bool=True) -> float:
        ancestral = self.gt_tree.get_ancestor_pairs(include_loss)
        ancestral_recall = sum((ancestral & self.inferred_tree.get_ancestor_pairs(include_loss)).values())\
                           / sum(ancestral.values())
        return ancestral_recall
    
    def clustered_pair_recall(self, include_loss: bool=True) -> float:
        clustered = self.gt_tree.get_cluster_pairs(include_loss)
        clustered_recall = sum((clustered & self.inferred_tree.get_cluster_pairs(include_loss)).values())\
                           / sum(clustered.values())
        return clustered_recall

    def incomparable_pair_recall(self, include_loss: bool=True) -> float:
        incomparable = self.gt_tree.get_incomparable_pairs(include_loss)
        if sum(incomparable.values()) == 0:
            return 1
        incomparable_recall = sum((incomparable & self.inferred_tree.get_incomparable_pairs(include_loss)).values())\
                           / sum(incomparable.values())
        return incomparable_recall

    def compute_cell_ari(self) -> float:
        gt_cell = self.gt_tree.get_cell_clusters()
        pred_cell = self.inferred_tree.get_cell_clusters()
        return self.ari(gt_cell, pred_cell)
    
    def compute_nclusters(self) -> float:
        pred_cell = self.inferred_tree.get_cell_clusters()

        return np.unique(pred_cell).shape[0]
    
       
    def compute_gtclusters(self) -> float:
        gt_cell = self.gt_tree.get_cell_clusters()

        return np.unique(gt_cell).shape[0]

    
    def compute_mut_ari(self) -> float:
        gt_mut = self.gt_tree.get_mut_clusters()

        try:
            pred_mut = self.inferred_tree.get_mut_clusters()
        except:
            pred_mut = self.inferred_tree.get_mut_clusters(len(gt_mut))

        return self.ari(gt_mut, pred_mut)

    @staticmethod
    def _event_dist(a, b) -> float:
        n = max(x.max() for x in a.values() if len(x) > 0) + 1
        events_a = np.zeros(n)
        events_b = np.zeros(n)
        events_a[a["loss"]] = 1
        events_a[a["neutral"]] = 2
        events_a[a["gain"]] = 3
        events_b[b["loss"]] = 1
        events_b[b["neutral"]] = 2
        events_b[b["gain"]] = 3
        return sum(events_a != events_b) / n

    def event_hamming(self) -> float:
        if len(self.gt_tree.event_mapping) == 0 or len(self.inferred_tree.event_mapping) == 0:
            return float("nan")
        scores = {}
        for gt_node, gt_events in self.gt_tree.event_mapping.items():
            for infer_node, infer_events in self.inferred_tree.event_mapping.items():
                scores[(gt_node, infer_node)] = self._event_dist(gt_events, infer_events)
        hamming = []
        for gt_clust, infer_clust in zip(self.gt_tree.get_cell_clusters(), self.inferred_tree.get_cell_clusters()):
            hamming.append(scores[(gt_clust, infer_clust)])
        return np.mean(hamming)

    @staticmethod
    def _genotype_dist(a, b) -> float:
        return sum(a != b) / len(a)

    def genotype_hamming(self) -> float:
        scores = {}
        # TODO: accumulate variants, including loss
        for gt_node, gt_genotype in self.gt_tree.snv_genotypes().items():
            for infer_node, infer_genotype in self.inferred_tree.snv_genotypes(self.n_mut).items():
                scores[(gt_node, infer_node)] = self._genotype_dist(gt_genotype, infer_genotype)
        hamming = []
        for gt_clust, infer_clust in zip(self.gt_tree.get_cell_clusters(), self.inferred_tree.get_cell_clusters()):
            hamming.append(scores[(int(gt_clust), int(infer_clust))])
        
        return np.mean(hamming)
    


    def loss_metrics(self) -> Dict[str, float]:
        if len(self.gt_tree.mut_loss_mapping) == 0:
            gt_loss = np.array([], dtype=int)
        else:
            gt_loss = np.concatenate([self.gt_tree.mut_loss_mapping[i] for i in self.gt_tree.mut_loss_mapping])
        if len(self.inferred_tree.mut_loss_mapping) == 0:
            infer_loss = np.array([], dtype=int)
        else:
            infer_loss = np.concatenate([self.inferred_tree.mut_loss_mapping[i] for i in self.inferred_tree.mut_loss_mapping])
        TP = len(np.intersect1d(gt_loss, infer_loss))
        FN = len(np.setdiff1d(gt_loss, infer_loss))
        FP = len(np.setdiff1d(infer_loss, gt_loss))
        if TP + FP > 0:
            precision = TP / (TP+FP)
        else:
            precision = float("nan")
        if TP + FN > 0:
            recall = TP / (TP+FN)
        else:
            recall = float("nan")
        return {
            "precision": precision,
            "recall": recall,
            "TP": TP,
            "FP": FP,
            "FN": FN,
        }
    

    
    def score(self, include_loss: bool=True) -> Dict[str, float]:
        """Compute cell & mutation ari; ancestor pair, incomparable pair, cluster recall; precision, event_score

        Returns:
            Dict[str, float]: _description_
        """
        loss_metric = self.loss_metrics()
        return {
            "cell_ari": self.compute_cell_ari(),
            "mut_ari": self.compute_mut_ari(),
            "ancestral_recall": self.ancestor_pair_recall(include_loss),
            "incomparable_recall": self.incomparable_pair_recall(include_loss),
            "clustered_recall": self.clustered_pair_recall(include_loss),
            "event_hamming": self.event_hamming(),
            "genotype_hamming": self.genotype_hamming(),
            "loss_precision": loss_metric["precision"],
            "loss_recall": loss_metric["recall"],
            "TP": loss_metric["TP"],
            "FP": loss_metric["FP"],
            "FN": loss_metric["FN"],
            "nclusters" : self.compute_nclusters(),
            "gtclusters" : self.compute_gtclusters()
        }



def get_baseline_clustering(clustering) -> Tuple[np.ndarray, np.ndarray]:
        with open(clustering) as ifile:
            cell_cluster = np.array(next(ifile).strip().split(',')).astype(int)
            mut_cluster = np.array(next(ifile).strip().split(',')).astype(int)
            return cell_cluster, mut_cluster

def score_no_tree(gt_tree, infer_genos, infer_cell_clust) -> float:
     
        #compute hamming 
        gt_clust = gt_tree.get_cell_clusters()
        gt_genos = gt_tree.snv_genotypes()
        print(infer_genos.index)

  

        scores = np.zeros(infer_genos.shape[0], dtype=float)

        for index, row in infer_genos.iterrows():
            cell_cluster = gt_clust[index]
            infer = row.values
            gt = gt_genos[cell_cluster]
            scores[index]= sum(infer!= gt) / len(gt)

        score = {
            "cell_ari" : adjusted_rand_score(gt_clust, infer_cell_clust),
            "genotype_hamming" : np.mean(scores)
             
        }

  
        
        return score



        # # TODO: accumulate variants, including loss
        # for gt_node, gt_genotype in gt_tree.snv_genotypes().items():
        #     for infer_cluster, infer_genotype in 
        #         scores[(gt_node, infer_node)] = self._genotype_dist(gt_genotype, infer_genotype)
        # hamming = []
        # for gt_clust, infer_clust in zip(self.gt_tree.get_cell_clusters(), self.inferred_tree.get_cell_clusters()):
        #     hamming.append(scores[(int(gt_clust), int(infer_clust))])
        
        # return np.mean(hamming)

def main(args):
    gt_cell = pd.read_csv(args.cell_clust)
    gt_mut =  pd.read_csv(args.mut_clust)
    edge_list = BuildTree().read_edge_list(args.tree)
    gt_loss = None
    if args.mut_loss is not None:
        gt_loss = pd.read_csv(args.mut_loss)
    gt_events = None
    if args.events is not None:
        gt_events = pd.read_csv(args.events)
    gt_tree = BuildTree().build(gt_cell, gt_mut, edge_list, gt_loss, gt_events)

    if args.method == PHERTILIZER or args.method == SCITE:
        infer_cell = pd.read_csv(args.infer_cell_clust)
        infer_mut =  pd.read_csv(args.infer_mut_clust)
        infer_loss = None
        if args.infer_mut_loss is not None:
            infer_loss = pd.read_csv(args.infer_mut_loss)
        infer_events = None
        if args.infer_events is not None:
            infer_events = pd.read_csv(args.infer_events)
        inferred_tree = BuildTree().build_phertilizer(infer_cell, infer_mut, args.infer_tree, infer_loss, infer_events)
        all_muts = inferred_tree.get_all_muts()
        tree_eval = TreeEval(gt_tree, inferred_tree, all_muts.shape[0])
    elif args.method == SPHYR:
        # SPhyR prediction
        inferred_tree, n_mut = BuildTree().build_sphyr(
            args.infer_cell_clust,
            args.infer_tree)
        tree_eval = TreeEval(gt_tree, inferred_tree, n_mut)
    elif args.method == SCITE:
        inferred_tree = pickle_load(args.infer_tree)
        inferred_tree.relabel()
        n_mut = len(inferred_tree.get_all_muts())
        tree_eval = TreeEval(gt_tree, inferred_tree, n_mut)
        
    elif args.method == BASELINE:
         inferred_genos = pd.read_table(args.infer_mut_clust, header=None, skiprows=2, delim_whitespace=True)
 
         inferred_snv_labels = pd.read_csv(args.snv_labels,header=None, names=["snv"])
         inferred_cell_labels = pd.read_csv(args.cell_labels, header=None, names=["cell"])

         inferred_cell_labels['cell'] = inferred_cell_labels["cell"].str.replace("cell", "").astype(int)
 
         inferred_genos["cell"] = inferred_cell_labels["cell"]

         inferred_genos =inferred_genos.set_index("cell")
         inferred_genos.columns = inferred_snv_labels['snv'].to_numpy()
         print(inferred_genos.head())

            #get the cell index
         cell_cluster, mut_cluster = get_baseline_clustering(args.infer_cell_clust)

         score =score_no_tree(gt_tree, inferred_genos,cell_cluster)
         metric = pd.DataFrame(score, index=[0])
         metric.to_csv(args.output, index=False)
         return 
    
 
    
    # # SciCloneFit prediction
    # infer_cell = pd.read_csv(args.infer_cell_clust)
    # infer_mut =  pd.read_csv(args.infer_mut_clust)
    # if args.infer_mut_loss is not None:
    #     infer_loss = pd.read_csv(args.infer_mut_loss)
    # infer_events = None
    # if args.infer_events is not None:
    #     infer_events = pd.read_csv(args.infer_events)
    # inferred_tree = BuildTree().build_sciclonefit(infer_cell, infer_mut, args.infer_tree, infer_loss, infer_events)


    metric = pd.DataFrame(tree_eval.score(not args.no_loss), index=[0])
    metric.to_csv(args.output, index=False)



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
    parser.add_argument("-l", "--likelihood", 
        help="ground truth likelihood file")
    parser.add_argument("-C", "--infer_cell_clust", 
        help="inferred cell clusters")
    parser.add_argument("-M", "--infer_mut_clust", 
        help="inferred_mut_clusters")
    parser.add_argument("-J", "--infer_mut_loss", 
        help="inferred_mutation loss clusters")
    parser.add_argument("-E", "--infer_events", 
        help="inferred of events")
    parser.add_argument("-T", "--infer_tree", 
        help="inferred tree file")
    parser.add_argument("-L", "--infer_likelihood", 
        help="inferred likelihood file")
    parser.add_argument( "--cell_labels", 
        help="list of cell labels")
    parser.add_argument( "--snv_labels", 
        help="list of snv labels")        
    parser.add_argument("-o", "--output", 
        help="output csv file")
    parser.add_argument("-z", "--method",
        type=str,
        choices=METHODS,
        default=PHERTILIZER,
        help="method")
    parser.add_argument("--no-loss", action='store_true', default=False,
        help="Set flag to disable loss computing in tree metrics")
    parser.add_argument("-p", "--png", 
        help="output file for png")
    

    args = parser.parse_args()

    # folder = "n5000_m2500"
    # # seed = 13
    # pth = "/scratch/data/leah/phertilizer/simulations/baseline/var_thresh2"
    # gt_pth = "/scratch/data/chuanyi/phertilizer/simulations"
    # gt_folder = 's12_n1500_m1500_c5_p0.01_cna1_l0_loh0_dcl2_dsnv2_dcnv2'
    # infer_folder = f"{pth}/{gt_folder}"
    # # gt_folder = "s16_n2500_m1500_c7_p0.01_cna1_l2_loh2_dcl2_dsnv2_dcnv2"
    # # infer_folder = "simulations/phertilizer/genome_biology/clones7_l2_loh2/s16_n2500_m1500_c7_p0.01_cna1_l2_loh2_dcl2_dsnv2_dcnv2/dollo/starts5_iterations10_minfrac0.1_minloss30_kneighbors5_lreadthresh5"
    # # # # # gt_folder = "s15_n5000_m2500_c5_p0.01_cna1_l2_loh2_dcl2_dsnv2_dcnv2"
    # # # # # infer_folder = "not_hierarchical/s15_n5000_m2500_c5_p0.01_cna1_l2_loh2_dcl2_dsnv2_dcnv2"
    # args = parser.parse_args([
    #     "-c", f"/scratch/data/chuanyi/phertilizer/simulations/preprocess/{gt_folder}/cellclust_gt.csv",
    #     "-m", f"/scratch/data/chuanyi/phertilizer/simulations/preprocess/{gt_folder}/mutclust_gt.csv",
    #     "-j", f"/scratch/data/chuanyi/phertilizer/simulations/preprocess/{gt_folder}/mut_loss_clust_gt.csv",
    #     "-e", f"/scratch/data/chuanyi/phertilizer/simulations/input/{gt_folder}_copy_number_profiles.csv",
    #     "-t", f"/scratch/data/chuanyi/phertilizer/simulations/input/{gt_folder}_tree.txt",
    #     "-M", f"{infer_folder}/sphyr_in.txt",
    #     "-C", f"{infer_folder}/cluster-assignments.txt",
    #     "--cell_labels",  f"{infer_folder}/sphyr_cell_labels.txt",
    #     "--snv_labels",  f"{infer_folder}/sphyr_snv_labels.txt",
    #     "-z", "baseline",
    #     "-o", f"{infer_folder}/metrics_baseline.csv"


    # # # #     # phertilizer inferred
    # #     "-C", f"/scratch/data/leah/phertilizer/{infer_folder}/pred_cell.csv",
    # #     "-M", f"/scratch/data/leah/phertilizer/{infer_folder}/pred_mut.csv",
    # #     "-J", f"/scratch/data/leah/phertilizer/{infer_folder}/pred_mut_loss.csv",
    # #     "-E", f"/scratch/data/leah/phertilizer/{infer_folder}/pred_event.csv",
    # #     "-T", f"/scratch/data/leah/phertilizer/{infer_folder}/best_tree.pickle",
    # # # #     "--no-loss",
    # # #     # SPhyR inferred
    # # #     "-z", "sphyr",
    # # #     "-C", f"{pth}/{gt_folder}/cluster-assignments.txt",
    # # #     "-T", f"{pth}/{gt_folder}/sphyr_output.dot",
    # # #     #     # # siclonefit inferred
    # # # #     # "-C", f"/scratch/data/leah/phertilizer/{infer_folder}/pred_cell.csv",
    # # # #     # "-M", f"/scratch/data/leah/phertilizer/{infer_folder}/pred_mut.csv",
    # # # #     # "-J", f"/scratch/data/leah/phertilizer/{infer_folder}/pred_mut_loss.csv",
    # # # #     # "-E", f"/scratch/data/leah/phertilizer/{infer_folder}/pred_event.csv",
    # # # #     # "-T", f"/scratch/data/chuanyi/phertilizer/simulations/siclonefit_sbmclone/hierarchical/s12_n1500_m1500_c5_p0.01_cna1_l0_loh0_dcl2_dsnv2_dcnv2/samples/best/best_MAP_tree.txt",
    # # # #     # output
    # # # #     # "-p", f"test/{folder}/s{seed}_{folder}_c4_p0.01_h0.8_f0.001_cna0.5_l0.25_d0.0_dcl2_dsnv2_dcnv0_true_tree.png",
    # #     "-o", "/scratch/data/leah/phertilizer/simulations/phertilizer/test.csv"
    # ])
    
    main(args)

    # "-c", f"/scratch/data/leah/phertilizer/test/s{seed}_{folder}_c5_p0.01_cna1_l0_loh0_dcl2_dsnv2_dcnv2_cellclust_gt.csv",
    # "-m", f"/scratch/data/leah/phertilizer/test/s{seed}_{folder}_c5_p0.01_cna1_l0_loh0_dcl2_dsnv2_dcnv2_mutclust_gt.csv",
    # "-j", f"/scratch/data/leah/phertilizer/test/s{seed}_{folder}_c5_p0.01_cna1_l0_loh0_dcl2_dsnv2_dcnv2_mut_loss_clust_gt.csv",
    # "-e", f"/scratch/data/leah/phertilizer/test/s{seed}_{folder}_c5_p0.01_cna1_l0_loh0_dcl2_dsnv2_dcnv2_copy_number_profiles.csv",
    # "-t", f"/scratch/data/leah/phertilizer/test/s{seed}_{folder}_c5_p0.01_cna1_l0_loh0_dcl2_dsnv2_dcnv2_tree.txt",
    # "-l", f"/scratch/data/leah/phertilizer/test/s{seed}_{folder}_c5_p0.01_cna1_l0_loh0_dcl2_dsnv2_dcnv2_likelihood.csv",
