from build_tree import BuildTree
import argparse
import pandas as pd
import pygraphviz as pgv
import numpy as np
import networkx as nx
from clonal_tree import ClonalTree
from utils import pickle_save


def scite_graphviz(tree_graphviz: str, cell_cluster: str=None, mut_cluster: str=None) -> list:
        """Parse SPhyR GraphViz dot file

        Args:
            tree_graphviz (str): graphviz tree file

        Returns:
            list: edge list
        """
        scite_tree = pgv.AGraph(tree_graphviz)

        mut_mapping = {}
        cell_mapping = {}
        recluster_mapping = {}
 
        edges = scite_tree.edges()

        tree = nx.DiGraph()
        for u,v in edges:
            if u =='Root':
                x = -1
            else:
                x = int(u.replace("SNV", ""))
         
            if 'SNV' in v:

                y = int(v.replace("SNV", ""))
                mut_mapping[y] = mut_cluster[y+1]
                tree.add_edge(x,y)
            else:
                y = int(v.replace("s", ""))
                # Need +1 ?
                if x in cell_mapping:
                    cell_mapping[x][0] = np.concatenate((cell_mapping[x][0], cell_cluster[y+1]))
                else:
                    cell_mapping[x] = {0 :cell_cluster[y+1]}
                recluster_mapping[y] = x
                


        #dfs search collapsing linear chains
        # collapse_linear(tree, -1, mut_mapping, parent=None)
        tree.remove_node(-1)
        if -1 in cell_mapping:
            del cell_mapping[-1]
        if -1 in mut_mapping:
            del mut_mapping[-1]
        ct = ClonalTree(tree, cell_mapping, mut_mapping)
        ct.relabel()

        #move cells up to parent node
        # for key in ct.cell_mapping:
        #     parent = list(ct.tree.successors(key))[0]
        #     cells = ct.cell_mapping[key][0]
        #     if key not in ct.mut_mapping:
            
            
        #         if parent in ct.cell_mapping:
        #             ct.cell_mapping[parent] = {0: np.concatenate([ct.cell_mapping[parent][0], cells])}
        #         else:
        #             ct.cell_mapping[parent] = {0:cells}
        #         del ct.cell_mapping[key]
        #     elif len(ct.mut_mapping[key])==0:
        #         if parent in ct.cell_mapping:
        #             ct.cell_mapping[parent] = {0: np.concatenate([ct.cell_mapping[parent][0], cells])}
        #         else:
        #             ct.cell_mapping[parent] = {0:cells}
        #         del ct.cell_mapping[key]

                



             
        return ct, recluster_mapping


def collapse_linear(tree, source, mut_mapping, parent):
    child = list(tree.successors(source))
    if len(child) == 0:
        return
    nodes_to_collapse = []
    last_node = None
    root= source
    while True:
        child = list(tree.successors(root))
        if len(child) > 1 or len(child) ==0:
            last_node = root
            tree.remove_nodes_from(nodes_to_collapse)
            if parent is not None:
                tree.add_edge(parent, last_node)
            nodes_to_collapse.append(last_node)

            mut_mapping[last_node] = np.concatenate([mut_mapping[n] for n in nodes_to_collapse if n != -1 ])
            for n in nodes_to_collapse:
                if n != -1 and n != last_node:
                    # print("deleting " + str(n))
                    del mut_mapping[n]
            for c in child:
                collapse_linear(tree,c, mut_mapping, last_node )
            break
        else:    
            nodes_to_collapse.append(root)
            root = child[0]

def construct_dict(dat, label):
    cluster_dict = {}
    for c in dat['cluster'].unique():
        df = dat[dat['cluster'] ==c]
        labs= df[label].to_numpy()
        cluster_dict[c] = labs 
    return cluster_dict 



      
      


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # parser.add_argument("-C", "--clusters", 
    #     help="inferred cell clusters")
    parser.add_argument("-T", "--infer_tree", 
        help="inferred tree file")
    parser.add_argument("--cell_lookup")
    parser.add_argument("--mut_lookup")
    parser.add_argument("-t", "--tree_text", 
        help="output text file for tree")
    parser.add_argument("--pred_cell")
    parser.add_argument("--pred_mut")
    parser.add_argument("--pred_loss")
    parser.add_argument("-p", "--png", 
        help="output file for png")
    parser.add_argument( "--pickle", 
        help="pickle clonal tree")
    

    args = parser.parse_args()

    # pth = "/scratch/data/leah/phertilizer/ACT/run/baseline_0.05/TN3/input"
    # out_pth = "/scratch/data/leah/phertilizer/ACT/run/baseline_0.05/TN3/scite/seed_5"
    # args = parser.parse_args([
    #     "-C", f"{pth}/clusters.csv",
    #     "-T", f"{out_pth}/opt_tree_ml0.gv_ml37.gv",
    #     "--cell_lookup", f"{pth}/cell_lookup.csv",
    #     "--mut_lookup", f"{pth}/mut_lookup.csv",
    #     "--pred_cell", f"{out_pth}/pred_cell.csv",
    #     "--pred_mut", f"{out_pth}/pred_mut.csv",
    #     "-t", f"{out_pth}/scite_tree.txt",
    #     "-p", f"{out_pth}/scite_tree_collapsed.png",
    # ])

    # pth = "/scratch/data/leah/phertilizer/DLP/baseline/input/"
    # out_pth = "/scratch/data/leah/phertilizer/DLP/baseline/scite/seed_4"
    # args = parser.parse_args([

    #     "-T", f"{out_pth}/opt_tree_ml0.gv",
    #     "--cell_lookup", f"{pth}/cell_lookup.csv",
    #     "--mut_lookup", f"{pth}/mut_lookup.csv",
    #     "--pred_cell", f"{out_pth}/pred_cell.csv",
    #     "--pred_mut", f"{out_pth}/pred_mut.csv",
    #     "-t", f"{out_pth}/tree0.txt",
    #     "-p", f"{out_pth}/tree0.png",
    # ])


    # pth = "/scratch/data/leah/phertilizer/simulations/baseline/scite/clones5_l0_loh0_p0.01_vaf0.05/s16_n5000_m5000"
    # out_pth = "/scratch/data/leah/phertilizer/simulations/baseline/scite/clones5_l0_loh0_p0.01_vaf0.05/s16_n5000_m5000"
    # args = parser.parse_args([

    #     "-T", f"{pth}/scite_ml0.gv",
    #     "--cell_lookup", f"{pth}/pred_cell.csv",
    #     "--mut_lookup", f"{pth}/pred_mut.csv",
    #     "--pred_cell", f"{out_pth}/pred_cell_out.csv",
    #     "--pred_mut", f"{out_pth}/pred_mut_out.csv",
    #     "-t", f"{out_pth}/tree0.txt",
    #     "-p", f"{out_pth}/tree0.png",
    #     "--pickle", f"{out_pth}/tree0.pickle",

    
    # ])
    cell_lookup_dat = pd.read_csv(args.cell_lookup)
    print(cell_lookup_dat.head())
    mut_lookup_dat = pd.read_csv(args.mut_lookup)
    print(mut_lookup_dat.head())
 
    # if True:
    #     cell_lookup_dat['cluster'] = cell_lookup_dat['cluster']-1
    # else:
    #     cell_lookup_dat.cluster = [ ord(x) - 64 for x in cell_lookup_dat.cluster ]
    #     cell_lookup_dat['cluster'] = cell_lookup_dat['cluster']-1
        # print(cell_lookup_dat.head())
    cell_clusters = construct_dict(cell_lookup_dat, 'cell_id')
    cell_lookup_dat= cell_lookup_dat.set_index("cell_id")
    cell_lookup_series = cell_lookup_dat['cell']

 
    # print(mut_lookup_dat.head())
    mut_clusters = construct_dict(mut_lookup_dat, 'mut_id')
    mut_lookup_dat =mut_lookup_dat.set_index("mut_id")
    mut_lookup_series = mut_lookup_dat['mutation']
    # mut_lookup_dat['cluster'] = mut_lookup_dat['cluster']-1


    inferred_tree, mapping = scite_graphviz(args.infer_tree, cell_clusters,mut_clusters)
    cell_lookup_dat =cell_lookup_dat['cluster'].map(mapping)


    if args.png is not None:
        inferred_tree.tree_png(args.png)
    

    if args.tree_text is not None:
        inferred_tree.save_text(args.tree_text)

    #TODO: collapse chains 


    
    #cell_lookup = pd.Series(data=cell_lookup_dat["cell_label"].to_numpy(), index=cell_lookup["cell_id"].to_numpy())   
    # cell_lookup['cell_id'] = "cell" + \
    #          variant_data['mutation_id'].astype(str)
    # mut_lookup = pd.read_csv(args.mut_lookup)
    # mut_lookup = pd.Series(data=mut_lookup["mut_label"].to_numpy(), index=mut_lookup["mut_id"].to_numpy())  
 
    pcell, pmut, ploss, pevents = inferred_tree.generate_results(cell_lookup_series, mut_lookup_series)

    if args.pred_cell is not None:
        pcell.to_csv(args.pred_cell, index=False)

    if args.pred_mut is not None:
        pmut.to_csv(args.pred_mut, index=False)
    
    if args.pred_loss is not None:
        ploss.to_csv(args.pred_loss, index=False)
    
    if args.pickle is not None:
        pickle_save(inferred_tree, args.pickle)

    print("Done")




 