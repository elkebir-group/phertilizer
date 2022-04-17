#!/usr/bin/env python 

import argparse
import sys
import pandas as pd 
from phertilizer import Phertilizer
import logging 
from utils import pickle_save


def main(args):

    col_names = ['chr', 'mutation_id', 'cell_id', 'base', 'var', 'total', 'copies']
    
    variant_data = pd.read_table(args.file, sep="\t", header=None, names=col_names, skiprows=[0])
   
    variant_data.drop('copies', inplace=True, axis=1)


    if args.bin_count_data is not None:
        if '.csv' in args.bin_count_data:
            bin_count_data= pd.read_csv(args.bin_count_data)
        else:
            bin_count_data= pd.read_table(args.bin_count_data)

        
        if '.csv' in args.bin_count_normal:
            bin_count_normal = pd.read_csv(args.bin_count_normal)
            logging.info(bin_count_normal.head())
        else:
            bin_count_normal = pd.read_table(args.bin_count_normal)

        snv_bin_mapping = pd.read_csv(args.snv_bin_mapping, names=['mutation_id', 'chr', 'bin'])


    else:
        bin_count_data = None    
        snv_bin_mapping = None
        

    print("Input data:")
    print(variant_data.head())


    ph = Phertilizer(variant_data, 
                    bin_count_data, 
                    bin_count_normal,
                    snv_bin_mapping,
                    debug = args.debug,
                    include_cna= args.include_cna_events,
                    )
 
    if args.min_frac is not None:
        min_cell = args.min_frac
        min_snvs = args.min_frac

    else:
        min_cell =args.min_cells
        min_snvs= args.min_snvs

  

    grow_tree, pre_process_list = ph.phertilize(
                                    alpha= args.alpha, 
                                    max_copies= args.copies, 
                                    min_cell = min_cell,
                                    min_snvs = min_snvs,
                                    max_iterations=args.iterations,
                                    starts=args.starts, 
                                    neutral_mean = args.neutral_mean,
                                    neutral_eps = args.neutral_eps,
                                    seed = args.seed,
                                    radius = args.radius,
                                    npass = args.npass )

    
    cell_lookup, mut_lookup = ph.get_id_mappings()
    print("\Phertilizer Tree....")
    print(grow_tree)


    print("Saving files.....")

    if args.tree_list is not None:
        pickle_save(pre_process_list, args.tree_list)
    
    if args.tree is not None:
        if '.dot' in args.tree:
            grow_tree.tree_dot(args.tree)
        else:
            grow_tree.tree_png(args.tree)
    
    if args.tree_pickle is not None:
        pickle_save(grow_tree, args.tree_pickle)
  

    pcell, pmut, _, event_df = ph.generate_results(grow_tree)
    cell_lookup, variant_data = ph.get_id_mappings()
    
    if args.cell_lookup is not None:
        cell_lookup = cell_lookup.rename("cell_label")
        cell_lookup.index.name = "cell_id"
        cell_lookup.to_csv(args.cell_lookup)
    
    if args.mut_lookup is not None:
        mut_lookup= mut_lookup.rename("mut_label")
        mut_lookup.index.name = "mut_id"
        mut_lookup.to_csv(args.mut_lookup)

    if args.pred_cell is not None:
        pcell.to_csv(args.pred_cell, index=False)

    if args.pred_mut is not None:
        pmut.to_csv(args.pred_mut, index=False)
    
    
    if args.pred_event is not None:
        event_df.to_csv(args.pred_event)
    
    
    if args.tree_path is not None:
        pre_process_list.save_the_trees(args.tree_path)
    

    print("Thanks for planting a tree! See you later, friend.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file", 
        help="input file for variant and total read counts with unlabled columns: [chr snv cell base var total]")
    parser.add_argument("--bin_count_data", 
        help="input binned read counts with headers containing bin ids")
    parser.add_argument("--bin_count_normal", 
        help="input binned read counts for normal cells with identical bins as the bin count data")
    parser.add_argument("--snv_bin_mapping", 
        help = "a comma delimited file with unlabeled columns: [snv chr bin]")
    parser.add_argument("-a", "--alpha", type=float,default= 0.001,
        help="per base read error rate" )
    parser.add_argument("--min_cells", type=int, default=100,
        help="smallest number of cells required to form a clone")
    parser.add_argument("--min_snvs", type=int, default=100,
        help="smallest number of SNVs required to form a cluster")
    parser.add_argument("--min_frac", type=float,
        help="smallest proportion of total cells(snvs) needed to form a cluster, if min_cells or min_snvs are given, min_frac is ignored")
    parser.add_argument("-j", "--iterations", type=int, default=10,
        help="maximum number of iterations")
    parser.add_argument("-s", "--starts", type=int, default=10,
        help="number of restarts")
    parser.add_argument("-d", "--seed", type=int, default=99059,
        help="seed")
    parser.add_argument("--npass", type=int, default=1)
    parser.add_argument("--radius", type=float, default=0.5)
    parser.add_argument("-c", "--copies", type= int, default=5,
        help="max number of copies" )
    parser.add_argument("--neutral_mean",  type= float, default=1.0,
        help="center of neutral RDR distribution" )
    parser.add_argument("--neutral_eps",  type= float, default=0.15,
        help="cutoff of neutral RDR distribution" )
    parser.add_argument( "-m", "--pred-mut", 
        help="output file for mutation clusters")
    parser.add_argument("-n", "--pred_cell", 
        help="output file cell clusters")  
    parser.add_argument("-e", "--pred_event", 
        help="output file cna genotypes")   
    parser.add_argument("--tree", 
        help= "output file for png (dot) of Phertilizer tree")
    parser.add_argument("--tree_pickle", 
        help= "output pickle of Phertilizer tree")
    parser.add_argument( "--tree_path", 
        help="path to directory where pngs of all trees are saved")
    parser.add_argument( "--tree_list", 
        help="pickle file to save a ClonalTreeList of all generated trees")
    parser.add_argument( "--cell_lookup", 
        help="output file that maps internal cell index to the input cell label")
    parser.add_argument( "--mut_lookup", 
        help="output file that maps internal mutation index to the input mutation label")
    parser.add_argument(
        '-v', '--verbose',
        help="Be verbose",
        action="store_const", dest="loglevel", const=logging.INFO,
    )
    args = parser.parse_args(None if sys.argv[1:] else ['-h'])
  
    logging.basicConfig(level=logging.INFO)
    

main(args)

