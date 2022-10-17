#!/usr/bin/env python

import argparse
from operator import truediv
import sys
import pandas as pd
import numpy as np


from phertilizer.phertilizer import Phertilizer
from phertilizer.utils import pickle_save


def main(args):

    col_names = ['chr', 'mutation_id', 'cell_id',
                 'base', 'var', 'total', 'copies']

    variant_data = pd.read_table(
        args.file, sep="\t", header=None, names=col_names, skiprows=[0])

    variant_data.drop('copies', inplace=True, axis=1)

    if '.csv' in args.bin_count_data:
        bin_count_data = pd.read_csv(args.bin_count_data)
    else:
        bin_count_data = pd.read_table(args.bin_count_data)

 




    print("\nInput variant read count data:")
    print(variant_data.head())
    print("\nInput binned read count data:")
    print(bin_count_data.head())
  

    ph = Phertilizer(variant_data,
                     bin_count_data,
                     alpha =args.alpha,
                     max_copies = args.copies,
                    mode = "rd",
                    dim_reduce = not args.no_umap )

    if args.embedding is not None:
        ph.save_embedding(args.embedding)

    
    seed  = args.seed
    best_like = np.NINF
    best_tree, best_list, best_loglikes = None, None, None

    for i in range(args.runs):
        print(f"Run {i+1}")
        grow_tree, pre_process_list, loglikelihood = ph.phertilize(
            max_iterations=args.iterations,
            starts=args.starts,
            seed=100*seed + i,
            radius=args.radius,
            gamma= args.gamma,
            d = args.min_obs,
            use_copy_kernel = True,
            post_process = args.post_process,
            low_cmb = args.low_cmb,
            high_cmb = args.high_cmb,
            nobs_per_cluster =args.nobs_per_cluster
            )
        
        if grow_tree.norm_loglikelihood  > best_like:
            best_like = grow_tree.norm_loglikelihood
            best_tree, best_list, best_loglikes = grow_tree, pre_process_list, loglikelihood
        print(f"\nRun {i+1} complete: Normalized Log Likelihood: {best_like}")




    cell_lookup, mut_lookup = ph.get_id_mappings()


 
    print("\nPhertilizer Tree")
    print(f"Normalized log likelihood: {best_like}")
    print(best_tree)


    
  
    print("\nSaving files.....")


    if args.tree_list is not None:
        pickle_save(best_list, args.tree_list)

    if args.tree is not None:
        if '.dot' in args.tree:
            best_tree.tree_dot(args.tree)
        else:
            best_tree.tree_png(args.tree)

    if args.likelihood is not None:
        # with open(args.likelihood,'w+') as f:
            best_loglikes.to_csv(args.likelihood)
            
    if args.tree_pickle is not None:
        pickle_save(best_tree, args.tree_pickle)

    cell_lookup, mut_lookup = ph.get_id_mappings()
    pcell, pmut, _, _ = best_tree.generate_results(
        cell_lookup, mut_lookup)


    if args.pred_cell is not None:
        pcell.to_csv(args.pred_cell, index=False)

    if args.pred_mut is not None:
        pmut.to_csv(args.pred_mut, index=False)

    if args.tree_path is not None:
        best_list.save_the_trees(args.tree_path)
    
    if args.tree_text is not None:
        best_tree.save_text(args.tree_text)




    print("\nThanks for planting a tree! See you later, friend.")


def get_options():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file", required=True,
                        help="input file for variant and total read counts with unlabled columns: [chr snv cell base var total]")
    parser.add_argument("--bin_count_data", required=True,
                        help="input binned read counts with headers containing bin ids or embedding dimensions")
    parser.add_argument("-a", "--alpha", type=float, default=0.001,
                        help="per base read error rate")
    parser.add_argument("-j", "--iterations", type=int, default=50, 
                        help="maximum number of iterations")
    parser.add_argument("-s", "--starts", type=int, default=16,
                        help="number of restarts")
    parser.add_argument("-d", "--seed", type=int, default=99059,
                        help="seed")
    parser.add_argument("--radius", type=float, default=1)
    parser.add_argument("-c", "--copies", type=int, default=5,
                        help="max number of copies")
    parser.add_argument("--runs", type=int, default=1,
                        help="number of Phertilizer runs")
    parser.add_argument("-g", "--gamma", type=float, default=0.95,
                        help="confidence level for power calculation to determine if there are sufficient observations for inference")
    parser.add_argument("--min_obs", type=int, default=10,
                        help = "lower bound on the minimum number of observations for a partition")
    parser.add_argument("-m", "--pred-mut",
                        help="output file for mutation clusters")
    parser.add_argument("-n", "--pred_cell",
                        help="output file cell clusters")
    parser.add_argument("--post_process", action="store_true", 
                        help="indicator if post processing should be performed on inferred tree")
    parser.add_argument("--tree",
                        help="output file for png (dot) of Phertilizer tree")
    parser.add_argument("--tree_pickle",
                        help="output pickle of Phertilizer tree")
    parser.add_argument("--tree_path",
                        help="path to directory where pngs of all candidate trees are saved")
    parser.add_argument("--tree_list",
                        help="pickle file to save a ClonalTreeList of all generated trees")
    parser.add_argument("--tree_text",
                        help="text file save edge list of best clonal tree")
    parser.add_argument("--likelihood",
                        help="output file where the likelihood of the best tree should be written")
    parser.add_argument("--embedding", type=str,
                        help="filename where the UMAP coordinates should be saved after embedding binned read counts") 
    parser.add_argument("--no-umap", action="store_true",
                        help="flag to indicate that input reads per bin file should NOT undergo additional dimensionality reduction")
    parser.add_argument("--low_cmb", type=float, default=0.05,
                        help="regularization parameter to assess the quality of a split where CMB should <= low_cmb for parts of an extension" )
    parser.add_argument("--high_cmb", type= float, default=0.15,
                        help="regularization parameter to assess the quality of a split where CMB should >= high_cmb for parts of an extension" )
    parser.add_argument("--nobs_per_cluster", type=int,default=3,
                        help="regularization parameter on the median number of reads per cell/SNV to accept extension")
  

    args = parser.parse_args(None if sys.argv[1:] else ['-h'])




    return(args)


def main_cli():
    args = get_options()
    main(args)


if __name__ == "__main__":
    args = get_options()
    main(args)
