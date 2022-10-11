#!/usr/bin/env python

import argparse
import sys
import pandas as pd
import numpy as np

from phertilizer import Phertilizer
from utils import pickle_save
# from phertilizer.phertilizer import Phertilizer
# from phertilizer.utils import pickle_save


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

    print(bin_count_data.head())
    if args.bin_count_normal is not None:
        if '.csv' in args.bin_count_normal:
            bin_count_normal = pd.read_csv(args.bin_count_normal)

        else:
            bin_count_normal = pd.read_table(args.bin_count_normal)
    else:
        bin_count_normal = None

    if args.snv_bin_mapping is not None:
        snv_bin_mapping = pd.read_csv(args.snv_bin_mapping, names=[
                                      'mutation_id', 'chr', 'bin'])
    else:
        snv_bin_mapping = None

    print("Input data:")
    print(variant_data.head())

  

    ph = Phertilizer(variant_data,
                     bin_count_data,
                     bin_count_normal,
                     snv_bin_mapping,
                     alpha =args.alpha,
                     max_copies = args.copies,
                    neutral_mean = args.neutral_mean,
                    neutral_eps = args.neutral_eps,
                    mode = args.mode

                     )
    if args.embedding is not None:
        ph.save_embedding(args.embedding)
    if args.min_frac is not None:
        min_cell = args.min_frac
        min_snvs = args.min_frac

    else:
        min_cell = args.min_cells
        min_snvs = args.min_snvs
    
    seed  = args.seed
    best_like = np.NINF
    best_tree, best_list, best_loglikes = None, None, None

    for i in range(args.runs):
        print(f"Run {i+1}")
        grow_tree, pre_process_list, loglikelihood = ph.phertilize(
            min_cell=min_cell,
            min_snvs=min_snvs,
            max_iterations=args.iterations,
            starts=args.starts,
            seed=100*seed + i,
            radius=args.radius,
            npass=args.npass,
            gamma= 0.95,
            d = args.min_obs,
            use_copy_kernel = args.use_copy_kernel,
            post_process = args.post_process
            )
        
        if grow_tree.norm_loglikelihood  > best_like:
            best_like = grow_tree.norm_loglikelihood
            best_tree, best_list, best_loglikes = grow_tree, pre_process_list, loglikelihood
        print(f"Run {i+1} complete")
        print(best_tree)
        print(f"Normalize Log Likelihood: {best_like}")



    cell_lookup, mut_lookup = ph.get_id_mappings()
    if args.data is not None:
        pickle_save(ph.data, args.data )



 
    print("\nPhertilizer Tree")
    print(f"Normalized log likelihood: {best_like}")
    print(best_tree)


    
  
    print("Saving files.....")


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
    pcell, pmut, _, event_df = best_tree.generate_results(
        cell_lookup, mut_lookup)

    if args.cell_lookup is not None:
        cell_lookup = cell_lookup.rename("cell_label")
        cell_lookup.index.name = "cell_id"
        cell_lookup.to_csv(args.cell_lookup)

    if args.mut_lookup is not None:
        mut_lookup = mut_lookup.rename("mut_label")
        mut_lookup.index.name = "mut_id"
        mut_lookup.to_csv(args.mut_lookup)

    if args.pred_cell is not None:
        pcell.to_csv(args.pred_cell, index=False)

    if args.pred_mut is not None:
        pmut.to_csv(args.pred_mut, index=False)

    # if args.pred_event is not None:
    #     event_df.to_csv(args.pred_event)

    if args.tree_path is not None:
        best_list.save_the_trees(args.tree_path)
    
    if args.tree_text is not None:
        best_tree.save_text(args.tree_text)

    if args.params is not None:
        pickle_save(ph.params, args.params)
    


    print("Thanks for planting a tree! See you later, friend.")


def get_options():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file", required=True,
                        help="input file for variant and total read counts with unlabled columns: [chr snv cell base var total]")
    parser.add_argument("--bin_count_data", required=True,
                        help="input binned read counts with headers containing bin ids")
    # parser.add_argument("--bin_count_normal",
    #                     help="input binned read counts for normal cells with identical bins as the bin count data")
    # parser.add_argument("--snv_bin_mapping",
    #                     help="a comma delimited file with unlabeled columns: [snv chr bin]")
    parser.add_argument("-a", "--alpha", type=float, default=0.001,
                        help="per base read error rate")
    # parser.add_argument("--min_cells", type=int, default=100,
    #                     help="smallest number of cells required to form a clone")
    # parser.add_argument("--min_snvs", type=int, default=100,
    #                     help="smallest number of SNVs required to form a cluster")
    # parser.add_argument("--min_frac", type=float,
    #                     help="smallest proportion of total cells(snvs) needed to form a cluster, if min_cells or min_snvs are given, min_frac is ignored")
    parser.add_argument("-j", "--iterations", type=int, default=5,
                        help="maximum number of iterations")
    parser.add_argument("-s", "--starts", type=int, default=3,
                        help="number of restarts")
    parser.add_argument("-d", "--seed", type=int, default=99059,
                        help="seed")
    parser.add_argument("--radius", type=float, default=1.0)
    parser.add_argument("-c", "--copies", type=int, default=5,
                        help="max number of copies")
    parser.add_argument( "--low_cmb", type=float, default=0.05)
    parser.add_argument( "--high_cmb", type=float, default=0.15)
    parser.add_argument( "--nobs_per_cluster", type=float, default=3.0)

    # parser.add_argument("--neutral_mean",  type=float, default=1.0,
    #                     help="center of neutral RDR distribution")
    # parser.add_argument("--neutral_eps",  type=float, default=0.15,
    #                     help="cutoff of neutral RDR distribution")
    parser.add_argument("-m", "--pred-mut",
                        help="output file for mutation clusters")
    parser.add_argument("-n", "--pred_cell",
                        help="output file cell clusters")
    parser.add_argument("--runs", type=int, default=1,
                        help="number of Phertilizer runs")
    parser.add_argument("--no_dim_reduce", action="store_true",
                        help="if dimensionality reduction should not be used")
    # parser.add_argument("-e", "--pred_event",
    #                     help="output file cna genotypes")
    parser.add_argument("-g", "--gamma", type=float, default=0.95,
                        help="confidence level for power calculation to determine if there are sufficient observations for inference")
    parser.add_argument("--min_obs", type=int, default=7,
                        help = "lower bound on the minimum number of observations for a partition")
    parser.add_argument("--use_copy_kernel", action="store_true",
                        help="indicator if copy number kernel should be used in cell clustering")
    parser.add_argument("--post_process", action="store_true", 
                        help="indicator low support cells and snvs should be placed into tree after inference")
    parser.add_argument("--tree",
                        help="output file for png (dot) of Phertilizer tree")
    parser.add_argument("--tree_pickle",
                        help="output pickle of Phertilizer tree")
    parser.add_argument("--tree_path",
                        help="path to directory where pngs of all trees are saved")
    parser.add_argument("--tree_list",
                        help="pickle file to save a ClonalTreeList of all generated trees")
    parser.add_argument("--tree_text",
                        help="text file save edge list of best clonal tree")
    parser.add_argument("--cell_lookup",
                        help="output file that maps internal cell index to the input cell label")
    parser.add_argument("--mut_lookup",
                        help="output file that maps internal mutation index to the input mutation label")
    parser.add_argument("--likelihood",
                        help="output file where the likelihood of the best tree should be written")
    parser.add_argument("--mode", choices=["var", "rd"],
                        help="the likelihood phertilizer should use to select the best tree")
    parser.add_argument("--embedding", type=str,
                        help="filename where the umap coordinates should be saved")  
                
    args = parser.parse_args(None if sys.argv[1:] else ['-h'])


    # parser.add_argument("--data", type=str,
    #                     help="filename where pickled data should be saved for post-processing")
    # parser.add_argument("--params", type=str,
    #                     help="filename where pickled parameters should be save")      

#/scratch/data/leah/phertilizer/simulations/phertilizer/recomb_rd/clones7_l0_loh0_p0.01_ck1/s12_n1500_m5000
#     inpath = "/scratch/data/leah/phertilizer/simulations/phertilizer/phert_input/s13_n5000_m5000_c5_p0.01_cna1_l0_loh0_dcl2_dsnv2_dcnv2"
#     outpath = "/scratch/data/leah/phertilizer/simulations/phertilizer/tst_snv"
    
#     args = parser.parse_args([ 
#         "-f", f"{inpath}/dataframe_mod.tsv",
#         "--bin_count_data", f"{inpath}/reads_per_bin_relabeled.csv",
#         # "-f", f"{outpath}/dataframe.tsv",
#         # "--bin_count_dat", f"{outpath}/bin_counts.csv",
#         "-d", "16",
#         "-c", "5",
#         "-j", "20",
#         "-s", "5",
#         "--runs", "1",
#         "-a", "0.001",
#         "--radius", "0.975",
#         "--gamma", "0.95",
#         "--min_obs", "7",
#         "--post_process",
#         "-m", f"{outpath}/pred_mut.csv",
#         "-n", f"{outpath}/pred_cell.csv",
#         "--tree", f"{outpath}/best_tree.png",
#         "--tree_text", f"{outpath}/best_tree.txt",
#         "--tree_pickle", f"{outpath}/best_tree.pickle",
#         # "--tree_list", f"{outpath}/tree_list.pickle",
#         "--likelihood", f"{outpath}/likelihood.txt",
#         "--cell_lookup", f"{outpath}/cell_lookup.csv",
#         "--mut_lookup", f"{outpath}/mut_lookup.csv",
#         "--npass", "2",
#         "--mode", "rd",
#      "--use_copy_kernel",
#      "--post_process"


# ])
  
    # inpath = "/scratch/data/leah/phertilizer/ACT/run/preprocess/TN3"
    # outpath = "/scratch/data/leah/phertilizer/ACT/run/TN3_tst/post2"
    # args = parser.parse_args([ 
    #     "-f", f"{inpath}/variant_data_t0.05.tsv",
    #     "--bin_count_data", f"{inpath}/binned_read_counts_t0.05.tsv",
    #     # "--bin_count_data", "/scratch/data/leah/phertilizer/ACT/input/TN2/paper_binned_read_counts.tsv",
    #     # "--bin_count_normal", "/scratch/data/leah/phertilizer/simulations/normal_samples/normal_cells_p0.01.tsv",
    #     # "--snv_bin_mapping",f"{inpath}/snv_bin_reformatted.csv",
 
    #     "--min_frac", "0.02",
    #     "-d", "5",
    #     "-c", "5",
    #     "-j", "10",
    #     "-s", "5",
    #     "-a", "0.001",
    #     "--radius", "0.25",
    #     "-m", f"{outpath}/pred_mut.csv",
    #     "-n", f"{outpath}/pred_cell.csv",
    #     "--tree", f"{outpath}/best_tree.png",
    #     "--tree_path", f"{outpath}",
    #     "--tree_text", f"{outpath}/best_tree.txt",
    #     "--tree_pickle", f"{outpath}/best_tree.pickle",
    #     "--tree_list", f"{outpath}/tree_list.pickle",
    #     "--likelihood", f"{outpath}/likelihood.txt",
    #     "--cell_lookup", f"{outpath}/cell_lookup.csv",
    #     "--mut_lookup", f"{outpath}/mut_lookup.csv",
    #     "--npass", "1",
    #     "--mode", "rd"



    # ])
   
    # inpath = "/scratch/data/leah/phertilizer/DLP/clones/phert_clade3"
    # outpath = "/scratch/data/leah/phertilizer/DLP/clones/phert_clade3"
    # # inpath ="/scratch/data/leah/phertilizer/DLP/input"
    # df = "variant_data.filt.tsv"
    # reads_per_bin_file = "input/binned_read_counts.merged.tsv"
    # args = parser.parse_args([ 
    #     "-f", f"{inpath}/dataframe.tsv",
    #     # "-f", f"{inpath}/{df}",
    #     "--bin_count_data", f"{inpath}/bin_counts.csv",
    #     # "--bin_count_data", f"{inpath}/binned_read_counts.merged.tsv",
    #     # "--bin_count_data", "/scratch/data/leah/phertilizer/ACT/input/TN2/paper_binned_read_counts.tsv",
    #     # "--bin_count_normal", "/scratch/data/leah/phertilizer/simulations/normal_samples/normal_cells_p0.01.tsv",
    #     # "--snv_bin_mapping",f"{inpath}/snv_bin_reformatted.csv",
 
    #     # "--min_frac", "0.015",
    #     # "--min_cells", "13",
    #     # "--min_snvs", "211",
    #     # "--min_cells", "13",
    #     # "--min_snvs", "50",
    #     "-d", "3",
    #     "-c", "5",
    #     "-j", "10",
    #     "-s", "30",
    #     "--runs", "3",
    #     "-a", "0.001",
    #     "--radius", "0.9",
    #     "--gamma", "0.95",
    #     "--min_obs", "10",
    #     "--post_process",
    #     "-m", f"{outpath}/pred_mut.csv",
    #     "-n", f"{outpath}/pred_cell.csv",
    #     "--tree", f"{outpath}/best_tree.png",
    #     "--tree_text", f"{outpath}/best_tree.txt",
    #     "--tree_pickle", f"{outpath}/best_tree.pickle",
    #     # "--tree_list", f"{outpath}/tree_list.pickle",
    #     "--likelihood", f"{outpath}/likelihood.txt",
    #     "--cell_lookup", f"{outpath}/cell_lookup.csv",
    #     "--mut_lookup", f"{outpath}/mut_lookup.csv",
    #     "--npass", "2",
    #     "--mode", "rd"
    # ])


    return(args)


def main_cli():
    args = get_options()
    main(args)


if __name__ == "__main__":
    args = get_options()
    main(args)
