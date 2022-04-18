#!/usr/bin/env python

import argparse
import pandas as pd
from phertilizer import Phertilizer
import logging
from utils import pickle_load, pickle_save


def main(args):

    col_names = ['chr', 'mutation_id', 'cell_id',
                 'base', 'var', 'total', 'copies']

    mut_lookup = pd.read_table(
        args.file, sep="\t", header=None, names=col_names, skiprows=[0])

    mut_lookup.drop('copies', inplace=True, axis=1)

    if args.bin_count_data is not None:
        if '.csv' in args.bin_count_data:
            bin_count_data = pd.read_csv(args.bin_count_data)
        else:
            bin_count_data = pd.read_table(args.bin_count_data)

    print("Input data:")
    print(mut_lookup.head())

    ph = Phertilizer(mut_lookup,
                     bin_count_data,
                     coverage=args.coverage)

    if args.min_frac is not None:
        min_cell = args.min_frac
        min_snvs = args.min_frac

    else:
        min_cell = args.min_cells
        min_snvs = args.min_snvs

    grow_tree, pre_process_list = ph.phertilize(
        alpha=args.alpha,
        max_copies=args.copies,
        min_cell=min_cell,
        min_snvs=min_snvs,
        max_iterations=args.iterations,
        starts=args.starts,
        seed=args.seed,
        radius=args.radius,
        npass=args.npass
    )

    cell_lookup, mut_lookup = ph.get_id_mappings()
    print("\nPhertilizer Tree....")
    print(grow_tree)

    print("Saving files.....")

    if args.likelihood is not None:
        f = open(args.likelihood, "w")
        f.write(f"{grow_tree.loglikelihood}")
        f.close()

    if args.tree_list is not None:
        pickle_save(pre_process_list, args.tree_list)

    if args.tree is not None:
        if '.dot' in args.tree:
            grow_tree.tree_dot(args.tree)
        else:
            grow_tree.tree_png(args.tree)

    if args.tree_pickle is not None:
        pickle_save(grow_tree, args.tree_pickle)

    cell_lookup, mut_lookup = ph.get_id_mappings()
    pcell, pmut = grow_tree.generate_results(cell_lookup, mut_lookup)
  

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

    if args.tree_path is not None:
        pre_process_list.save_the_trees(args.tree_path)

    print("Thanks for planting a tree! See you later, friend.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file", required=True,
                        help="input dataframe file")
    parser.add_argument("--bin_count_data", required=True,
                        help="input dataframe file for copy number matrix")
    parser.add_argument("-a", "--alpha", type=float, default=0.001,
                        help="per base read error rate")
    parser.add_argument("--coverage", type=float, default=0.01,
                        help="coverage per base")
    parser.add_argument("--include_cna_events", action="store_true")
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
    parser.add_argument("--npass", type=int, default=1,
                        help="number of clustering stopping heuristics to pass")
    parser.add_argument("-d", "--seed", type=int, default=99059,
                        help="seed")
    parser.add_argument("--radius", type=float, default=0.5)
    parser.add_argument("-c", "--copies", type=int, default=5,
                        help="max number of copies")
    parser.add_argument("-m", "--pred-mut",
                        help="output file mutation clusters")
    parser.add_argument("-n", "--pred_cell",
                        help="output file cell clusters")
    parser.add_argument("-e", "--pred_event",
                        help="output file cna genotypes")
    parser.add_argument("--tree",
                        help="output file for png (dot) of phertilizer tree after grow")
    parser.add_argument("--tree_pickle",
                        help="output pickle of phertilizer tree after grow")
    parser.add_argument("--tree_path",
                        help="path to directory where pngs of post-processed trees are saved")
    parser.add_argument("--tree_list",
                        help="where to save the list of pre processed trees")
    parser.add_argument("--cell_lookup",
                        help="where to save the file that maps internal cell index to the input cell label")
    parser.add_argument("--mut_lookup",
                        help="where to save the file that maps internal mutation index to the input mutation label")
    parser.add_argument("--likelihood",
                        help="where to save the file with the tree likelihood")
    parser.add_argument(
        '-v', '--verbose',
        help="Be verbose",
        action="store_const", dest="loglevel", const=logging.INFO,
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)


main(args)
