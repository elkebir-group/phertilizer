

import argparse
import sys
from utils import pickle_load, pickle_save
import pandas as pd




if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--phert_tree", required=True,
                        help="pickle file of inferred tree")
    parser.add_argument("-d", "--data", required=True,
                        help="input file for variant and total read counts with unlabled columns: [chr snv cell base var total]")
    parser.add_argument("-j", "--iterations", type=int, default=5,
                        help="maximum number of iterations")
    parser.add_argument("-q", "--quantile", type=float, default=0.05,
                        help="quantile of to resassign")
    parser.add_argument("-m", "--pred_mut",
                        help="output file for mutation clusters")
    parser.add_argument("-n", "--pred_cell",
                        help="output file cell clusters")
    parser.add_argument("--tree",
                        help="output file for png (dot) of Phertilizer tree")
    parser.add_argument("--tree_pickle",
                        help="output pickle of Phertilizer tree")
    parser.add_argument("--tree_path",
                        help="path to directory where pngs of all trees are saved")
    parser.add_argument("--tree_text",
                        help="text file save edge list of best clonal tree")
    parser.add_argument("--likelihood",
                        help="output file where the likelihood of the best tree should be written")
    parser.add_argument("--embedding", type=str, help="umap coordinates")
    args = parser.parse_args(None if sys.argv[1:] else ['-h'])

    # inpth  = "/scratch/data/leah/phertilizer/DLP/nonzero_var_test2/s5/starts5_iterations10_minfrac0.015_r0.5"
    # outpth ="/scratch/data/leah/phertilizer/DLP/nonzero_var_test2/s5/starts5_iterations10_minfrac0.015_r0.5/post0.1"
    # args = parser.parse_args([
    #     "-i", f"{inpth}/best_tree.pickle",
    #     "-d", f"{inpth}/data.pickle",
    #     "-n", f"{outpth}/pred_cell.csv",
    #     "-m", f"{outpth}/pred_mut.csv",
    #     "--likelihood",   f"{outpth}/likelihood.csv",
    #     "--tree_text",  f"{outpth}/post_tree.txt",
    #     "--tree",  f"{outpth}/post_tree.png",
    #     "--tree_pickle",  f"{outpth}/post_tree.pickle",
    #     "-j", "20",
    #     "-q", "0.1"
    # ])

    data = pickle_load(args.data)
    tree = pickle_load(args.phert_tree)

    print(data.read_depth.shape)
    embedding = pd.DataFrame(data.read_depth, columns=["V1", "V2"])

    cell_lookup= data.cell_lookup
  

    embedding = pd.concat([embedding, cell_lookup], axis=1)
    embedding.columns = ["V1", "V2", "cell"]
    # embedding = embedding.rename(columns={"0": "cell"})
    if args.embedding is not None:
        embedding.to_csv(args.embedding, index=False)


    print(tree)
    print(tree.loglikelihood)

    tree.post_process(data, args.quantile, args.iterations, place_missing=True)

    if args.tree_text is not None:
        tree.save_text(args.tree_text)

    if args.pred_cell or args.pred_mut is not None:
        pcell, pmut, _, _ = tree.generate_results( data.cell_lookup, data.mut_lookup)

        if args.pred_cell is not None:
            pcell.to_csv(args.pred_cell, index=False)

        if args.pred_mut is not None:
            pmut.to_csv(args.pred_mut, index=False)
    
    if args.tree is not None:
            if '.dot' in args.tree:
                tree.tree_dot(args.tree)
            else:
                tree.tree_png(args.tree)

    if args.likelihood is not None:
            with open(args.likelihood,'w+') as f:
                f.write(",0\n")
                f.write(f"0,{tree.loglikelihood}")
                
    if args.tree_pickle is not None:
            pickle_save(tree, args.tree_pickle)
        

                    