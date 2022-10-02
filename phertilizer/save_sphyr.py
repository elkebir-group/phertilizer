from build_tree import BuildTree
import argparse
import pandas as pd



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-C", "--clusters", 
        help="inferred cell clusters")
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
    

    args = parser.parse_args()

    # pth = "/scratch/data/leah/phertilizer/ACT/run/test"
    # args = parser.parse_args([
    #     "-C", f"{pth}/clusters.csv",
    #     "-T", f"{pth}/sphyr_output.dot",
    #     "--cell_lookup", f"{pth}/cell_lookup.csv",
    #     "--mut_lookup", f"{pth}/mut_lookup.csv",
    #     "--pred_cell", f"{pth}/pred_cell.csv",
    #     "--pred_mut", f"{pth}/pred_mut.csv",
    #     "-t", f"{pth}/sphyr_tree.txt",
    #     "-p", f"{pth}/sphyr_tree.png",
    # ])

    inferred_tree, _ = BuildTree().build_sphyr(args.clusters, args.infer_tree)
    
    if args.png is not None:
        inferred_tree.tree_png(args.png)

    if args.tree_text is not None:
        inferred_tree.save_text(args.tree_text)


    cell_lookup = pd.read_csv(args.cell_lookup)
    cell_lookup = pd.Series(data=cell_lookup["cell_label"].to_numpy(), index=cell_lookup["cell_id"].to_numpy())   
    # cell_lookup['cell_id'] = "cell" + \
    #          variant_data['mutation_id'].astype(str)
    mut_lookup = pd.read_csv(args.mut_lookup)
    mut_lookup = pd.Series(data=mut_lookup["mut_label"].to_numpy(), index=mut_lookup["mut_id"].to_numpy())  
 
    pcell, pmut, ploss, pevents = inferred_tree.generate_results(cell_lookup, mut_lookup)

    if args.pred_cell is not None:
        pcell.to_csv(args.pred_cell, index=False)

    if args.pred_mut is not None:
        pmut.to_csv(args.pred_mut, index=False)
    
    if args.pred_loss is not None:
        ploss.to_csv(args.pred_loss, index=False)





 