
import numpy as np
import pygraphviz as pgv


def bins_to_string(bins, bin2chrom=None):
    s = []
    sorted(bins)
    if len(bins) == 1:
        return f"{bins[0]}"
    elif len(bins) == 2:
        return f"{bins[0]}-{bins[1]}"
    else:
        leftmost = None
        for right in bins:
            if leftmost == None:
                leftmost = right
                left = right
                continue
            if right == left + 1:
                left = right
            else:
                if bin2chrom is not None:
                    start_chrom = bin2chrom.loc[bin2chrom["start"] == leftmost, :]
                    end_chrom = bin2chrom.loc[abs(bin2chrom["end"] - left) <= 1, :]
                    if len(start_chrom) > 0 and len(end_chrom) > 0:
                        if all(start_chrom.index == end_chrom.index):
                            s.append(f"{start_chrom['chrom'].item()}{start_chrom['arm'].item()}")
                        elif start_chrom["chrom"].item() == end_chrom["chrom"].item():
                            s.append(f"{start_chrom['chrom'].item()}")
                        else:
                            s.append(f"{start_chrom['chrom'].item()}{start_chrom['arm'].item()}-"
                                    f"{end_chrom['chrom'].item()}{end_chrom['arm'].item()}")
                    else:
                        s.append(f"{leftmost}-{left}")
                else:
                    s.append(f"{leftmost}-{left}")
                leftmost = right
                left = right
        if bin2chrom is not None:
            start_chrom = bin2chrom.loc[bin2chrom["start"] == leftmost, :]
            end_chrom = bin2chrom.loc[abs(bin2chrom["end"] - right) <= 1, :]
            if len(start_chrom) > 0 and len(end_chrom) > 0:
                if all(start_chrom.index == end_chrom.index):
                    s.append(f"{start_chrom['chrom'].item()}{start_chrom['arm'].item()}")
                elif start_chrom["chrom"].item() == end_chrom["chrom"].item():
                    s.append(f"{start_chrom['chrom'].item()}")
                else:
                    s.append(f"{start_chrom['chrom'].item()}{start_chrom['arm'].item()}-"
                            f"{end_chrom['chrom'].item()}{end_chrom['arm'].item()}")
            else:
                s.append(f"{leftmost}-{right}")
        else:
            s.append(f"{leftmost}-{right}")
        return ', '.join(s)

class DrawClonalTree:
    def __init__(self, clonal_tree, bin2chrom=None):
        self.T = clonal_tree.tree
        self.nodes = tuple(self.T.nodes())
        self.cm = clonal_tree.cell_mapping
        self.mm = clonal_tree.mut_mapping
        self.ml = clonal_tree.mut_loss_mapping
        self.em = clonal_tree.event_mapping

        self.node_likelihood = clonal_tree.node_likelihood
        self.likelihood, var_like, bin_like = clonal_tree.get_loglikelihood()
        self.bin2chrom = bin2chrom

        self.include_subclusters = any([len(self.cm[n]) > 1 for n in self.nodes])
        
        

        

        self.cell_count = {n : len(np.concatenate([self.cm[n][k]  for k in self.cm[n]])) for n in self.nodes}
        self.mut_count = {n : len(self.mm[n]) for n in self.nodes}
        
        

        if not clonal_tree.has_loss():
            self.labels = {n: str(n) + "\nCells:" + str(self.cell_count[n]) + "\n+SNVs:" + str(self.mut_count[n])  for n in self.nodes}
            for n in self.node_likelihood:
                for key in self.node_likelihood[n]:
                    like_value = np.round(self.node_likelihood[n][key])
                    self.labels[n] += f"\n{key}: {like_value}"
        else:
            self.mut_loss_count = {n: 0 for n in self.nodes}
            for key in self.mut_loss_count:
                if key in self.ml:
                    self.mut_loss_count[key] = len(self.ml[key])
            self.labels = {}
            for n in self.nodes:
                self.labels[n] = str(n)
                if n in self.node_likelihood:
                    for key in self.node_likelihood[n]:
                        self.labels[n] += f"\n{key}: {self.node_likelihood[n][key]}"
                self.labels[n] += "\nCells:" + str(self.cell_count[n])
                # SNV
                if self.mut_count[n] > 0:
                    self.labels[n] += "\nSNVs: +" + str(self.mut_count[n])
                elif self.mut_loss_count[n] > 0:
                    self.labels[n] += "\nSNVs: -" + str(self.mut_loss_count[n])
                    
        # CNA
        for n in self.em:
            parents = list(self.T.predecessors(n))
            cna = []

            if len(parents) > 0:
                parent = parents[0]
                if parent not in self.em:
                    continue
                delta = np.intersect1d(self.em[n]["loss"], self.em[parent]["neutral"])
                if len(delta) > 0:
                    cna.append(f"n↓l: {bins_to_string(delta, self.bin2chrom)}")
                delta = np.intersect1d(self.em[n]["loss"], self.em[parent]["gain"])
                if len(delta) > 0:
                    cna.append(f"g↓l: {bins_to_string(delta, self.bin2chrom)}")
                delta = np.intersect1d(self.em[n]["neutral"], self.em[parent]["gain"])
                if len(delta) > 0:
                    cna.append(f"g↓n: {bins_to_string(delta, self.bin2chrom)}")
                delta = np.intersect1d(self.em[n]["gain"], self.em[parent]["neutral"])
                if len(delta) > 0:
                    cna.append(f"n↑g: {bins_to_string(delta, self.bin2chrom)}")
                delta = np.intersect1d(self.em[n]["gain"], self.em[parent]["loss"])
                if len(delta) > 0:
                    cna.append(f"l↑g: {bins_to_string(delta, self.bin2chrom)}")
                delta = np.intersect1d(self.em[n]["neutral"], self.em[parent]["loss"])
                if len(delta) > 0:
                    cna.append(f"l↑n: {bins_to_string(delta, self.bin2chrom)}")
            if len(cna) > 0:
                cna[0] = "\nCNAs: " + cna[0]
                self.labels[n] += '\n'.join(cna)

        like_label = ""
        self.tree = pgv.AGraph(strict=False, directed=False)
        if self.likelihood is not None:
            total_like = np.round(self.likelihood)
            like_label += f"Log Likelihood: {total_like}"
        if var_like is not None:
                var_like = np.round(var_like)
                like_label += f"\nVariant Data Likelihood: {var_like}"
        if bin_like is not None:
                bin_like = np.round(bin_like)
                like_label += f"\nBin Count Data Likelihood: {bin_like}"


        self.tree.graph_attr["label"] = like_label
        index = max(list(self.T.nodes()))
        for n in self.nodes:

            self.tree.add_node(n, label=self.labels[n])
            if len(self.cm[n]) > 1 and self.include_subclusters:
                subgraph = [n]

                for s in self.cm[n]:
                    index += 1
                    subgraph.append(index)
                    self.tree.add_node(index, label=f"Cells {n}_{s}: {len(self.cm[n][s])}")
           
    
                    self.tree.add_subgraph(subgraph, name=f"cluster_{n}", label=f"Cluster {n}")
        self.tree.add_edges_from(list(self.T.edges))


    def savePNG(self, file):
        self.tree.layout("dot")
        self.tree.draw(file)


    def saveDOT(self, file):
        self.tree.write(file)


