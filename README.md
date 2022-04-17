## Phertilizer: growing a clonal tree from ultra-low coverage single-cell DNA sequencing data of tumors


![Overview of Phertilizer](overview.png)
Phertilizer infers a clonal tree with SNV and CNA genotypes given ultra-low coverage single-cell sequencing data.
(a) A tumor is composed of groups of cells, or clones with distinct genotypes.
(b) Ultra-low coverage single-cell DNA seequencing produces total read counts and variant read countsfor n cells and m SNV loci, and read-depth ratios for the same cells for an input set of bins.
(c) Phertilizer infers a cell clustering, SNV genotypes, CNA genotypes and a clonal tree  with maximum posterior probability.


## Contents

  1. [Installation](#install)
     * [Using github](#compilation)
          * [Dependencies](#pre-requisites)
  2. [Usage instructions](#usage)
     * [Modes](#modes)    
     * [I/O formats](#io)
     * [Phertilizer](#phertilizer)

<a name="install"></a>

## Installation

<a name="install"></a>
  1. Clone the repository
      ```bash
            $ git clone https://github.com/elkebir-group/phertilizer.git

<a name="pre-requisites"></a>
#### Pre-requisites
+ python3 (>=3.7)
+ [numpy](https://numpy.org/doc/)
+ [pandas](https://pandas.pydata.org/pandas-docs/stable/index.html)
+ [numba](http://numba.pydata.org)
+ [scipy](https://scipy.org)
+ [networkx](https://networkx.org)
+ [sklearn.cluster](https://scikit-learn.org/stable/modules/clustering.html#clustering)
+ [pickle](https://docs.python.org/3/library/pickle.html)

#### Optional
+ [umap](https://umap-learn.readthedocs.io/en/latest/)

<a name="modes"></a>
## Modes
Phertilizer can be run in two modes:
 1. *CNA Mode* 
    + Input: variant/total read counts, binned read counts for tumor cells, binned read counts for normal cells, mapping of SNVs to bin 
    + Phertilizer returns a clonal tree, a cell clustering and **both SNV and CNA genotypes**
 2. *SNV Mode* 
    + Input: variant/total read counts, binned read counts for tumor cells 
    + Phertilizer returns a clonal tree, a cell clustering and **only SNV genotypes** 
