
import numpy as np
from pandas import Series, DataFrame, concat
from time import time
from functools import wraps
import pickle
from scipy.linalg import eigh
from sklearn.cluster import AgglomerativeClustering
import networkx as nx
from scipy.special import comb 


def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        print('func:%r args:[%r, %r] took: %2.4f sec' % (f.__name__, args, kw, te-ts))
        return result
    return wrap


def mapping_to_dataframe(mapping, id_name, cell=False):
        #TODO: Streamline below code using convert_array functions
        pred_list = []
        if len(mapping)==0:
            return DataFrame(columns=[id_name, "cluster"])
        for k in mapping:
            if cell:
                for s in mapping[k]:
                    temp = DataFrame(mapping[k][s], columns=[id_name])
                    temp['cluster'] =k
                    temp['subcluster'] = s
                    pred_list.append(temp)
            else:
                temp = DataFrame(mapping[k], columns=[id_name])
                temp['cluster'] =k
                pred_list.append(temp)

        pred = concat(pred_list)
        pred= pred.sort_values(by=[id_name])
        

        return pred

def check_stats(stats_dict, jump_perc_threshold, first_eig_thresh, num_pass):
    if stats_dict is None:
        return False 

    if len(stats_dict) ==0:
        return False 
    test_result = []
    
    test_result.append(stats_dict['jump_perc'] > jump_perc_threshold)

    test_result.append(stats_dict['first_gap'] <= first_eig_thresh)

    test_result.append(stats_dict['spectral_num_k'] > 1)

    return sum(test_result) >= num_pass
    
def find_cells_with_no_reads( mat, cells, muts):
    total_reads_by_cell = np.count_nonzero(mat[np.ix_(cells, muts)],axis=1).reshape(-1)

    return cells[total_reads_by_cell==0]       

def find_muts_with_no_reads( mat, cells, muts):
    total_reads_by_mut = np.count_nonzero(mat[np.ix_(cells, muts)],axis=0).reshape(-1)

    return muts[total_reads_by_mut==0]   



def snv_kernel_width(dmat):
    kw = 0.25*(np.max(dmat)-np.min(dmat))
  
    return kw

def cnv_kernel_width(dmat):

    kw = 0.25*(np.max(dmat)-np.min(dmat))

    return kw


def impute_mut_features( c_mat, mut_features, cells):
        
        
    cell_series= Series(cells)
    identity_matrix = np.identity(c_mat.shape[0])

    na_features = mut_features.sum(axis=1).reshape(-1)

    #if we don't have enough cells to impute then we can't cluster
    if sum(np.isnan(na_features)) > 0.5*len(cells):
        return mut_features
    
    cells_to_impute = cells[np.isnan(na_features)]

    mask = identity_matrix==1

    masked_cdist= np.ma.array(c_mat, mask=mask, dtype=float)
    
    all_indices = np.arange(c_mat.shape[0])

    non_cell_indices = np.setdiff1d(all_indices, cells)
    non_cell_indices = np.union1d(cells_to_impute, non_cell_indices)

    masked_cdist[:,non_cell_indices] = np.NAN

    replacement_cell = np.argsort(masked_cdist, axis=1)

    for c in cells_to_impute:

        cell_index = cell_series[cell_series == c].index[0]
        
        #find the cell closest in distance based on copy number 
    
        replacement_cell_id = replacement_cell[c,0]

        replacement_cell_index = cell_series[cell_series == replacement_cell_id].index[0]

        #impute the replacement
        mut_features[cell_index,:] = mut_features[replacement_cell_index,:]
    
    assert not np.any(np.isnan(mut_features))

    return mut_features

def normalizedMinCut( W, index):

    stats = {}
    if W.shape[0] ==1 or np.any(np.isnan(W)):
        return (index, np.empty(shape=0)), None, None, stats


    D = np.diag(W.sum(axis=1))
    eig_vals, vecs = eigh(D-W, D)
    y_vals = Series(vecs[:,1], index=index)
    # y_vals.to_csv("/scratch/data/leah/phertilizer/DLP/clones17/vec_vals.csv", index=False)
    v= np.sort(vecs[:,1])
    
    #calculate statistics on clustering 
    
    jump_percentage = np.max(np.abs(np.diff(v))) / (np.max(v) - np.min(v))
  
    #calculate the spectral gap and find the max 
    first_gap = eig_vals[1] - eig_vals[0]
    
    spectral_k = np.argmax(np.diff(eig_vals)) + 1
    largest_gap = np.max(np.diff(eig_vals))
    
    # print(f"Jump Percentage: {jump_percentage} First Eig Val: {eig_vals[0]} 
    # First Gap: {first_gap} Largest Gap: {largest_gap} Number of Clusters:{spectral_k}")
    
    stats["jump_perc"] = jump_percentage
    stats["first_gap"] = first_gap
    stats["largest_gap"] = largest_gap
    stats["spectral_num_k"] = spectral_k

  
    labels = AgglomerativeClustering(n_clusters=2, affinity="euclidean", linkage="ward").fit_predict(vecs[:,1].reshape(-1,1))
    cluster1 = index[labels== 0]
    cluster2 = index[labels==1]
  
    if len(cluster1) < 10 or len(cluster2) < 10:
        cluster1 = y_vals[y_vals > 0].index.to_numpy()
        cluster2 = y_vals[y_vals <= 0].index.to_numpy()
      


    labels = Series(labels, index)
    return (cluster1, cluster2), y_vals,  labels, stats
    
   

# def check_obs(cells, muts, total, axis=0):
#     nobs =np.count_nonzero(total[np.ix_(cells,muts)], axis=axis)
#     return nobs.mean()

def success_prob(N,parts, d):
  denom = comb(N+ parts-1, N)
  obs_freedom = N -parts*d
  num = comb(obs_freedom + parts -1, obs_freedom)
  return num/denom

 
def power_calc(d,gamma, parts, Nmax):
    
    for n in range(1,Nmax+1):
        p = success_prob(n,parts, d)
        if p >= gamma:
            break 
    return n   

def check_obs(cells, muts, total):
    nobs =np.count_nonzero(total[np.ix_(cells,muts)])
    return nobs

def get_next_label(T):
    """Computes the next incremental node label in the tree
    :return: number. The next incremental integer node label in the tree
    """
    nodes = np.array(list(T.nodes()))
    last_node = np.max(nodes)
    return last_node + 1


def dataframe_to_mapping(df, name="cell"):
    mapping  = {}
    clusters = df['cluster'].unique()
    for c in clusters:
        mapping[c] =df[df['cluster']==c][name].to_numpy()

    return(mapping)

    
def pickle_save(obj, fname):
        with open(fname, 'wb') as handle:
            pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)

        
def pickle_load(fname):
    with open(fname, 'rb') as handle:
        obj = pickle.load(handle)
    
    return obj




def reverse_dict(init_dict):
    output_dict = {}
    for k in init_dict:
        for val in init_dict[k]:
            output_dict[val] = k
    
    return output_dict

def dict_to_series(event_mapping):
    bin_series = []
    for s in event_mapping:
        states = np.full(shape=len(event_mapping[s]), fill_value=s)
        bin_series.append(Series(states, index = event_mapping[s]))
    
    bin_series = concat(bin_series).sort_index()
    return bin_series


def dict_to_dataframe(event_mapping):
    if event_mapping is None or len(event_mapping) ==0:
        return DataFrame()
    series_list = []
    for n in event_mapping:
       series_list.append( dict_to_series(event_mapping[n]))
    
    df = concat(series_list, axis=1)
    df.columns = [ f"node_{n}" for n in list(event_mapping.keys())]
    df.index.rename("bin", inplace=True)
    return df 


 
def compare_dict( d1, d2):

    if d1.keys() != d2.keys():
        return False
    
    for k in d1:
        arr1 = np.sort(d1[k])
        arr2 = np.sort(d2[k])
        if not np.array_equal(arr1, arr2):
            return False 
    
    return True

def compare_dict_of_dict(self, d1, d2):
    if d1.keys() != d2.keys():
        return False
    
    for k in d1:
        if not compare_dict(d1[k], d2[k]):
            return False
    
    return True

def generate_cell_dataframe(cell_mapping, cell_lookup):
    #convert mapping to series in order of cells
    pred_cell_df= mapping_to_dataframe(cell_mapping, "cell_id", cell=True)
    pred_cell_df["cell"] = cell_lookup[pred_cell_df['cell_id']].values
    pred_cell_df = pred_cell_df.drop(['cell_id'], axis=1)


    return pred_cell_df

def generate_mut_dataframe( mut_mapping, mut_lookup):
    #convert mapping to series in order of mutations
    pred_mut_df= mapping_to_dataframe(mut_mapping, "mutation_id")

    pred_mut_df["mutation"] = mut_lookup[pred_mut_df['mutation_id']].values

    pred_mut_df = pred_mut_df.drop(['mutation_id'], axis=1)


    return pred_mut_df
    
def generate_event_dataframe(self, event_mapping):
    #convert mapping to series in order of mutations
    

    return dict_to_dataframe(event_mapping)



def find_largest_component(adj, labels):
    graph = nx.from_numpy_matrix(adj)


    node_mapping = {i : n for i, n in enumerate(labels) }
    graph =nx.relabel_nodes(graph, node_mapping, copy=True)

    # #find connected components 
    connected_components = [graph.subgraph(c).copy() for c in nx.connected_components(graph)]
    # print(f" Graph:\nNodes: {len(labels)} Edges: {len(list.graph.edges())}   Connected Components: {len(connected_components)}  ")
    
    component_size = [len(list(c.nodes())) for c in connected_components]

    for i,c in enumerate(component_size):
        if c > 5:
            print(f"Component {i} Size: {c}")

  
    
    largest = np.argmax(np.array(component_size))

    largest_comp = connected_components[largest]

    nodes_in_comp = np.array(list(largest_comp.nodes()))

    return nodes_in_comp


