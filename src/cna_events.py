

import numpy as np
import pandas as pd
from gaussian_mixture import RDRGaussianMixture



class CNA_HMM():
    def __init__(self,  bins, rdr, chrom_bin_mapping, alpha =0.1,neutral_mean=1.00, neutral_eps=1.5,  debug=False):
        

        self.bins = bins
        self.chrom_bin_mapping = chrom_bin_mapping
     
        self.rdr = rdr
        self.states = ["gain", "loss", "neutral"]
        self.alpha = alpha
        self.neutral_mean = neutral_mean
        self.neutral_eps = neutral_eps
        self.state_to_index = {s: idx for idx, s in enumerate(self.states)}
        self.transitionlogprobs = np.log(np.array([[1-alpha, alpha/2, alpha/2], [ alpha/2, 1-alpha, alpha/2], [alpha/2, alpha/2, 1-alpha]]))
        #uniform prior on initial probabilities
        self.initiallogprobs = {s: np.log(1/3) for s in self.states}
        
        self.emission_dist = RDRGaussianMixture(neutral_mean, neutral_eps)
        self.emission_dist.fit(self.rdr)

        self.debug = True
        
 
    def get_bins(self):
        return self.bins    

  
    def emissionLogProbVec(self, s, vec):
         return self.emission_dist.compute_log_pdf(s, vec)


    def emissionLogProb(self, s,rdr_vector):
    


        obs = rdr_vector.mean()
        emission_log_prob_by_cell = self.emission_dist.compute_log_pdf(s, obs)
        return  emission_log_prob_by_cell.sum()


    def compute_log_transition(self, bin_states):
        obs_states = bin_states.to_numpy()
        obs_states_shifted = np.delete(obs_states, 0)
        obs_states_shifted = np.append(obs_states_shifted, "nan")
        mask = obs_states == obs_states_shifted
        trans_probs = np.full(mask.shape, fill_value=np.log(self.alpha/2))
        trans_probs[mask] = np.log(1-self.alpha)
        trans_probs = np.delete(trans_probs, len(trans_probs)-1)
        trans_probs =np.insert(trans_probs, 0, np.log(1/3))



        return trans_probs

    def compute_likelihood(self, cells, bin_states):

      

        bin_states= bin_states.sort_index()
        rdr_data = self.rdr[cells,]
        chromosomes = self.chrom_bin_mapping[self.bins]
        chroms_present = np.unique(chromosomes)
        log_likelihood = 0
        cell_totals = np.zeros_like(cells, dtype=float)
        for c in chroms_present:
            bins = self.chrom_bin_mapping[self.chrom_bin_mapping==c].index.to_numpy()

            trans_probs =self.compute_log_transition(bin_states.loc[bins])
    
            for i,b in enumerate(bins):
                state = bin_states.loc[b]
                emission_probs = self.emissionLogProbVec(state, rdr_data[:,b])
                tprobs = trans_probs[i]
                cell_totals += emission_probs + tprobs
           
        

        return pd.Series(cell_totals, index=cells)

    
    
    def viterbi(self,  cells, bins):

      
        bin_states_by_chrom = {}
        chromosomes = self.chrom_bin_mapping[self.bins]
        chroms_present = np.unique(chromosomes)
        for c in chroms_present:
            bins = self.chrom_bin_mapping[self.chrom_bin_mapping==c].index.to_numpy()
            

            #run viterbi algorithm and save back points
            V, BT = self.viterbi_bt( cells, bins, self.states)


    
            last_bin = bins[-1]

            max_prob = max([V[s][bins[-1]] for s in self.states])


            bin_states = {}
        
            for s in self.states:
                if V[s][last_bin] == max_prob:
                    bin_states[last_bin]= s
            
            
            for idx in range(len(bins)-2, -1, -1):
                bin = bins[idx]
                next_bin = bins[idx + 1]
                next_bin_state = bin_states[next_bin]
                bin_states[bin] = BT[next_bin_state][next_bin]
            
            bin_states_by_chrom[c] = bin_states
        return bin_states_by_chrom
        
       

    def viterbi_bt(self,cells, bins, states):
        # Positions

        bins = np.sort(bins)

        
        # Initialization v[state][bin] = 0
        # Initialization bt[state][bin]
        v = {s: { bin : 0 for bin in bins } for s in states}
        bt = {s: { bin : None for bin in bins } for s in states }
        rdr = self.rdr[cells, :]
        
        for idx, bin in enumerate(bins):
            rdr_vector = rdr[:,bin]
            
            if idx == 0:
                for s in states:
                    v[s][bin] = self.initiallogprobs[s] + self.emissionLogProb(s, rdr_vector)
            else:
                prev_bin = bins[idx-1]
                for s in states:
                    index_s = self.state_to_index[s]
                    v[s][bin] = self.emissionLogProb(s, rdr_vector)
                    max_d_val =  np.NINF
                    max_d = None
                    for d in states:
                        index_d =self.state_to_index[d]
                        if max_d_val < v[d][prev_bin] + self.transitionlogprobs[index_d, index_s]:
                            max_d = d
                            max_d_val = v[d][prev_bin] + self.transitionlogprobs[index_d, index_s]
                    bt[s][bin] = max_d
                    v[s][bin] += max_d_val
            ### END SOLUTION 
        return v, bt
    
    def reverse_dictionary(self, bin_states):
        state_list = {s: [] for s in self.states}
        for c in bin_states:
            chrom_dict = bin_states[c]
            for b in chrom_dict:
                state_list[chrom_dict[b]].append(b)
        
        return state_list
            
    def convert_to_dataframe(self, state_list):

        
        df_frame_list = []
        for s in self.states:
            df = pd.DataFrame({"bin_idx":state_list[s]})
          
            df["state"] = s
    
            df_frame_list.append(df)
        
        df_results = pd.concat(df_frame_list)
    

        return df_results

    def run(self, cells):
        
        if len(cells)==0:
            return None
     
        bin_states = self.viterbi(cells,self.bins)
        event_mapping = self.reverse_dictionary(bin_states)
            
        return event_mapping

            
