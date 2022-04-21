

import numpy as np
import pandas as pd
from gaussian_mixture import RDRGaussianMixture


class CNA_HMM():
    """
    A hidden Markov model class to model CNA genotypes with three possible states ("gain", "loss", "neutral") 

    ...

    Attributes
    ----------
    states : tuple
        the names of allowable CNA genotype states ("gain", "loss", "neutral")
    bins : np.array
        an array of specified bins
    chrom_bin_mapping : pandas Series
        a mapping of chromosomes to bins   
    rdr : np.array
        a cell x bin matrix containing the read-depth ratio
    beta : float 
        the parameter that specifies 1- beta probability of not transitioning to a different state  
    neutral_mean : float
        the parameter specifying the approximate location of the neutral state 
        used to infer the emission distribution from the rdr data
    neutral_eps : float
        the parmaeter specifying the approximate standard deviation of the neutral state
        used to the emission distribution from the rdr data
    state_to_index : dict
        a mapping of the state variable to the index in the transition probability matrix
    transitionlogprobs : np.array   
        a 3 x 3 matrix specifiying the log transition probabilities of the HMM
    initiallogprobs : dict
        a dictionary specifying the initial log probability of each state. Uses
        a uniform distribution over the states
    emission_dist : RDRGaussianMixture
        an object that captures the emission distribution of each state


    Methods
    -------

    get_bins()
        returns the bins used for the HMM

    emissionLogProbVec(s, vec)
        computes the emission probability of a vector of RDR values for a given state s

    emissionLogProb
         computes the emission probability of the mean of a vector of RDR values for a given state s

    compute_log_transition(bin_states)
        compute the transition probabilities of a series of bin states

    compute_likelihood(cells, bin_states)
        computes the likelihood of RDR valus for a set of cells under the specified HMM and given bin states

    viterbi( cells, bins)
        implements the Viterbi algorithm to decode the most likely set of states for the given data

    viterbi_bt( cells, bins, states)
        performs backtrace of the Viterbi algorithm to decode the most likely set of states for the given data

    reverse_dictionary( bin_states)
        reverses the dictionary resulting in states as keys and arrays of bins having that state as data

    run(cells)
         main flow control to perform decoding of CNA genotype states for a give set of cells

    """

    def __init__(self,  bins, rdr, chrom_bin_mapping, beta=0.1, neutral_mean=1.00, neutral_eps=0.15):

        self.bins = bins
        self.chrom_bin_mapping = chrom_bin_mapping

        self.rdr = rdr
        self.states = ["gain", "loss", "neutral"]
        self.alpha = beta
        self.neutral_mean = neutral_mean
        self.neutral_eps = neutral_eps
        self.state_to_index = {s: idx for idx, s in enumerate(self.states)}
        self.transitionlogprobs = np.log(np.array(
            [[1-beta, beta/2, beta/2], [beta/2, 1-beta, beta/2], [beta/2, beta/2, 1-beta]]))
        # uniform prior on initial probabilities
        self.initiallogprobs = {s: np.log(1/3) for s in self.states}

        self.emission_dist = RDRGaussianMixture(neutral_mean, neutral_eps)
        self.emission_dist.fit(self.rdr)

    def get_bins(self):
        '''gets the bins used for the HMM

        Returns
        -------
        a np.array of bins used for HMM

        '''
        return self.bins

    def emissionLogProbVec(self, s, vec):
        '''computes the emission probability of a vector of RDR values for a given state s

        Parameters
        ----------
        s : str
            the CNA genotype for which the emission probability should be computed
        vec: np.array 
            the RDR values for which the mission probability should be computed

        Returns
        -------
        a np.array of probabilities of the given vector under the specified state 
        '''
        return self.emission_dist.compute_log_pdf(s, vec)

    def emissionLogProb(self, s, rdr_vector):
        '''computes the emission probability of the mean of a vector of RDR values for a given state s

        Parameters
        ----------
        s : str
            the CNA genotype for which the emission probability should be computed
        rdr_vector: np.array 
            the RDR values for which the mission probability should be computed

        Returns
        -------
        a float representing the emiission probability of the mean of RDR vector values for the given state
        '''
        obs = rdr_vector.mean()
        emission_log_prob_by_cell = self.emission_dist.compute_log_pdf(s, obs)
        return emission_log_prob_by_cell.sum()

    def compute_log_transition(self, bin_states):
        ''' compute the transition probabilities of a series of bin states

        Parameters
        ----------
        bin_states : a pandas Series
            a mapping of bins (index) to a give state of CNA genotype states (data)


        Returns
        -------
        an np.array representing the initial and transition probabilities of the given bin states
        '''
        obs_states = bin_states.to_numpy()
        obs_states_shifted = np.delete(obs_states, 0)
        obs_states_shifted = np.append(obs_states_shifted, "nan")
        mask = obs_states == obs_states_shifted
        trans_probs = np.full(mask.shape, fill_value=np.log(self.alpha/2))
        trans_probs[mask] = np.log(1-self.alpha)
        trans_probs = np.delete(trans_probs, len(trans_probs)-1)
        trans_probs = np.insert(trans_probs, 0, np.log(1/3))

        return trans_probs

    def compute_likelihood(self, cells, bin_states):
        ''' computes the likelihood of RDR valus for a set of cells under the specified HMM and given bin states

        Parameters
        ----------
        cells : np.array
            the indices of cells for which the likelihood of RDR values should be computed

        bin_states : a pandas Series
            a mapping of bins (index) to a give state of CNA genotype states (data)


        Returns
        -------
        a pandas Series with cells as indices and the loglikelihood as the data
        '''

        bin_states = bin_states.sort_index()
        rdr_data = self.rdr[cells, ]
        chromosomes = self.chrom_bin_mapping[self.bins]
        chroms_present = np.unique(chromosomes)

        cell_totals = np.zeros_like(cells, dtype=float)
        for c in chroms_present:
            bins = self.chrom_bin_mapping[self.chrom_bin_mapping == c].index.to_numpy(
            )

            trans_probs = self.compute_log_transition(bin_states.loc[bins])

            for i, b in enumerate(bins):
                state = bin_states.loc[b]
                emission_probs = self.emissionLogProbVec(state, rdr_data[:, b])
                tprobs = trans_probs[i]
                cell_totals += emission_probs + tprobs

        return pd.Series(cell_totals, index=cells)

    def viterbi(self,  cells, bins):
        ''' implements the Viterbi algorithm to decode the most likely set of states for the given data

        Parameters
        ----------
        cells : np.array
            the indices of cells for which the likelihood of RDR values should be computed

        bins : np.array
            the bin ids for which decoding should be performed


        Returns
        -------
        a dictionary with chromosomes as keys and bin states as data
        '''

        bin_states_by_chrom = {}
        chromosomes = self.chrom_bin_mapping[self.bins]
        chroms_present = np.unique(chromosomes)
        for c in chroms_present:
            bins = self.chrom_bin_mapping[self.chrom_bin_mapping == c].index.to_numpy(
            )

            # run viterbi algorithm and save back points
            V, BT = self.viterbi_bt(cells, bins, self.states)

            last_bin = bins[-1]

            max_prob = max([V[s][bins[-1]] for s in self.states])

            bin_states = {}

            for s in self.states:
                if V[s][last_bin] == max_prob:
                    bin_states[last_bin] = s

            for idx in range(len(bins)-2, -1, -1):
                bin = bins[idx]
                next_bin = bins[idx + 1]
                next_bin_state = bin_states[next_bin]
                bin_states[bin] = BT[next_bin_state][next_bin]

            bin_states_by_chrom[c] = bin_states
        return bin_states_by_chrom

    def viterbi_bt(self, cells, bins, states):
        ''' performs backtrace of the Viterbi algorithm to decode the most likely set of states for the given data

        Parameters
        ----------
        cells : np.array
            the indices of cells for which the likelihood of RDR values should be computed

        bins : np.array
            the bin ids for which decoding should be performed

        states : list
            list of states to be considered

        Returns
        -------
        v : dict
            max probabilites of the series of hidden states
        bt : dict
            backtrace containing the most likely state path to arrive at the MAP probabilities 
        '''

        bins = np.sort(bins)

        v = {s: {bin: 0 for bin in bins} for s in states}
        bt = {s: {bin: None for bin in bins} for s in states}
        rdr = self.rdr[cells, :]

        for idx, bin in enumerate(bins):
            rdr_vector = rdr[:, bin]

            if idx == 0:
                for s in states:
                    v[s][bin] = self.initiallogprobs[s] + \
                        self.emissionLogProb(s, rdr_vector)
            else:
                prev_bin = bins[idx-1]
                for s in states:
                    index_s = self.state_to_index[s]
                    v[s][bin] = self.emissionLogProb(s, rdr_vector)
                    max_d_val = np.NINF
                    max_d = None
                    for d in states:
                        index_d = self.state_to_index[d]
                        if max_d_val < v[d][prev_bin] + self.transitionlogprobs[index_d, index_s]:
                            max_d = d
                            max_d_val = v[d][prev_bin] + \
                                self.transitionlogprobs[index_d, index_s]
                    bt[s][bin] = max_d
                    v[s][bin] += max_d_val

        return v, bt

    def reverse_dictionary(self, bin_states):
        ''' reverses the dictionary resulting in states as keys and arrays of bins having that state as data

        Parameters
        ----------
        bin_states : dict
            a dictionary with chromosomese as keys and the decoded state sequence for the bins in that chromosomes 


        Returns
        -------
        state_list : dict
            a dictionary with states as keys and np.array of bins decoded to those states

        '''

        state_list = {s: [] for s in self.states}
        for c in bin_states:
            chrom_dict = bin_states[c]
            for b in chrom_dict:
                state_list[chrom_dict[b]].append(b)

        return state_list

    def run(self, cells):
        ''' main flow control to perform decoding of CNA genotype states for a give set of cells

        Parameters
        ----------
        cells : np.array
            the cell indices for which the bin states for that cluster should be decoded

        Returns
        -------
        event_mapping : dict
            a dictionary with states as keys and np.array of bins decoded to those states

        '''

        if len(cells) == 0:
            return None

        bin_states = self.viterbi(cells, self.bins)
        event_mapping = self.reverse_dictionary(bin_states)

        return event_mapping
