# -*- coding: utf-8 -*-

import codecs
import collections
import gc
import math
import multiprocessing
import nltk
import numpy
import os
import Queue
import resource
import scipy
import string
import sys
import time
import util

"""
Filter for sub parse trees
"""
def filter(parse_tree, adapted_non_terminal):
    return parse_tree.node==adapted_non_terminal.symbol();

"""
Compute the aggregate digamma values, for phi update.
"""
def compute_E_log_stick_weights(telescope_1, telescope_2):
    assert(telescope_1.shape==telescope_2.shape);
    
    psi_telescope_1 = scipy.special.psi(telescope_1);
    psi_telescope_2 = scipy.special.psi(telescope_2);
    psi_telescope_all = scipy.special.psi(telescope_1 + telescope_2);

    aggregate_psi_nu_2_minus_psi_nu_all_k = numpy.cumsum(psi_telescope_2 - psi_telescope_all, axis=1);
    E_leftover_stick_weight = aggregate_psi_nu_2_minus_psi_nu_all_k[:, -1];
    
    aggregate_psi_nu_2_minus_psi_nu_all_k = numpy.hstack((numpy.zeros((telescope_1.shape[0], 1)), aggregate_psi_nu_2_minus_psi_nu_all_k[:, :-1]));
    assert(aggregate_psi_nu_2_minus_psi_nu_all_k.shape==telescope_1.shape);

    E_stick_weights = psi_telescope_1 - psi_telescope_all + aggregate_psi_nu_2_minus_psi_nu_all_k;
    
    assert numpy.all(E_stick_weights<0), (E_stick_weights, E_stick_weights<0, telescope_1, telescope_2)
    return E_stick_weights, E_leftover_stick_weight;

"""
Compute the aggregate digamma values, for phi update.
"""
def compute_log_stick_weights(telescope_1, telescope_2):
    assert(telescope_1.shape==telescope_2.shape);
    
    log_telescope_1 = numpy.log(telescope_1);
    log_telescope_2 = numpy.log(telescope_2);
    log_telescope_all = numpy.log(telescope_1 + telescope_2);

    aggregate_log_nu_2_minus_log_nu_all_k = numpy.cumsum(log_telescope_2 - log_telescope_all, axis=1);
    E_leftover_stick_weight = aggregate_log_nu_2_minus_log_nu_all_k[:, -1];
    
    aggregate_log_nu_2_minus_log_nu_all_k = numpy.hstack((numpy.zeros((telescope_1.shape[0], 1)), aggregate_log_nu_2_minus_log_nu_all_k[:, :-1]));
    assert(aggregate_log_nu_2_minus_log_nu_all_k.shape==telescope_1.shape);

    E_stick_weights = log_telescope_1 - log_telescope_all + aggregate_log_nu_2_minus_log_nu_all_k;
    
    assert numpy.all(E_stick_weights<0), (E_stick_weights, E_stick_weights<0, telescope_1, telescope_2)
    return E_stick_weights, E_leftover_stick_weight;

"""
"""
def reverse_cumulative_sum_matrix_over_axis(matrix, axis):
    cumulative_sum = numpy.zeros(matrix.shape);
    (k, n) = matrix.shape;
    if axis == 1:
        for j in xrange(n - 2, -1, -1):
            cumulative_sum[:, j] = cumulative_sum[:, j + 1] + matrix[:, j + 1];
    elif axis == 0:
        for i in xrange(k - 2, -1, -1):
            cumulative_sum[i, :] = cumulative_sum[i + 1, :] + matrix[i + 1, :];

    return cumulative_sum;

def retrieve_tokens_by_pre_order_traversal_of_adapted_non_terminal(production_list, adapted_non_terminal):
    token_list = [];
    for candidate_production in production_list:
        if isinstance(candidate_production, util.AdaptedProduction):
            token_list += candidate_production.retrieve_tokens_of_adapted_non_terminal(adapted_non_terminal);
        else:
            continue;

    return token_list;

'''
def sample_tree_by_pre_order_traversal(hybrid, current_hyper_node, input_string, adapted_sufficient_statistics=None, pcfg_sufficient_statistics=None):
    if pcfg_sufficient_statistics!=None:
        if current_hyper_node._node not in pcfg_sufficient_statistics:
            pcfg_sufficient_statistics[current_hyper_node._node] = numpy.zeros((1, len(hybrid._gamma_index_to_pcfg_production_of_lhs[current_hyper_node._node])));

    sampled_production, unsampled_hyper_nodes, log_probability_of_sampled_production = current_hyper_node.random_sample_derivation();
    if isinstance(sampled_production, util.AdaptedProduction):
        # if sampled production is an adapted production
        assert (unsampled_hyper_nodes==None or len(unsampled_hyper_nodes)==0), "incomplete adapted production: %s" % sampled_production
        #nu_index = self._active_adapted_production_to_nu_index_of_lhs[current_hyper_node._node][sampled_production];
        
        #adapted_sufficient_statistics[current_hyper_node._node][0, nu_index] += 1;
        if adapted_sufficient_statistics!=None:
            if sampled_production not in adapted_sufficient_statistics:
                adapted_sufficient_statistics[sampled_production] = 0;
            adapted_sufficient_statistics[sampled_production] += 1;
        return [sampled_production]
    elif isinstance(sampled_production, nltk.grammar.Production):
        # if sampled production is an pcfg production
        gamma_index = hybrid._pcfg_production_to_gamma_index_of_lhs[current_hyper_node._node][sampled_production];
        if pcfg_sufficient_statistics!=None:
            pcfg_sufficient_statistics[current_hyper_node._node][0, gamma_index] += 1;

        # if sampled production is a pre-terminal pcfg production
        if unsampled_hyper_nodes==None or len(unsampled_hyper_nodes)==0:
            assert (not hybrid.is_adapted_non_terminal(current_hyper_node._node)), "adapted pre-terminal found: %s" % current_hyper_node._node
            return [sampled_production]

        # if sampled production is a regular pcfg production
        production_list = [sampled_production];
        for unsampled_hyper_node in unsampled_hyper_nodes:
            production_list += sample_tree_by_pre_order_traversal(hybrid, unsampled_hyper_node, input_string, adapted_sufficient_statistics, pcfg_sufficient_statistics);

        # if current node is a non-adapted non-terminal node
        if not hybrid.is_adapted_non_terminal(current_hyper_node._node):
            return production_list
        
        new_adapted_production = util.AdaptedProduction(current_hyper_node._node, input_string[current_hyper_node._span[0]:current_hyper_node._span[1]], production_list);
        
        if adapted_sufficient_statistics!=None:
            if new_adapted_production not in adapted_sufficient_statistics:
                adapted_sufficient_statistics[new_adapted_production] = 0;
            adapted_sufficient_statistics[new_adapted_production] += 1;
            
        return [new_adapted_production];
    else:
        sys.stderr.write("Error in recognizing the production class %s @ checkpoint 2...\n" % sampled_production.__class__);
        sys.exit();
'''

class Process_E_Step_dict(multiprocessing.Process):
    def __init__(self,
                 task_queue,
                 hybrid,
                 # for training
                 adapted_sufficient_statistics_dict_queue=None,
                 new_adapted_sufficient_statistics_dict_queue=None,
                 pcfg_sufficient_statistics_dict_queue=None,
                 # for testing
                 retrieve_tokens_at_adapted_non_terminal=None,
                 result_output_path=None,
                 result_model_name=None,
                 
                 number_of_samples=10
                ):
        multiprocessing.Process.__init__(self)
        
        self._task_queue = task_queue;
        
        self._hybrid = hybrid;
        self._number_of_samples = number_of_samples;
       
        # for training only
        self._pcfg_sufficient_statistics_dict_queue = pcfg_sufficient_statistics_dict_queue;
        self._adapted_sufficient_statistics_dict_queue = adapted_sufficient_statistics_dict_queue;
        self._new_adapted_sufficient_statistics_dict_queue = new_adapted_sufficient_statistics_dict_queue;
        
        # for testing only
        self._retrieve_tokens_at_adapted_non_terminal = retrieve_tokens_at_adapted_non_terminal;
        self._result_output_path = result_output_path;
        self._result_model_name = result_model_name;
        
    #@profile
    def run(self):
        E_log_stick_weights, E_log_theta = self._hybrid.propose_pcfg();
        
        if self._result_output_path!=None:
            output_truth_file_stream = open(os.path.join(self._result_output_path, "%s.%s.avg.truth.%s" % (self._retrieve_tokens_at_adapted_non_terminal, self._result_model_name, self.name)), 'w');
            output_test_file_stream = open(os.path.join(self._result_output_path, "%s.%s.avg.test.%s" % (self._retrieve_tokens_at_adapted_non_terminal, self._result_model_name, self.name)), 'w');
        else:
            pcfg_sufficient_statistics = {};
            for non_terminal in E_log_theta:
                pcfg_sufficient_statistics[non_terminal] = numpy.zeros(E_log_theta[non_terminal].shape);
            adapted_sufficient_statistics = {};
            for adapted_non_terminal in E_log_stick_weights:
                adapted_sufficient_statistics[adapted_non_terminal] = numpy.zeros(E_log_stick_weights[adapted_non_terminal].shape);
            new_adapted_sufficient_statistics = {}
            for adapted_non_terminal in E_log_stick_weights:
                new_adapted_sufficient_statistics[adapted_non_terminal] = {};
                
        #process_data_clock = time.time()
        while not self._task_queue.empty():
            try:
                (input_string, reference_string) = self._task_queue.get_nowait();
            except Queue.Empty:
                continue;

            #print 'process %s is processing string %s...' % (self.name, input_string)
            parsed_string = input_string.split();
            
            #compute_inside_probabilities_clock = time.time()
            root_node = self._hybrid.compute_inside_probabilities(E_log_stick_weights, E_log_theta, parsed_string);
            #compute_inside_probabilities_clock = time.time() - compute_inside_probabilities_clock
            #print "time to compute inside probabilities", compute_inside_probabilities_clock

            #sample_tree_clock = time.time()
            for sample_index in xrange(self._number_of_samples):
                if self._result_output_path!=None:
                    production_list = self._hybrid._sample_tree_process_dict(root_node, parsed_string)
                    retrieved_tokens = retrieve_tokens_by_pre_order_traversal_of_adapted_non_terminal(production_list, self._retrieve_tokens_at_adapted_non_terminal);

                    output_truth_file_stream.write("%s\n" % (reference_string));
                    output_test_file_stream.write("%s\n" % (" ".join(retrieved_tokens)));
                else:
                    production_list = self._hybrid._sample_tree_process_dict(root_node, parsed_string, adapted_sufficient_statistics, new_adapted_sufficient_statistics, pcfg_sufficient_statistics);
            #sample_tree_clock = time.time() - sample_tree_clock
            #print "time to sample tree", sample_tree_clock
                
            del root_node;
            
            self._task_queue.task_done()
            
        #process_data_clock = time.time() - process_data_clock;
        #print "time to process data", process_data_clock
        
        if self._result_output_path!=None:
            output_truth_file_stream.close();
            output_test_file_stream.close();
        else:
            #accumulation_sufficient_statistics_clock = time.time();
            
            self._adapted_sufficient_statistics_dict_queue.put(adapted_sufficient_statistics);
            self._new_adapted_sufficient_statistics_dict_queue.put(new_adapted_sufficient_statistics);
            self._pcfg_sufficient_statistics_dict_queue.put(pcfg_sufficient_statistics);
            
            '''
            (adapted_sufficient_statistics_mutex, new_adapted_sufficient_statistics_mutex, pcfg_sufficient_statistics_mutex) = self._sufficient_statistics_mutex;
            (adapted_sufficient_statistics_dict, new_adapted_sufficient_statistics_dict, pcfg_sufficient_statistics_dict) = self._sufficient_statistics_dict;
            
            adapted_sufficient_statistics_mutex.acquire();
            try:
                for adapted_non_terminal in adapted_sufficient_statistics:
                    temp_sufficient_statistics = self._adapted_sufficient_statistics_dict_queue[adapted_non_terminal];
                    #print "+++++ adapted +++++", self.name, adapted_non_terminal, adapted_sufficient_statistics[adapted_non_terminal].shape, temp_sufficient_statistics.shape;
                    if temp_sufficient_statistics.shape!=adapted_sufficient_statistics[adapted_non_terminal].shape:
                        print temp_sufficient_statistics.shape, adapted_sufficient_statistics[adapted_non_terminal].shape
                        print temp_sufficient_statistics,
                        print adapted_sufficient_statistics[adapted_non_terminal]
                    temp_sufficient_statistics += adapted_sufficient_statistics[adapted_non_terminal];
                    self._adapted_sufficient_statistics_dict_queue[adapted_non_terminal] = temp_sufficient_statistics;
            finally:
                adapted_sufficient_statistics_mutex.release();
                
            new_adapted_sufficient_statistics_mutex.acquire();
            try:
                for adapted_non_terminal in new_adapted_sufficient_statistics:
                    #print "+++++ new-adapted +++++", self.name, adapted_non_terminal, len(new_adapted_sufficient_statistics[adapted_non_terminal])
                    temp_sufficient_statistics = self._new_adapted_sufficient_statistics_dict_queue[adapted_non_terminal];
                    for new_adapted_production in new_adapted_sufficient_statistics[adapted_non_terminal]:
                        if new_adapted_production not in temp_sufficient_statistics:
                            temp_sufficient_statistics[new_adapted_production] = 0;
                        temp_sufficient_statistics[new_adapted_production] += new_adapted_sufficient_statistics[adapted_non_terminal][new_adapted_production];
                    self._new_adapted_sufficient_statistics_dict_queue[adapted_non_terminal] = temp_sufficient_statistics;
            finally:
                new_adapted_sufficient_statistics_mutex.release();
                
            pcfg_sufficient_statistics_mutex.acquire();
            try:
                for non_terminal in pcfg_sufficient_statistics:
                    #print "+++++ pcfg +++++", self.name, non_terminal, pcfg_sufficient_statistics[non_terminal].shape
                    temp_sufficient_statistics = self._pcfg_sufficient_statistics_dict_queue[non_terminal];
                    temp_sufficient_statistics += pcfg_sufficient_statistics[non_terminal];
                    self._pcfg_sufficient_statistics_dict_queue[non_terminal] = temp_sufficient_statistics;
            finally:
                pcfg_sufficient_statistics_mutex.release();
            '''
            
            #accumulation_sufficient_statistics_clock = time.time() - accumulation_sufficient_statistics_clock;
            #print "time to accumulate sufficient statistics", accumulation_sufficient_statistics_clock

class Process_E_Step_queue(multiprocessing.Process):
    def __init__(self,
                 task_queue,
                 hybrid,
                 E_log_theta,
                 E_log_stick_weights,
                 result_queue_adapted_sufficient_statistics=None,
                 result_queue_pcfg_sufficient_statistics=None,
                 retrieve_tokens_at_adapted_non_terminal=None,
                 result_output_path=None,
                 result_model_name=None,
                 number_of_samples=10
                ):
        multiprocessing.Process.__init__(self)
        
        self._task_queue = task_queue;
        
        self._hybrid = hybrid;
        self._number_of_samples = number_of_samples;
       
        self._E_log_theta = E_log_theta;
        self._E_log_stick_weights = E_log_stick_weights;

        # for training only
        self._result_queue_pcfg_sufficient_statistics = result_queue_pcfg_sufficient_statistics;
        self._result_queue_adapted_sufficient_statistics = result_queue_adapted_sufficient_statistics;
        
        # for testing only
        #self._result_queue_production_list = result_queue_production_list;
        self._retrieve_tokens_at_adapted_non_terminal = retrieve_tokens_at_adapted_non_terminal;
        self._result_output_path = result_output_path;
        self._result_model_name = result_model_name;
        
    #@profile
    def run(self):
        if self._result_output_path!=None:
            output_truth_file_stream = open(os.path.join(self._result_output_path, "%s.%s.avg.truth.%s" % (self._retrieve_tokens_at_adapted_non_terminal, self._result_model_name, self.name)), 'w');
            output_test_file_stream = open(os.path.join(self._result_output_path, "%s.%s.avg.test.%s" % (self._retrieve_tokens_at_adapted_non_terminal, self._result_model_name, self.name)), 'w');

        while not self._task_queue.empty():
            try:
                (input_string, reference_string) = self._task_queue.get_nowait();
            except Queue.Empty:
                continue;

            #print 'process %s is processing string %s...' % (self.name, input_string)
            parsed_string = input_string.split();
            
            #compute_inside_probabilities_clock = time.time()
            root_node = self._hybrid.compute_inside_probabilities(self._E_log_stick_weights, self._E_log_theta, parsed_string);
            #compute_inside_probabilities_clock = time.time() - compute_inside_probabilities_clock
            #print "time to compute inside probabilities", compute_inside_probabilities_clock
        
            #sample_tree_clock = time.time()
            for sample_index in xrange(self._number_of_samples):
                if self._result_output_path!=None:
                    production_list = self._hybrid._sample_tree_process_queue(root_node, parsed_string)
                    retrieved_tokens = retrieve_tokens_by_pre_order_traversal_of_adapted_non_terminal(production_list, self._retrieve_tokens_at_adapted_non_terminal);
                    #print " ".join(retrieved_tokens), reference_string
                    #self._result_queue_production_list.put((" ".join(retrieved_tokens), reference_string));

                    output_truth_file_stream.write("%s\n" % (reference_string));
                    output_test_file_stream.write("%s\n" % (" ".join(retrieved_tokens)));
                else:
                    production_list = self._hybrid._sample_tree_process_queue(root_node, parsed_string, self._result_queue_adapted_sufficient_statistics, self._result_queue_pcfg_sufficient_statistics)
                        
                #self.model_state_assertion();
            #sample_tree_clock = time.time() - sample_tree_clock
            #print "time to sample tree", sample_tree_clock
            
            del root_node;
            
            self._task_queue.task_done()
        
        if self._result_output_path!=None:
            output_truth_file_stream.close();
            output_test_file_stream.close();

class Hybrid(object):
    def __init__(self,
                 start_symbol,
                 pcfg_productions,
                 adapted_non_terminals,
                 number_of_samples=10
                    ):
        self._number_of_samples = number_of_samples;
        
        self._start_symbol = start_symbol;
        
        self._adapted_non_terminals = set(adapted_non_terminals);
        self._non_terminals = set(pcfg_production.lhs() for pcfg_production in pcfg_productions)
        self._terminals = set();
        for pcfg_production in pcfg_productions:
            self._terminals |= set(pcfg_production.rhs()) - self._non_terminals;

        print "terminals:", " ".join([terminal.encode('utf-8') for terminal in self._terminals])
        print "non-terminals:", self._non_terminals
        print "adapted non-terminal:", self._adapted_non_terminals;
        
        assert(self._non_terminals.isdisjoint(self._terminals))
        assert(self._adapted_non_terminals.isdisjoint(self._terminals))
        assert(self._adapted_non_terminals.issubset(self._non_terminals))
        
        self._pcfg_productions = collections.defaultdict(set);
        
        self._number_of_productions = 0;
        self._number_of_productions_prime = 0;
        
        self._lhs_to_pcfg_production = collections.defaultdict(set);
        self._rhs_to_pcfg_production = collections.defaultdict(set);
        self._lhs_rhs_to_pcfg_production = collections.defaultdict();
        self._rhs_to_unary_pcfg_production = collections.defaultdict(set);

        self._gamma_index_to_pcfg_production_of_lhs = collections.defaultdict(collections.defaultdict);
        self._pcfg_production_to_gamma_index_of_lhs = collections.defaultdict(collections.defaultdict);

        for pcfg_production in pcfg_productions:
            # make sure all pcfg production is in CNF
            assert(len(pcfg_production.rhs())>=1 and len(pcfg_production.rhs())<=2);
            
            lhs_node = pcfg_production.lhs();
            rhs_nodes = pcfg_production.rhs();
            
            self._pcfg_productions[(lhs_node, rhs_nodes)].add(pcfg_production);
            
            self._lhs_rhs_to_pcfg_production[(lhs_node, rhs_nodes)] = pcfg_production
            self._lhs_to_pcfg_production[lhs_node].add(pcfg_production);
            self._rhs_to_pcfg_production[rhs_nodes].add(pcfg_production);
            if len(rhs_nodes)==1:
                self._rhs_to_unary_pcfg_production[rhs_nodes[0]].add(pcfg_production);

            self._gamma_index_to_pcfg_production_of_lhs[lhs_node][len(self._gamma_index_to_pcfg_production_of_lhs[lhs_node])] = pcfg_production;
            self._pcfg_production_to_gamma_index_of_lhs[lhs_node][pcfg_production] = len(self._pcfg_production_to_gamma_index_of_lhs[lhs_node]);

        topology_order, order_topology = self._topological_sort();
        
        self._incremental_build_up = False;
        self._non_terminal_to_level = topology_order;
        self._level_to_non_terminal = order_topology;
        
        self._ordered_adaptor_top_down = [];
        for x in xrange(len(order_topology)):
            for non_terminal in order_topology[x]:
                if non_terminal in self._adapted_non_terminals:
                    self._ordered_adaptor_top_down.append(non_terminal);
        self._ordered_adaptor_down_top = self._ordered_adaptor_top_down[::-1];
        
        print "adaptors in top-down order:", self._ordered_adaptor_top_down
        
    def _initialize(self,
                    number_of_strings,
                    batch_size,
                    tau=1.,
                    kappa=0.5,
                    alpha_pi=None,
                    beta_pi=None,
                    alpha_theta=None,
                    truncation_level=None,
                    reorder_interval=10,
                    table_relabel_interval=500,
                    table_relabel_iterations=100,
                    sufficient_statistics_scale=0,
                    #pcfg_rhs_scale_coefficient=10
                    ):
        self._number_of_strings = number_of_strings;
        self._batch_size = batch_size;
        
        self._counter = 0;
        self._tau = tau;
        self._kappa = kappa;
        self._epsilon = pow(self._tau + self._counter, -self._kappa);
        
        self._reorder_interval = reorder_interval;
        self._table_relabel_iterations = 500/batch_size;
        self._table_relabel_interval = table_relabel_interval;
        self._initial_table_relable = False;
        
        self._alpha_theta = {};
        if alpha_theta==None:
            for non_terminal in self._non_terminals:
                #self._alpha_theta[non_terminal] = numpy.ones((1, len(self._gamma_index_to_pcfg_production_of_lhs[non_terminal]))) / 10;
                self._alpha_theta[non_terminal] = numpy.ones((1, len(self._gamma_index_to_pcfg_production_of_lhs[non_terminal]))) / len(self._gamma_index_to_pcfg_production_of_lhs[non_terminal]);
                #self._alpha_theta[non_terminal] = 1. / len(self._gamma_index_to_pcfg_production_of_lhs[non_terminal]);
            
        self._gamma = {};
        self._pcfg_sufficient_statistics_of_lhs = {};
        self._pcfg_production_usage_counts_of_lhs = {};
        for non_terminal in self._non_terminals:
            self._gamma[non_terminal] = numpy.ones((1, len(self._gamma_index_to_pcfg_production_of_lhs[non_terminal]))) / len(self._gamma_index_to_pcfg_production_of_lhs[non_terminal]);
            #self._gamma[non_terminal] = numpy.ones((1, len(self._gamma_index_to_pcfg_production_of_lhs[non_terminal]))) / 10;
            #for gamma_index in xrange(len(self._gamma_index_to_pcfg_production_of_lhs[non_terminal])):
                #self._gamma[non_terminal][0, gamma_index] *= len(self._gamma_index_to_pcfg_production_of_lhs[non_terminal][gamma_index].rhs());
            self._pcfg_sufficient_statistics_of_lhs[non_terminal] = numpy.zeros((1, len(self._gamma_index_to_pcfg_production_of_lhs[non_terminal])));
            self._pcfg_production_usage_counts_of_lhs[non_terminal] = numpy.zeros((1, len(self._gamma_index_to_pcfg_production_of_lhs[non_terminal])), dtype=int);

            '''
            # TODO: scale the thetas
            for pcfg_production in self._pcfg_production_to_gamma_index_of_lhs[non_terminal]:
                gamma_index = self._pcfg_production_to_gamma_index_of_lhs[non_terminal][pcfg_production];
                self._alpha_theta[non_terminal][0, gamma_index] *= pcfg_rhs_scale_coefficient**(len(pcfg_production.rhs())-1);
                self._gamma[non_terminal][0, gamma_index] *= pcfg_rhs_scale_coefficient**(len(pcfg_production.rhs())-1);
            #print self._alpha_theta[non_terminal], self._gamma[non_terminal]
            '''
            
        if truncation_level==None:
            truncation_level = {};
            for adapted_non_terminal in self._ordered_adaptor_top_down[::-1]:
                truncation_level[adapted_non_terminal] = 1000;
        self._truncation_level = truncation_level;
        assert(len(self._truncation_level)==len(self._adapted_non_terminals))                
        print "desired_truncation_level:", self._truncation_level;
          
        if sufficient_statistics_scale<=0:
            self._ranking_statistics_scale = 1.0/pow(self._tau, -self._kappa);
            #self._ranking_statistics_scale = 1.0;
        
        self._lhs_rhs_to_active_adapted_production = collections.defaultdict(set);
        self._lhs_to_active_adapted_production = collections.defaultdict(set);
        self._rhs_to_active_adapted_production = collections.defaultdict(set);
        
        if alpha_pi==None:
            alpha_pi = {}
            for adapted_non_terminal in self._adapted_non_terminals:
                alpha_pi[adapted_non_terminal] = 1e3;
        self._alpha_pi = alpha_pi;                
        assert(len(self._alpha_pi)==len(self._adapted_non_terminals))
        print "alpha_pi:", self._alpha_pi
        
        if beta_pi==None:
            beta_pi = {}
            for adapted_non_terminal in self._adapted_non_terminals:
                beta_pi[adapted_non_terminal] = 0;
        self._beta_pi = beta_pi;
        assert(len(self._beta_pi)==len(self._adapted_non_terminals))
        print "beta_pi:", self._beta_pi

        self._nu_1 = {};
        self._nu_2 = {};

        self._nu_index_to_active_adapted_production_of_lhs = {};
        self._active_adapted_production_to_nu_index_of_lhs = {};

        self._active_adapted_production_sufficient_statistics_of_lhs = {};
        self._active_adapted_production_usage_counts_of_lhs = {};
        self._active_adapted_production_length_of_lhs = {};

        for adapted_non_terminal in self._adapted_non_terminals:
            self._nu_index_to_active_adapted_production_of_lhs[adapted_non_terminal] = {};
            self._active_adapted_production_to_nu_index_of_lhs[adapted_non_terminal] = {};
            self._nu_1[adapted_non_terminal] = numpy.ones((1, len(self._nu_index_to_active_adapted_production_of_lhs[adapted_non_terminal])));
            self._nu_2[adapted_non_terminal] = numpy.ones((1, len(self._nu_index_to_active_adapted_production_of_lhs[adapted_non_terminal])))*self._alpha_pi[adapted_non_terminal];
            self._active_adapted_production_sufficient_statistics_of_lhs[adapted_non_terminal] = numpy.zeros((1, len(self._nu_index_to_active_adapted_production_of_lhs[adapted_non_terminal])));
            self._active_adapted_production_usage_counts_of_lhs[adapted_non_terminal] = numpy.zeros((1, len(self._nu_index_to_active_adapted_production_of_lhs[adapted_non_terminal])), dtype=int);
            self._active_adapted_production_length_of_lhs[adapted_non_terminal] = numpy.zeros((1, len(self._nu_index_to_active_adapted_production_of_lhs[adapted_non_terminal])), dtype=int);
            #self._adapted_production_sufficient_statistics_of_lhs[adapted_non_terminal] = nltk.probability.FreqDist();
            
        #self._adapted_production_usage_freqdist = collections.defaultdict(nltk.probability.FreqDist);
        self._adapted_production_usage_freqdist = nltk.probability.FreqDist();
        self._adapted_production_dependents_of_adapted_production = collections.defaultdict(set);
        self._adapted_production_sufficient_statistics_of_lhs = collections.defaultdict(nltk.probability.FreqDist);
        
    def _topological_sort(self):
        dag = collections.defaultdict(set);
        #unlinked_nodes[self._start_symbol] = set();
        unlinked_nodes = set();
        unlinked_nodes.add(self._start_symbol);
        while(len(unlinked_nodes)>0):
            candidate_node = unlinked_nodes.pop();
            for candidate_pcfg_production in self.get_pcfg_productions(lhs=candidate_node):
                for non_terminal in candidate_pcfg_production.rhs():
                    if not isinstance(non_terminal, nltk.grammar.Nonterminal):
                        continue;
                    if non_terminal==candidate_node:
                        continue;
                    dag[candidate_node].add(non_terminal);
                    unlinked_nodes.add(non_terminal);

        topology_ordering = {};
        ordering_topology = collections.defaultdict(set);
        topology_ordering[self._start_symbol] = 0;
        ordering_topology[0].add(self._start_symbol);
        unprocessed_nodes = [(self._start_symbol, 0)];
        while len(unprocessed_nodes)>0:
            (unprocessed_node, depth) = unprocessed_nodes.pop(0);
            for child_node in dag[unprocessed_node]:
                if child_node in topology_ordering:
                    #topology_ordering[child_node] = min(depth+1, topology_ordering[child_node]);
                    topology_ordering[child_node] = max(depth+1, topology_ordering[child_node]);
                    ordering_topology[max(depth+1, topology_ordering[child_node])].add(child_node);
                else:
                    topology_ordering[child_node] = depth+1;
                    ordering_topology[depth+1].add(child_node);
                unprocessed_nodes.append((child_node, topology_ordering[child_node]));
                
        print "topological ordering is:", topology_ordering
        
        return topology_ordering, ordering_topology

    def propose_pcfg(self):
        E_log_stick_weights = {};
        E_log_left_over_stick_weights = {};
        for adapted_non_terminal in self._adapted_non_terminals:
            if len(self._nu_index_to_active_adapted_production_of_lhs[adapted_non_terminal])<=0:
                E_log_stick_weights[adapted_non_terminal] = numpy.zeros((1, len(self._nu_index_to_active_adapted_production_of_lhs[adapted_non_terminal])));
                E_log_left_over_stick_weights[adapted_non_terminal] = 0;
            else:
                E_log_stick_weights[adapted_non_terminal], E_log_left_over_stick_weights[adapted_non_terminal] = compute_E_log_stick_weights(self._nu_1[adapted_non_terminal], self._nu_2[adapted_non_terminal]);
                #E_log_stick_weights[adapted_non_terminal], E_log_left_over_stick_weights[adapted_non_terminal] = compute_log_stick_weights(self._nu_1[adapted_non_terminal], self._nu_2[adapted_non_terminal]);
            assert E_log_stick_weights[adapted_non_terminal].shape==(1, len(self._nu_index_to_active_adapted_production_of_lhs[adapted_non_terminal])), (adapted_non_terminal, len(self._nu_index_to_active_adapted_production_of_lhs[adapted_non_terminal]), E_log_stick_weights[adapted_non_terminal].shape, E_log_left_over_stick_weights[adapted_non_terminal].shape);

        E_log_theta = {};
        for non_terminal in self._non_terminals:
            E_log_theta[non_terminal] = scipy.special.psi(self._gamma[non_terminal]) - scipy.special.psi(numpy.sum(self._gamma[non_terminal]));
            assert(E_log_theta[non_terminal].shape==(1, len(self._gamma_index_to_pcfg_production_of_lhs[non_terminal])));
            
            if self.is_adapted_non_terminal(non_terminal):
                E_log_theta[non_terminal] += E_log_left_over_stick_weights[non_terminal];
                
        return E_log_stick_weights, E_log_theta
    
    def compute_inside_probabilities(self, E_log_stick_weights, E_log_theta, input_sequence, sentence_root=None, candidate_adaptors=None):
        #E_log_stick_weights, E_log_theta = self.propose_pcfg();
        if candidate_adaptors==None:
            candidate_adaptors = self._adapted_non_terminals;
        
        sequence_length = len(input_sequence);
        
        root_and_position_to_node = collections.defaultdict(dict);
        position_and_root_to_node = collections.defaultdict(dict);
        
        for span in xrange(1, sequence_length+1):
            for i in xrange(sequence_length-span+1):
                j = i+span;
                for non_terminal in self._non_terminals:
                    lhs = non_terminal;
                    
                    # find the adapted production that spans over i to j
                    if non_terminal in candidate_adaptors:
                        #print self._active_adapted_production_to_nu_index_of_lhs[non_terminal]
                        candidate_adapted_productions = self.get_adapted_productions(lhs=non_terminal, rhs=tuple(input_sequence[i:j]));
                        for candidate_adapted_production in candidate_adapted_productions:
                            nu_index = self._active_adapted_production_to_nu_index_of_lhs[lhs][candidate_adapted_production];
                            
                            # this is to prevent searching new sampled rules
                            if nu_index>=E_log_stick_weights[lhs].shape[1]:
                                continue;
                            
                            if (i, j) not in root_and_position_to_node[lhs]:
                                hyper_node = util.HyperNode(lhs, (i, j));
                                root_and_position_to_node[lhs][(i, j)] = hyper_node;
                                position_and_root_to_node[(i, j)][lhs] = hyper_node;
                                
                            root_and_position_to_node[lhs][(i, j)].add_new_derivation(candidate_adapted_production, E_log_stick_weights[lhs][0, nu_index], hyper_nodes=None);
                            
                    # find the pcfg productions
                    candidate_pcfg_productions = self.get_pcfg_productions(lhs=non_terminal, rhs=None);
                    for candidate_pcfg_production in candidate_pcfg_productions:
                        # make sure all pcfg production is in CNF  
                        #assert(len(candidate_pcfg_production.rhs())==1 or len(candidate_pcfg_production.rhs())==2)
                        
                        rhs_0 = candidate_pcfg_production.rhs()[0];

                        gamma_index = self._pcfg_production_to_gamma_index_of_lhs[lhs][candidate_pcfg_production];

                        if len(candidate_pcfg_production.rhs())==1:
                            if span==1:
                                if rhs_0==input_sequence[i:j][0]:
                                    # this is a terminal initialization rule, otherwise, we don't consider
                                    hyper_node = util.HyperNode(lhs, (i, j));
                                    hyper_node.add_new_derivation(candidate_pcfg_production, E_log_theta[lhs][0, gamma_index], hyper_nodes=None);
                                    root_and_position_to_node[lhs][(i, j)] = hyper_node;
                                    position_and_root_to_node[(i, j)][lhs] = hyper_node;
                                else:
                                    continue;
                            else:
                                continue;
                        elif len(candidate_pcfg_production.rhs())==2:
                            if rhs_0 not in root_and_position_to_node:
                                continue;

                            rhs_1 = candidate_pcfg_production.rhs()[1];
                            assert(self.is_non_terminal(rhs_1));
                            
                            if rhs_1 in root_and_position_to_node:
                                for k in xrange(i+1, j):
                                    if (i, k) not in root_and_position_to_node[rhs_0]:
                                        continue;
                                    if (k, j) not in root_and_position_to_node[rhs_1]:
                                        continue;
                                    
                                    if (i, j) not in root_and_position_to_node[lhs]:
                                        hyper_node = util.HyperNode(lhs, (i, j));
                                        root_and_position_to_node[lhs][(i, j)] = hyper_node;
                                        position_and_root_to_node[(i, j)][lhs] = hyper_node;
                                    log_probability = E_log_theta[lhs][0, gamma_index] + root_and_position_to_node[rhs_0][(i, k)]._accumulated_log_probability + root_and_position_to_node[rhs_1][(k, j)]._accumulated_log_probability;
                                    root_and_position_to_node[lhs][(i, j)].add_new_derivation(candidate_pcfg_production, log_probability, [root_and_position_to_node[rhs_0][(i, k)], root_and_position_to_node[rhs_1][(k, j)]]);
                                    #root_and_position_to_node[lhs][(i, j)].add_new_derivation(candidate_pcfg_production, E_log_theta[lhs][0, gamma_index], [root_and_position_to_node[rhs_0][(i, k)], root_and_position_to_node[rhs_1][(k, j)]]);
                        else:
                            sys.stderr.write('Error: pcfg production not in CNF...\n');
                            sys.exit();
                
                unary_node_set = set(position_and_root_to_node[(i, j)]);
                while len(unary_node_set)>0:
                    non_terminal = unary_node_set.pop();
                    
                    unary_productions = self.get_unary_pcfg_productions_by_rhs(rhs=non_terminal);
                    for unary_production in unary_productions:
                        if len(unary_production.rhs())!=1:
                            continue;
                        
                        unary_node_set.add(unary_production.lhs());
                        lhs = unary_production.lhs();
                        rhs = unary_production.rhs()[0];

                        gamma_index = self._pcfg_production_to_gamma_index_of_lhs[lhs][unary_production];
                        
                        if (i, j) not in root_and_position_to_node[lhs]:
                            hyper_node = util.HyperNode(lhs, (i, j));
                            root_and_position_to_node[lhs][(i, j)] = hyper_node;
                            position_and_root_to_node[(i, j)][lhs] = hyper_node;
                        log_probability = E_log_theta[lhs][0, gamma_index] + root_and_position_to_node[rhs][(i, j)]._accumulated_log_probability;
                        root_and_position_to_node[lhs][(i, j)].add_new_derivation(unary_production, log_probability, [root_and_position_to_node[rhs][(i, j)]]);
                        #root_and_position_to_node[lhs][(i, j)].add_new_derivation(unary_production, E_log_theta[lhs][0, gamma_index], [root_and_position_to_node[rhs][(i, j)]]);
        
        if sentence_root==None:
            return root_and_position_to_node[self._start_symbol][(0, sequence_length)];
        else:
            assert isinstance(sentence_root, nltk.grammar.Nonterminal);
            return root_and_position_to_node[sentence_root][(0, sequence_length)];
    
    '''
    def e_step_inference(self, input_strings, reference_strings, retrieve_tokens_at_adapted_non_terminal, output_path, model_name, number_of_samples=10):
        assert(retrieve_tokens_at_adapted_non_terminal in self._adapted_non_terminals);
        #if number_of_samples==None:
            #number_of_samples = self._number_of_samples;

        output_average_truth_file = open(os.path.join(output_path, "%s.%s.avg.truth" % (retrieve_tokens_at_adapted_non_terminal, model_name)), 'w');
        output_average_test_file = open(os.path.join(output_path, "%s.%s.avg.test" % (retrieve_tokens_at_adapted_non_terminal, model_name)), 'w');
        #output_maximum_truth_file = open(os.path.join(output_path, "%s.%s.max.truth" % (retrieve_tokens_at_adapted_non_terminal, model_name)), 'w');
        #output_maximum_test_file = open(os.path.join(output_path, "%s.%s.max.test" % (retrieve_tokens_at_adapted_non_terminal, model_name)), 'w');
        
        pcfg_sufficient_statistics = {};
        adapted_sufficient_statistics = {};
        
        E_log_stick_weights, E_log_theta = self.propose_pcfg();

        counter = 0;
        for (input_string, reference_string) in zip(input_strings, reference_strings):
            retrieved_tokens_lists = nltk.probability.FreqDist();

            #parsed_string = [ch for ch in input_string if ch not in string.whitespace];
            parsed_string = input_string.split();
            root_node = self.compute_inside_probabilities(E_log_stick_weights, E_log_theta, parsed_string, );
        
            for sample_index in xrange(number_of_samples):
                production_list = self._sample_tree(root_node, parsed_string, pcfg_sufficient_statistics, adapted_sufficient_statistics, inference_mode=True);
                #production_list = sample_tree_by_pre_order_traversal(self, root_node, parsed_string, None, None);

                #output_average_test_file.write("%s\n" % production_list);
                retrieved_tokens = retrieve_tokens_by_pre_order_traversal_of_adapted_non_terminal(production_list, retrieve_tokens_at_adapted_non_terminal);
                #retrieved_tokens_lists[input_string].append(" ".join(retrieved_tokens));
                retrieved_tokens_lists.inc(" ".join(retrieved_tokens), 1);
            
            assert(retrieved_tokens_lists.N()==number_of_samples)

            maximum_tokens = retrieved_tokens_lists.max();
            #output_maximum_truth_file.write("%s\n" % reference_string);
            #output_maximum_test_file.write("%s\n" % maximum_tokens);
            
            for average_tokens in retrieved_tokens_lists.samples():
                for x in xrange(retrieved_tokens_lists[average_tokens]):
                    output_average_truth_file.write("%s\n" % reference_string);
                    output_average_test_file.write("%s\n" % average_tokens);
            
            counter += 1;
            if counter % 5000 == 0:
                print "processed %g%% data..." % (counter * 100.0 / len(input_strings));

        return
    '''
    
    def e_step(self, input_strings, number_of_samples, inference_parameter=None):
        if inference_parameter==None:
            reference_strings = None;
            retrieve_tokens_at_adapted_non_terminal = None;
            output_path = None;
            model_name = None;
        else:
            (reference_strings, retrieve_tokens_at_adapted_non_terminal, output_path, model_name) = inference_parameter;
            assert retrieve_tokens_at_adapted_non_terminal in self._adapted_non_terminals;
            assert len(input_strings)==len(reference_strings);
            output_average_truth_file = open(os.path.join(output_path, "%s.%s.avg.truth" % (retrieve_tokens_at_adapted_non_terminal, model_name)), 'w');
            output_average_test_file = open(os.path.join(output_path, "%s.%s.avg.test" % (retrieve_tokens_at_adapted_non_terminal, model_name)), 'w');
        
        if inference_parameter==None:
            pcfg_sufficient_statistics = {};
            for non_terminal in self._non_terminals:
                pcfg_sufficient_statistics[non_terminal] = numpy.zeros((1, len(self._gamma_index_to_pcfg_production_of_lhs[non_terminal])));
            
            adapted_sufficient_statistics = {};
            for adapted_non_terminal in self._adapted_non_terminals:
                adapted_sufficient_statistics[adapted_non_terminal] = numpy.zeros((1, len(self._nu_index_to_active_adapted_production_of_lhs[adapted_non_terminal])));
        else:
            pcfg_sufficient_statistics = None;
            adapted_sufficient_statistics = None;
            log_likelihood = 0;
        
        E_log_stick_weights, E_log_theta = self.propose_pcfg();
        
        #for input_string in input_strings:
        for string_index in xrange(len(input_strings)):
            input_string = input_strings[string_index];
            
            #compute_inside_probabilities_clock = time.time()
            parsed_string = input_string.split();
            root_node = self.compute_inside_probabilities(E_log_stick_weights, E_log_theta, parsed_string);
            #self.model_state_assertion();
            #compute_inside_probabilities_clock = time.time() - compute_inside_probabilities_clock
            #print "time to compute inside probabilities", compute_inside_probabilities_clock

            if inference_parameter!=None:
                retrieved_tokens_lists = nltk.probability.FreqDist();
            
            #sample_tree_clock = time.time()
            for sample_index in xrange(number_of_samples):
                production_list = self._sample_tree(root_node, parsed_string, pcfg_sufficient_statistics, adapted_sufficient_statistics);

                if inference_parameter!=None:
                    log_likelihood += self._compute_log_likelihood(production_list, E_log_stick_weights, E_log_theta);
                    
                    '''
                    for sampled_production in production_list:
                        if isinstance(sampled_production, util.AdaptedProduction):
                            nu_index = self._active_adapted_production_to_nu_index_of_lhs[sampled_production.lhs()][sampled_production];
                            log_likelihood += E_log_stick_weights[sampled_production.lhs()][0, nu_index];
                        else:
                            gamma_index = self._pcfg_production_to_gamma_index_of_lhs[sampled_production.lhs()][sampled_production];
                            log_likelihood += E_log_theta[sampled_production.lhs()][0, gamma_index];                                    
                    '''
                    
                    retrieved_tokens = retrieve_tokens_by_pre_order_traversal_of_adapted_non_terminal(production_list, retrieve_tokens_at_adapted_non_terminal);
                    # Warning: if you are using nltk 2.x, please use inc()
                    #retrieved_tokens_lists.inc(" ".join(retrieved_tokens), 1);
                    retrieved_tokens_lists[" ".join(retrieved_tokens)] += 1;
            
            if inference_parameter!=None:
                assert(retrieved_tokens_lists.N()==number_of_samples);
                
                reference_string = reference_strings[string_index];
    
                #maximum_tokens = retrieved_tokens_lists.max();
                #output_maximum_truth_file.write("%s\n" % reference_string);
                #output_maximum_test_file.write("%s\n" % maximum_tokens);
                
                for average_tokens in retrieved_tokens_lists.samples():
                    for x in xrange(retrieved_tokens_lists[average_tokens]):
                        output_average_truth_file.write("%s\n" % reference_string);
                        output_average_test_file.write("%s\n" % average_tokens);

        if inference_parameter==None:
            return pcfg_sufficient_statistics, adapted_sufficient_statistics;
        else:
            log_likelihood -= numpy.log(number_of_samples);
            print "Held-out likelihood of test data is %g..." % log_likelihood;
            
    def _compute_log_likelihood(self, production_list, E_log_stick_weights, E_log_theta):
        log_likelihood = 0;
        for sampled_production in production_list:
            if isinstance(sampled_production, util.AdaptedProduction):
                if sampled_production in self._active_adapted_production_to_nu_index_of_lhs[sampled_production.lhs()]:
                    nu_index = self._active_adapted_production_to_nu_index_of_lhs[sampled_production.lhs()][sampled_production];
                    log_likelihood += E_log_stick_weights[sampled_production.lhs()][0, nu_index];
                else:
                    log_likelihood += self._compute_log_likelihood(sampled_production.get_production_list(), E_log_stick_weights, E_log_theta);
            else:
                gamma_index = self._pcfg_production_to_gamma_index_of_lhs[sampled_production.lhs()][sampled_production];
                log_likelihood += E_log_theta[sampled_production.lhs()][0, gamma_index];
        return log_likelihood;
                
    def _sample_tree(self, current_hyper_node, input_string, pcfg_sufficient_statistics=None, adapted_sufficient_statistics=None):
        assert (pcfg_sufficient_statistics==None and adapted_sufficient_statistics==None) or (pcfg_sufficient_statistics!=None and adapted_sufficient_statistics!=None)
        
        sampled_production, unsampled_hyper_nodes, log_probability_of_sampled_production = current_hyper_node.random_sample_derivation();
        if isinstance(sampled_production, util.AdaptedProduction):
            # if sampled production is an adapted production
            assert (unsampled_hyper_nodes==None or len(unsampled_hyper_nodes)==0), "incomplete adapted production: %s" % sampled_production
            nu_index = self._active_adapted_production_to_nu_index_of_lhs[current_hyper_node._node][sampled_production];
            if adapted_sufficient_statistics!=None:
                adapted_sufficient_statistics[current_hyper_node._node][0, nu_index] += 1;
            
            return [sampled_production]
        elif isinstance(sampled_production, nltk.grammar.Production):
            # if sampled production is an pcfg production
            gamma_index = self._pcfg_production_to_gamma_index_of_lhs[current_hyper_node._node][sampled_production];
            if pcfg_sufficient_statistics!=None:
                pcfg_sufficient_statistics[current_hyper_node._node][0, gamma_index] += 1;

            # if sampled production is a pre-terminal pcfg production
            if unsampled_hyper_nodes==None or len(unsampled_hyper_nodes)==0:
                assert (not self.is_adapted_non_terminal(current_hyper_node._node)), "adapted pre-terminal found: %s" % current_hyper_node._node
                return [sampled_production]

            # if sampled production is a regular pcfg production
            production_list = [sampled_production];
            for unsampled_hyper_node in unsampled_hyper_nodes:
                production_list += self._sample_tree(unsampled_hyper_node, input_string, pcfg_sufficient_statistics, adapted_sufficient_statistics);

            # if current node is a non-adapted non-terminal node
            if not self.is_adapted_non_terminal(current_hyper_node._node):
                return production_list

            '''
            # if current hyper-node is an adapted non-terminal, and sampled production is a pcfg production
            for candidate_production in production_list:
                if isinstance(candidate_production, util.AdaptedProduction):
                    nu_index = self._active_adapted_production_to_nu_index_of_lhs[candidate_production.lhs()][candidate_production];
                    adapted_sufficient_statistics[candidate_production.lhs()][0, nu_index] -= 1;
                elif isinstance(candidate_production, nltk.grammar.Production):
                    gamma_index = self._pcfg_production_to_gamma_index_of_lhs[candidate_production.lhs()][candidate_production];
                    pcfg_sufficient_statistics[candidate_production.lhs()][0, gamma_index] -= 1;
                else:
                    print "Error in recognizing the production @ checkpoint 1..."
            '''
            
            new_adapted_production = util.AdaptedProduction(current_hyper_node._node, input_string[current_hyper_node._span[0]:current_hyper_node._span[1]], production_list);
            
            if pcfg_sufficient_statistics==None and adapted_sufficient_statistics==None:
                return [new_adapted_production]
            
            if new_adapted_production not in self.get_adapted_productions(current_hyper_node._node, tuple(input_string[current_hyper_node._span[0]:current_hyper_node._span[1]])):
                # if this is an inactive adapted production
                adapted_production_count = 0;
                pcfg_production_count = 0;
                for candidate_production in production_list:
                    if isinstance(candidate_production, util.AdaptedProduction):
                        adapted_production_count += 1;
                        #print candidate_production.rhs(), len(candidate_production.rhs())
                        if len(candidate_production.rhs())==1:
                            print "skip singleton adapted production:", candidate_production 
                            continue;
                        nu_index = self._active_adapted_production_to_nu_index_of_lhs[candidate_production.lhs()][candidate_production];
                        # Warning: if you are using nltk 2.x, please use inc()
                        #self._adapted_production_usage_freqdist.inc(candidate_production, 1);
                        self._adapted_production_usage_freqdist[candidate_production] += 1;
                        self._adapted_production_dependents_of_adapted_production[candidate_production].add(new_adapted_production);
                        self._active_adapted_production_usage_counts_of_lhs[candidate_production.lhs()][0, nu_index] += 1;
                    elif isinstance(candidate_production, nltk.grammar.Production):
                        pcfg_production_count += 1;
                        gamma_index = self._pcfg_production_to_gamma_index_of_lhs[candidate_production.lhs()][candidate_production];
                        self._pcfg_production_usage_counts_of_lhs[candidate_production.lhs()][0, gamma_index] += 1;
                    else:
                        sys.stderr.write("Error in recognizing the production @ checkpoint 1...\n");
                        sys.exit();
                    
                # activate this adapted rule
                self._lhs_rhs_to_active_adapted_production[(current_hyper_node._node, tuple(input_string[current_hyper_node._span[0]:current_hyper_node._span[1]]))].add(new_adapted_production);
                self._lhs_to_active_adapted_production[current_hyper_node._node].add(new_adapted_production);
                self._rhs_to_active_adapted_production[tuple(input_string[current_hyper_node._span[0]:current_hyper_node._span[1]])].add(new_adapted_production);
                
                self._nu_index_to_active_adapted_production_of_lhs[current_hyper_node._node][len(self._nu_index_to_active_adapted_production_of_lhs[current_hyper_node._node])] = new_adapted_production;
                self._active_adapted_production_to_nu_index_of_lhs[current_hyper_node._node][new_adapted_production] = len(self._active_adapted_production_to_nu_index_of_lhs[current_hyper_node._node]);

                self._nu_1[current_hyper_node._node] = numpy.hstack((self._nu_1[current_hyper_node._node], numpy.ones((1, 1))));
                self._nu_2[current_hyper_node._node] = numpy.hstack((self._nu_2[current_hyper_node._node], numpy.ones((1, 1))*self._alpha_pi[current_hyper_node._node]));
                
                nu_index = self._active_adapted_production_to_nu_index_of_lhs[current_hyper_node._node][new_adapted_production];
                adapted_sufficient_statistics[current_hyper_node._node] = numpy.hstack((adapted_sufficient_statistics[current_hyper_node._node], numpy.zeros((1, 1))));
                
                self._active_adapted_production_usage_counts_of_lhs[current_hyper_node._node] = numpy.hstack((self._active_adapted_production_usage_counts_of_lhs[current_hyper_node._node], numpy.zeros((1, 1))));
                self._active_adapted_production_usage_counts_of_lhs[current_hyper_node._node][0, nu_index] = self._adapted_production_usage_freqdist[new_adapted_production];

                self._active_adapted_production_length_of_lhs[current_hyper_node._node] = numpy.hstack((self._active_adapted_production_length_of_lhs[current_hyper_node._node], numpy.zeros((1, 1))));
                #self._active_adapted_production_length_of_lhs[current_hyper_node._node][0, nu_index] = len(new_adapted_production.rhs());
                if current_hyper_node._node==self._ordered_adaptor_top_down[-1]:
                    self._active_adapted_production_length_of_lhs[current_hyper_node._node][0, nu_index] = len(new_adapted_production.rhs());
                else:
                    self._active_adapted_production_length_of_lhs[current_hyper_node._node][0, nu_index] = adapted_production_count;

                self._active_adapted_production_sufficient_statistics_of_lhs[current_hyper_node._node] = numpy.hstack((self._active_adapted_production_sufficient_statistics_of_lhs[current_hyper_node._node], numpy.zeros((1, 1))));
                if nu_index>self._truncation_level[current_hyper_node._node]:
                    self._active_adapted_production_sufficient_statistics_of_lhs[current_hyper_node._node][0, nu_index] = self._active_adapted_production_sufficient_statistics_of_lhs[current_hyper_node._node][0, nu_index-1];
            else:
                nu_index = self._active_adapted_production_to_nu_index_of_lhs[current_hyper_node._node][new_adapted_production];
                
            adapted_sufficient_statistics[current_hyper_node._node][0, nu_index] += 1;
            return [new_adapted_production]
        else:
            sys.stderr.write("Error in recognizing the production class %s @ checkpoint 2...\n" % sampled_production.__class__);
            sys.exit();


    '''
    def _sample_parse_tree(self, current_hyper_node, input_string, inference_mode=False):
        sampled_production, unsampled_hyper_nodes, log_probability_of_sampled_production = current_hyper_node.random_sample_derivation();
        
        production_list_log_probability = log_probability_of_sampled_production;
        production_list = [sampled_production];
        for unsampled_hyper_node in unsampled_hyper_nodes:
            production_sublist, log_probability_of_sampled_subtrees = self._sample_parse_tree(unsampled_hyper_node, input_string);
            production_list += production_sublist;
            production_list_log_probability += log_probability_of_sampled_subtrees;
        
        if self.is_adapted_non_terminal(current_hyper_node._node):
            new_adapted_production = util.AdaptedProduction(current_hyper_node._node, input_string[current_hyper_node._span[0]:current_hyper_node._span[1]], production_list);
            return [new_adapted_production], production_list_log_probability;
        else:
            return production_list, production_list_log_probability;

    def _tree_log_likelihood(self, production_list, E_log_stick_weights, E_log_theta):
        log_likelihood = 0;
        for candidate_production in production_list:
            if isinstance(candidate_production, nltk.grammar.Production):
                lhs = candidate_production.lhs();
                gamma_index = self._pcfg_production_to_gamma_index_of_lhs[lhs][candidate_production];
                log_likelihood += E_log_theta[lhs][0, gamma_index];
            elif isinstance(candidate_production, nltk.grammar.Production):
                print ""
            else:
                sys.stderr.write("Error in recognizing the production...\n");
                sys.exit();
    '''

    def m_step(self):
        for non_terminal in self._pcfg_sufficient_statistics_of_lhs:
            temp_sufficient_statistics = self._pcfg_sufficient_statistics_of_lhs[non_terminal] * self._ranking_statistics_scale;
            temp_sufficient_statistics += self._pcfg_production_usage_counts_of_lhs[non_terminal];

            self._gamma[non_terminal] = self._alpha_theta[non_terminal] + temp_sufficient_statistics;
            
        for adapted_non_terminal in self._active_adapted_production_sufficient_statistics_of_lhs:
            temp_sufficient_statistics = self._active_adapted_production_sufficient_statistics_of_lhs[adapted_non_terminal] * self._ranking_statistics_scale;
            temp_sufficient_statistics += self._active_adapted_production_usage_counts_of_lhs[adapted_non_terminal];

            assert numpy.all(self._active_adapted_production_usage_counts_of_lhs[adapted_non_terminal]>=0), adapted_non_terminal
            assert numpy.all(self._active_adapted_production_sufficient_statistics_of_lhs[adapted_non_terminal]>=0), adapted_non_terminal
            assert numpy.all(self._active_adapted_production_length_of_lhs[adapted_non_terminal]>=0), adapted_non_terminal
            
            reversed_cumulated_temp_sufficient_statistics = reverse_cumulative_sum_matrix_over_axis(temp_sufficient_statistics, 1);
            cumulated_counts = numpy.r_[0:len(self._active_adapted_production_to_nu_index_of_lhs[adapted_non_terminal])][numpy.newaxis, :]+1;
            
            self._nu_1[adapted_non_terminal] = temp_sufficient_statistics + self._beta_pi[adapted_non_terminal] + 1;
            self._nu_2[adapted_non_terminal] = reversed_cumulated_temp_sufficient_statistics + self._alpha_pi[adapted_non_terminal] + cumulated_counts * self._beta_pi[adapted_non_terminal];

            assert(numpy.all(self._nu_1[adapted_non_terminal]>0))
            assert(numpy.all(self._nu_2[adapted_non_terminal]>0))
            
        return;

    def stochastic_m_step(self, pcfg_sufficient_statistics, adapted_sufficient_statistics):
        for non_terminal in pcfg_sufficient_statistics:#self._pcfg_sufficient_statistics_of_lhs:
            #temp_sufficient_statistics = self._pcfg_sufficient_statistics_of_lhs[non_terminal];
            #self._gamma[non_terminal] = self._alpha_theta[non_terminal] + temp_sufficient_statistics;
            
            temp_sufficient_statistics = pcfg_sufficient_statistics[non_terminal] / (self._number_of_samples * self._batch_size) * self._number_of_strings;
            self._gamma[non_terminal] = (1-self._epsilon)*self._gamma[non_terminal] + self._epsilon*(self._alpha_theta[non_terminal] + temp_sufficient_statistics);
            
        for adapted_non_terminal in adapted_sufficient_statistics:#self._active_adapted_production_sufficient_statistics_of_lhs:
            #temp_sufficient_statistics = self._active_adapted_production_usage_counts_of_lhs[adapted_non_terminal] + self._active_adapted_production_sufficient_statistics_of_lhs[adapted_non_terminal];
            #temp_sufficient_statistics = self._active_adapted_production_sufficient_statistics_of_lhs[adapted_non_terminal];
            
            #reversed_cumulated_temp_sufficient_statistics = reverse_cumulative_sum_matrix_over_axis(temp_sufficient_statistics, 1);
            #cumulated_counts = numpy.r_[0:len(self._active_adapted_production_to_nu_index_of_lhs[adapted_non_terminal])][numpy.newaxis, :]+1;
            
            #self._nu_1[adapted_non_terminal] = temp_sufficient_statistics + self._beta_pi + 1;
            #self._nu_2[adapted_non_terminal] = reversed_cumulated_temp_sufficient_statistics + self._alpha_pi + cumulated_counts * self._beta_pi;

            temp_sufficient_statistics = adapted_sufficient_statistics[adapted_non_terminal] / (self._number_of_samples * self._batch_size) * self._number_of_strings;
            reversed_cumulated_temp_sufficient_statistics = reverse_cumulative_sum_matrix_over_axis(temp_sufficient_statistics, 1);
            cumulated_counts = numpy.r_[0:len(self._active_adapted_production_to_nu_index_of_lhs[adapted_non_terminal])][numpy.newaxis, :]+1;
            
            self._nu_1[adapted_non_terminal] = (1-self._epsilon)*self._nu_1[adapted_non_terminal] + self._epsilon*(temp_sufficient_statistics + self._beta_pi[adapted_non_terminal] + 1);
            self._nu_2[adapted_non_terminal] = (1-self._epsilon)*self._nu_2[adapted_non_terminal] + self._epsilon*(reversed_cumulated_temp_sufficient_statistics + self._alpha_pi[adapted_non_terminal] + cumulated_counts * self._beta_pi[adapted_non_terminal]);
            
        return;

    def _accumulate_sufficient_statistics(self, pcfg_sufficient_statistics, adapted_sufficient_statistics, number_of_samples, batch_size):
        sufficient_statistics_scale = self._number_of_strings / (number_of_samples * batch_size);
        #sufficient_statistics_scale = 1. / number_of_samples;
        
        for non_terminal in self._non_terminals:
            pcfg_sufficient_statistics[non_terminal] *= sufficient_statistics_scale;
        
        for adapted_non_terminal in self._adapted_non_terminals:
            adapted_sufficient_statistics[adapted_non_terminal] *= sufficient_statistics_scale;
            
        '''
        for adapted_non_terminal in self._adapted_non_terminals:
            for adapted_production in self._adapted_production_sufficient_statistics_of_lhs[adapted_non_terminal]:
                downgrading_ranking_sufficient_statistics = -self._epsilon*self._adapted_production_sufficient_statistics_of_lhs[adapted_non_terminal][adapted_production];
                if adapted_non_terminal in adapted_sufficient_statistics:
                    nu_index = self._active_adapted_production_to_nu_index_of_lhs[adapted_non_terminal][adapted_production];
                    downgrading_ranking_sufficient_statistics += self._epsilon*adapted_sufficient_statistics[adapted_non_terminal][0, nu_index];
                self._adapted_production_sufficient_statistics_of_lhs[adapted_non_terminal].inc(adapted_production, downgrading_ranking_sufficient_statistics*self._ranking_statistics_scale);
        '''
    
        for non_terminal in self._non_terminals:
            downgrading_pcfg_sufficient_statistics = -self._epsilon*self._pcfg_sufficient_statistics_of_lhs[non_terminal];
            if non_terminal in pcfg_sufficient_statistics:
                downgrading_pcfg_sufficient_statistics += self._epsilon*pcfg_sufficient_statistics[non_terminal];
            self._pcfg_sufficient_statistics_of_lhs[non_terminal] += downgrading_pcfg_sufficient_statistics;

        for adapted_non_terminal in self._adapted_non_terminals:
            downgrading_ranking_sufficient_statistics = -self._epsilon*self._active_adapted_production_sufficient_statistics_of_lhs[adapted_non_terminal];
            
            if adapted_non_terminal in adapted_sufficient_statistics:
                downgrading_ranking_sufficient_statistics += self._epsilon*adapted_sufficient_statistics[adapted_non_terminal];
            self._active_adapted_production_sufficient_statistics_of_lhs[adapted_non_terminal] += downgrading_ranking_sufficient_statistics;
        
        return;
    
    """
    @deprecated: 
    """
    '''
    def reorder_adapted_productions_by_tree(self):
        for adapted_non_terminal in self._adapted_production_sufficient_statistics_of_lhs:
            new_nu_index_to_adapted_production = {};
            new_adapted_production_to_nu_index = {};
            
            new_nu_1 = numpy.zeros((1, len(self._nu_index_to_active_adapted_production_of_lhs[adapted_non_terminal])));
            new_nu_2 = numpy.zeros((1, len(self._nu_index_to_active_adapted_production_of_lhs[adapted_non_terminal])));
            new_active_adapted_production_usage_counts_of_lhs = numpy.zeros((1, len(self._nu_index_to_active_adapted_production_of_lhs[adapted_non_terminal])));
            new_active_adapted_production_sufficient_statistics_of_lhs = numpy.zeros((1, len(self._nu_index_to_active_adapted_production_of_lhs[adapted_non_terminal])));
            
            for adapted_production in self._adapted_production_sufficient_statistics_of_lhs[adapted_non_terminal]:
                if adapted_production not in self._active_adapted_production_to_nu_index_of_lhs[adapted_non_terminal]:
                    continue;
                
                new_nu_index_to_adapted_production[len(new_nu_index_to_adapted_production)] = adapted_production;
                new_adapted_production_to_nu_index[adapted_production] = len(new_adapted_production_to_nu_index);
                
                old_nu_index = self._active_adapted_production_to_nu_index_of_lhs[adapted_non_terminal][adapted_production];
                new_nu_index = new_adapted_production_to_nu_index[adapted_production];
                
                new_nu_1[0, new_nu_index] = self._nu_1[adapted_non_terminal][0, old_nu_index];
                new_nu_2[0, new_nu_index] = self._nu_2[adapted_non_terminal][0, old_nu_index];

                new_active_adapted_production_usage_counts_of_lhs[0, new_nu_index] = self._active_adapted_production_usage_counts_of_lhs[adapted_non_terminal][0, old_nu_index];
                new_active_adapted_production_sufficient_statistics_of_lhs[0, new_nu_index] = self._active_adapted_production_sufficient_statistics_of_lhs[adapted_non_terminal][0, old_nu_index];

            self._nu_index_to_active_adapted_production_of_lhs[adapted_non_terminal] = new_nu_index_to_adapted_production;
            self._active_adapted_production_to_nu_index_of_lhs[adapted_non_terminal] = new_adapted_production_to_nu_index;

            self._nu_1[adapted_non_terminal] = new_nu_1;
            self._nu_2[adapted_non_terminal] = new_nu_2;
            
            self._active_adapted_production_usage_counts_of_lhs[adapted_non_terminal] = new_active_adapted_production_usage_counts_of_lhs;
            self._active_adapted_production_sufficient_statistics_of_lhs[adapted_non_terminal] = new_active_adapted_production_sufficient_statistics_of_lhs;
            
        return;
    '''
    
    """
    @deprecated:
    """
    '''
    def reorder_adapted_productions_by_string(self):
        for adapted_non_terminal in self._adapted_production_sufficient_statistics_of_lhs:
            new_nu_index_to_adapted_production = {};
            new_adapted_production_to_nu_index = {};
            
            new_nu_1 = numpy.random.random((1, len(self._nu_index_to_active_adapted_production_of_lhs[adapted_non_terminal])));
            new_nu_2 = numpy.random.random((1, len(self._nu_index_to_active_adapted_production_of_lhs[adapted_non_terminal])));
            new_active_adapted_production_usage_counts_of_lhs = numpy.zeros((1, len(self._nu_index_to_active_adapted_production_of_lhs[adapted_non_terminal])));
            new_active_adapted_production_sufficient_statistics_of_lhs = numpy.zeros((1, len(self._nu_index_to_active_adapted_production_of_lhs[adapted_non_terminal])));
            
            for adapted_production in self._adapted_production_sufficient_statistics_of_lhs[adapted_non_terminal]:
                if adapted_production not in self._active_adapted_production_to_nu_index_of_lhs[adapted_non_terminal]:
                    continue;
                
                new_nu_index_to_adapted_production[len(new_nu_index_to_adapted_production)] = adapted_production;
                new_adapted_production_to_nu_index[adapted_production] = len(new_adapted_production_to_nu_index);
                
                old_nu_index = self._active_adapted_production_to_nu_index_of_lhs[adapted_non_terminal][adapted_production];
                new_nu_index = new_adapted_production_to_nu_index[adapted_production];
                
                new_nu_1[0, new_nu_index] = self._nu_1[adapted_non_terminal][0, old_nu_index];
                new_nu_2[0, new_nu_index] = self._nu_2[adapted_non_terminal][0, old_nu_index];
                new_active_adapted_production_sufficient_statistics_of_lhs[0, new_nu_index] = self._active_adapted_production_sufficient_statistics_of_lhs[adapted_non_terminal][0, old_nu_index];
                new_active_adapted_production_usage_counts_of_lhs[0, new_nu_index] = self._active_adapted_production_usage_counts_of_lhs[adapted_non_terminal][0, old_nu_index];
        
            self._nu_index_to_active_adapted_production_of_lhs[adapted_non_terminal] = new_nu_index_to_adapted_production;
            self._active_adapted_production_to_nu_index_of_lhs[adapted_non_terminal] = new_adapted_production_to_nu_index;

            self._nu_1[adapted_non_terminal] = new_nu_1;
            self._nu_2[adapted_non_terminal] = new_nu_2;
            self._active_adapted_production_sufficient_statistics_of_lhs[adapted_non_terminal] = new_active_adapted_production_sufficient_statistics_of_lhs;
            self._active_adapted_production_usage_counts_of_lhs[adapted_non_terminal] = new_active_adapted_production_usage_counts_of_lhs;
            
        return 
    '''
    
    def model_state_assertion(self):
        for non_terminal in self._non_terminals:
            assert(numpy.all(self._pcfg_sufficient_statistics_of_lhs[non_terminal]>=0));
            assert(numpy.all(self._pcfg_production_usage_counts_of_lhs[non_terminal]>=0));
            assert(self._pcfg_sufficient_statistics_of_lhs[non_terminal].shape==self._pcfg_production_usage_counts_of_lhs[non_terminal].shape);
            assert(self._pcfg_sufficient_statistics_of_lhs[non_terminal].shape==(1, len(self._pcfg_production_to_gamma_index_of_lhs[non_terminal])));
            assert(len(self._gamma_index_to_pcfg_production_of_lhs[non_terminal])==len(self._pcfg_production_to_gamma_index_of_lhs[non_terminal]));
            assert(self._gamma[non_terminal].shape==(1, len(self._pcfg_production_to_gamma_index_of_lhs[non_terminal])));
        
        #==================================================================
        adapted_production_freqdist = nltk.probability.FreqDist();
        pcfg_production_freqdist = nltk.probability.FreqDist();
        adapted_production_dependent = collections.defaultdict(set);        
        for adapted_non_terminal in self._ordered_adaptor_top_down:
            for active_adapted_production in self._active_adapted_production_to_nu_index_of_lhs[adapted_non_terminal]:
                for active_production in active_adapted_production._productions:
                    if isinstance(active_production, util.AdaptedProduction):
                        adapted_production_dependent[active_production].add(active_adapted_production);
                        # Warning: if you are using nltk 2.x, please use inc()
                        #adapted_production_freqdist.inc(active_production, 1);
                        adapted_production_freqdist[active_production] += 1;
                    elif isinstance(active_production, nltk.grammar.Production):
                        # Warning: if you are using nltk 2.x, please use inc()
                        #pcfg_production_freqdist.inc(active_production, 1);
                        pcfg_production_freqdist[active_production] += 1;
                    else:
                        sys.stderr.write("Error: %s\n" % active_production.__class__);
            
        for adapted_non_terminal in self._ordered_adaptor_top_down:
            for adapted_production in self._active_adapted_production_to_nu_index_of_lhs[adapted_non_terminal]:
                nu_index = self._active_adapted_production_to_nu_index_of_lhs[adapted_non_terminal][adapted_production];

                #print adapted_non_terminal, adapted_production
                #print self._active_adapted_production_usage_counts_of_lhs[adapted_non_terminal][0, nu_index], adapted_production_freqdist[adapted_production]
                
                assert (self._adapted_production_dependents_of_adapted_production[adapted_production]==adapted_production_dependent[adapted_production]), \
                        ("<adapted production>: %s\n<test>: %d\t%s\n<truth>: %d\t%s" % (
                         adapted_production, \
                         len(self._adapted_production_dependents_of_adapted_production[adapted_production]), \
                         self._adapted_production_dependents_of_adapted_production[adapted_production], \
                         len(adapted_production_dependent[adapted_production]), \
                         adapted_production_dependent[adapted_production]))
                assert (self._active_adapted_production_usage_counts_of_lhs[adapted_non_terminal][0, nu_index]==adapted_production_freqdist[adapted_production]), \
                        (adapted_non_terminal, adapted_production, \
                         len(self._active_adapted_production_usage_counts_of_lhs[adapted_non_terminal]), \
                         self._active_adapted_production_usage_counts_of_lhs[adapted_non_terminal][0, nu_index], \
                         self._adapted_production_usage_freqdist[adapted_production], \
                         adapted_production_freqdist[adapted_production])

                assert (self._active_adapted_production_usage_counts_of_lhs[adapted_non_terminal][0, nu_index]==adapted_production_freqdist[adapted_production]);
                assert (self._adapted_production_usage_freqdist[adapted_production]==adapted_production_freqdist[adapted_production]);
        for non_terminal in self._non_terminals:
            for pcfg_production in self._pcfg_production_to_gamma_index_of_lhs[non_terminal]:
                gamma_index = self._pcfg_production_to_gamma_index_of_lhs[non_terminal][pcfg_production];
                assert (self._pcfg_production_usage_counts_of_lhs[non_terminal][0, gamma_index]==pcfg_production_freqdist[pcfg_production]), (self._pcfg_production_usage_counts_of_lhs[non_terminal][0, gamma_index], pcfg_production_freqdist[pcfg_production]);
        #==================================================================
        
        for adapted_non_terminal in self._adapted_non_terminals:
            assert(numpy.all(self._active_adapted_production_usage_counts_of_lhs[adapted_non_terminal]>=0));
            assert(numpy.all(self._active_adapted_production_sufficient_statistics_of_lhs[adapted_non_terminal]>=0));
            
            #assert(len(self._adapted_production_usage_freqdist[adapted_non_terminal])==len(self._adapted_production_sufficient_statistics_of_lhs[adapted_non_terminal]));
            #if adapted_non_terminal in self._adapted_production_sufficient_statistics_of_lhs:
                #assert(self._adapted_production_sufficient_statistics_of_lhs[adapted_non_terminal].values()[-1]>=0);
            #if adapted_non_terminal in self._adapted_production_usage_freqdist:
                #assert(self._adapted_production_usage_freqdist[adapted_non_terminal].values()[-1]>=0);
            
            assert(self._active_adapted_production_sufficient_statistics_of_lhs[adapted_non_terminal].shape==self._nu_1[adapted_non_terminal].shape)
            assert(self._active_adapted_production_sufficient_statistics_of_lhs[adapted_non_terminal].shape==self._nu_2[adapted_non_terminal].shape)
            assert(self._active_adapted_production_sufficient_statistics_of_lhs[adapted_non_terminal].shape==self._active_adapted_production_usage_counts_of_lhs[adapted_non_terminal].shape)

            '''
            for nu_index in self._nu_index_to_active_adapted_production_of_lhs[adapted_non_terminal]:
                adapted_production = self._nu_index_to_active_adapted_production_of_lhs[adapted_non_terminal][nu_index];
                assert(abs(self._ranking_statistics_scale*self._active_adapted_production_sufficient_statistics_of_lhs[adapted_non_terminal][0, nu_index]-self._adapted_production_sufficient_statistics_of_lhs[adapted_non_terminal][adapted_production]) < 1e-6);
            '''
                
            for nu_index in self._nu_index_to_active_adapted_production_of_lhs[adapted_non_terminal]:
                adapted_production = self._nu_index_to_active_adapted_production_of_lhs[adapted_non_terminal][nu_index];
                assert(self._active_adapted_production_usage_counts_of_lhs[adapted_non_terminal][0, nu_index]==self._adapted_production_usage_freqdist[adapted_production]);
                #assert (self._active_adapted_production_usage_counts_of_lhs[adapted_non_terminal][0, nu_index]==len(self._adapted_production_dependents_of_adapted_production[adapted_production])), (len(self._adapted_production_dependents_of_adapted_production[adapted_production]), self._active_adapted_production_usage_counts_of_lhs[adapted_non_terminal][0, nu_index]);

        for adapted_non_terminal in self._adapted_non_terminals:
            candidate_adapted_productions = self.get_adapted_productions(lhs=adapted_non_terminal);
            assert(len(candidate_adapted_productions)==len(self._active_adapted_production_to_nu_index_of_lhs[adapted_non_terminal]));
            assert(len(candidate_adapted_productions)==len(self._nu_index_to_active_adapted_production_of_lhs[adapted_non_terminal]));

            for candidate_adapted_production in candidate_adapted_productions - set(self._active_adapted_production_to_nu_index_of_lhs[adapted_non_terminal]):
                print "NOT in indexed adapted production:", candidate_adapted_production
            for candidate_adapted_production in set(self._active_adapted_production_to_nu_index_of_lhs[adapted_non_terminal]) - candidate_adapted_productions:
                print "NOT in hashed adapted production:", candidate_adapted_production
            for candidate_adapted_productions in candidate_adapted_productions ^ set(self._active_adapted_production_to_nu_index_of_lhs[adapted_non_terminal]):
                print "NOT in either:", candidate_adapted_productions
                
            assert (candidate_adapted_productions==set(self._active_adapted_production_to_nu_index_of_lhs[adapted_non_terminal])), "%d\t%d\t%d" % (len(candidate_adapted_productions), len(self._active_adapted_production_to_nu_index_of_lhs[adapted_non_terminal]), len(set(self._active_adapted_production_to_nu_index_of_lhs[adapted_non_terminal])));
            
            for candidate_adapted_production in self._active_adapted_production_to_nu_index_of_lhs[adapted_non_terminal]:
                assert(candidate_adapted_production in candidate_adapted_productions)

            for candidate_adapted_production in candidate_adapted_productions:
                assert(candidate_adapted_production in self._active_adapted_production_to_nu_index_of_lhs[adapted_non_terminal]);

    def prune_adapted_productions(self, reorder_only=False):
        #for adapted_non_terminal in self._ordered_active_adaptor_down_top[::-1]:
        for adapted_non_terminal in self._ordered_adaptor_top_down:
            aggregate_sufficient_statistics_and_usage_counts = self._active_adapted_production_sufficient_statistics_of_lhs[adapted_non_terminal]
            #aggregate_sufficient_statistics_and_usage_counts += self._active_adapted_production_usage_counts_of_lhs[adapted_non_terminal]

            aggregate_sufficient_statistics_and_usage_counts *= numpy.log(self._epsilon * self._active_adapted_production_length_of_lhs[adapted_non_terminal] + 1);
            #aggregate_sufficient_statistics_and_usage_counts += 0.2 * self._number_of_strings * numpy.log(self._active_adapted_production_length_of_lhs[adapted_non_terminal]+1);
            #aggregate_sufficient_statistics_and_usage_counts += self._epsilon * self._number_of_strings * numpy.log(self._active_adapted_production_length_of_lhs[adapted_non_terminal]+1);
            #aggregate_sufficient_statistics_and_usage_counts += 0.2 * self._number_of_strings * numpy.log(self._active_adapted_production_length_of_lhs[adapted_non_terminal]+1);
            #aggregate_sufficient_statistics_and_usage_counts += 2*numpy.log(self._active_adapted_production_length_of_lhs[adapted_non_terminal]+1);
            #aggregate_sufficient_statistics_and_usage_counts *= numpy.log(self._active_adapted_production_length_of_lhs[adapted_non_terminal]/10+1)
            #aggregate_sufficient_statistics_and_usage_counts *= numpy.log(self._active_adapted_production_length_of_lhs[adapted_non_terminal]/100+1)
            #aggregate_sufficient_statistics_and_usage_counts *= numpy.log(1. / self._counter * self._active_adapted_production_length_of_lhs[adapted_non_terminal] + 1)
            #if adapted_non_terminal == self._ordered_adaptor_top_down[-1]:
                #aggregate_sufficient_statistics_and_usage_counts *= numpy.log(self._epsilon * self._active_adapted_production_length_of_lhs[adapted_non_terminal] + 1)
                #aggregate_sufficient_statistics_and_usage_counts *= numpy.log(1. / self._counter * self._active_adapted_production_length_of_lhs[adapted_non_terminal] + 1)            

            #if adapted_non_terminal != self._ordered_active_adaptor_down_top[-1]:
            if adapted_non_terminal != self._ordered_adaptor_top_down[0]:
                aggregate_sufficient_statistics_and_usage_counts *= self._active_adapted_production_usage_counts_of_lhs[adapted_non_terminal]>0
            
            nu_index_in_descending_order = numpy.argsort(aggregate_sufficient_statistics_and_usage_counts[0, :])[::-1];
            non_zeros = numpy.sum(aggregate_sufficient_statistics_and_usage_counts>0);
            
            if reorder_only:
                desired_truncation_level = len(nu_index_in_descending_order);
            else:
                #desired_truncation_level = non_zeros;
                #'''
                #if adapted_non_terminal == self._ordered_active_adaptor_down_top[-1]:
                if adapted_non_terminal == self._ordered_adaptor_top_down[0]:
                    desired_truncation_level = min(self._truncation_level[adapted_non_terminal], non_zeros);
                else:
                    desired_truncation_level = non_zeros;
                #'''
                #desired_truncation_level = min(self._truncation_level[adapted_non_terminal], len(nu_index_in_descending_order), non_zeros);
                #desired_truncation_level = min(max(self._truncation_level[adapted_non_terminal], len(nu_index_in_descending_order)), non_zeros);
            print "prune %s: truncation_level %d, actual truncation_level %d, non_zero_truncation_level %d" %(adapted_non_terminal, self._truncation_level[adapted_non_terminal], len(nu_index_in_descending_order), non_zeros);

            new_nu_index_to_active_adapted_production_of_lhs = {};
            new_active_adapted_production_to_nu_index_of_lhs = {};
            
            new_nu_1 = numpy.random.random((1, desired_truncation_level));
            new_nu_2 = numpy.random.random((1, desired_truncation_level)) * self._alpha_pi[adapted_non_terminal];
            new_active_adapted_production_usage_counts_of_lhs = numpy.zeros((1, desired_truncation_level));
            new_active_adapted_production_sufficient_statistics_of_lhs = numpy.zeros((1, desired_truncation_level));
            new_active_adapted_production_length_of_rhs = numpy.zeros((1, desired_truncation_level));

            #self.model_state_assertion();
 
            #for adapted_production in adapted_productions[0:desired_truncation_level]:
            for adapted_production_index in nu_index_in_descending_order[:desired_truncation_level]:
                adapted_production = self._nu_index_to_active_adapted_production_of_lhs[adapted_non_terminal][adapted_production_index];
                
                new_nu_index_to_active_adapted_production_of_lhs[len(new_nu_index_to_active_adapted_production_of_lhs)] = adapted_production;
                new_active_adapted_production_to_nu_index_of_lhs[adapted_production] = len(new_active_adapted_production_to_nu_index_of_lhs);
                
                old_nu_index = self._active_adapted_production_to_nu_index_of_lhs[adapted_non_terminal][adapted_production];
                new_nu_index = new_active_adapted_production_to_nu_index_of_lhs[adapted_production];
                
                new_nu_1[0, new_nu_index] = self._nu_1[adapted_non_terminal][0, old_nu_index];
                new_nu_2[0, new_nu_index] = self._nu_2[adapted_non_terminal][0, old_nu_index];
                
                new_active_adapted_production_sufficient_statistics_of_lhs[0, new_nu_index] = self._active_adapted_production_sufficient_statistics_of_lhs[adapted_non_terminal][0, old_nu_index];
                new_active_adapted_production_usage_counts_of_lhs[0, new_nu_index] = self._active_adapted_production_usage_counts_of_lhs[adapted_non_terminal][0, old_nu_index];
                new_active_adapted_production_length_of_rhs[0, new_nu_index] = self._active_adapted_production_length_of_lhs[adapted_non_terminal][0, old_nu_index];

            #old_active_adapted_production_to_nu_index_of_lhs = self._active_adapted_production_to_nu_index_of_lhs[adapted_non_terminal];
            old_nu_index_to_active_adapted_production_of_lhs = self._nu_index_to_active_adapted_production_of_lhs[adapted_non_terminal];
            
            self._nu_index_to_active_adapted_production_of_lhs[adapted_non_terminal] = new_nu_index_to_active_adapted_production_of_lhs;
            self._active_adapted_production_to_nu_index_of_lhs[adapted_non_terminal] = new_active_adapted_production_to_nu_index_of_lhs;
            
            #print adapted_non_terminal, self._nu_1[adapted_non_terminal].shape, self._nu_2[adapted_non_terminal].shape;
            
            self._nu_1[adapted_non_terminal] = new_nu_1;
            self._nu_2[adapted_non_terminal] = new_nu_2;
            self._active_adapted_production_sufficient_statistics_of_lhs[adapted_non_terminal] = new_active_adapted_production_sufficient_statistics_of_lhs;
            self._active_adapted_production_usage_counts_of_lhs[adapted_non_terminal] = new_active_adapted_production_usage_counts_of_lhs;
            self._active_adapted_production_length_of_lhs[adapted_non_terminal] = new_active_adapted_production_length_of_rhs;

            #print adapted_non_terminal, self._nu_1[adapted_non_terminal].shape, self._nu_2[adapted_non_terminal].shape;

            #assert numpy.all(self._active_adapted_production_usage_counts_of_lhs[adapted_non_terminal]>=0), adapted_non_terminal
            #assert numpy.all(self._active_adapted_production_sufficient_statistics_of_lhs[adapted_non_terminal]>=0), adapted_non_terminal
            #assert numpy.all(self._active_adapted_production_length_of_lhs[adapted_non_terminal]>=0), adapted_non_terminal

            #for adapted_production in adapted_productions[desired_truncation_level:len(adapted_productions)]:
            for adapted_production_index in nu_index_in_descending_order[desired_truncation_level:]:
                adapted_production = old_nu_index_to_active_adapted_production_of_lhs[adapted_production_index];
                for candidate_production in adapted_production._productions:
                    if isinstance(candidate_production, util.AdaptedProduction):
                        #print candidate_production.rhs(), len(candidate_production.rhs())
                        if len(candidate_production.rhs())==1:
                            print "skip singleton adapted production:", candidate_production 
                            continue;
                        # Warning: if you are using nltk 2.x, please use inc()
                        #self._adapted_production_usage_freqdist.inc(candidate_production, -1);
                        self._adapted_production_usage_freqdist[candidate_production] += -1;
                        self._adapted_production_dependents_of_adapted_production[candidate_production].discard(adapted_production);
                        if candidate_production in self._active_adapted_production_to_nu_index_of_lhs[candidate_production.lhs()]:
                            nu_index = self._active_adapted_production_to_nu_index_of_lhs[candidate_production.lhs()][candidate_production];
                            #assert numpy.all(self._active_adapted_production_usage_counts_of_lhs[candidate_production.lhs()])>=0, (adapted_non_terminal, candidate_production.lhs())
                            self._active_adapted_production_usage_counts_of_lhs[candidate_production.lhs()][0, nu_index] -= 1;
                            #assert numpy.all(self._active_adapted_production_usage_counts_of_lhs[candidate_production.lhs()])>=0, (adapted_non_terminal, candidate_production.lhs())
                    elif isinstance(candidate_production, nltk.grammar.Production):
                        gamma_index = self._pcfg_production_to_gamma_index_of_lhs[candidate_production.lhs()][candidate_production];
                        self._pcfg_production_usage_counts_of_lhs[candidate_production.lhs()][0, gamma_index] -= 1;
                    else:
                        print "Error in recognizing the candidate production."

                self._lhs_rhs_to_active_adapted_production[(adapted_non_terminal, adapted_production.rhs())].discard(adapted_production);
                self._lhs_to_active_adapted_production[adapted_non_terminal].discard(adapted_production);
                self._rhs_to_active_adapted_production[adapted_production.rhs()].discard(adapted_production);

            #self.model_state_assertion();
            #assert numpy.all(self._active_adapted_production_usage_counts_of_lhs[adapted_non_terminal]>=0), adapted_non_terminal
            #assert numpy.all(self._active_adapted_production_sufficient_statistics_of_lhs[adapted_non_terminal]>=0), adapted_non_terminal
            #assert numpy.all(self._active_adapted_production_length_of_lhs[adapted_non_terminal]>=0), adapted_non_terminal
            
        #sys.exit();
        
    def _recursive_replace_grammaton(self, adapted_non_terminal, old_adapted_production, new_adapted_production):
        if new_adapted_production==old_adapted_production:
            # if new adapted production is the same as the old adapted production
            return 0;
        else:
            # if new adapted production is different than the old adapted production
            old_nu_index = self._active_adapted_production_to_nu_index_of_lhs[adapted_non_terminal][old_adapted_production];
            
            if new_adapted_production not in self._active_adapted_production_to_nu_index_of_lhs[adapted_non_terminal]:
                # if new sampled adapted production is a new adapted production
                
                self._active_adapted_production_to_nu_index_of_lhs[adapted_non_terminal].pop(old_adapted_production);
                self._active_adapted_production_to_nu_index_of_lhs[adapted_non_terminal][new_adapted_production] = old_nu_index;
                self._nu_index_to_active_adapted_production_of_lhs[adapted_non_terminal][old_nu_index] = new_adapted_production;
                
                self._lhs_rhs_to_active_adapted_production[adapted_non_terminal, old_adapted_production.rhs()].remove(old_adapted_production);
                self._lhs_rhs_to_active_adapted_production[adapted_non_terminal, new_adapted_production.rhs()].add(new_adapted_production);
                
                self._lhs_to_active_adapted_production[adapted_non_terminal].remove(old_adapted_production);
                self._lhs_to_active_adapted_production[adapted_non_terminal].add(new_adapted_production);
                
                self._rhs_to_active_adapted_production[old_adapted_production.rhs()].remove(old_adapted_production);
                self._rhs_to_active_adapted_production[new_adapted_production.rhs()].add(new_adapted_production);
                
                for candidate_production in old_adapted_production._productions:
                    if isinstance(candidate_production, util.AdaptedProduction):
                        #print candidate_production.rhs(), len(candidate_production.rhs())
                        if len(candidate_production.rhs())==1:
                            print "skip singleton adapted production:", candidate_production 
                            continue;
                        nu_index = self._active_adapted_production_to_nu_index_of_lhs[candidate_production.lhs()][candidate_production];
                        
                        # Warning: if you are using nltk 2.x, please use inc()
                        #self._adapted_production_usage_freqdist.inc(candidate_production, -1);
                        self._adapted_production_usage_freqdist[candidate_production] += -1;
                        self._active_adapted_production_usage_counts_of_lhs[candidate_production.lhs()][0, nu_index] -= 1;
                        self._adapted_production_dependents_of_adapted_production[candidate_production].discard(old_adapted_production);
                        
                    elif isinstance(candidate_production, nltk.grammar.Production):
                        #print self._pcfg_production_to_gamma_index_of_lhs[candidate_production.lhs()];
                        gamma_index = self._pcfg_production_to_gamma_index_of_lhs[candidate_production.lhs()][candidate_production];
                        self._pcfg_production_usage_counts_of_lhs[candidate_production.lhs()][0, gamma_index] -= 1;
                    else:
                        print "Error in recognizing the production @ checkpoint 3..."
                        
                #assert numpy.all(self._active_adapted_production_usage_counts_of_lhs[adapted_non_terminal]>=0), adapted_non_terminal
                #assert numpy.all(self._active_adapted_production_sufficient_statistics_of_lhs[adapted_non_terminal]>=0), adapted_non_terminal
                #assert numpy.all(self._active_adapted_production_length_of_lhs[adapted_non_terminal]>=0), adapted_non_terminal                        
                        
                for candidate_production in new_adapted_production._productions:
                    if isinstance(candidate_production, util.AdaptedProduction):
                        nu_index = self._active_adapted_production_to_nu_index_of_lhs[candidate_production.lhs()][candidate_production];
                        # Warning: if you are using nltk 2.x, please use inc()
                        #self._adapted_production_usage_freqdist.inc(candidate_production, +1);
                        self._adapted_production_usage_freqdist[candidate_production] += 1;
                        self._active_adapted_production_usage_counts_of_lhs[candidate_production.lhs()][0, nu_index] += 1;
                        self._adapted_production_dependents_of_adapted_production[candidate_production].add(new_adapted_production);
                    elif isinstance(candidate_production, nltk.grammar.Production):
                        #print self._pcfg_production_to_gamma_index_of_lhs[candidate_production.lhs()];
                        gamma_index = self._pcfg_production_to_gamma_index_of_lhs[candidate_production.lhs()][candidate_production];
                        self._pcfg_production_usage_counts_of_lhs[candidate_production.lhs()][0, gamma_index] += 1;
                    else:
                        print "Error in recognizing the production @ checkpoint 3..."
    
                new_nu_index = old_nu_index;
            else:
                # if new sampled adapted production is an active adapted production
                new_nu_index = self._active_adapted_production_to_nu_index_of_lhs[adapted_non_terminal][new_adapted_production];
                
                if new_nu_index>old_nu_index:
                    temp_nu_index = new_nu_index;
                    temp_adapted_production = new_adapted_production;
                    
                    new_nu_index = old_nu_index;
                    new_adapted_production = old_adapted_production;
                    
                    old_nu_index = temp_nu_index;
                    old_adapted_production = temp_adapted_production;
    
                self._active_adapted_production_sufficient_statistics_of_lhs[adapted_non_terminal][0, new_nu_index] += self._active_adapted_production_sufficient_statistics_of_lhs[adapted_non_terminal][0, old_nu_index];
                self._active_adapted_production_sufficient_statistics_of_lhs[adapted_non_terminal][0, old_nu_index] = 0;
                self._active_adapted_production_usage_counts_of_lhs[adapted_non_terminal][0, new_nu_index] += self._active_adapted_production_usage_counts_of_lhs[adapted_non_terminal][0, old_nu_index];
                self._active_adapted_production_usage_counts_of_lhs[adapted_non_terminal][0, old_nu_index] = 0;
                
                adapted_production_usage_counts = self._adapted_production_usage_freqdist[old_adapted_production];
                # Warning: if you are using nltk 2.x, please use inc()
                #self._adapted_production_usage_freqdist.inc(old_adapted_production, -adapted_production_usage_counts);
                #self._adapted_production_usage_freqdist.inc(new_adapted_production, adapted_production_usage_counts);
                self._adapted_production_usage_freqdist[old_adapted_production] += -adapted_production_usage_counts;
                self._adapted_production_usage_freqdist[new_adapted_production] += adapted_production_usage_counts;
    
            for old_dependent_adapted_production in self._adapted_production_dependents_of_adapted_production[old_adapted_production]:
                new_production_list = []
                for candidate_production in old_dependent_adapted_production._productions:
                    if isinstance(candidate_production, util.AdaptedProduction):
                        if candidate_production==old_adapted_production:
                            new_production_list.append(new_adapted_production);
                        else:
                            new_production_list.append(candidate_production);
                    elif isinstance(candidate_production, nltk.grammar.Production):
                        new_production_list.append(candidate_production);
                    else:
                        print "Error in recognizing the production @ checkpoint 3..."
                
                lhs = old_dependent_adapted_production.lhs();
                rhs = old_dependent_adapted_production.rhs();
                new_dependent_adapted_production = util.AdaptedProduction(lhs, rhs, new_production_list);
                self._recursive_replace_grammaton(lhs, old_dependent_adapted_production, new_dependent_adapted_production);
                self._adapted_production_dependents_of_adapted_production[new_adapted_production].add(new_dependent_adapted_production);
                
            return 1;
            
    def table_label_resampling(self):
        pcfg_sufficient_statistics = {};
        for non_terminal in self._non_terminals:
            pcfg_sufficient_statistics[non_terminal] = numpy.zeros((1, len(self._gamma_index_to_pcfg_production_of_lhs[non_terminal])));
        adapted_sufficient_statistics = {};
        for adapted_non_terminal in self._adapted_non_terminals:
            adapted_sufficient_statistics[adapted_non_terminal] = numpy.zeros((1, len(self._nu_index_to_active_adapted_production_of_lhs[adapted_non_terminal])));
        
        #for adapted_non_terminal_index in xrange(len(self._ordered_adaptor_top_down[::-1][1:])):
        for adapted_non_terminal_index in xrange(1, len(self._ordered_adaptor_down_top)):
            adapted_non_terminal = self._ordered_adaptor_down_top[adapted_non_terminal_index];
            candidate_adaptors = self._ordered_adaptor_down_top[:adapted_non_terminal_index];
            
            number_of_tables_relabelled = 0;
            
            #for active_adapted_production_index in xrange(len(self._active_adapted_production_to_nu_index_of_lhs[adapted_non_terminal])-1, -1, -1):
            for active_adapted_production_index in xrange(len(self._active_adapted_production_to_nu_index_of_lhs[adapted_non_terminal])):

                #self.model_state_assertion();
                old_adapted_production = self._nu_index_to_active_adapted_production_of_lhs[adapted_non_terminal][active_adapted_production_index];
                old_nu_index = self._active_adapted_production_to_nu_index_of_lhs[adapted_non_terminal][old_adapted_production];
                
                if self._active_adapted_production_usage_counts_of_lhs[adapted_non_terminal][0, old_nu_index]==0 and self._active_adapted_production_sufficient_statistics_of_lhs[adapted_non_terminal][0, old_nu_index]==0:
                    # if such adapted production is useless, i.e., never used in any other derivations.
                    continue;

                E_log_stick_weights, E_log_theta = self.propose_pcfg();    
                
                root_node = self.compute_inside_probabilities(E_log_stick_weights, E_log_theta, list(old_adapted_production.rhs()), adapted_non_terminal, candidate_adaptors);
                #new_adapted_production = sample_tree_by_pre_order_traversal(self, root_node, list(old_adapted_production.rhs()), None, None);
                #new_adapted_production = self._sample_tree_by_pre_order_traversal(root_node, list(old_adapted_production.rhs()), pcfg_sufficient_statistics, adapted_sufficient_statistics);
                new_adapted_production = self._sample_tree(root_node, list(old_adapted_production.rhs()), pcfg_sufficient_statistics, adapted_sufficient_statistics);
                assert isinstance(new_adapted_production, list) and len(new_adapted_production)==1;
                new_adapted_production = new_adapted_production[0];
                assert isinstance(new_adapted_production, util.AdaptedProduction), new_adapted_production
                assert old_adapted_production.lhs()==new_adapted_production.lhs()
                assert " ".join(old_adapted_production.rhs())==" ".join(new_adapted_production.rhs())
                
                '''
                sampled = False;
                while(True):
                    root_node = self.compute_inside_probabilities(E_log_stick_weights, E_log_theta, list(old_adapted_production.rhs()), adapted_non_terminal, candidate_adaptors);
                    #new_adapted_production = sample_tree_by_pre_order_traversal(self, root_node, list(old_adapted_production.rhs()), None, None);
                    #new_adapted_production = self._sample_tree_by_pre_order_traversal(root_node, list(old_adapted_production.rhs()), pcfg_sufficient_statistics, adapted_sufficient_statistics);
                    new_adapted_production = self._sample_tree(root_node, list(old_adapted_production.rhs()), pcfg_sufficient_statistics, adapted_sufficient_statistics);
                    assert isinstance(new_adapted_production, list) and len(new_adapted_production)==1;
                    new_adapted_production = new_adapted_production[0];
                    assert isinstance(new_adapted_production, util.AdaptedProduction), new_adapted_production
                    assert old_adapted_production.lhs()==new_adapted_production.lhs()
                    assert " ".join(old_adapted_production.rhs())==" ".join(new_adapted_production.rhs())
                    
                    if new_adapted_production not in self._active_adapted_production_to_nu_index_of_lhs[adapted_non_terminal]:
                        break;
                    new_nu_index = self._active_adapted_production_to_nu_index_of_lhs[adapted_non_terminal][new_adapted_production];                    
                    if self._active_adapted_production_usage_counts_of_lhs[adapted_non_terminal][0, old_nu_index]>0:
                        break;
                    if self._active_adapted_production_sufficient_statistics_of_lhs[adapted_non_terminal][0, new_nu_index]>0:
                        break;
                    
                    print "old:", old_adapted_production
                    print "new:", new_adapted_production
                    sampled = True;
                
                if sampled:
                    sys.exit();
                '''
                    
                #self.model_state_assertion();
                #replace_grammaton_clock = time.time()
                number_of_tables_relabelled += self._recursive_replace_grammaton(adapted_non_terminal, old_adapted_production, new_adapted_production);
                #replace_grammaton_clock = time.time() - replace_grammaton_clock
                #print "time to replace grammaton", replace_grammaton_clock
                #self.model_state_assertion();

            print "%d out of %d tables get relabelled for adaptor %s..." % (number_of_tables_relabelled, len(self._active_adapted_production_to_nu_index_of_lhs[adapted_non_terminal]), adapted_non_terminal);
            
            assert numpy.all(self._active_adapted_production_usage_counts_of_lhs[adapted_non_terminal]>=0), adapted_non_terminal
            assert numpy.all(self._active_adapted_production_sufficient_statistics_of_lhs[adapted_non_terminal]>=0), adapted_non_terminal
            assert numpy.all(self._active_adapted_production_length_of_lhs[adapted_non_terminal]>=0), adapted_non_terminal

        return;

    def optimize_alpha_theta(self, step_factor=1, decay=0.1, converge_threshold=1e-3):
        for non_terminal in self._non_terminals:
            temp_step_factor = step_factor;
            
            old_alpha_theta = -1e9;
            while(numpy.any(abs(old_alpha_theta-self._alpha_theta[non_terminal])>converge_threshold)):
                first_derivative = -scipy.special.psi(self._alpha_theta[non_terminal]);
                first_derivative += scipy.special.psi(numpy.sum(self._alpha_theta[non_terminal]));
                first_derivative += scipy.special.psi(self._gamma[non_terminal]) - scipy.special.psi(numpy.sum(self._gamma[non_terminal]))
                assert(first_derivative.shape==(1, len(self._gamma_index_to_pcfg_production_of_lhs[non_terminal])))
            
                second_derivative = -scipy.special.polygamma(1, self._alpha_theta[non_terminal]);
                second_derivative += scipy.special.polygamma(1, numpy.sum(self._alpha_theta[non_terminal]));
                assert(second_derivative.shape==(1, len(self._gamma_index_to_pcfg_production_of_lhs[non_terminal])))
                
                if numpy.any(second_derivative==0) or not numpy.all(numpy.isfinite(first_derivative/second_derivative)):
                    #print "warning: natural gradient of alpha_theta throws arithmetic exception..."
                    break;
                
                old_alpha_theta = self._alpha_theta[non_terminal];
                while(numpy.any(temp_step_factor*first_derivative/second_derivative>=self._alpha_theta[non_terminal])):
                    temp_step_factor *= decay;
                self._alpha_theta[non_terminal] -= temp_step_factor*first_derivative/second_derivative;
                assert (numpy.all(self._alpha_theta[non_terminal]>0)), (old_alpha_theta, self._alpha_theta[non_terminal]);
                
        print "optimize alpha_theta:\n%s" % ("\n".join(["\t%s: %s" % (non_terminal, self._alpha_theta[non_terminal]) for non_terminal in self._non_terminals]));
        
        return;

    def optimize_alpha_pi(self, step_factor=1e3, decay=0.5, converge_threshold=1.):
        for adapted_non_terminal in self._adapted_non_terminals:
            indices = numpy.r_[0:len(self._active_adapted_production_to_nu_index_of_lhs[adapted_non_terminal])][numpy.newaxis, :]+1
            assert(indices.shape==(1, len(self._active_adapted_production_to_nu_index_of_lhs[adapted_non_terminal])))
            
            temp_step_factor = step_factor;
            
            old_alpha_pi = -1e9;
            while(abs(old_alpha_pi-self._alpha_pi[adapted_non_terminal])>converge_threshold):
                index_beta_pi = indices * self._beta_pi[adapted_non_terminal];
                alpha_pi_and_index_beta_pi = self._alpha_pi[adapted_non_terminal] + index_beta_pi;
                
                first_derivative = numpy.sum(scipy.special.psi(self._nu_2[adapted_non_terminal]) - scipy.special.psi(self._nu_1[adapted_non_terminal] + self._nu_2[adapted_non_terminal]));
                first_derivative += numpy.sum(scipy.special.psi(1-self._beta_pi[adapted_non_terminal]+alpha_pi_and_index_beta_pi));
                first_derivative -= numpy.sum(scipy.special.psi(alpha_pi_and_index_beta_pi));
                
                second_derivative = numpy.sum(scipy.special.polygamma(1, 1-self._beta_pi[adapted_non_terminal]+alpha_pi_and_index_beta_pi));
                second_derivative -= numpy.sum(scipy.special.polygamma(1, alpha_pi_and_index_beta_pi));

                if numpy.any(second_derivative==0) or not numpy.all(numpy.isfinite(first_derivative/second_derivative)):
                    #print "warning: natural gradient of alpha_pi throws arithmetic exception..."
                    break;
    
                old_alpha_pi = self._alpha_pi[adapted_non_terminal];
                while(temp_step_factor*first_derivative/second_derivative>=self._alpha_pi[adapted_non_terminal]):
                    temp_step_factor *= decay;
                self._alpha_pi[adapted_non_terminal] = self._alpha_pi[adapted_non_terminal] - temp_step_factor*first_derivative/second_derivative;
                assert (self._alpha_pi[adapted_non_terminal]>0), (old_alpha_pi, self._alpha_pi[adapted_non_terminal], temp_step_factor, first_derivative, second_derivative, numpy.sum(scipy.special.polygamma(1, 1-self._beta_pi[adapted_non_terminal]+alpha_pi_and_index_beta_pi)), numpy.sum(scipy.special.polygamma(1, alpha_pi_and_index_beta_pi)));
        
        #print ["%s - %f" % (adapted_non_terminal, self._alpha_pi[adapted_non_terminal]) for adapted_non_terminal in self._adapted_non_terminals]
        print "optimize alpha_pi: %s" % ("\t".join(["%s: %f" % (adapted_non_terminal, self._alpha_pi[adapted_non_terminal]) for adapted_non_terminal in self._adapted_non_terminals]));
        
        return;
    
    def optimize_beta_pi(self, step_factor=1e3, decay=0.5, converge_threshold=1e-3):
        for adapted_non_terminal in self._adapted_non_terminals:
            indices = numpy.r_[0:len(self._active_adapted_production_to_nu_index_of_lhs[adapted_non_terminal])][numpy.newaxis, :]+1;
            assert(indices.shape==(1, len(self._active_adapted_production_to_nu_index_of_lhs[adapted_non_terminal])))
            
            temp_step_factor = step_factor;
            
            old_beta_pi = -1e9;
            while(abs(old_beta_pi-self._beta_pi[adapted_non_terminal])>converge_threshold):
            
                first_derivative = len(self._active_adapted_production_to_nu_index_of_lhs[adapted_non_terminal]) * scipy.special.psi(1-self._beta_pi[adapted_non_terminal]);
                first_derivative -= numpy.sum(indices * scipy.special.psi(self._alpha_pi[adapted_non_terminal] + indices * self._beta_pi[adapted_non_terminal]));
                first_derivative += numpy.sum((indices-1) * scipy.special.psi(1 + self._alpha_pi[adapted_non_terminal] + (indices-1)*self._beta_pi[adapted_non_terminal]));
                first_derivative -= numpy.sum(scipy.special.psi(self._nu_1[adapted_non_terminal]) - scipy.special.psi(self._nu_1[adapted_non_terminal] + self._nu_2[adapted_non_terminal]));
                first_derivative += numpy.sum(indices * (scipy.special.psi(self._nu_2[adapted_non_terminal]) - scipy.special.psi(self._nu_1[adapted_non_terminal] + self._nu_2[adapted_non_terminal])));
                
                second_derivative = -len(self._active_adapted_production_to_nu_index_of_lhs[adapted_non_terminal]) * scipy.special.polygamma(1, (1-self._beta_pi[adapted_non_terminal]));
                second_derivative += numpy.sum(numpy.power(indices-1, 2) * scipy.special.polygamma(1, (1 + self._alpha_pi[adapted_non_terminal] + (indices-1) * self._beta_pi[adapted_non_terminal])));
                second_derivative -= numpy.sum(numpy.power(indices, 2) * scipy.special.polygamma(1, (self._alpha_pi[adapted_non_terminal] + indices * self._beta_pi[adapted_non_terminal])));
                
                if numpy.any(second_derivative==0) or not numpy.all(numpy.isfinite(first_derivative/second_derivative)):
                    print "warning: natural gradient of beta_pi throws arithmetic exception..."
                    break;
                
                old_beta_pi = self._beta_pi[adapted_non_terminal];
                while(temp_step_factor*first_derivative/second_derivative>self._beta_pi[adapted_non_terminal]):
                    temp_step_factor *= decay;
                    #print adapted_non_terminal, self._beta_pi[adapted_non_terminal], temp_step_factor, first_derivative, second_derivative;
                self._beta_pi[adapted_non_terminal] = self._beta_pi[adapted_non_terminal] - temp_step_factor*first_derivative/second_derivative;
                if self._beta_pi[adapted_non_terminal]<0:
                    # this is floating point error
                    self._beta_pi[adapted_non_terminal] = 0;
                    #self._beta_pi[adapted_non_terminal] = -self._beta_pi[adapted_non_terminal]
                
                #print adapted_non_terminal, self._beta_pi[adapted_non_terminal];
                assert(self._beta_pi[adapted_non_terminal]>=0), (self._beta_pi[adapted_non_terminal], temp_step_factor, first_derivative, second_derivative);
                assert(self._beta_pi[adapted_non_terminal]<1);
    
        print "optimize beta_pi: %s" % ("\t".join(["%s: %f" % (adapted_non_terminal, self._beta_pi[adapted_non_terminal]) for adapted_non_terminal in self._adapted_non_terminals]));
    
        return;

    '''
    def e_step_inference_process_queue(self, input_strings, reference_strings, retrieve_tokens_at_adapted_non_terminal, output_path, model_name, number_of_samples=10, number_of_processes=2):
        assert(retrieve_tokens_at_adapted_non_terminal in self._adapted_non_terminals);
        
        #output_average_truth_file = codecs.open(os.path.join(output_path, "avg.truth"), 'a', encoding='utf-8');
        #output_average_test_file = codecs.open(os.path.join(output_path, "avg.test"), 'a', encoding='utf-8');
        #output_maximum_truth_file = codecs.open(os.path.join(output_path, "max.truth"), 'a', encoding='utf-8');
        #output_maximum_test_file = codecs.open(os.path.join(output_path, "max.test"), 'a', encoding='utf-8');
        
        #output_average_truth_file = open(os.path.join(output_path, "avg.truth"), 'a');
        #output_average_test_file = open(os.path.join(output_path, "avg.test"), 'a');
        #output_maximum_truth_file = open(os.path.join(output_path, "max.truth"), 'a');
        #output_maximum_test_file = open(os.path.join(output_path, "max.test"), 'a');
            
        # establish communication queues
        task_queue = multiprocessing.JoinableQueue()
        #result_queue_pcfg_sufficient_statistics = multiprocessing.Queue()
        #result_queue_adapted_sufficient_statistics = multiprocessing.Queue()
        result_queue_production_list = multiprocessing.Queue();
    
        # enqueue jobs
        #for (input_string, referece_string) in input_strings:
        for (input_string, reference_string) in zip(input_strings, reference_strings):
            task_queue.put((input_string, reference_string));

        E_log_stick_weights, E_log_theta = self.propose_pcfg();
        
        # start consumers
        #number_of_processes = multiprocessing.cpu_count();
        print 'creating %d processes_e_step' % number_of_processes
        processes_e_step = [ Process_E_Step_queue(task_queue,
                                                self,
                                                E_log_theta,
                                                E_log_stick_weights,
                                                None,
                                                None,
                                                retrieve_tokens_at_adapted_non_terminal,
                                                output_path,
                                                model_name,
                                                number_of_samples)
                                                for process_index in xrange(number_of_processes)];
    
        #output_average_truth_file = os.path.join(output_path, "avg.truth")
        #output_average_test_file = os.path.join(output_path, "avg.test")
        #process_inference_output = Process_Inference_Output(result_queue_production_list,
                                                            #output_average_truth_file,
                                                            #output_average_test_file);
        
        for process_e_step in processes_e_step:
            process_e_step.start();
        #for process_e_step in processes_e_step:
            #process_e_step.join();
    
        # Add a poison pill for each consumer
        #for i in xrange(number_of_processes):
            #task_queue.put(None)
    
        # Wait for all of the task_queue to finish
        task_queue.join();
        
        #process_inference_output.start();
        #process_inference_output.join();
        
        #for process_e_step in processes_e_step:
            #process_e_step.terminate();
        task_queue.close();
        
        return
    '''

    def e_step_process_queue(self, input_strings, number_of_samples=10, number_of_processes=2, inference_parameter=None):
        if inference_parameter==None:
            reference_strings = None;
            retrieve_tokens_at_adapted_non_terminal = None;
            output_path = None;
            model_name = None;
        else:
            (reference_strings, retrieve_tokens_at_adapted_non_terminal, output_path, model_name) = inference_parameter;
            assert retrieve_tokens_at_adapted_non_terminal in self._adapted_non_terminals;
            assert len(input_strings)==len(reference_strings);
        
        # establish communication queues
        task_queue = multiprocessing.JoinableQueue()
        if inference_parameter==None:
            result_queue_pcfg_sufficient_statistics = multiprocessing.Queue()
            result_queue_adapted_sufficient_statistics = multiprocessing.Queue()
        else:
            result_queue_pcfg_sufficient_statistics = None;
            result_queue_adapted_sufficient_statistics = None;
        
        # enqueue jobs
        if inference_parameter==None:
            for input_string in input_strings:
                task_queue.put((input_string, None));
        else:
            for (input_string, reference_string) in zip(input_strings, reference_strings):
                task_queue.put((input_string, reference_string));
        
        E_log_stick_weights, E_log_theta = self.propose_pcfg();
        
        # start consumers
        #number_of_processes = multiprocessing.cpu_count();
        print 'creating %d processes' % number_of_processes
        processes_e_step = [Process_E_Step_queue(task_queue,
                                                 self,
                                                 E_log_theta,
                                                 E_log_stick_weights,
                                                 result_queue_adapted_sufficient_statistics,
                                                 result_queue_pcfg_sufficient_statistics,
                                                 retrieve_tokens_at_adapted_non_terminal,
                                                 output_path,
                                                 model_name,
                                                 number_of_samples)
                                                 for process_index in xrange(number_of_processes) ];                               
        
        for process_e_step in processes_e_step:
            process_e_step.start();
        #for process_e_step in processes_e_step:
            #process_e_step.join();
            
        # Wait for all of the task_queue to finish
        task_queue.join();
        
        #for process_e_step in processes_e_step:
            #process_e_step.terminate();
        task_queue.close();
        
        if inference_parameter==None:
            pcfg_sufficient_statistics, adapted_sufficient_statistics = self._format_tree_samples_queue(result_queue_adapted_sufficient_statistics, result_queue_pcfg_sufficient_statistics);
            return pcfg_sufficient_statistics, adapted_sufficient_statistics;
        else:
            return;
        
    def _sample_tree_process_queue(self, current_hyper_node, input_string, result_queue_adapted_sufficient_statistics=None, result_queue_pcfg_sufficient_statistics=None):
        sampled_production, unsampled_hyper_nodes, log_probability_of_sampled_production = current_hyper_node.random_sample_derivation();
        if isinstance(sampled_production, util.AdaptedProduction):
            # if sampled production is an adapted production
            assert (unsampled_hyper_nodes==None or len(unsampled_hyper_nodes)==0), "incomplete adapted production: %s" % sampled_production

            if result_queue_adapted_sufficient_statistics != None:
                #result_queue_adapted_sufficient_statistics.put((sampled_production, current_hyper_node._node, tuple(input_string[current_hyper_node._span[0]:current_hyper_node._span[1]])));
                nu_index = self._active_adapted_production_to_nu_index_of_lhs[current_hyper_node._node][sampled_production];
                result_queue_adapted_sufficient_statistics.put((current_hyper_node._node, nu_index));
            
            return [sampled_production]
        elif isinstance(sampled_production, nltk.grammar.Production):
            # if sampled production is an pcfg production
            gamma_index = self._pcfg_production_to_gamma_index_of_lhs[current_hyper_node._node][sampled_production];
            
            if result_queue_pcfg_sufficient_statistics != None:
                result_queue_pcfg_sufficient_statistics.put((current_hyper_node._node, gamma_index));
            
            # if sampled production is a pre-terminal pcfg production
            if unsampled_hyper_nodes==None or len(unsampled_hyper_nodes)==0:
                assert (not self.is_adapted_non_terminal(current_hyper_node._node)), "adapted pre-terminal found: %s" % current_hyper_node._node
                return [sampled_production]
    
            # if sampled production is a regular pcfg production
            production_list = [sampled_production];
            for unsampled_hyper_node in unsampled_hyper_nodes:
                production_list += self._sample_tree_process_queue(unsampled_hyper_node, input_string, result_queue_adapted_sufficient_statistics, result_queue_pcfg_sufficient_statistics);
    
            # if current node is a non-adapted non-terminal node
            if not self.is_adapted_non_terminal(current_hyper_node._node):
                return production_list
    
            new_adapted_production = util.AdaptedProduction(current_hyper_node._node, input_string[current_hyper_node._span[0]:current_hyper_node._span[1]], production_list);
    
            if result_queue_adapted_sufficient_statistics != None:
                if new_adapted_production in self.get_adapted_productions(current_hyper_node._node, tuple(input_string[current_hyper_node._span[0]:current_hyper_node._span[1]])):
                    # if this is an active adapted production
                    nu_index = self._active_adapted_production_to_nu_index_of_lhs[current_hyper_node._node][new_adapted_production];
                    result_queue_adapted_sufficient_statistics.put((current_hyper_node._node, nu_index));
                    #result_queue_adapted_sufficient_statistics.put((new_adapted_production, current_hyper_node._node, tuple(input_string[current_hyper_node._span[0]:current_hyper_node._span[1]])));
                else:
                    # if this is an inactive adapted production
                    result_queue_adapted_sufficient_statistics.put((current_hyper_node._node, tuple(input_string[current_hyper_node._span[0]:current_hyper_node._span[1]]), new_adapted_production));
                    
            return [new_adapted_production];
        else:
            sys.stderr.write("Error in recognizing the production class %s @ checkpoint 2...\n" % sampled_production.__class__);
            sys.exit();
            
    def _format_tree_samples_queue(self, result_queue_adapted_sufficient_statistics, result_queue_pcfg_sufficient_statistics):
        adapted_sufficient_statistics = {};
        for adapted_non_terminal in self._adapted_non_terminals:
            adapted_sufficient_statistics[adapted_non_terminal] = numpy.zeros((1, len(self._nu_index_to_active_adapted_production_of_lhs[adapted_non_terminal])));

        pcfg_sufficient_statistics = {};
        for non_terminal in self._non_terminals:
            pcfg_sufficient_statistics[non_terminal] = numpy.zeros((1, len(self._gamma_index_to_pcfg_production_of_lhs[non_terminal])));
            
        counter = 0
        #while not result_queue_pcfg_sufficient_statistics.empty():
        for result_queue_element_index in xrange(result_queue_pcfg_sufficient_statistics.qsize()):
            (non_terminal, gamma_index) = result_queue_pcfg_sufficient_statistics.get();
            #print non_terminal, gamma_index
            
            pcfg_sufficient_statistics[non_terminal][0, gamma_index] += 1;
            counter += 1;

        counter = 0
        #while not result_queue_adapted_sufficient_statistics.empty():
        for result_queue_element_index in xrange(result_queue_adapted_sufficient_statistics.qsize()):
            result_queue_element = result_queue_adapted_sufficient_statistics.get();
            if len(result_queue_element)==2:
                # if this is an active adapted production
                (adapted_production_node, nu_index) = result_queue_element;
                assert adapted_production_node in self._adapted_non_terminals
            elif len(result_queue_element)==3:
                # if this is an inactive adapted production
                (adapted_production_node, adapted_production_yields, new_adapted_production) = result_queue_element;
                assert adapted_production_node in self._adapted_non_terminals
                
                if new_adapted_production in self.get_adapted_productions(adapted_production_node, adapted_production_yields):
                    # if this adapted production is actived *just now*
                    nu_index = self._active_adapted_production_to_nu_index_of_lhs[adapted_production_node][new_adapted_production];
                else:
                    adapted_production_count = 0;
                    pcfg_production_count = 0;
                    for candidate_production in new_adapted_production.get_production_list():
                        if isinstance(candidate_production, util.AdaptedProduction):
                            adapted_production_count += 1;
                            nu_index = self._active_adapted_production_to_nu_index_of_lhs[candidate_production.lhs()][candidate_production];
                            # Warning: if you are using nltk 2.x, please use inc()
                            #self._adapted_production_usage_freqdist.inc(candidate_production, 1);
                            self._adapted_production_usage_freqdist[candidate_production] += 1;
                            self._adapted_production_dependents_of_adapted_production[candidate_production].add(new_adapted_production);
                            self._active_adapted_production_usage_counts_of_lhs[candidate_production.lhs()][0, nu_index] += 1;
                        elif isinstance(candidate_production, nltk.grammar.Production):
                            pcfg_production_count += 1;
                            gamma_index = self._pcfg_production_to_gamma_index_of_lhs[candidate_production.lhs()][candidate_production];
                            self._pcfg_production_usage_counts_of_lhs[candidate_production.lhs()][0, gamma_index] += 1;
                        else:
                            sys.stderr.write("Error in recognizing the production @ checkpoint 1...\n");
                            sys.exit();
                            
                    # activate this adapted rule
                    self._lhs_rhs_to_active_adapted_production[(adapted_production_node, adapted_production_yields)].add(new_adapted_production);
                    self._lhs_to_active_adapted_production[adapted_production_node].add(new_adapted_production);
                    self._rhs_to_active_adapted_production[adapted_production_yields].add(new_adapted_production);
                    
                    self._nu_index_to_active_adapted_production_of_lhs[adapted_production_node][len(self._nu_index_to_active_adapted_production_of_lhs[adapted_production_node])] = new_adapted_production;
                    self._active_adapted_production_to_nu_index_of_lhs[adapted_production_node][new_adapted_production] = len(self._active_adapted_production_to_nu_index_of_lhs[adapted_production_node]);
    
                    self._nu_1[adapted_production_node] = numpy.hstack((self._nu_1[adapted_production_node], numpy.ones((1, 1))));
                    self._nu_2[adapted_production_node] = numpy.hstack((self._nu_2[adapted_production_node], numpy.ones((1, 1))*self._alpha_pi[adapted_production_node]));
                    
                    nu_index = self._active_adapted_production_to_nu_index_of_lhs[adapted_production_node][new_adapted_production];
                    adapted_sufficient_statistics[adapted_production_node] = numpy.hstack((adapted_sufficient_statistics[adapted_production_node], numpy.zeros((1, 1))));
                    
                    self._active_adapted_production_usage_counts_of_lhs[adapted_production_node] = numpy.hstack((self._active_adapted_production_usage_counts_of_lhs[adapted_production_node], numpy.zeros((1, 1))));
                    self._active_adapted_production_usage_counts_of_lhs[adapted_production_node][0, nu_index] = self._adapted_production_usage_freqdist[new_adapted_production];
    
                    self._active_adapted_production_length_of_lhs[adapted_production_node] = numpy.hstack((self._active_adapted_production_length_of_lhs[adapted_production_node], numpy.zeros((1, 1))));
                    #self._active_adapted_production_length_of_lhs[adapted_production_node][0, nu_index] = len(adapted_production_list.rhs());
                    if adapted_production_node==self._ordered_adaptor_top_down[-1]:
                        self._active_adapted_production_length_of_lhs[adapted_production_node][0, nu_index] = len(new_adapted_production.rhs());
                    else:
                        self._active_adapted_production_length_of_lhs[adapted_production_node][0, nu_index] = adapted_production_count;
    
                    self._active_adapted_production_sufficient_statistics_of_lhs[adapted_production_node] = numpy.hstack((self._active_adapted_production_sufficient_statistics_of_lhs[adapted_production_node], numpy.zeros((1, 1))));
                    if nu_index>self._truncation_level[adapted_production_node]:
                        self._active_adapted_production_sufficient_statistics_of_lhs[adapted_production_node][0, nu_index] = self._active_adapted_production_sufficient_statistics_of_lhs[adapted_production_node][0, nu_index-1];
            else:
                sys.stderr.write("Error in result_queue_adapted_sufficient_statistics...\n");
                sys.exit();
            
            counter += 1;
            
            adapted_sufficient_statistics[adapted_production_node][0, nu_index] += 1;

        return pcfg_sufficient_statistics, adapted_sufficient_statistics;

    '''
    def e_step_inference_process_dict(self, input_strings, reference_strings, retrieve_tokens_at_adapted_non_terminal, output_path, model_name, number_of_samples=10, number_of_processes=2):
        assert(retrieve_tokens_at_adapted_non_terminal in self._adapted_non_terminals);
        
        #output_average_truth_file = codecs.open(os.path.join(output_path, "avg.truth"), 'a', encoding='utf-8');
        #output_average_test_file = codecs.open(os.path.join(output_path, "avg.test"), 'a', encoding='utf-8');
        #output_maximum_truth_file = codecs.open(os.path.join(output_path, "max.truth"), 'a', encoding='utf-8');
        #output_maximum_test_file = codecs.open(os.path.join(output_path, "max.test"), 'a', encoding='utf-8');
        
        #output_average_truth_file = open(os.path.join(output_path, "avg.truth"), 'a');
        #output_average_test_file = open(os.path.join(output_path, "avg.test"), 'a');
        #output_maximum_truth_file = open(os.path.join(output_path, "max.truth"), 'a');
        #output_maximum_test_file = open(os.path.join(output_path, "max.test"), 'a');
            
        # establish communication queues
        task_queue = multiprocessing.JoinableQueue()
        #result_queue_pcfg_sufficient_statistics = multiprocessing.Queue()
        #result_queue_adapted_sufficient_statistics = multiprocessing.Queue()
        result_queue_production_list = multiprocessing.Queue();
    
        # enqueue jobs
        #for (input_string, referece_string) in input_strings:
        for (input_string, reference_string) in zip(input_strings, reference_strings):
            task_queue.put((input_string, reference_string));

        E_log_stick_weights, E_log_theta = self.propose_pcfg();
        
        # start consumers
        #number_of_processes = multiprocessing.cpu_count();
        print 'creating %d processes_e_step' % number_of_processes
        processes_e_step = [Process_E_Step_dict(task_queue,
                                                self,
                                                E_log_theta,
                                                E_log_stick_weights,
                                                None,
                                                None,
                                                None,
                                                retrieve_tokens_at_adapted_non_terminal,
                                                output_path,
                                                model_name,
                                                number_of_samples)
                                                for process_index in xrange(number_of_processes)];
    
        #output_average_truth_file = os.path.join(output_path, "avg.truth")
        #output_average_test_file = os.path.join(output_path, "avg.test")
        #process_inference_output = Process_Inference_Output(result_queue_production_list,
                                                            #output_average_truth_file,
                                                            #output_average_test_file);
        
        for process_e_step in processes_e_step:
            process_e_step.start();
        #for process_e_step in processes_e_step:
            #process_e_step.join();
    
        # Add a poison pill for each consumer
        #for i in xrange(number_of_processes):
            #task_queue.put(None)
    
        # Wait for all of the task_queue to finish
        task_queue.join();
        
        #process_inference_output.start();
        #process_inference_output.join();
        
        #for process_e_step in processes_e_step:
            #process_e_step.terminate();
        task_queue.close();
        
        return
    '''

    def e_step_process_dict(self, input_strings, number_of_samples=10, number_of_processes=2, inference_parameter=None):
        if inference_parameter==None:
            reference_strings = None;
            retrieve_tokens_at_adapted_non_terminal = None;
            output_path = None;
            model_name = None;
        else:
            (reference_strings, retrieve_tokens_at_adapted_non_terminal, output_path, model_name) = inference_parameter;
            assert retrieve_tokens_at_adapted_non_terminal in self._adapted_non_terminals;
            assert len(input_strings)==len(reference_strings);
        
        '''
        # establish communication queues
        if inference_parameter==None:
            multiprocessing_manager = multiprocessing.Manager();
            pcfg_sufficient_statistics_dict = multiprocessing_manager.dict();
            for non_terminal in self._non_terminals:
                pcfg_sufficient_statistics_dict[non_terminal] = numpy.zeros((1, len(self._gamma_index_to_pcfg_production_of_lhs[non_terminal])));
            adapted_sufficient_statistics_dict = multiprocessing_manager.dict();
            for adapted_non_terminal in self._adapted_non_terminals:
                adapted_sufficient_statistics_dict[adapted_non_terminal] = numpy.zeros((1, len(self._nu_index_to_active_adapted_production_of_lhs[adapted_non_terminal])));
                #print "+++++ initialize +++++", adapted_sufficient_statistics_dict[adapted_non_terminal].shape;
            new_adapted_sufficient_statistics_dict = multiprocessing_manager.dict();
            for adapted_non_terminal in self._adapted_non_terminals:
                new_adapted_sufficient_statistics_dict[adapted_non_terminal] = {};

            adapted_sufficient_statistics_mutex = multiprocessing.Lock();
            pcfg_sufficient_statistics_mutex = multiprocessing.Lock();
            new_adapted_sufficient_statistics_mutex = multiprocessing.Lock();
        
            sufficient_statistics_mutex = (adapted_sufficient_statistics_mutex, new_adapted_sufficient_statistics_mutex, pcfg_sufficient_statistics_mutex);    
            #sufficient_statistics_dictionary = (adapted_sufficient_statistics_dict, new_adapted_sufficient_statistics_dict, pcfg_sufficient_statistics_dict);
        else:
            sufficient_statistics_mutex = None;
            #sufficient_statistics_dictionary = None;
        '''
        
        # enqueue jobs
        task_queue = multiprocessing.JoinableQueue()
        if inference_parameter==None:
            for input_string in input_strings:
                task_queue.put((input_string, None));
            result_queue_pcfg_sufficient_statistics = multiprocessing.Queue()
            result_queue_adapted_sufficient_statistics = multiprocessing.Queue()
            result_queue_new_adapted_sufficient_statistics = multiprocessing.Queue()
        else:
            for (input_string, reference_string) in zip(input_strings, reference_strings):
                task_queue.put((input_string, reference_string));
            result_queue_pcfg_sufficient_statistics = None;
            result_queue_adapted_sufficient_statistics = None;
            result_queue_new_adapted_sufficient_statistics = None;
            
        #process_e_step_clock = time.time()
        
        # start consumers
        #number_of_processes = multiprocessing.cpu_count();
        print 'creating %d processes' % number_of_processes
        processes_e_step = [Process_E_Step_dict(task_queue,
                                                self,
                                                result_queue_adapted_sufficient_statistics,
                                                result_queue_new_adapted_sufficient_statistics,
                                                result_queue_pcfg_sufficient_statistics,
                                                retrieve_tokens_at_adapted_non_terminal,
                                                output_path,
                                                model_name,
                                                number_of_samples)
                                                for process_index in xrange(number_of_processes)];
        
        for process_e_step in processes_e_step:
            process_e_step.start();
            
        task_queue.join();
        
        task_queue.close();
        
        if inference_parameter==None:
            pcfg_sufficient_statistics, adapted_sufficient_statistics = self._format_tree_samples_dict(result_queue_adapted_sufficient_statistics, result_queue_new_adapted_sufficient_statistics, result_queue_pcfg_sufficient_statistics);
            return pcfg_sufficient_statistics, adapted_sufficient_statistics;
        else:
            return;

    def _sample_tree_process_dict(self,
                                  current_hyper_node,
                                  input_string,
                                  adapted_sufficient_statistics_dictionary,
                                  new_adapted_sufficient_statistics_dictionary,
                                  pcfg_sufficient_statistics_dictionary,
                                  new_adapted_sufficient_statistics_mutex=None,
                                  sufficient_statistics_dictionary=None):
        #(adapted_sufficient_statistics_dictionary, new_adapted_sufficient_statistics_dictionary, pcfg_sufficient_statistics_dictionary) = sufficient_statistics_dictionary;
        
        sampled_production, unsampled_hyper_nodes, log_probability_of_sampled_production = current_hyper_node.random_sample_derivation();
        if isinstance(sampled_production, util.AdaptedProduction):
            # if sampled production is an adapted production
            assert (unsampled_hyper_nodes==None or len(unsampled_hyper_nodes)==0), "incomplete adapted production: %s" % sampled_production

            if adapted_sufficient_statistics_dictionary != None:
                #result_queue_adapted_sufficient_statistics.put((sampled_production, current_hyper_node._node, tuple(input_string[current_hyper_node._span[0]:current_hyper_node._span[1]])));
                nu_index = self._active_adapted_production_to_nu_index_of_lhs[current_hyper_node._node][sampled_production];
                adapted_sufficient_statistics_dictionary[current_hyper_node._node][0, nu_index] += 1;
                    
                '''
                adapted_sufficient_statistics_mutex.acquire();
                try:
                    temp_sufficient_statistics = adapted_sufficient_statistics_dictionary[current_hyper_node._node]
                    temp_sufficient_statistics[0, nu_index] += 1;
                    adapted_sufficient_statistics_dictionary[current_hyper_node._node] = temp_sufficient_statistics;
                finally:
                    adapted_sufficient_statistics_mutex.release();
                '''
                
            return [sampled_production]
        elif isinstance(sampled_production, nltk.grammar.Production):
            # if sampled production is an pcfg production
            if pcfg_sufficient_statistics_dictionary != None:
                gamma_index = self._pcfg_production_to_gamma_index_of_lhs[current_hyper_node._node][sampled_production];
                pcfg_sufficient_statistics_dictionary[current_hyper_node._node][0, gamma_index] += 1;
                
                '''
                pcfg_sufficient_statistics_mutex.acquire();
                try:
                    temp_sufficient_statistics = pcfg_sufficient_statistics_dictionary[current_hyper_node._node]
                    temp_sufficient_statistics[0, gamma_index] += 1;
                    pcfg_sufficient_statistics_dictionary[current_hyper_node._node] = temp_sufficient_statistics;
                finally:
                    pcfg_sufficient_statistics_mutex.release();
                '''
            
            # if sampled production is a pre-terminal pcfg production
            if unsampled_hyper_nodes==None or len(unsampled_hyper_nodes)==0:
                assert (not self.is_adapted_non_terminal(current_hyper_node._node)), "adapted pre-terminal found: %s" % current_hyper_node._node
                #print "------>sampled pre-terminal production", sampled_production
                return [sampled_production]
    
            # if sampled production is a regular pcfg production
            production_list = [sampled_production];
            for unsampled_hyper_node in unsampled_hyper_nodes:
                production_list += self._sample_tree_process_dict(unsampled_hyper_node, input_string, adapted_sufficient_statistics_dictionary, new_adapted_sufficient_statistics_dictionary, pcfg_sufficient_statistics_dictionary, sufficient_statistics_dictionary);
            #print "------>formulate proudction list", production_list
    
            # if current node is a non-adapted non-terminal node
            if not self.is_adapted_non_terminal(current_hyper_node._node):
                return production_list
            
            new_adapted_production = util.AdaptedProduction(current_hyper_node._node, input_string[current_hyper_node._span[0]:current_hyper_node._span[1]], production_list);
            #print "------>insert new adapted production", new_adapted_production
            
            if new_adapted_production in self.get_adapted_productions(current_hyper_node._node, tuple(input_string[current_hyper_node._span[0]:current_hyper_node._span[1]])):
                # if this is an active adapted production
                if adapted_sufficient_statistics_dictionary != None:
                    nu_index = self._active_adapted_production_to_nu_index_of_lhs[current_hyper_node._node][new_adapted_production];
                    adapted_sufficient_statistics_dictionary[current_hyper_node._node][0, nu_index] += 1;
                    
                    '''
                    adapted_sufficient_statistics_mutex.acquire();
                    try:
                        temp_sufficient_statistics = adapted_sufficient_statistics_dictionary[current_hyper_node._node]
                        temp_sufficient_statistics[0, nu_index] += 1;
                        adapted_sufficient_statistics_dictionary[current_hyper_node._node] = temp_sufficient_statistics;
                    finally:
                        adapted_sufficient_statistics_mutex.release();
                    '''
            else:
                # if this is an inactive adapted production
                if new_adapted_sufficient_statistics_dictionary != None:
                    '''
                    if (tuple(input_string[current_hyper_node._span[0]:current_hyper_node._span[1]]), new_adapted_production) not in new_adapted_sufficient_statistics_dictionary[current_hyper_node._node]:
                        new_adapted_sufficient_statistics_dictionary[current_hyper_node._node][(tuple(input_string[current_hyper_node._span[0]:current_hyper_node._span[1]]), new_adapted_production)] = 0;
                    new_adapted_sufficient_statistics_dictionary[current_hyper_node._node][(tuple(input_string[current_hyper_node._span[0]:current_hyper_node._span[1]]), new_adapted_production)] += 1;
                    '''
                    
                    if new_adapted_production not in new_adapted_sufficient_statistics_dictionary[current_hyper_node._node]:
                        new_adapted_sufficient_statistics_dictionary[current_hyper_node._node][new_adapted_production] = 0;
                    new_adapted_sufficient_statistics_dictionary[current_hyper_node._node][new_adapted_production] += 1;                            
                    
                    '''
                    for temp_production in new_adapted_production.get_production_list():
                        if isinstance(temp_production, util.AdaptedProduction):
                            if (temp_production not in new_adapted_sufficient_statistics_dictionary[temp_production.lhs()]) and (temp_production not in self.get_adapted_productions(temp_production.lhs(), temp_production.rhs())):
                                print "Error:", temp_production, new_adapted_production;
                                print temp_production in new_adapted_sufficient_statistics_dictionary[temp_production.lhs()]
                                
                                for adapted_non_terminal in new_adapted_sufficient_statistics_dictionary:
                                    print "----------", adapted_non_terminal
                                    for key_tuple in new_adapted_sufficient_statistics_dictionary[adapted_non_terminal]:
                                        print key_tuple[1], new_adapted_sufficient_statistics_dictionary[adapted_non_terminal][key_tuple];
                                        
                            assert (temp_production in new_adapted_sufficient_statistics_dictionary[temp_production.lhs()]) or (temp_production in self.get_adapted_productions(temp_production.lhs(), temp_production.rhs()));
                    '''
                    
                    '''
                    new_adapted_sufficient_statistics_mutex.acquire();
                    try:
                        temp_sufficient_statistics = new_adapted_sufficient_statistics_dictionary[current_hyper_node._node];
                        if (tuple(input_string[current_hyper_node._span[0]:current_hyper_node._span[1]]), new_adapted_production) not in temp_sufficient_statistics:
                            temp_sufficient_statistics[(tuple(input_string[current_hyper_node._span[0]:current_hyper_node._span[1]]), new_adapted_production)] = 0;
                        temp_sufficient_statistics[(tuple(input_string[current_hyper_node._span[0]:current_hyper_node._span[1]]), new_adapted_production)] += 1;
                        new_adapted_sufficient_statistics_dictionary[current_hyper_node._node] = temp_sufficient_statistics;
                    finally:
                        new_adapted_sufficient_statistics_mutex.release();
                    '''
                    
            return [new_adapted_production];
        else:
            sys.stderr.write("Error in recognizing the production class %s @ checkpoint 2...\n" % sampled_production.__class__);
            sys.exit();

    def _format_tree_samples_dict(self, result_queue_adapted_sufficient_statistics, result_queue_new_adapted_sufficient_statistics, result_queue_pcfg_sufficient_statistics):
        adapted_sufficient_statistics = {};
        for adapted_non_terminal in self._adapted_non_terminals:
            adapted_sufficient_statistics[adapted_non_terminal] = numpy.zeros((1, len(self._nu_index_to_active_adapted_production_of_lhs[adapted_non_terminal])));

        pcfg_sufficient_statistics = {};
        for non_terminal in self._non_terminals:
            pcfg_sufficient_statistics[non_terminal] = numpy.zeros((1, len(self._gamma_index_to_pcfg_production_of_lhs[non_terminal])));
            
        for result_queue_element_index in xrange(result_queue_pcfg_sufficient_statistics.qsize()):
            pcfg_sufficient_statistics_dictionary = result_queue_pcfg_sufficient_statistics.get();
            for pcfg_production_node in self._non_terminals:
                pcfg_sufficient_statistics[pcfg_production_node] += pcfg_sufficient_statistics_dictionary[pcfg_production_node];
        
        for result_queue_element_index in xrange(result_queue_adapted_sufficient_statistics.qsize()):
            adapted_sufficient_statistics_dictionary = result_queue_adapted_sufficient_statistics.get();
            for adapted_production_node in self._ordered_adaptor_top_down[::-1]:
                adapted_sufficient_statistics[adapted_production_node] += adapted_sufficient_statistics_dictionary[adapted_production_node];
        
        for result_queue_element_index in xrange(result_queue_new_adapted_sufficient_statistics.qsize()):
            new_adapted_sufficient_statistics_dictionary = result_queue_new_adapted_sufficient_statistics.get();
            
            for adapted_production_node in self._ordered_adaptor_top_down[::-1]:
                temp_sufficient_statistics = new_adapted_sufficient_statistics_dictionary[adapted_production_node];
                    
                for new_adapted_production in temp_sufficient_statistics.keys():
                    counts = temp_sufficient_statistics[new_adapted_production];
                    
                    if new_adapted_production not in self.get_adapted_productions(new_adapted_production.lhs(), new_adapted_production.rhs()):
                        # if this is an inactive adapted production
                        assert adapted_production_node in self._adapted_non_terminals
                        
                        adapted_production_count = 0;
                        pcfg_production_count = 0;
                        for candidate_production in new_adapted_production.get_production_list():
                            if isinstance(candidate_production, util.AdaptedProduction):
                                adapted_production_count += 1;
                                #if candidate_production not in self._active_adapted_production_to_nu_index_of_lhs[candidate_production.lhs()]:
                                    #print self.get_adapted_productions(candidate_production.lhs(), candidate_production.rhs())
                                    #print self._active_adapted_production_to_nu_index_of_lhs[candidate_production.lhs()][candidate_production];
                                
                                nu_index = self._active_adapted_production_to_nu_index_of_lhs[candidate_production.lhs()][candidate_production];
                                # Warning: if you are using nltk 2.x, please use inc()
                                #self._adapted_production_usage_freqdist.inc(candidate_production, 1);
                                self._adapted_production_usage_freqdist[candidate_production] += 1;
                                self._adapted_production_dependents_of_adapted_production[candidate_production].add(new_adapted_production);
                                self._active_adapted_production_usage_counts_of_lhs[candidate_production.lhs()][0, nu_index] += 1;
                            elif isinstance(candidate_production, nltk.grammar.Production):
                                pcfg_production_count += 1;
                                gamma_index = self._pcfg_production_to_gamma_index_of_lhs[candidate_production.lhs()][candidate_production];
                                self._pcfg_production_usage_counts_of_lhs[candidate_production.lhs()][0, gamma_index] += 1;
                            else:
                                sys.stderr.write("Error in recognizing the production @ checkpoint 1...\n");
                                sys.exit();
                                
                        # activate this adapted rule
                        self._lhs_rhs_to_active_adapted_production[(adapted_production_node, new_adapted_production.rhs())].add(new_adapted_production);
                        self._lhs_to_active_adapted_production[adapted_production_node].add(new_adapted_production);
                        self._rhs_to_active_adapted_production[new_adapted_production.rhs()].add(new_adapted_production);
                        
                        self._nu_index_to_active_adapted_production_of_lhs[adapted_production_node][len(self._nu_index_to_active_adapted_production_of_lhs[adapted_production_node])] = new_adapted_production;
                        self._active_adapted_production_to_nu_index_of_lhs[adapted_production_node][new_adapted_production] = len(self._active_adapted_production_to_nu_index_of_lhs[adapted_production_node]);
            
                        self._nu_1[adapted_production_node] = numpy.hstack((self._nu_1[adapted_production_node], numpy.ones((1, 1))));
                        self._nu_2[adapted_production_node] = numpy.hstack((self._nu_2[adapted_production_node], numpy.ones((1, 1))*self._alpha_pi[adapted_production_node]));
                        
                        nu_index = self._active_adapted_production_to_nu_index_of_lhs[adapted_production_node][new_adapted_production];
                        adapted_sufficient_statistics[adapted_production_node] = numpy.hstack((adapted_sufficient_statistics[adapted_production_node], numpy.zeros((1, 1))));
                        
                        self._active_adapted_production_usage_counts_of_lhs[adapted_production_node] = numpy.hstack((self._active_adapted_production_usage_counts_of_lhs[adapted_production_node], numpy.zeros((1, 1))));
                        self._active_adapted_production_usage_counts_of_lhs[adapted_production_node][0, nu_index] = self._adapted_production_usage_freqdist[new_adapted_production];
            
                        self._active_adapted_production_length_of_lhs[adapted_production_node] = numpy.hstack((self._active_adapted_production_length_of_lhs[adapted_production_node], numpy.zeros((1, 1))));
                        #self._active_adapted_production_length_of_lhs[adapted_production_node][0, nu_index] = len(adapted_production_list.rhs());
                        #'''
                        if adapted_production_node==self._ordered_adaptor_top_down[-1]:
                            self._active_adapted_production_length_of_lhs[adapted_production_node][0, nu_index] = len(new_adapted_production.rhs());
                        else:
                            self._active_adapted_production_length_of_lhs[adapted_production_node][0, nu_index] = adapted_production_count;
                        #'''
            
                        self._active_adapted_production_sufficient_statistics_of_lhs[adapted_production_node] = numpy.hstack((self._active_adapted_production_sufficient_statistics_of_lhs[adapted_production_node], numpy.zeros((1, 1))));
                        if nu_index>self._truncation_level[adapted_production_node]:
                            self._active_adapted_production_sufficient_statistics_of_lhs[adapted_production_node][0, nu_index] = self._active_adapted_production_sufficient_statistics_of_lhs[adapted_production_node][0, nu_index-1];
                    else:
                        # if this is an active adapted production
                        nu_index = self._active_adapted_production_to_nu_index_of_lhs[adapted_production_node][new_adapted_production];

                    adapted_sufficient_statistics[adapted_production_node][0, nu_index] += counts;
                    
        return pcfg_sufficient_statistics, adapted_sufficient_statistics;
    
    def learning(self, input_strings, number_of_processes=0):
        self._counter += 1;
        self._epsilon = pow(self._tau + self._counter, -self._kappa);
        #self._epsilon = 1.0/self._counter;
        
        #self.model_state_assertion()
        clock = time.time();
        assert len(input_strings)==self._batch_size;
        if number_of_processes<=1:
            pcfg_sufficient_statistics, adapted_sufficient_statistics = self.e_step(input_strings, self._number_of_samples);
        else:
            #pcfg_sufficient_statistics, adapted_sufficient_statistics = self.e_step_process_queue(input_strings, self._number_of_samples, number_of_processes);
            pcfg_sufficient_statistics, adapted_sufficient_statistics = self.e_step_process_dict(input_strings, self._number_of_samples, number_of_processes);
            #pcfg_sufficient_statistics, adapted_sufficient_statistics = self.e_step_process_manager(input_strings, self._number_of_samples, number_of_processes);
        clock_e_step = time.time() - clock;
        #self.model_state_assertion()
        
        clock = time.time();
        self._accumulate_sufficient_statistics(pcfg_sufficient_statistics, adapted_sufficient_statistics, self._number_of_samples, self._batch_size);

        #self.model_state_assertion()
        self.m_step();
        #self.stochastic_m_step(pcfg_sufficient_statistics, adapted_sufficient_statistics);

        clock_m_step = time.time() - clock;

        if numpy.random.random()<1./self._counter or self._batch_size * self._counter % 1000 == 0:
            table_label_resampling_time = time.time()
            self.table_label_resampling();
            #self.model_state_assertion();
            table_label_resampling_time = time.time() - table_label_resampling_time;
            print "Table relabeling takes %f seconds..." % table_label_resampling_time;
            #self.prune_adapted_productions(reorder_only=True);
            #self.prune_adapted_productions()

        if self._counter % self._reorder_interval==0:
            #self.prune_adapted_productions(reorder_only=True);
            self.prune_adapted_productions();
            #self.model_state_assertion();
            
            #self.optimize_alpha_theta();
            #self.optimize_alpha_pi()

        return clock_e_step, clock_m_step
    
    def inference(self, input_strings, inference_parameter, number_of_samples=10, number_of_processes=0):
        test_clock = time.time();
        
        if number_of_processes<=1:
            self.e_step(input_strings, number_of_samples, inference_parameter);
        else:
            #self.e_step_process_queue(input_strings, number_of_samples, number_of_processes, inference_parameter);
            self.e_step_process_dict(input_strings, number_of_samples, number_of_processes, inference_parameter);
            
        test_clock = time.time() - test_clock;
        print "Inference test data takes %g seconds..." % test_clock
        
        return;
        
    def seed(self, input_strings):
        self._epsilon = pow(self._tau, -self._kappa);
        
        #self.model_state_assertion()
        clock = time.time();
        pcfg_sufficient_statistics, adapted_sufficient_statistics = self.e_step(input_strings, 1);
        clock_e_step = time.time() - clock;
        #self.model_state_assertion()
        
        clock = time.time();
        self._accumulate_sufficient_statistics(pcfg_sufficient_statistics, adapted_sufficient_statistics, 1, len(input_strings));
        #self.model_state_assertion()
        self.m_step();
        #self.stochastic_m_step(pcfg_sufficient_statistics, adapted_sufficient_statistics);
        
        self.prune_adapted_productions(reorder_only=True);
        
        #if self._counter % self._reorder_interval==0:
            #self.prune_adapted_productions()
            #self.model_state_assertion();
            
        '''
        if numpy.random.random()<0.01:
            self.optimize_alpha_pi();
            self.optimize_beta_pi();
        '''
        
        clock_m_step = time.time() - clock;
        
        return clock_e_step, clock_m_step
    
    def is_adapted_non_terminal(self, node):
        return node in self._adapted_non_terminals;
    
    def is_non_terminal(self, node):
        return node in self._non_terminals;

    def is_terminal(self, node):
        return not self.is_non_terminal(node);
    
    def is_pcfg_production(self, production):
        return isinstance(production, nltk.grammar.Production) and (production in self._pcfg_productions);
    
    def start(self):
        return self._start_symbol            

    def get_adapted_productions(self, lhs=None, rhs=None):
        assert(rhs==None or isinstance(rhs, tuple));
        assert(lhs!=None or rhs!=None);

        if lhs==None and rhs==None:
            # no constraints so return everything
            #return self._adapted_productions;
            print "Seriously?"
            sys.exit();
        elif lhs and rhs==None:
            # only lhs specified so look up its index
            return self._lhs_to_active_adapted_production[lhs];
        elif rhs and lhs==None:
            # only rhs specified so look up its index
            return self._rhs_to_active_adapted_production[rhs];
        else:
            # intersect
            return self._lhs_rhs_to_active_adapted_production[(lhs, rhs)];

    """
    Return the grammar get_pcfg_productions, filtered by the left-hand side
    or the first item in the right-hand side.

    :param lhs: Only return get_pcfg_productions with the given left-hand side.
    :param rhs: Only return get_pcfg_productions with the given first item
        in the right-hand side.
    :return: A list of get_pcfg_productions matching the given constraints.
    :rtype: list(Production)
    """
    def get_pcfg_productions(self, lhs=None, rhs=None):
        assert(rhs==None or isinstance(rhs, tuple));
        
        if lhs==None and rhs==None:
            # no constraints so return everything
            return self._pcfg_productions
        elif lhs and rhs==None:
            # only lhs specified so look up its index
            return self._lhs_to_pcfg_production[lhs];
        elif rhs and lhs==None:
            # only rhs specified so look up its index
            return self._rhs_to_pcfg_production[rhs];
        else:
            # intersect
            return self._lhs_rhs_to_pcfg_production[(lhs, rhs)];
        
    def get_unary_pcfg_productions_by_rhs(self, rhs=None):
        assert(nltk.grammar.is_nonterminal(rhs));
        return self._rhs_to_unary_pcfg_production[rhs];

    def export_adaptor_grammar(self, adaptor_grammar_path):
        E_log_stick_weights, E_log_theta = self.propose_pcfg();
        #adaptor_grammar_output = codecs.open(adaptor_grammar_path, 'w', encoding='utf-8');
        adaptor_grammar_output = open(adaptor_grammar_path, 'w');
        
        for non_terminal in self._non_terminals:
            production_freqdist = nltk.probability.FreqDist();

            for gamma_index in self._gamma_index_to_pcfg_production_of_lhs[non_terminal]:
                pcfg_production = self._gamma_index_to_pcfg_production_of_lhs[non_terminal][gamma_index];
                # Warning: if you are using nltk 2.x, please use inc()
                #production_freqdist.inc(pcfg_production, numpy.exp(E_log_theta[non_terminal][0, gamma_index]));
                production_freqdist[pcfg_production] += numpy.exp(E_log_theta[non_terminal][0, gamma_index]);
            
            if non_terminal in self._adapted_non_terminals:
                for nu_index in self._nu_index_to_active_adapted_production_of_lhs[non_terminal]:
                    adapted_production = self._nu_index_to_active_adapted_production_of_lhs[non_terminal][nu_index];
                    # Warning: if you are using nltk 2.x, please use inc()
                    #production_freqdist.inc(adapted_production, numpy.exp(E_log_stick_weights[non_terminal][0, nu_index]));
                    production_freqdist[adapted_production] += numpy.exp(E_log_stick_weights[non_terminal][0, nu_index]);
 
            adaptor_grammar_output.write("==========%s==========\n" % non_terminal);
            for production in production_freqdist:
                adaptor_grammar_output.write("%s\t%g\t%s\n" % (isinstance(production, util.AdaptedProduction), production_freqdist[production], production));
                '''
                if isinstance(production, util.AdaptedProduction):
                    nu_index = self._active_adapted_production_to_nu_index_of_lhs[non_terminal][production];
                    adaptor_grammar_output.write("%d\t%g\t%g\t%g\t%g\t%g\t%s\n" % (nu_index, self._nu_1[non_terminal][0, nu_index], self._nu_2[non_terminal][0, nu_index], production_freqdist[production], self._active_adapted_production_usage_counts_of_lhs[non_terminal][0, nu_index], self._active_adapted_production_sufficient_statistics_of_lhs[non_terminal][0, nu_index], production));
                elif isinstance(production, nltk.grammar.Production):
                    adaptor_grammar_output.write("%s\t%g\t%s\n" % ((isinstance(production, util.AdaptedProduction)), production_freqdist[production], production));
                '''
                #print production, "\t", is_adapted, "\t", production_freqdist[(production, is_adapted)];

    def export_aggregated_adaptor_grammar(self, aggregated_adaptor_grammar_path):
        E_log_stick_weights, E_log_theta = self.propose_pcfg();
        #adaptor_grammar_output = codecs.open(aggregated_adaptor_grammar_path, 'w', encoding='utf-8');
        adaptor_grammar_output = open(aggregated_adaptor_grammar_path, 'w');
        
        for non_terminal in self._non_terminals:
            production_freqdist = nltk.probability.FreqDist();

            for gamma_index in self._gamma_index_to_pcfg_production_of_lhs[non_terminal]:
                pcfg_production = self._gamma_index_to_pcfg_production_of_lhs[non_terminal][gamma_index];
                # Warning: if you are using nltk 2.x, please use inc()
                #production_freqdist.inc((pcfg_production.lhs(), pcfg_production.rhs(), False), numpy.exp(E_log_theta[non_terminal][0, gamma_index]));
                production_freqdist[(pcfg_production.lhs(), pcfg_production.rhs(), False)] += numpy.exp(E_log_theta[non_terminal][0, gamma_index]);
            
            if non_terminal in self._adapted_non_terminals:
                for nu_index in self._nu_index_to_active_adapted_production_of_lhs[non_terminal]:
                    adapted_production = self._nu_index_to_active_adapted_production_of_lhs[non_terminal][nu_index];
                    # Warning: if you are using nltk 2.x, please use inc()
                    #production_freqdist.inc((adapted_production.lhs(), adapted_production.rhs(), True), numpy.exp(E_log_stick_weights[non_terminal][0, nu_index]));
                    production_freqdist[(adapted_production.lhs(), adapted_production.rhs(), True)] += numpy.exp(E_log_stick_weights[non_terminal][0, nu_index]);
 
            adaptor_grammar_output.write("==========%s==========\n" % non_terminal);
            for (production_lhs, production_rhs, is_adapted) in production_freqdist:
                if is_adapted:
                    #print " ".join([terminal for terminal in production_rhs])
                    adaptor_grammar_output.write("%s -> %s\t%s\t%g\n" % (production_lhs, " ".join([terminal for terminal in production_rhs]), is_adapted, production_freqdist[(production_lhs, production_rhs, is_adapted)]));
                else:
                    adaptor_grammar_output.write("%s -> %s\t%s\t%g\n" % (production_lhs, production_rhs, is_adapted, production_freqdist[(production_lhs, production_rhs, is_adapted)]));

    def e_step_process_manager(self, input_strings, number_of_samples=10, number_of_processes=2, inference_parameter=None):
        if inference_parameter==None:
            reference_strings = None;
            retrieve_tokens_at_adapted_non_terminal = None;
            output_path = None;
            model_name = None;
        else:
            (reference_strings, retrieve_tokens_at_adapted_non_terminal, output_path, model_name) = inference_parameter;
            assert retrieve_tokens_at_adapted_non_terminal in self._adapted_non_terminals;
            assert len(input_strings)==len(reference_strings);
        
        # establish communication queues
        if inference_parameter==None:
            multiprocessing_manager = multiprocessing.Manager();
            pcfg_sufficient_statistics_dictionary = multiprocessing_manager.dict();
            for non_terminal in self._non_terminals:
                pcfg_sufficient_statistics_dictionary[non_terminal] = numpy.zeros((1, len(self._gamma_index_to_pcfg_production_of_lhs[non_terminal])));
            adapted_sufficient_statistics_dictionary = multiprocessing_manager.dict();
            for adapted_non_terminal in self._adapted_non_terminals:
                adapted_sufficient_statistics_dictionary[adapted_non_terminal] = numpy.zeros((1, len(self._nu_index_to_active_adapted_production_of_lhs[adapted_non_terminal])));
            new_adapted_sufficient_statistics_dictionary = multiprocessing_manager.dict();
            for adapted_non_terminal in self._adapted_non_terminals:
                new_adapted_sufficient_statistics_dictionary[adapted_non_terminal] = {};
        else:
            pcfg_sufficient_statistics_dictionary = None;
            adapted_sufficient_statistics_dictionary = None;
            new_adapted_sufficient_statistics_dictionary = None;

        E_log_stick_weights, E_log_theta = self.propose_pcfg();
        
        jobs = [];
        for input_string in input_strings:
            jobs.append(multiprocessing.Process(target=worker, args=(self,
                                                                     (input_string, None),
                                                                     E_log_stick_weights,
                                                                     E_log_theta,
                                                                     adapted_sufficient_statistics_dictionary,
                                                                     new_adapted_sufficient_statistics_dictionary,
                                                                     pcfg_sufficient_statistics_dictionary)))
            
        for j in jobs:
            j.start()
        for j in jobs:
            j.join()
        
        if inference_parameter==None:
            pcfg_sufficient_statistics = pcfg_sufficient_statistics_dictionary;
            adapted_sufficient_statistics = adapted_sufficient_statistics_dictionary;
            adapted_sufficient_statistics = self._format_tree_samples_dict(adapted_sufficient_statistics_dictionary, new_adapted_sufficient_statistics_dictionary);
            return pcfg_sufficient_statistics, adapted_sufficient_statistics;
        else:
            return;

if __name__ == '__main__':
    sufficient_statistics = numpy.random.random((1, 0))*100;
    reversed_cumulated_temp_sufficient_statistics = reverse_cumulative_sum_matrix_over_axis(sufficient_statistics, 1);
    nu_1 = sufficient_statistics + 1;
    nu_2 = reversed_cumulated_temp_sufficient_statistics + 50;
    a, b = compute_E_log_stick_weights(nu_1, nu_2)
    c = numpy.hstack((numpy.exp(a), [numpy.exp(b)]));
    print c/numpy.sum(c);
    a, b = compute_log_stick_weights(nu_1, nu_2)
    c = numpy.hstack((numpy.exp(a), [numpy.exp(b)]));
    print c/numpy.sum(c);
