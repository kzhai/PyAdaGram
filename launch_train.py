#!/usr/bin/python
import pickle
import string
import numpy
import getopt
import sys
import random
import time
import math
import re
import pprint
import codecs
import datetime
import optparse
import os
import nltk
import numpy
import scipy
import scipy.io
import collections


# bash scrip to terminate all sub-processes
# kill $(ps aux | grep 'python infag' | awk '{print $2}')

def shuffle_lists(list1, list2):
    assert (len(list1) == len(list2))
    list1_shuf = []
    list2_shuf = []
    index_shuf = list(range(len(list1)))
    random.shuffle(index_shuf)
    for i in index_shuf:
        list1_shuf.append(list1[i])
        list2_shuf.append(list2[i])


def parse_args():
    parser = optparse.OptionParser()
    parser.set_defaults(
        # parameter set 1
        input_directory=None,
        output_directory=None,
        # corpus_name=None,
        grammar_file=None,

        # parameter set 2
        # number_of_topics=25,
        number_of_documents=-1,
        batch_size=-1,
        training_iterations=-1,
        number_of_processes=0,
        # multiprocesses=False,

        # parameter set 3
        grammaton_prune_interval=10,
        snapshot_interval=10,
        kappa=0.75,
        tau=64.0,

        # parameter set 4
        # desired_truncation_level={},
        # alpha_theta={},
        # alpha_pi={},
        # beta_pi={},

        # parameter set 5
        train_only=False,
        heldout_data=0.0,
        enable_word_model=False

        # fix_vocabulary=False
    )
    # parameter set 1
    parser.add_option("--input_directory", type="string", dest="input_directory",
                      help="input directory [None]")
    parser.add_option("--output_directory", type="string", dest="output_directory",
                      help="output directory [None]")
    # parser.add_option("--corpus_name", type="string", dest="corpus_name",
    # help="the corpus name [None]")
    parser.add_option("--grammar_file", type="string", dest="grammar_file",
                      help="the grammar file [None]")

    # parameter set 2
    # parser.add_option("--number_of_topics", type="int", dest="number_of_topics",
    # help="second level truncation [25]")
    parser.add_option("--number_of_documents", type="int", dest="number_of_documents",
                      help="number of documents [-1]")
    parser.add_option("--batch_size", type="int", dest="batch_size",
                      help="batch size [-1 in batch mode]")
    parser.add_option("--training_iterations", type="int", dest="training_iterations",
                      help="max iteration to run training [number_of_documents/batch_size]")
    parser.add_option("--number_of_processes", type="int", dest="number_of_processes",
                      help="number of processes [0]")
    # parser.add_option("--multiprocesses", action="store_true", dest="multiprocesses",
    # help="multiprocesses [false]")

    # parameter set 3
    parser.add_option("--kappa", type="float", dest="kappa",
                      help="learning rate [0.5]")
    parser.add_option("--tau", type="float", dest="tau",
                      help="slow down [1.0]")
    parser.add_option("--grammaton_prune_interval", type="int", dest="grammaton_prune_interval",
                      help="vocabuary rank interval [10]")
    parser.add_option("--snapshot_interval", type="int", dest="snapshot_interval",
                      help="snapshot interval [grammaton_prune_interval]")

    # parameter set 4
    '''
    parser.add_option("--desired_truncation_level", dest="desired_truncation_level",
                      action="callback", callback=process_ints, help="desired truncation level")    
    parser.add_option("--alpha_theta", dest="alpha_theta",
                      action="callback", callback=process_floats,
                      help="hyper-parameter for Dirichlet distribution of PCFG productions [1.0/number_of_pcfg_productions]")
    parser.add_option("--alpha_pi", dest="alpha_pi",
                      action="callback", callback=process_floats,
                      help="hyper-parameter for Pitman-Yor process")
    parser.add_option("--beta_pi", dest="beta_pi",
                      action="callback", callback=process_floats,
                      help="hyper-parameter for Pitman-Yor process")
    '''

    # parameter set 5
    parser.add_option("--train_only", action="store_true", dest="train_only",
                      help="train mode only [false]")
    parser.add_option("--heldout_data", type="int", dest="heldout_data",
                      help="portion of heldout data [0.0]")
    parser.add_option("--enable_word_model", action="store_true", dest="enable_word_model",
                      help="enable word model [false]")

    '''
    parser.add_option("--fix_vocabulary", action="store_true", dest="fix_vocabulary",
                      help="run this program with fix vocabulary")
    '''

    (options, args) = parser.parse_args()
    return options


def main():
    options = parse_args()

    # parameter set 1
    # assert(options.corpus_name!=None)
    assert (options.input_directory is not None)
    assert (options.output_directory is not None)

    input_directory = options.input_directory
    input_directory = input_directory.rstrip("/")
    corpus_name = os.path.basename(input_directory)

    output_directory = options.output_directory
    if not os.path.exists(output_directory):
        os.mkdir(output_directory)
    output_directory = os.path.join(output_directory, corpus_name)
    if not os.path.exists(output_directory):
        os.mkdir(output_directory)

    assert (options.grammar_file is not None)
    grammar_file = options.grammar_file
    assert (os.path.exists(grammar_file))

    # Documents
    train_docs = []
    input_stream = open(os.path.join(input_directory, 'train.dat'), 'r')
    for line in input_stream:
        train_docs.append(line.strip())
    input_stream.close()
    print("successfully load all training documents...")

    # parameter set 2
    if options.number_of_documents > 0:
        number_of_documents = options.number_of_documents
    else:
        number_of_documents = len(train_docs)
    if options.batch_size > 0:
        batch_size = options.batch_size
    else:
        batch_size = number_of_documents
    # assert(number_of_documents % batch_size==0)
    training_iterations = number_of_documents // batch_size
    if options.training_iterations > 0:
        training_iterations = options.training_iterations
    # training_iterations=int(math.ceil(1.0*number_of_documents/batch_size))
    # multiprocesses = options.multiprocesses
    assert (options.number_of_processes >= 0)
    number_of_processes = options.number_of_processes

    # parameter set 3
    assert (options.grammaton_prune_interval > 0)
    grammaton_prune_interval = options.grammaton_prune_interval
    snapshot_interval = grammaton_prune_interval
    if options.snapshot_interval > 0:
        snapshot_interval = options.snapshot_interval
    assert (options.tau >= 0)
    tau = options.tau
    # assert(options.kappa>=0.5 and options.kappa<=1)
    assert (options.kappa >= 0 and options.kappa <= 1)
    kappa = options.kappa
    if batch_size <= 0:
        print("warning: running in batch mode...")
        kappa = 0

    # read in adaptor grammars
    desired_truncation_level = {}
    alpha_pi = {}
    beta_pi = {}

    grammar_rules = []
    adapted_non_terminals = set()
    # for line in codecs.open(grammar_file, 'r', encoding='utf-8'):
    for line in open(grammar_file, 'r'):
        line = line.strip()
        if line.startswith("%"):
            continue
        if line.startswith("@"):
            tokens = line.split()
            assert (len(tokens) == 5)
            adapted_non_terminal = nltk.Nonterminal(tokens[1])
            adapted_non_terminals.add(adapted_non_terminal)
            desired_truncation_level[adapted_non_terminal] = int(tokens[2])
            alpha_pi[adapted_non_terminal] = float(tokens[3])
            beta_pi[adapted_non_terminal] = float(tokens[4])
            continue
        grammar_rules.append(line)
    grammar_rules = "\n".join(grammar_rules)

    # Warning: if you are using nltk 2.x, please use parse_grammar()
    # from nltk.grammar import parse_grammar, standard_nonterm_parser
    # start, productions = parse_grammar(grammar_rules, standard_nonterm_parser, probabilistic=False)
    from nltk.grammar import read_grammar, standard_nonterm_parser
    start, productions = read_grammar(grammar_rules, standard_nonterm_parser, probabilistic=False)

    # create output directory
    now = datetime.datetime.now()
    suffix = now.strftime("%y%b%d-%H%M%S") + ""
    # desired_truncation_level_string = "".join(["%s%d" % (symbol, desired_truncation_level[symbol]) for symbol in desired_truncation_level])
    # alpha_pi_string = "".join(["%s%d" % (symbol, alpha_pi[symbol]) for symbol in alpha_pi])
    # beta_pi_string = "".join(["%s%d" % (symbol, beta_pi[symbol]) for symbol in beta_pi])
    # output_directory += "-" + str(now.microsecond) + "/"
    suffix += "-D%d-P%d-S%d-B%d-O%d-t%d-k%g-G%s/" % (number_of_documents,
                                                     # number_of_topics,
                                                     grammaton_prune_interval,
                                                     snapshot_interval,
                                                     batch_size,
                                                     training_iterations,
                                                     tau,
                                                     kappa,
                                                     # alpha_theta,
                                                     # alpha_pi_string,
                                                     # beta_pi_string,
                                                     # desired_truncation_level_string,
                                                     os.path.basename(grammar_file)
                                                     )

    output_directory = os.path.join(output_directory, suffix)
    os.mkdir(os.path.abspath(output_directory))

    # store all the options to a input_stream
    options_output_file = open(output_directory + "option.txt", 'w')
    # parameter set 1
    options_output_file.write("input_directory=" + input_directory + "\n")
    options_output_file.write("corpus_name=" + corpus_name + "\n")
    options_output_file.write("grammar_file=" + str(grammar_file) + "\n")
    # parameter set 2
    options_output_file.write("number_of_processes=" + str(number_of_processes) + "\n")
    # options_output_file.write("multiprocesses=" + str(multiprocesses) + "\n")
    options_output_file.write("number_of_documents=" + str(number_of_documents) + "\n")
    options_output_file.write("batch_size=" + str(batch_size) + "\n")
    options_output_file.write("training_iterations=" + str(training_iterations) + "\n")

    # parameter set 3
    options_output_file.write("grammaton_prune_interval=" + str(grammaton_prune_interval) + "\n")
    options_output_file.write("snapshot_interval=" + str(snapshot_interval) + "\n")
    options_output_file.write("tau=" + str(tau) + "\n")
    options_output_file.write("kappa=" + str(kappa) + "\n")

    # parameter set 4
    # options_output_file.write("alpha_theta=" + str(alpha_theta) + "\n")
    options_output_file.write("alpha_pi=%s\n" % alpha_pi)
    options_output_file.write("beta_pi=%s\n" % beta_pi)
    options_output_file.write("desired_truncation_level=%s\n" % desired_truncation_level)
    # parameter set 5    
    # options_output_file.write("heldout_data=" + str(heldout_data) + "\n")
    options_output_file.close()

    print("========== ========== ========== ========== ==========")
    # parameter set 1
    print(("output_directory=" + output_directory))
    print(("input_directory=" + input_directory))
    print(("corpus_name=" + corpus_name))
    print(("grammar_file=" + str(grammar_file)))

    # parameter set 2
    print(("number_of_documents=" + str(number_of_documents)))
    print(("batch_size=" + str(batch_size)))
    print(("training_iterations=" + str(training_iterations)))
    print(("number_of_processes=" + str(number_of_processes)))
    # print "multiprocesses=" + str(multiprocesses)

    # parameter set 3
    print(("grammaton_prune_interval=" + str(grammaton_prune_interval)))
    print(("snapshot_interval=" + str(snapshot_interval)))
    print(("tau=" + str(tau)))
    print(("kappa=" + str(kappa)))

    # parameter set 4
    # print "alpha_theta=" + str(alpha_theta)
    print(("alpha_pi=%s" % alpha_pi))
    print(("beta_pi=%s" % beta_pi))
    print(("desired_truncation_level=%s" % desired_truncation_level))
    # parameter set 5
    # print "heldout_data=" + str(heldout_data)
    print("========== ========== ========== ========== ==========")

    import hybrid
    adagram_inferencer = hybrid.Hybrid(start,
                                       productions,
                                       adapted_non_terminals
                                       )

    adagram_inferencer._initialize(number_of_documents,
                                   batch_size,
                                   tau,
                                   kappa,
                                   alpha_pi,
                                   beta_pi,
                                   None,
                                   desired_truncation_level,
                                   grammaton_prune_interval
                                   )

    '''
    clock_iteration = time.time()
    clock_e_step, clock_m_step = adagram_inferencer.seed(train_docs)
    clock_iteration = time.time()-clock_iteration
    print 'E-step, M-step and Seed take %g, %g and %g seconds respectively...' % (clock_e_step, clock_m_step, clock_iteration)p
    '''

    # adagram_inferencer.export_adaptor_grammar(os.path.join(output_directory, "infag-0"))
    # adagram_inferencer.export_aggregated_adaptor_grammar(os.path.join(output_directory, "ag-0"))

    random.shuffle(train_docs)
    training_clock = time.time()
    snapshot_clock = time.time()
    for iteration in range(training_iterations):
        start_index = batch_size * iteration
        end_index = batch_size * (iteration + 1)
        if start_index // number_of_documents < end_index // number_of_documents:
            # train_doc_set = train_docs[(batch_size * iteration) % (number_of_documents) :] + train_docs[: (batch_size * (iteration+1)) % (number_of_documents)]
            train_doc_set = train_docs[(batch_size * iteration) % (number_of_documents):]
            random.shuffle(train_docs)
            train_doc_set += train_docs[: (batch_size * (iteration + 1)) % (number_of_documents)]
        else:
            train_doc_set = train_docs[(batch_size * iteration) % (number_of_documents): (batch_size * (
                    iteration + 1)) % number_of_documents]

        clock_iteration = time.time()
        # print "processing document:", train_doc_set
        clock_e_step, clock_m_step = adagram_inferencer.learning(train_doc_set, number_of_processes)

        if (iteration + 1) % snapshot_interval == 0:
            # cpickle_file = open(os.path.join(output_directory, "model-%d" % (adagram_inferencer._counter+1)), 'wb')
            # cPickle.dump(adagram_inferencer, cpickle_file)
            # cpickle_file.close()
            adagram_inferencer.export_adaptor_grammar(os.path.join(output_directory, "adagram-" + str((iteration + 1))))
            # adagram_inferencer.export_aggregated_adaptor_grammar(os.path.join(output_directory, "ag-" + str((iteration+1))))

        if (iteration + 1) % 1000 == 0:
            snapshot_clock = time.time() - snapshot_clock
            print(('Processing 1000 mini-batches take %g seconds...' % (snapshot_clock)))
            snapshot_clock = time.time()

        clock_iteration = time.time() - clock_iteration
        print(('E-step, M-step and iteration %d take %g, %g and %g seconds respectively...' % (
            adagram_inferencer._counter, clock_e_step, clock_m_step, clock_iteration)))

    adagram_inferencer.export_adaptor_grammar(
        os.path.join(output_directory, "adagram-" + str(adagram_inferencer._counter + 1)))
    # adagram_inferencer.export_aggregated_adaptor_grammar(os.path.join(output_directory, "ag-" + str((iteration+1))))

    cpickle_file = open(os.path.join(output_directory, "model-%d" % (iteration + 1)), 'wb')
    pickle.dump(adagram_inferencer, cpickle_file)
    cpickle_file.close()

    training_clock = time.time() - training_clock
    print(('Training finished in %g seconds...' % (training_clock)))


if __name__ == '__main__':
    main()
