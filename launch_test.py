#!/usr/bin/python
import cPickle
import string
import numpy
import getopt
import sys
import random
import time
import re
import pprint
import codecs
import datetime, os;
import scipy.io;
import nltk;
import numpy;
import optparse

# bash scrip to terminate all sub-processes 
#kill $(ps aux | grep 'python infag' | awk '{print $2}')

def parse_args():
    parser = optparse.OptionParser()
    parser.set_defaults(# parameter set 1
                        data_directory=None,
                        model_directory=None,
                        non_terminal_symbol="Word",
                        number_of_samples=10,
                        number_of_processes=0,
                        )
    # parameter set 1
    parser.add_option("--data_directory", type="string", dest="data_directory",
                      help="data directory [None]");
    parser.add_option("--model_directory", type="string", dest="model_directory",
                      help="model directory [None]");
    parser.add_option("--non_terminal_symbol", type="string", dest="non_terminal_symbol",
                      help="non-terminal symbol [Word]");
    parser.add_option("--number_of_samples", type="int", dest="number_of_samples",
                      help="number of samples [10]");
    parser.add_option("--number_of_processes", type="int", dest="number_of_processes",
                      help="number of processes [0]");

    (options, args) = parser.parse_args();
    return options;

def main():
    options = parse_args();

    # parameter set 1
    assert(options.data_directory!=None);
    data_directory = options.data_directory;
    assert(options.model_directory!=None);
    model_directory = options.model_directory;
    assert(options.non_terminal_symbol!=None);
    non_terminal_symbol = options.non_terminal_symbol;
    
    assert(options.number_of_samples>0);
    number_of_samples = options.number_of_samples;
    assert(options.number_of_processes>=0);
    number_of_processes = options.number_of_processes;

    print "========== ========== ========== ========== =========="
    # parameter set 1
    print "data_directory=" + data_directory
    print "model_directory=" + model_directory
    print "non_terminal_symbol=" + non_terminal_symbol
    
    print "number_of_samples=" + str(number_of_samples)
    print "number_of_processes=" + str(number_of_processes)
    print "========== ========== ========== ========== =========="

    # Documents
    train_docs = [];
    input_stream = open(os.path.join(data_directory, 'train.dat'), 'r');
    for line in input_stream:
        train_docs.append(line.strip());
    input_stream.close();
    print "successfully load %d training documents..." % (len(train_docs))

    test_docs = [];
    input_stream = open(os.path.join(data_directory, 'test.dat'), 'r');
    for line in input_stream:
        test_docs.append(line.strip());
    input_stream.close()
    print "successfully load %d testing documents..." % (len(test_docs))    

    for model_file in os.listdir(model_directory):
        if not model_file.startswith("model-"):
            continue;
        
        model_file_path = os.path.join(model_directory, model_file);
        
        try:
            cpickle_file = open(model_file_path, 'r');
            infinite_adaptor_grammar = cPickle.load(cpickle_file);
            print "successfully load model from %s" % (model_file_path);
            cpickle_file.close();
        except ValueError:
            print "warning: unsuccessfully load model from %s due to value error..." % (model_file_path);
            continue;
        except EOFError:
            print "warning: unsuccessfully load model from %s due to EOF error..." % (model_file_path);
            continue;
        
        non_terminal = nltk.grammar.Nonterminal(non_terminal_symbol);
        assert(non_terminal in infinite_adaptor_grammar._adapted_non_terminals);
        
        inference_parameter = (test_docs, non_terminal, model_directory, model_file);
        infinite_adaptor_grammar.inference(train_docs, inference_parameter, number_of_samples, number_of_processes);

        '''    
        #from launch_train import shuffle_lists
        #shuffle_lists(train_docs, test_docs);
        if number_of_processes==0:
            infinite_adaptor_grammar.inference(train_docs, test_docs, non_terminal, model_directory, model_file, number_of_samples);
        else:
            from hybrid_process import inference_process;
            inference_process(infinite_adaptor_grammar, train_docs, test_docs, non_terminal, model_directory, number_of_samples, number_of_processes);
        '''
        
if __name__ == '__main__':
    main()