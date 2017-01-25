#!/usr/bin/python
import cPickle
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
import os;
import scipy.io;
import nltk;
import numpy;
import collections;
import optparse

# bash scrip to terminate all sub-processes 
#kill $(ps aux | grep 'python infag' | awk '{print $2}')

def parse_args():
    parser = optparse.OptionParser()
    parser.set_defaults(# parameter set 1
                        input_directory=None,
                        output_directory=None,
                        corpus_name=None,
                        model_directory=None,
                        
                        # parameter set 2
                        online_iterations=-1,
                        snapshot_interval=-1,
                        number_of_processes=0,
                        )
    # parameter set 1
    parser.add_option("--input_directory", type="string", dest="input_directory",
                      help="input directory [None]");
    parser.add_option("--output_directory", type="string", dest="output_directory",
                      help="output directory [None]");
    parser.add_option("--corpus_name", type="string", dest="corpus_name",
                      help="the corpus name [None]")
    parser.add_option("--model_directory", type="string", dest="model_directory",
                      help="the model directory [None]")
    
    # parameter set 2
    parser.add_option("--online_iterations", type="int", dest="online_iterations",
                      help="resume iteration to run training [nonpos=previous settings]");
    parser.add_option("--number_of_processes", type="int", dest="number_of_processes",
                      help="number of processes [0]");                      
    parser.add_option("--snapshot_interval", type="int", dest="snapshot_interval",
                      help="snapshot interval [nonpos=previous settings]");
                      
    (options, args) = parser.parse_args();
    return options;

model_setting_pattern = re.compile(r"\w+\-\d+\-D\d+\-P\d+\-S(?P<snapshot>\d+)\-B\d+\-O(?P<online>\d+)\-t\d+\-k[\d\.]+\-G\w+\-T[\w\&\.]+\-ap[\w\&\.]+\-bp[\w\&\.]+");
snapshot_pattern = re.compile(r"\-S(?P<message>\d+)\-");
online_pattern = re.compile(r"\-O(?P<message>\d+)\-");

def main():
    options = parse_args();
    
    # parameter set 1
    assert(options.corpus_name!=None);
    assert(options.input_directory!=None);
    assert(options.output_directory!=None);
    assert(options.model_directory!=None);
    corpus_name = options.corpus_name;
    input_directory = options.input_directory;
    input_directory = os.path.join(input_directory, corpus_name);
    output_directory = options.output_directory;
    if not os.path.exists(output_directory):
        os.mkdir(output_directory);
    output_directory = os.path.join(output_directory, corpus_name);
    if not os.path.exists(output_directory):
        os.mkdir(output_directory);    
    model_directory = options.model_directory;
    if not model_directory.endswith("/"):
        model_directory += "/";
    
    # look for model snapshot
    model_setting = os.path.basename(os.path.dirname(model_directory));
    model_pattern_match_object = re.match(model_setting_pattern, model_setting);
    model_pattern_match_dictionary = model_pattern_match_object.groupdict();
    previous_online_iterations = int(model_pattern_match_dictionary["online"]);
    previous_snapshot_interval = int(model_pattern_match_dictionary["snapshot"]);
    model_file_path = os.path.join(model_directory, "model-%d" % (previous_online_iterations))
    
    # load model snapshot
    try:
        cpickle_file = open(model_file_path, 'r');
        infinite_adaptor_grammar = cPickle.load(cpickle_file);
        print "successfully load model from %s" % (model_directory);
        cpickle_file.close();
    except ValueError:
        print "warning: unsuccessfully load model from %s due to value error..." % (model_file_path);
        return;
    except EOFError:
        print "warning: unsuccessfully load model from %s due to EOF error..." % (model_file_path);
        return;
    
    batch_size = infinite_adaptor_grammar._batch_size;
    number_of_documents = infinite_adaptor_grammar._number_of_strings;
    
    # parameter set 2
    online_iterations=number_of_documents/batch_size;
    if options.online_iterations>0:
        online_iterations=options.online_iterations;
    assert(options.number_of_processes>=0);
    number_of_processes = options.number_of_processes;
    snapshot_interval = previous_snapshot_interval;
    if options.snapshot_interval>0:
        snapshot_interval=options.snapshot_interval;

    # adjust model output path name
    model_setting = re.sub(snapshot_pattern, "-S%d-" % (snapshot_interval), model_setting);
    model_setting = re.sub(online_pattern, "-O%d-" % (online_iterations+previous_online_iterations), model_setting);
    
    output_directory = os.path.join(output_directory, model_setting);
    os.mkdir(os.path.abspath(output_directory));
    
    # store all the options to a output stream
    options_output_file = open(os.path.join(output_directory, "option.txt"), 'w');
    # parameter set 1
    options_output_file.write("input_directory=" + input_directory + "\n");
    options_output_file.write("corpus_name=" + corpus_name + "\n");
    options_output_file.write("model_directory=" + model_directory + "\n");
    # parameter set 2
    options_output_file.write("snapshot_interval=" + str(snapshot_interval) + "\n");
    options_output_file.write("online_iterations=" + str(online_iterations) + "\n");    
    options_output_file.write("number_of_processes=" + str(number_of_processes) + "\n");
    # parameter set 3
    options_output_file.write("number_of_documents=" + str(number_of_documents) + "\n");
    options_output_file.write("batch_size=" + str(batch_size) + "\n");
    options_output_file.close()
    
    print "========== ========== ========== ========== =========="
    # parameter set 1
    print "input_directory=" + input_directory;
    print "corpus_name=" + corpus_name;
    print "model_directory=" + model_directory;
    # parameter set 2
    print "snapshot_interval=" + str(snapshot_interval);
    print "online_iterations=" + str(online_iterations);
    print "number_of_processes=" + str(number_of_processes);
    # parameter set 3
    print "number_of_documents=" + str(number_of_documents);
    print "batch_size=" + str(batch_size);
    print "========== ========== ========== ========== =========="
    
    # Documents
    train_docs = [];
    input_stream = open(os.path.join(input_directory, 'train.dat'), 'r');
    for line in input_stream:
        train_docs.append(line.strip());
    input_stream.close();
    print "successfully load all training documents..."
    
    random.shuffle(train_docs);    
    training_clock = time.time();
    snapshot_clock = time.time();
    for iteration in xrange(previous_online_iterations, previous_online_iterations+online_iterations):
        start_index = batch_size * iteration;
        end_index = batch_size * (iteration + 1);
        if start_index / number_of_documents < end_index / number_of_documents:
            #train_doc_set = train_docs[(batch_size * iteration) % (number_of_documents) :] + train_docs[: (batch_size * (iteration+1)) % (number_of_documents)];
            train_doc_set = train_docs[(batch_size * iteration) % (number_of_documents) :];
            random.shuffle(train_docs);
            train_doc_set += train_docs[: (batch_size * (iteration+1)) % (number_of_documents)];
        else:
            train_doc_set = train_docs[(batch_size * iteration) % (number_of_documents) : (batch_size * (iteration+1)) % number_of_documents];

        clock_iteration = time.time();
        #print "processing document:", train_doc_set
        clock_e_step, clock_m_step = infinite_adaptor_grammar.learning(train_doc_set, number_of_processes);
        
        if (iteration+1)%snapshot_interval==0:
            #cpickle_file = open(os.path.join(output_directory, "model-%d" % (iteration+1)), 'wb');
            #cPickle.dump(infinite_adaptor_grammar, cpickle_file);
            #cpickle_file.close();
            infinite_adaptor_grammar.export_adaptor_grammar(os.path.join(output_directory, "adagram-" + str((iteration+1))))
            #infinite_adaptor_grammar.export_aggregated_adaptor_grammar(os.path.join(output_directory, "ag-" + str((iteration+1))))
        
        if (iteration+1) % 1000==0:
            snapshot_clock = time.time() - snapshot_clock;
            print 'Processing 1000 mini-batches take %g seconds...' % (snapshot_clock);
            snapshot_clock = time.time()
    
        clock_iteration = time.time()-clock_iteration;
        print 'E-step, M-step and iteration %d take %g, %g and %g seconds respectively...' % (infinite_adaptor_grammar._counter, clock_e_step, clock_m_step, clock_iteration);
    
    infinite_adaptor_grammar.export_adaptor_grammar(os.path.join(output_directory, "adagram-" + str((iteration+1))))
    #infinite_adaptor_grammar.export_aggregated_adaptor_grammar(os.path.join(output_directory, "ag-" + str((iteration+1))))

    cpickle_file = open(os.path.join(output_directory, "model-%d" % (iteration+1)), 'wb');
    cPickle.dump(infinite_adaptor_grammar, cpickle_file);
    cpickle_file.close();
    
    training_clock = time.time()-training_clock;
    print 'Training finished in %g seconds...' % (training_clock);

if __name__ == '__main__':
    main();
