import optparse;

delimiter = '-';

def floatable(str):
    try:
        float(str)
        return True
    except ValueError:
        return False
    
def intable(str):
    try:
        int(str)
        return True
    except ValueError:
        return False

def process_floats(option, opt_str, value, parser):
    assert value is None
    value = {};

    for arg in parser.rargs:
        # stop on --foo like options
        if arg[:2] == "--" and len(arg) > 2:
            break
        # stop on -a, but not on -3 or -3.0
        if arg[:1] == "-" and len(arg) > 1 and not floatable(arg):
            break
        
        tokens = arg.split("=");
        value[tokens[0]]=float(tokens[1]);
        
    del parser.rargs[:len(value)]
    setattr(parser.values, option.dest, value)
    
    return

def process_ints(option, opt_str, value, parser):
    assert value is None
    value = {};

    for arg in parser.rargs:
        # stop on --foo like options
        if arg[:2] == "--" and len(arg) > 2:
            break
        # stop on -a, but not on -3 or -3.0
        if arg[:1] == "-" and len(arg) > 1 and not int(arg):
            break
        
        tokens = arg.split("=");
        value[tokens[0]]=int(tokens[1]);
        
    del parser.rargs[:len(value)]
    setattr(parser.values, option.dest, value)
    
    return
    
def parse_args():
    parser = optparse.OptionParser()
    parser.set_defaults(# parameter set 1
                        input_directory=None,
                        output_directory=None,
                        #corpus_name=None,
                        grammar_file=None,
                        
                        # parameter set 2
                        #number_of_topics=25,
                        number_of_documents=-1,
                        batch_size=-1,
                        training_iterations=-1,
                        number_of_processes=0,
                        #multiprocesses=False,

                        # parameter set 3
                        grammaton_prune_interval=10,
                        snapshot_interval=10,
                        kappa=0.75,
                        tau=64.0,
                        
                        # parameter set 4
                        #desired_truncation_level={},
                        #alpha_theta={},
                        #alpha_pi={},
                        #beta_pi={},
                        
                        # parameter set 5
                        train_only=False,
                        heldout_data=0.0,
                        enable_word_model=False

                        #fix_vocabulary=False
                        )
    # parameter set 1
    parser.add_option("--input_directory", type="string", dest="input_directory",
                      help="input directory [None]");
    parser.add_option("--output_directory", type="string", dest="output_directory",
                      help="output directory [None]");
    #parser.add_option("--corpus_name", type="string", dest="corpus_name",
                      #help="the corpus name [None]")
    parser.add_option("--grammar_file", type="string", dest="grammar_file",
                      help="the grammar file [None]")
    
    # parameter set 2
    #parser.add_option("--number_of_topics", type="int", dest="number_of_topics",
                    #help="second level truncation [25]");
    parser.add_option("--number_of_documents", type="int", dest="number_of_documents",
                      help="number of documents [-1]");
    parser.add_option("--batch_size", type="int", dest="batch_size",
                      help="batch size [-1 in batch mode]");
    parser.add_option("--training_iterations", type="int", dest="training_iterations",
                      help="max iteration to run training [number_of_documents/batch_size]");
    parser.add_option("--number_of_processes", type="int", dest="number_of_processes",
                      help="number of processes [0]");
    #parser.add_option("--multiprocesses", action="store_true", dest="multiprocesses",
                      #help="multiprocesses [false]");                      

    # parameter set 3
    parser.add_option("--kappa", type="float", dest="kappa",
                      help="learning rate [0.5]")
    parser.add_option("--tau", type="float", dest="tau",
                      help="slow down [1.0]")    
    parser.add_option("--grammaton_prune_interval", type="int", dest="grammaton_prune_interval",
                      help="vocabuary rank interval [10]");
    parser.add_option("--snapshot_interval", type="int", dest="snapshot_interval",
                      help="snapshot interval [grammaton_prune_interval]");
                      
    # parameter set 4
    '''
    parser.add_option("--desired_truncation_level", dest="desired_truncation_level",
                      action="callback", callback=process_ints, help="desired truncation level");    
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
                      help="train mode only [false]");
    parser.add_option("--heldout_data", type="int", dest="heldout_data",
                      help="portion of heldout data [0.0]")
    parser.add_option("--enable_word_model", action="store_true", dest="enable_word_model",
                      help="enable word model [false]");

    '''
    parser.add_option("--fix_vocabulary", action="store_true", dest="fix_vocabulary",
                      help="run this program with fix vocabulary");
    '''

    (options, args) = parser.parse_args();
    return options;