import collections
import nltk
import numpy
import scipy
import sys
import math
import time

"""
"""
def log_add(log_a, log_b):
    if log_a < log_b:
        return log_b + numpy.log(1 + numpy.exp(log_a - log_b))
    else:
        return log_a + numpy.log(1 + numpy.exp(log_b - log_a))

class GraphNode():
    def __init__(self, non_terminal, next_nodes=set()): 
        self._non_terminal = non_terminal
        self._next_nodes = next_nodes;
        
    def add_node(self, next_node):
        assert(isinstance(next_node, self.__class__));
        self._next_nodes.add(next_node);

class AdaptedProduction(nltk.grammar.Production):
    def __init__(self, lhs, rhs, productions):
        super(AdaptedProduction, self).__init__(lhs, rhs);
        self._productions = productions;
        productions_hash = 0;
        for production in productions:
            productions_hash /= 1e9;
            productions_hash += production.__hash__();
        self._hash = hash((self._lhs, self._rhs, productions_hash));
        #self._hash = hash((self._lhs, self._rhs, "%f%f" % (time.time(), numpy.random.random())));

    def get_production_list(self):
        return self._productions;

    def match_grammaton(self, production_list):
        if len(self._productions)!=len(production_list):
            return False;
        for x in xrange(len(self._productions)):
            if self._productions[x]!=production_list[x]:
                return False;
        return True;

    def __eq__(self, other):
        """
        @return: true if this C{Production} is equal to C{other}.
        @rtype: C{boolean}
        """
        if not isinstance(other, self.__class__):
            return False;
        if self._lhs!=other._lhs:
            return False;
        if self._rhs!=other._rhs:
            return False;
        return self.match_grammaton(other._productions);

    def __str__(self):
        str = "%s -> %s (" % (self._lhs, " ".join(["%s" % elt for elt in self._rhs]));
        str += ", ".join(["%s" % production for production in self._productions])
        str += ")"
        return str;
    
    def retrieve_tokens_of_adapted_non_terminal(self, adapted_non_terminal):
        if self.lhs()==adapted_non_terminal:
            return ["".join(self.rhs())];
        else:
            token_list = [];
            for candidate_production in self._productions:
                if isinstance(candidate_production, self.__class__):
                    token_list += candidate_production.retrieve_tokens_of_adapted_non_terminal(adapted_non_terminal);
                else:
                    continue;
            return token_list;
        
class HyperNode():
    def __init__(self,
                 node,
                 span):
        self._node = node;
        self._span = span;
        
        self._derivation = [];
        self._log_probability = [];
        
        # TODO: this would incur duplication in derivation
        #self._derivation_log_probability = {}
        
        self._accumulated_log_probability = float('NaN');
    
    def add_new_derivation(self, production, log_probability, hyper_nodes=None):
        self._derivation.append((production, hyper_nodes));
        self._log_probability.append(log_probability);

        '''
        if (production, hyper_nodes) not in self._derivation_log_probability:
            self._derivation_log_probability[(production, hyper_nodes)] = log_probability;
        else:
            self._derivation_log_probability[(production, hyper_nodes)] = log_add(log_probability, self._derivation_log_probability[(production, hyper_nodes)]);
        '''
            
        if math.isnan(self._accumulated_log_probability):
            self._accumulated_log_probability = log_probability;
        else:
            self._accumulated_log_probability = log_add(log_probability, self._accumulated_log_probability);
            
        return;
    
    def random_sample_derivation(self):
        #print self._derivation_log_probability.keys()
        random_number = numpy.random.random();
        
        '''
        for (production, hyper_nodes) in self._derivation_log_probability:
            current_probability = numpy.exp(self._derivation_log_probability[(production, hyper_nodes)] - self._accumulated_log_probability);
            if random_number>current_probability:
                random_number -= current_probability;
            else:
                return production, hyper_nodes
        '''
        
        #print "<<<<<<<<<<debug>>>>>>>>>>"
        #for x in xrange(len(self._derivation)):
            #print self._derivation[x], numpy.exp(self._log_probability[x] - self._accumulated_log_probability);
        #sys.exit();
        
        assert(len(self._derivation)==len(self._log_probability))
        for x in xrange(len(self._derivation)):
            current_probability = numpy.exp(self._log_probability[x] - self._accumulated_log_probability);
            if random_number>current_probability:
                random_number -= current_probability;
            else:
                #return self._derivation[x][0], self._derivation[x][1]
                return self._derivation[x][0], self._derivation[x][1], self._log_probability[x] - self._accumulated_log_probability
            
    def __len__(self):
        #return len(self._derivation_log_probability);
        return len(self._log_probability);
        
    def __str__(self):
        output_string = "[%s (%d:%d) " % (self._node, self._span[0], self._span[1]);

        '''
        for (production, hyper_nodes) in self._derivation_log_probability:
            
            output_string += "<%s" % (production);
            print production, len(hyper_nodes);
            if hyper_nodes!=None:
                print len(hyper_nodes)
                for hyper_node in hyper_nodes:
                    output_string += " %s" % (hyper_node);
            output_string += "> ";
        return output_string
        '''
        
        for x in xrange(len(self._derivation)):
            production = self._derivation[x][0];
            hyper_nodes = self._derivation[x][1];
            
            output_string += "<%s" % (production);
            #print production, len(hyper_nodes);
            if hyper_nodes!=None:
                #print len(hyper_nodes)
                for hyper_node in hyper_nodes:
                    output_string += " %s" % (hyper_node);
            output_string += "> ";
        return output_string

    def __repr__(self):
        return self.__str__()
    
    def __hash__(self):
        return hash((self._node, self._span));

class PassiveEdge():
    def __init__(self,
                 node,
                 left,
                 right
                 ):
        self._node = node;
        self._left = left;
        self._right = right;
        
class ActiveEdge():
    def __init__(self,
                 lhs,
                 rhs,
                 left,
                 right,
                 parsed=0
                 ):
        self._lhs = lhs;
        self._rhs = rhs;
        self._left = left;
        self._right = right;
        self._parsed = parsed;
        
class HyperGraph():
    def __init__(self):
        self._top_down_approach = True;
        return;
    
    def parse(self, sentence, grammar):
        words = sentence.split();
        self.initialize(words, grammar, PassiveEdge(grammar._start_symbol, 0, len(words)));
        
        while len(self._finishing_agenda)>0:
            while len(self._explore_agenda)>0:
                break;
            
            edge = self._finishing_agenda.pop();
            self.finish_edge(edge);
            
        return;
    
    def initialize(self, words, grammar, goal=None):
        # TODO: create new chart and agenda
        self._explore_agenda = [];
        self._finishing_agenda = [];
        for x in len(words):
            self._finishing_agenda.append(PassiveEdge(words[x], x, x+1));
            
        if self._top_down_approach:
            for production in grammar.productions(lhs=grammar._start_symbol):
                self._finishing_agenda.append(ActiveEdge(production.lhs(), production.rhs(), 0, 0));
            
    def finish_edge(self, edge):
        # TODO: add edge to chart
        self.do_fundamental_rule(edge);
        self.do_rule_introduction(edge);
        
    def do_fundamental_rule(self, edge):
        return;
    
    def do_rule_introduction(self, edge):
        return;                
    
def demo():
    graph_node_2 = GraphNode("NP", set(GraphNode("VP")));
    graph_node_1 = GraphNode("NP");
    print graph_node_1.__eq__(graph_node_2);

if __name__ == '__main__':
    demo()