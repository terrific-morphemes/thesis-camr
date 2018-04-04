# -*- coding:utf-8 -*-

from __future__ import print_function
import json
from collections import defaultdict
from constants import ROOT_FORM,ROOT_LEMMA,ROOT_POS,EMPTY,ROOT_PCPT


class Data():
    '''represent dota instance of one sentence'''
    
    counter_sen = 1
    
    def __init__(self):
        self.tree = None
        self.coreference = None
        #self.dependency = []
        self.text = None
        self.tokens = []
        self.amr = None
        self.gold_graph = None
        self.sentID = self.counter_sen
        self.annoID = None
        self.comment = None
        self.trace_dict = defaultdict(set)
        
        self.tokens.append({'id':0,'form':ROOT_FORM,'pos':ROOT_POS,'ne':'O','rel':EMPTY, 'cpt_label_pred':ROOT_PCPT})

    @staticmethod
    def reset():
        Data.counter_sen = 1
        
    @staticmethod
    def newSen():
        Data.counter_sen += 1  # won't be pickled
        #self.dependency.append([])
        #self.tokens.append([])

    def get_tokenized_sent(self):
        return [tok['form'] for tok in self.tokens][1:]
        
    def addTree(self, tree):
        self.tree = tree
        
    def addText(self, sentence):
        self.text = sentence
        
    def addToken(self, token, pop, ne, norm_ne=None):
        tok_inst = {}
        tok_inst['id'] = len(self.tokens)
        tok_inst['form'] = token
        tok_inst['pos'] = pop
        tok_inst['ne'] = ne
        tok_inst['rel'] = EMPTY
        tok_inst['cpt_label_pred'] = ROOT_PCPT
        tok_inst['norm_ne'] = norm_ne
        self.tokens.append(tok_inst)

    def addCpt(self,cpt_labels):
        for i, cpt_label in enumerate(cpt_labels):
            self.tokens[i+1]['cpt_label_pred'] = cpt_label
            
    def addCoref( self, coref_set):
        self.coreference = coref_set

    # def addTrace(self, rel, gov, trace):
    #     self.trace_dict[int(gov)].add((rel, int(trace)))
        
    def addDependency( self, rel, l_index, r_index):
        '''CoNLL dependency format'''
        try:
            assert int(r_index) == self.tokens[int(r_index)]['id'] and int(l_index) == self.tokens[int(l_index)]['id']
        except IndexError as e:
            print(str(__file__)+':'+str(e))
            print(str(self.tokens))
            print('_%s_%s' % (l_index, r_index))
            #import pdb; pdb.set_trace()
            raise

        self.tokens[int(r_index)]['head'] = int(l_index)
        self.tokens[int(r_index)]['rel'] = rel
        
    def addProp(self, prd, frmset, arg, label):
        self.tokens[prd]['frmset'] = frmset
        if 'args' in self.tokens[prd]:
            self.tokens[prd]['args'][arg]=label
        else:
            self.tokens[prd]['args']={arg:label}

        # bi-directional
        if 'pred' in self.tokens[arg]:
            self.tokens[arg]['pred'][prd]=label
        else:
            self.tokens[arg]['pred']={prd:label}

    def addAMR(self,amr):
        self.amr = amr
        
    def addComment(self,comment):
        self.comment = comment
        self.annoID = comment['id']
        
    def addGoldGraph(self,gold_graph):
        self.gold_graph = gold_graph


    def get_ne_span(self,tags_to_merge):
        pre_ne_id = None
        ne_span_dict = defaultdict(list)
        for tok in self.tokens:
            if tok['ne'] in tags_to_merge:
                if pre_ne_id is None:
                    ne_span_dict[tok['id']].append(tok['id'])
                    pre_ne_id = tok['id']
                else:
                    ne_span_dict[pre_ne_id].append(tok['id'])
            else:
                pre_ne_id = None
        return ne_span_dict

    def printDep(self,tagged=False):
        out_str = ''
        for tok in self.tokens:
            if 'head' in tok:
                gov_id = tok['head']
                if tagged:
                    out_str += "%s(%s-%s:%s, %s-%s:%s)\n" % (tok['rel'], self.tokens[gov_id]['form'], gov_id, self.tokens[gov_id]['pos'], tok['form'], tok['id'], tok['pos'])
                else:
                    out_str += "%s(%s-%s, %s-%s)\n" % (tok['rel'], self.tokens[gov_id]['form'], gov_id, tok['form'], tok['id'])
        return out_str

    def to_string(self):
        output = 'Sentence #%d--%s\n' % (self.sentID, self.annoID)
        output += ' '.join('%s_%s_%s' %  (tok['form'], tok['pos'], tok['ne']) for tok in self.tokens) + '\n'
        output += '\n'
        output += 'Dependency:\n'
        output += self.printDep() + '\n'
        output += 'AMR:' + '\n'
        output +=  self.amr.to_amr_string() if self.amr else 'None' 
        output += '\n'
        return output.encode('utf8')
        
    def toJSON(self):
        json = {}
        json['tree'] = self.tree
        json['coreference'] = self.coreference
        #json['dependency'] = self.dependency
        json['text'] = self.text
        json['tokens'] = self.tokens
        json['amr'] = self.amr
        return json

##    def find
