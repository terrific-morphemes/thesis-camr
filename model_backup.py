#!/usr/bin/python

from __future__ import print_function
# import bz2, contextlib
import numpy as np
import sys
# import json
import cPickle as pickle
# import simplejson as json
from constants import *
from constants import ACTION_TYPE_TABLE
from common.util import Alphabet, ETag, ConstTag, get_unk_temlate
import importlib
from collections import defaultdict
# from parser import Parser


_FEATURE_TEMPLATES_FILE = './feature/basic_abt_feats.templates'


class Model():

    """weights and templates"""

    # weight = None
    # n_class = None
    # n_rel = None
    # n_tag = None
    indent = " " * 4
    # feature_codebook = None
    # class_codebook = None
    # feats_generator = None

    def __init__(self, elog=sys.stdout):

        self.elog = elog
        self.weight = None
        self.aux_weight = None
        self.avg_weight = None  # for store the averaged weights
        # self.n_class = n_class
        # self.n_rel = n_rel
        # self.n_tag = n_tag
        self._feats_templates_file = _FEATURE_TEMPLATES_FILE
        self._feature_templates_list = []
        self._feats_gen_filename = None
        self.feats_generator = None
        self.token_to_concept_table = defaultdict(set)
        self.pp_count_dict = defaultdict(int)
        self.total_num_words = 0
        self.token_label_set = defaultdict(set)
        self.class_codebook = None
        self.feature_codebook = None
        self.rel_codebook = Alphabet()
        self.tag_codebook = {
            'Concept': Alphabet(),
            'ETag': Alphabet(),
            'ConstTag': Alphabet(),
            'ABTTag': Alphabet()
        }

        # we use fuzzy matching templates
        # to generate candidate tag for unkown words when decoding
        self.unk_template_list = defaultdict(int)
        self.abttag_count = defaultdict(int)

    def setup(self, action_type, instances, parser,
              feature_templates_file=None):
        if feature_templates_file:
            self._feats_templates_file = feature_templates_file
        self.class_codebook = Alphabet.from_dict(
            dict(
                (i, k) for i, (k, v) in
                enumerate(ACTION_TYPE_TABLE[action_type])
            ), True)
        self.feature_codebook = dict(
            [(i, Alphabet()) for i in self.class_codebook._index_to_label.keys()])
        self.read_templates()

        # n_rel,n_tag = self._set_rel_tag_codebooks(instances, parser)
        n_subclass = self._set_rel_tag_codebooks(instances, parser)
        self._set_class_weight(self.class_codebook.size(), n_subclass)
        self._set_statistics(instances)
        self.output_feature_generator()

    def _set_statistics(self, instances):
        #pp_count_dict = defaultdict(int)
        for inst in instances:
            sent = inst.tokens
            self.total_num_words += len(sent)
            for token in sent:
                if token['pos'] == 'IN' and token['rel'] == 'prep':
                    self.pp_count_dict[token['form'].lower()] += 1

    def _set_rel_tag_codebooks(self, instances, parser):
        '''
        set up relation and concept tag base on gold graph
        '''
        # TODO
        self.rel_codebook.add(NULL_EDGE)
        self.rel_codebook.add(START_EDGE)
        # self.tag_codebook['Concept'].add(NULL_TAG)

        for i, inst in enumerate(instances):
            if i % 1000 == 0:
                print("%s...." % (str(i)), file=sys.stdout, end='')
                sys.stdout.flush()
            gold_graph = inst.gold_graph
            gold_nodes = gold_graph.nodes
            #gold_edges = gold_graph.edges
            sent_tokens = inst.tokens
            #state = parser.testOracleGuide(inst)

            for g, d in gold_graph.tuples():
                if g.startswith('x'):  # TODO wrap this up: is_abstract
                    gnode = gold_nodes[g]
                    g_nset = gnode.index_set
                    g_nset_srt = sorted(list(g_nset))
                    g_span_wds = [tok['form']
                                  for tok in sent_tokens if tok['id'] in g_nset_srt]

                    g_span_form = ''.join(g_span_wds)

                    #possible hacky fix
                    try:
                        g_span_ne = sent_tokens[g_nset_srt[0]]['ne']
                    except IndexError:
                        g_span_ne = 'O'
                    #g_span_ne = sent_tokens[g_nset_srt[0]]['ne']
                    g_entity_tag = gold_graph.get_node_tag(g)

                    if g_span_ne not in ['O', 'NUMBER']:  # is name entity
                        self.token_to_concept_table[g_span_ne].add(
                            g_entity_tag)

                    #########################
                    # # TODO record unk templates
                    # if g_span_ne == 'O':
                    #     unk_template = get_unk_temlate(g_span_form,
                    #                                    g_entity_tag)
                    #     if unk_template:
                    #         self.unk_template_list[unk_template] += 1
                    #########################

                    self.token_to_concept_table[g_span_form].add(g_entity_tag)

                    if isinstance(g_entity_tag, ETag):
                        self.tag_codebook['ETag'].add(g_entity_tag)
                    elif isinstance(g_entity_tag, ConstTag):
                        self.tag_codebook['ConstTag'].add(g_entity_tag)
                    else:
                        self.tag_codebook['Concept'].add(g_entity_tag)
                else:
                    g_entity_tag = gold_graph.get_node_tag(g)
                    self.tag_codebook['ABTTag'].add(g_entity_tag)
                    self.abttag_count[g_entity_tag] += 1

                if d.startswith('x'):
                    dnode = gold_nodes[d]
                    d_nset = dnode.index_set
                    d_nset_srt = sorted(list(d_nset))
                    d_span_wds = [tok['form'] for tok in sent_tokens
                                  if tok['id'] in d_nset_srt]
                    #for debugging
                    #print(sent_tokens[d_nset_srt[0]]['ne'])
                    #d_span_ne = sent_tokens[d_nset_srt[0]]['ne']
                    try:
                        d_span_ne = sent_tokens[d_nset_srt[0]]['ne']
                    except IndexError:
                        d_span_ne = 'O'
                    d_entity_tag = gold_graph.get_node_tag(d)
                    d_span_form = ''.join(d_span_wds)

                    if d_span_ne not in ['O', 'NUMBER']:
                        self.token_to_concept_table[d_span_ne].add(
                            d_entity_tag)
                    self.token_to_concept_table[d_span_form].add(d_entity_tag)

                    # # TODO
                    # if d_span_ne == 'O':  # record unk templates
                    #     unk_template = get_unk_temlate(d_span_form,
                    #                                    d_entity_tag)
                    #     if unk_template:
                    #         self.unk_template_list[unk_template] += 1

                    if isinstance(d_entity_tag, ETag):
                        self.tag_codebook['ETag'].add(d_entity_tag)
                    elif isinstance(d_entity_tag, ConstTag):
                        self.tag_codebook['ConstTag'].add(d_entity_tag)
                    else:
                        self.tag_codebook['Concept'].add(d_entity_tag)

                else:
                    d_entity_tag = gold_graph.get_node_tag(d)
                    self.tag_codebook['ABTTag'].add(d_entity_tag)
                    self.abttag_count[d_entity_tag] += 1

                g_edge_label = gold_graph.get_edge_label(g, d)

                self.rel_codebook.add(g_edge_label)

        print('\n')
        n_subclass = [1] * self.class_codebook.size()
        self._pruning()

        for k, v in self.class_codebook._index_to_label.items():
            if v in ACTION_WITH_TAG:
                #n_tag[k] = reduce(lambda x,y: x+y, map(lambda z: self.tag_codebook[z].size(), self.tag_codebook.keys()))
                n_subclass[k] = self.tag_codebook['ABTTag'].size()
            if v in ACTION_WITH_EDGE:
                #n_rel[k] = self.rel_codebook.size()
                n_subclass[k] = self.rel_codebook.size()
        # return n_rel,n_tag
        return n_subclass

    def _pruning(self, threshold=10, topN=30):
        # prune abttag
        pruned_abttag_codebook = Alphabet()
        for v in self.tag_codebook['ABTTag'].labels():
            if self.abttag_count[v] >= threshold:
                pruned_abttag_codebook.add(v)
        self.tag_codebook['ABTTag'] = pruned_abttag_codebook

        # prune unk template
        self.unk_template_list = sorted(self.unk_template_list.items(),
                                        key=lambda x: x[1],
                                        reverse=True)[:topN]

    def _set_class_weight(self, n_class, n_subclass=None,
                          init_feature_dim=10**5):

        #self.weight = [np.zeros(shape = (init_feature_dim,nt,nr),dtype=WEIGHT_DTYPE) for nr,nt in zip(n_rel,n_tag)]
        #self.aux_weight = [np.zeros(shape = (init_feature_dim,nt,nr),dtype=WEIGHT_DTYPE) for nr,nt in zip(n_rel,n_tag)]
        #self.avg_weight = [np.zeros(shape = (init_feature_dim,nt,nr),dtype=WEIGHT_DTYPE) for nr,nt in zip(n_rel,n_tag)]

        self.weight = [np.zeros(
            shape=(init_feature_dim, ns), dtype=WEIGHT_DTYPE) for ns in n_subclass]
        self.aux_weight = [np.zeros(
            shape=(init_feature_dim, ns), dtype=WEIGHT_DTYPE) for ns in n_subclass]
        self.avg_weight = [np.zeros(
            shape=(init_feature_dim, ns), dtype=WEIGHT_DTYPE) for ns in n_subclass]

    def read_templates(self):

        ff_name = self._feats_templates_file
        for line in open(ff_name, 'r'):
            line = line.strip()
            if not line:
                pass
            elif line.startswith('#'):
                pass
            else:
                elements = line.split()
                # elements.extend(['tx'])
                template = "'%s=%s' %% (%s)" % (
                    '&'.join(elements), '%s_' * len(elements), ','.join(elements))
                self._feature_templates_list.append((template, elements))

    def output_feature_generator(self):
        """based on feature autoeval method in (Huang,2010)'s parser"""

        import time
        self._feats_gen_filename = 'feats_gen_' + \
            self._feats_templates_file.split(
                '/')[-1].split('.')[0]  # str(int(time.time()))
        output = open('./temp/' + self._feats_gen_filename + '.py', 'w')

        output.write('#generated by model.py\n')
        output.write('from constants import *\n')
        output.write('def generate_features(state,action):\n')
        output.write(Model.indent +
                     's0,b0,a0=state.get_feature_context_window(action)\n')

        element_set = set([])
        definition_str = Model.indent + 'feats=[]\n'
        append_feats_str = ''

        definition_str += Model.indent + \
            "act_idx = state.model.class_codebook.get_index(action['type'])\n"
        definition_str += Model.indent + \
            "tx = action['tag'] if 'tag' in action else EMPTY\n"
        #definition_str += Model.indent+"txv = len(tx.split('-'))==2 if tx is not EMPTY else EMPTY\n"
        #definition_str += Model.indent+"lx = action['edge_label'] if 'edge_label' in action else EMPTY\n"
        #definition_str += Model.indent+"print state.model.class_codebook._label_to_index\n"
        # print self._feature_templates_list

        for template, elements in self._feature_templates_list:

            for e in elements:  # definition
                if e not in element_set:
                    sub_elements = e.split('_')
                    if len(sub_elements) == 2:
                        definition_str += "%s%s=%s['%s'] if %s else EMPTY\n" % (
                            Model.indent, e, sub_elements[0], FEATS_ABBR[sub_elements[1]], sub_elements[0])
                    elif len(sub_elements) == 3:
                        definition_str += "%s%s=%s['%s']['%s'] if %s and %s['%s'] else EMPTY\n" % (
                            Model.indent, e, sub_elements[0], FEATS_ABBR[sub_elements[1]], FEATS_ABBR[sub_elements[2]], sub_elements[0], sub_elements[0], FEATS_ABBR[sub_elements[1]])
                    else:
                        pass
                    element_set.add(e)
                else:
                    pass

            append_feats_str += "%sif [%s] != %s*[None]:feats.append(%s)\n" % (
                Model.indent, ','.join(elements), len(elements), template)
            #append_feats_str += "%sfeats.append(%s)\n" % (Model.indent,template)

        definition_str += "%sdist1=abs(s0['id']-b0['id']) if b0 and b0 is not ABT_TOKEN and s0 is not ABT_TOKEN else EMPTY\n" % (
            Model.indent)
        definition_str += "%sif dist1 > 10: dist1=10\n" % (Model.indent)
        definition_str += "%sdist2=abs(a0['id']-b0['id']) if b0 and a0 and b0 is not ABT_TOKEN and a0 is not ABT_TOKEN else EMPTY\n" % (
            Model.indent)
        definition_str += "%sif dist2 > 10: dist2=10\n" % (Model.indent)

        #definition_str += "%seqfrmset=s0['eqfrmset']\n"%(Model.indent)
        output.write(definition_str)
        output.write(append_feats_str)

        output.write('%sreturn feats' % Model.indent)
        output.close()

        # sys.path.append('/temp/')
        print("Importing feature generator!")
        self.feats_generator = importlib.import_module(
            'temp.' + self._feats_gen_filename).generate_features

    def toJSON(self):
        print('Converting model to JSON')
        print('class size: %s \nrelation size: %s \ntag size: %s' % (self.class_codebook.size(), self.rel_codebook.size(
        ), map(lambda x: '%s->%s ' % (x, self.tag_codebook[x].size()), self.tag_codebook.keys())))
        print('feature codebook size: %s' % (','.join(('%s:%s') %
                                                      (i, f.size()) for i, f in self.feature_codebook.items())))
        print('weight shape: %s' % (','.join(('%s:%s') % (i, w.shape)
                                             for i, w in enumerate(self.avg_weight))))
        print('token to concept table: %s' %
              (len(self.token_to_concept_table)))
        model_dict = {
            '_feature_templates_list': self._feature_templates_list,
            '_feats_gen_filename': self._feats_gen_filename,
            #'weight':[w.tolist() for w in self.weight],
            #'aux_weight':[axw.tolist() for axw in self.aux_weight],
            'avg_weight': [agw.tolist() for agw in self.avg_weight],
            'token_to_concept_table': dict([(k, list(v)) for k, v in self.token_to_concept_table.items()]),
            'class_codebook': self.class_codebook.to_dict(),
            'feature_codebook': self.feature_codebook.to_dict(),
            'rel_codebook': self.rel_codebook.to_dict(),
            'tag_codebook': dict([(k, self.tag_codebook[k].to_dict()) for k in self.tag_codebook])
        }
        return model_dict

    def save_model(self, model_filename):
        # pickle.dump(self,open(model_filename,'wb'),pickle.HIGHEST_PROTOCOL)
        print('Model info:', file=self.elog)
        print('class size: %s \nrelation size: %s \ntag size: %s' % (self.class_codebook.size(), self.rel_codebook.size(
        ), map(lambda x: '%s->%s ' % (x, self.tag_codebook[x].size()), self.tag_codebook.keys())), file=self.elog)
        print('feature codebook size: %s' % (','.join(('%s:%s') % (i, f.size())
                                                      for i, f in self.feature_codebook.items())), file=self.elog)
        # print 'weight shape: %s' % (self.avg_weight.shape)
        print('weight shape: %s' % (','.join(('%s:%s') % (i, w.shape)
                                             for i, w in enumerate(self.avg_weight))), file=self.elog)
        print('token to concept table: %s' %
              (len(self.token_to_concept_table)), file=self.elog)
        weight = self.weight
        aux_weight = self.aux_weight
        #avg_weight = self.avg_weight

        self.weight = None
        self.aux_weight = None

        try:
            # with contextlib.closing(bz2.BZ2File(model_filename, 'wb')) as f:
            with open(model_filename, 'wb') as f:
                pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)
        except:
            #print >> sys.stderr, 'Saving model error', sys.exc_info()[0]
            print(sys.exc_info()[0], file=sys.stderr)
            #raise
            pass

        self.weight = weight
        self.aux_weight = aux_weight
        #self.avg_weight = avg_weight

    @staticmethod
    def load_model(model_filename):

        # with contextlib.closing(bz2.BZ2File(model_filename, 'rb')) as f:
        with open(model_filename, 'rb') as f:
            model = pickle.load(f)

        return model
