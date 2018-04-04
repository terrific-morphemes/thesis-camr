#!/usr/bin/python

'''
Implementation for graph of which nodes are spans of sentence 

@author Chuan Wang
@since: 2017-05-10
'''
from __future__ import print_function

import copy
import sys
import re
from util import StrLiteral, Polarity, Quantity, ConstTag, Interrogative, ETag
from util import ispunctuation
import constants
from common.amr_graph import *
from collections import defaultdict
from span import Span

log = sys.stderr
debuglevel = 1


class SpanNode(object):
    '''
    this class models the node which maps to a span of tokens (continuous)
    '''

    def __init__(self, start, end, words, tag=constants.NULL_TAG):
        self.start = start
        self.id = start
        self.end = end
        self.tag = tag
        self.words = words
        self.children = []
        self.parents = []
        self.SWAPPED = False
        self.num_swap = 0
        self.num_parent_infer = 0
        self.num_parent_infer_in_chain = 0
        self.num_child_insert = 0
        self.num_child_insert_in_chain = 0
        self.del_child = []  # record the replaced or deleted child
        self.rep_parent = []  # record the parent replaced
        self.outgoing_traces = set()
        self.incoming_traces = set()

    @staticmethod
    def from_span(span):
        """initialize from span object"""
        return SpanNode(span.start, span.end, span.words, span.entity_tag)

    def addChild(self, child):
        # if isinstance(c,list):
        #    self.children.extend(c)
        # else:
        if child not in self.children:
            self.children.append(child)

    def contains(self, other_node):
        if other_node.start >= self.start and other_node.end <= self.end and \
           not (other_node.start == self.start and other_node.end == self.end):
            return True
        else:
            return False

    def addParent(self, parent):
        if parent not in self.parents:
            self.parents.append(parent)

    def removeChild(self, child):
        self.children.remove(child)

    def removeParent(self, parent):
        self.parents.remove(parent)

    def __str__(self):
        return 'Span node:(%s,%s) Children: %s SWAPPED:%s' % (self.start, self.end, self.children, self.SWAPPED)

    def __repr__(self):
        return 'Span node:(%s,%s) Children: %s SWAPPED:%s' % (self.start, self.end, self.children, self.SWAPPED)


class DSpanNode(object):
    '''
    this class models the node which maps to a set of tokens (may not be continuous)
    '''

    def __init__(self, id, index_set, tag=constants.NULL_TAG):

        self.id = id
        self.index_set = index_set  # maps to multiple words
        self.tag = tag
        self.chr_index_set = set()  # multiple amr concepts

        self.children = []
        self.parents = []

        self.SWAPPED = False
        self.num_swap = 0
        self.num_parent_infer = 0
        self.num_parent_infer_in_chain = 0
        self.num_child_insert = 0
        self.num_child_insert_in_chain = 0
        self.del_child = []  # record the replaced or deleted child
        self.rep_parent = []  # record the parent replaced
        self.outgoing_traces = set()
        self.incoming_traces = set()

    def addChild(self, child_id):
        if child_id not in self.children:
            self.children.append(child_id)

    def contains(self, other_node):
        other_node_set = other_node.index_set
        return len(other_node_set) > 0 and other_node_set.issubset(self.index_set)

    def addParent(self, parent_id):
        if parent_id not in self.parents:
            self.parents.append(parent_id)

    def removeChild(self, child_id):
        self.children.remove(child_id)

    def removeParent(self, parent_id):
        self.parents.remove(parent_id)

    def __str__(self):
        return 'Node:(%s,%s) Children: %s SWAPPED:%s' % (self.id, self.index_set, self.children, self.SWAPPED)

    def __repr__(self):
        return 'Node:(%s,%s) Children: %s SWAPPED:%s' % (self.id, self.index_set, self.children, self.SWAPPED)


class SpanGraph(object):
    """
    Graph of span nodes
    """
    LABELED = False
    #graphID = 0

    def __init__(self, graphID=None, root='r'):

        self.graphID = graphID
        self.root = root  # this is actually the unique top
        self.multi_roots = []
        self.nodes = {}  # refer to spans by start index
        self.edges = {}  # refer to edges by tuple (parent,child)
        self.sent = None
        self.sent_end = None

        # cache the tuples so that we don't have to traverse the graph everytime
        self.static_tuples = set([])
        self.abt_node_num = 0
        self.abt_node_table = {}
        self.index_to_id = {}
        self.collapse_map = {}

        self.nodes_error_table = defaultdict(str)
        self.edges_error_table = defaultdict(str)

    def fix_root(self):
        # for other disconnected multi roots, we all link them to the root, here we ignore graphs with circle
        for node_id in self.nodes:
            # multi-roots
            if self.nodes[node_id].parents == [] and node_id != 0 and not self.isContained(node_id):
                if self.nodes[node_id].children:
                    self.add_edge(0, node_id, constants.FAKE_ROOT_EDGE)
                self.multi_roots.append(node_id)

    def set_sent(self, sent):
        self.sent = sent
        self.sent_end = len(sent)

    def set_collapse_map(self, orig, new):
        self.collapse_map[orig] = new

    @staticmethod
    def init_ref_graph_abt(amr, inst):
        """
        Instantiate graph from AMR(zh) graph;
        alignment information is stored in the amr variables
        """
        def init_span_node(h, prev_collapsed_vars,
                           embedded_edge, spgraph,
                           amr, h_incoming_edge):

            h_index_list = h.split('_')
            h_base = h_index_list[0]
            is_multi_words = all(x.startswith('x') for x in h_index_list)
            hconcept = amr.node_to_concepts[h] if h in amr.node_to_concepts else h
            h_node = None

            if h in spgraph.nodes:
                h_node = spgraph.nodes[h]
                if h in prev_collapsed_vars:
                    embedded_edge = True
            elif h_base in spgraph.nodes and is_multi_words:
                h_node = spgraph.nodes[h_base]
                if h in prev_collapsed_vars:
                    embedded_edge = True
            elif len(h_index_list) == 1:  # single  word
                h_index = int(h_index_list[0][1:])
                if h_index < spgraph.sent_end:  # aligned
                    h_index_set = set([h_index])
                    h_node = DSpanNode(h, h_index_set, hconcept)
                    spgraph.add_node(h_node)
                    spgraph.index_to_id[h_index] = h

                else:  # unaligned
                    h_abs = 'a' + h[1:]
                    if h_abs in spgraph.nodes:
                        h_node = spgraph.nodes[h_abs]

                    # collapse vars in advance
                    elif 'name' in amr[h]:  # named entity
                        name_var = amr[h]['name'][0]
                        name_base = name_var.split('_')[0]
                        if name_base not in spgraph.nodes:
                            name_index_set = set(int(x[1:])
                                                 for x in name_var.split('_'))
                            named_entity_tag = '%s|name@name' % hconcept
                            name_node = DSpanNode(
                                name_base, name_index_set, ETag(named_entity_tag))
                            spgraph.add_node(name_node)
                            for name_index in name_index_set:
                                spgraph.index_to_id[name_index] = name_base
                            h_node = name_node
                            # mark advance visited vars
                            prev_collapsed_vars |= set([name_var])
                            if h in amr.roots:
                                spgraph.set_collapse_map(h, name_base)
                        else:
                            h_node = spgraph.nodes[name_base]
                    # elif re.match('have-(org|rel)-role', hconcept):
                    #    pass
                    else:
                        h_node = DSpanNode(h_abs, set(), hconcept)
                        spgraph.add_node(h_node)

            else:  # multiple aligned
                is_multi_words = all(x.startswith('x') for x in h_index_list)
                h_base = h_index_list[0]
                if is_multi_words:  # many to one
                    h_index_set = set(int(x[1:]) for x in h_index_list)
                    h_node = DSpanNode(h_base, h_index_set, hconcept)
                    spgraph.add_node(h_node)
                    for index in h_index_set:
                        spgraph.index_to_id[index] = h
                elif h_index_list[0].startswith('x') \
                        and all(not x.startswith('x') for x in h_index_list[1:]):  # one to many (multiple graph nodes)

                    h_base_index = int(h_base[1:])
                    if h_base not in spgraph.nodes:  # first visit
                        h_index_set = set([h_base_index])
                        h_node = DSpanNode(h_base, h_index_set, ETag(hconcept))
                        h_node.chr_index_set |= set(
                            int(num) for num in h_index_list[1:])
                        spgraph.add_node(h_node)
                        spgraph.index_to_id[h_base_index] = h_base
                    else:
                        h_node = spgraph.nodes[h_base]
                        old_h_chr_index_set = h_node.chr_index_set
                        curr_chr_index_set = set(int(num)
                                                 for num in h_index_list[1:])
                        # update concept tag
                        if not curr_chr_index_set.issubset(old_h_chr_index_set):
                            h_node.chr_index_set |= curr_chr_index_set
                            h_node.tag = ETag('%s|%s@%s' % (
                                h_node.tag, h_incoming_edge, hconcept))
                            prev_collapsed_vars |= set([h])
                            embedded_edge = True
                else:  # many to many
                    h_index_set = set(int(x[1:])
                                      for x in h_index_list if x.startswith('x'))
                    h_node = DSpanNode(h_base, h_index_set, hconcept)
                    spgraph.add_node(h_node)
                    for index in h_index_set:
                        spgraph.index_to_id[index] = h

            return h_node, prev_collapsed_vars, embedded_edge

        spgraph = SpanGraph(inst.sentID)
        spgraph.set_sent(inst.tokens)
        sequences = amr.dfs()[0]
        prev_collapsed_vars = set()

        for seq in sequences:
            h = seq.node_label
            h_incoming_edge = seq.trace
            embedded_edge = False

            if h not in amr.node_to_concepts:  # ignore unaligned/un-instantiated constants
                # TODO
                continue

            h_node, prev_collapsed_vars, embedded_edge = init_span_node(h, prev_collapsed_vars,
                                                                        embedded_edge, spgraph,
                                                                        amr, h_incoming_edge)

            for edge, ds in amr[h].items():
                d = ds[0]

                if d not in amr.node_to_concepts:  # ignore unaligned/un-instantiated constants
                    # TODO
                    continue

                d_node, prev_collapsed_vars, _ = init_span_node(d, prev_collapsed_vars,
                                                                embedded_edge, spgraph,
                                                                amr, edge)

                if d in prev_collapsed_vars:  # ignore vars visited in advance
                    continue

                # if edge == 'wiki':
                #     continue

                # add edge
                if embedded_edge:
                    edge = '*' + edge

                spgraph.add_edge(h_node.id, d_node.id, edge)

        sym_root = DSpanNode('r', set([0]), constants.NULL_TAG)
        spgraph.add_node(sym_root)

        amr_root = amr.roots[0]
        amr_root_base = amr_root.split('_')[0]
        amr_root_abs = 'a' + amr_root_base[1:]
        root_node = None

        if amr_root in spgraph.nodes:
            root_node = spgraph.nodes[amr_root]
        elif amr_root_base in spgraph.nodes:
            root_node = spgraph.nodes[amr_root_base]
        elif amr_root_abs in spgraph.nodes:
            root_node = spgraph.nodes[amr_root_abs]
        elif amr_root in spgraph.collapse_map:
            new_root = spgraph.collapse_map[amr_root]
            root_node = spgraph.nodes[new_root]
        else:
            print(spgraph.nodes, file=sys.stderr)
            raise Exception('root variable %s not initialized' % amr_root)

        spgraph.add_edge(sym_root.id, root_node.id, constants.NULL_EDGE)
        # else:
        #     if debuglevel > 1:
        #         print >> log, "GraphID:%s WARNING:root %s not aligned!"%(SpanGraph.graphID,amr.node_to_concepts[amr.roots[0]])

        return spgraph

    @staticmethod
    def init_dep_graph(instance, sent=None):
        """instantiate graph from data instance, keep the unaligned (abstract) concepts in the generated span graph."""
        dpg = SpanGraph(instance.sentID)
        dpg.set_sent(instance.tokens)
        for tok in instance.tokens:
            if tok['id'] == 0:  # root
                root_form = tok['form']
                root_id = 'r'
                curr_node = DSpanNode(root_id, set([0]))
                dpg.add_node(curr_node)
                dpg.multi_roots.append(root_id)
            elif 'head' in tok:
                gov_index = tok['head']
                gov_id = 'x' + str(gov_index) if gov_index > 0 else 'r'
                gov_form = instance.tokens[gov_index]['form']
                gov_netag = instance.tokens[gov_index]['ne']
                if gov_id not in dpg.nodes:
                    gov_node = DSpanNode(gov_id, set([gov_index]))
                    dpg.add_node(gov_node)
                if 'head' not in instance.tokens[gov_index] and gov_id not in dpg.multi_roots:
                    dpg.multi_roots.append(gov_id)

                dep_index = tok['id']
                dep_id = 'x' + str(dep_index)
                dep_form = tok['form']
                dep_netag = tok['ne']
                dep_label = tok['rel']
                if dep_id not in dpg.nodes:
                    dep_node = DSpanNode(dep_id, set([dep_index]))
                    dpg.add_node(dep_node)
                dpg.add_edge(gov_id, dep_id)

            else:  # punctuation
                punc_index = tok['id']
                punc_id = 'x' + str(punc_index)
                punc_form = tok['form']
                if punc_id not in dpg.multi_roots:
                    dpg.multi_roots.append(punc_id)
                if punc_id not in dpg.nodes:
                    dpg.add_node(DSpanNode(punc_id, set([punc_index])))

        if not dpg.nodes:  # dependency tree is empty
            root = DSpanNode('r', set([0]))
            dpg.multi_roots.append(root.id)
            dpg.add_node(root)

        if constants.FLAG_COREF:
            dpg.add_trace_info(instance)
            dpg.add_coref_info(instance)

        return dpg

    def add_trace_info(self, instance):
        # adding trace info
        for node_id in instance.trace_dict:
            node = self.nodes[node_id]
            node.outgoing_traces = node.outgoing_traces.union(
                instance.trace_dict[node_id])
            for rel, trace_id in instance.trace_dict[node_id]:
                trace_node = self.nodes[trace_id]
                trace_node.incoming_traces.add((rel, node_id))

    def add_coref_info(self, instance):
        if instance.coreference:
            for cpair in instance.coreference:
                src_word, src_i, src_pos, src_l, src_r = cpair[0]
                sink_work, sink_i, sink_pos, sink_l, sink_r = cpair[1]
                assert src_i == sink_i
                src_node = self.nodes[src_pos]
                sink_node = self.nodes[sink_pos]
                src_head = self.sent[src_pos]['head']
                sink_head = self.sent[sink_pos]['head']
                src_rel = self.sent[src_pos]['rel']
                sink_rel = self.sent[sink_pos]['rel']
                if (not self.sent[src_pos]['pos'].startswith('PRP') and not self.sent[sink_pos]['pos'].startswith('PRP') and src_pos < sink_pos) or (self.sent[sink_pos]['pos'].startswith('PRP')):
                    while self.sent[sink_head]['pos'] in constants.FUNCTION_TAG:
                        sink_rel = self.sent[sink_head]['rel']
                        sink_head = self.sent[sink_head]['head']
                    if sink_head != src_pos:
                        src_node.incoming_traces.add((sink_rel, sink_head))
                elif (not self.sent[src_pos]['pos'].startswith('PRP') and not self.sent[sink_pos]['pos'].startswith('PRP') and sink_pos < src_pos) or (self.sent[src_pos]['pos'].startswith('PRP')):
                    while self.sent[src_head]['pos'] in constants.FUNCTION_TAG:
                        src_rel = self.sent[src_head]['rel']
                        src_head = self.sent[src_head]['head']
                    if src_head != sink_pos:
                        sink_node.incoming_traces.add((src_rel, src_head))
                else:
                    pass

    def post_process(self, root='r'):
        if len(self.nodes[root].children) == 0:
            for mr in self.multi_roots:
                if mr in self.nodes and len(self.nodes[mr].children) > 0:
                    self.add_edge(root, mr)

    def pre_merge_date(self, instance):
        date_spans = instance.get_ne_span(['DATE'])
        for start_id, span in date_spans.items():
            norm_ne = instance.tokens[start_id]['norm_ne']
            if norm_ne and re.match('({0}{0}{0}{0})(\-{0}{0})?(\-{0}{0})?'.format('[0-9]'), norm_ne):
                instance.tokens[start_id]['form'] = norm_ne
                for sp_id in span:
                    if sp_id != start_id and start_id in self.nodes and sp_id in self.nodes:
                        #self.merge_node(start_id, sp_id, ext_wds=False)
                        self.remove_node(sp_id)
                        if sp_id in self.multi_roots:
                            self.multi_roots.remove(sp_id)

    def pre_merge_netag(self, instance):
        ne_spans = instance.get_ne_span(PRE_MERGE_NETAG)
        for ne_id, span in ne_spans.items():
            for sp_id in span:
                if sp_id != ne_id and ne_id in self.nodes and sp_id in self.nodes:
                    self.merge_node(ne_id, sp_id)
                    if sp_id in self.multi_roots:
                        self.multi_roots.remove(sp_id)

    def clear_up(self, parent_to_add, cidx):
        '''
        clear up the possible duplicate coreference surface form here
        '''
        deleted_nodes = set([])
        if isinstance(parent_to_add, int) and isinstance(cidx, int) and self.sent[cidx]['ne'] != 'O':
            cnode = self.nodes[cidx]
            ctok_set = set(self.sent[i]['form']
                           for i in range(cnode.start, cnode.end))
            for c in self.nodes[parent_to_add].children:
                if isinstance(c, int):
                    pcnode = self.nodes[c]
                    pctok_set = set(self.sent[j]['form']
                                    for j in range(pcnode.start, pcnode.end))
                    if self.sent[c]['ne'] == self.sent[cidx]['ne'] and self.sent[c]['ne'] and len(ctok_set & pctok_set) > 0:
                        self.remove_subgraph(c, deleted_nodes)

        return deleted_nodes

    def remove_subgraph(self, idx, deleted_nodes):
        '''
        remove the subgraph rooted at idx; no cycle should be in the subgraph
        '''
        for child in self.nodes[idx].children:
            self.remove_subgraph(child, deleted_nodes)

        self.remove_node(idx)
        deleted_nodes.add(idx)

    def make_root(self):
        first = sorted(self.nodes.keys())[0]
        root = SpanNode(0, 1, ['root'], 'O')
        self.multi_roots.append(0)
        self.add_node(root)
        self.add_edge(0, first)
        # for c in self.nodes[first].children:
        #    self.remove_edge(first,c)
        #    self.add_edge(0,c)

    def is_empty(self):
        return len(self.nodes.keys()) == 0

    def is_root(self):
        return self.nodes.keys() == [0]

    def numNodes(self):
        return len(self.nodes.keys())

    def nodes_list(self):
        return self.nodes.keys()

    def isContained(self, other_node):
        """check whether node is contained by some node in graph"""

        for k in self.nodes:
            curr_node = self.nodes[k]
            if k == other_node.id:
                continue
            if curr_node.contains(other_node):
                return k

        return False

    def add_node(self, node):
        self.nodes[node.id] = node

    def add_edge(self, gov_index, dep_index, edge=constants.NULL_EDGE):
        self.nodes[gov_index].addChild(dep_index)
        self.nodes[dep_index].addParent(gov_index)
        self.edges[tuple((gov_index, dep_index))] = edge

    def new_abt_node(self, gov_index, tag, reverse=False):
        '''
        adding new abt node, if reverse adding as child instead
        '''
        gov_node = self.nodes[gov_index]
        abt_node_index = constants.ABT_PREFIX + str(self.abt_node_num)
        abt_node = DSpanNode(abt_node_index, set(), tag)
        self.add_node(abt_node)
        self.abt_node_num += 1

        if not reverse:
            gov_node.num_parent_infer += 1
            abt_node.num_parent_infer_in_chain = gov_node.num_parent_infer_in_chain + 1
            abt_node.incoming_traces = gov_node.incoming_traces.copy()
            abt_node.outgoing_traces = gov_node.outgoing_traces.copy()
            gov_parents = gov_node.parents[:]
            for p in gov_parents:
                edge_label = self.get_edge_label(p, gov_index)
                self.add_edge(p, abt_node_index, edge_label)
                self.remove_edge(p, gov_index)

            self.add_edge(abt_node_index, gov_index)
        else:
            gov_node.num_child_insert += 1
            abt_node.num_child_insert_in_chain = gov_node.num_child_insert_in_chain + 1
            self.add_edge(gov_index, abt_node_index)
        return abt_node_index
    '''
    work with infer1 
    def new_abt_node(self,gov_index):
        gov_node = self.nodes[gov_index]
        abt_node_index=constants.ABT_PREFIX+str(self.abt_node_num)
        abt_node = SpanNode(abt_node_index,abt_node_index,[constants.ABT_FORM],constants.NULL_TAG)
        self.add_node(abt_node)
        self.add_edge(gov_index,abt_node_index)
        self.abt_node_num += 1
        gov_node.num_child_infer += 1
        return abt_node_index
    '''

    def add_abt_mapping(self, key, value):
        #assert key not in self.abt_node_table
        self.abt_node_table[key] = value

    def set_node_tag(self, idx, tag):
        self.nodes[idx].tag = tag

    def get_node_tag(self, idx):
        return self.nodes[idx].tag

    def get_edge_label(self, gov_index, dep_index):
        return self.edges[tuple((gov_index, dep_index))]

    def set_edge_label(self, gov_index, dep_index, edge_label):
        self.edges[tuple((gov_index, dep_index))] = edge_label

    def get_direction(self, i, j):
        """left or right or no arc"""
        if j in self.nodes[i].children:
            return 0
        elif i in self.nodes[j].children:
            return 1
        else:
            return -1

    def record_rep_head(self, cidx, idx):
        self.nodes[cidx].rep_parent.append(idx)

    def remove_node(self, idx, RECORD=False):
        for p in self.nodes[idx].parents[:]:
            self.remove_edge(p, idx)
            if RECORD:
                self.nodes[p].del_child.append(idx)
        for c in self.nodes[idx].children[:]:
            self.remove_edge(idx, c)
            # if self.nodes[c].parents == [] and c not in self.multi_roots: # disconnected graph formed
            #    self.multi_roots.append(c)
        del self.nodes[idx]
        if idx in self.multi_roots:
            self.multi_roots.remove(idx)

    # ignore the multiedge between same nodes
    def remove_edge(self, gov_index, dep_index):
        self.nodes[gov_index].removeChild(dep_index)
        self.nodes[dep_index].removeParent(gov_index)
        if (gov_index, dep_index) in self.edges:
            del self.edges[(gov_index, dep_index)]

    def swap_head(self, gov_index, dep_index):
        """
        just flip the position of gov and dep
        """
        '''
        self.nodes[dep_index].addChild(gov_index)
        self.nodes[gov_index].removeChild(dep_index)
        children = self.nodes[gov_index].children
        self.nodes[dep_index].addChild(children)
        self.nodes[gov_index].children = []
        for c in children[:]:
            self.nodes[c].removeParent(gov_index)
            self.nodes[c].addParent(dep_index)

        parents = self.nodes[gov_index].parents
        self.nodes[dep_index].parents = parents
        self.nodes[gov_index].parents = [dep_index]
        for p in parents[:]:
            self.nodes[p].removeChild(gov_index)
            self.nodes[p].addChild(dep_index)
        '''
        tmp_parents = self.nodes[dep_index].parents[:]
        tmp_children = self.nodes[dep_index].children[:]
        for p in self.nodes[gov_index].parents[:]:
            if p != dep_index:
                edge_label = self.get_edge_label(p, gov_index)
                self.remove_edge(p, gov_index)
                self.add_edge(p, dep_index, edge_label)
        for c in self.nodes[gov_index].children[:]:
            if c != dep_index:
                edge_label = self.get_edge_label(gov_index, c)
                self.remove_edge(gov_index, c)
                self.add_edge(dep_index, c, edge_label)
            else:
                self.remove_edge(gov_index, c)
                self.add_edge(c, gov_index)

        for sp in tmp_parents:
            if sp != gov_index:
                edge_label = self.get_edge_label(sp, dep_index)
                self.remove_edge(sp, dep_index)
                self.add_edge(sp, gov_index, edge_label)
            # else:
            #    self.remove_edge(sp,dep_index)

        for sc in tmp_children:
            if sc != gov_index:
                edge_label = self.get_edge_label(dep_index, sc)
                self.remove_edge(dep_index, sc)
                self.add_edge(gov_index, sc, edge_label)

        self.nodes[gov_index].SWAPPED = True

    def reattach_node(self, idx, cidx, parent_to_attach, edge_label):
        self.remove_edge(idx, cidx)
        if parent_to_attach is not None:
            self.add_edge(parent_to_attach, cidx, edge_label)

    def is_cycle(self, root):
        visited = set()
        return self.is_cycle_aux(root, visited)

    def is_cycle_aux(self, rt, visited):
        if rt not in visited:
            visited.add(rt)
        else:
            return True

        for c in self.nodes[rt].children:
            if self.is_cycle_aux(c, visited):
                return True

        return False

    def find_true_head(self, index):
        true_index = index
        # if current node has done inferred; find the true head of abstract struture
        if self.nodes[index].num_parent_infer > 0:
            assert self.nodes[true_index].parents
            ancestor_index = self.nodes[true_index].parents[0]
            # TODO wrap this with is_abstract method
            while ancestor_index.startswith('a') and self.nodes[ancestor_index].parents:
                true_index = ancestor_index
                ancestor_index = self.nodes[true_index].parents[0]

        return true_index

    def swap_head2(self, gov_index, dep_index, sigma, edge_label=None):
        """
        keep dep and gov's dependents unchanged, only switch the dependency edge 
        direction, also all gov's parents become dep's parents
        """
        #
        origin_index = gov_index
        gov_index = self.find_true_head(gov_index)

        for p in self.nodes[gov_index].parents[:]:
            if p != dep_index and p in sigma:
                self.remove_edge(p, gov_index)
                self.add_edge(p, dep_index)

        if dep_index in self.nodes[gov_index].parents:
            self.remove_edge(origin_index, dep_index)
        else:
            # self.nodes[gov_index].removeChild(dep_index)
            # self.nodes[gov_index].addParent(dep_index)

            # self.nodes[dep_index].removeParent(gov_index)
            # self.nodes[dep_index].addChild(gov_index)
            self.remove_edge(origin_index, dep_index)
            self.add_edge(dep_index, gov_index, edge_label)
            self.nodes[gov_index].SWAPPED = True
            self.nodes[dep_index].num_swap += 1

    def replace_head(self, idx1, idx2):
        for c in self.nodes[idx1].children[:]:
            if c != idx2 and c not in self.nodes[idx2].children:
                if c not in self.nodes[idx2].parents:  # no multi-edge no circle
                    self.add_edge(idx2, c)

        for p in self.nodes[idx1].parents[:]:
            if p != idx2:
                self.add_edge(p, idx2)

        self.remove_node(idx1, RECORD=True)

    def merge_node(self, idx1, idx2):
        '''merge nodes connecting current arc (idx1,idx2) '''
        tmp1 = idx1
        tmp2 = idx2

        inorder = int(tmp1[1:]) < int(tmp2[1:])

        idx1 = tmp1 if inorder else tmp2
        idx2 = tmp2 if inorder else tmp1

        node1 = self.nodes[idx1]
        node2 = self.nodes[idx2]
        node1.index_set |= node2.index_set

        for p in self.nodes[idx2].parents[:]:

            if p != idx1 and p not in self.nodes[idx1].parents:
                edge_label = self.get_edge_label(p, idx2)
                self.add_edge(p, idx1, edge_label)

        for c in self.nodes[idx2].children[:]:
            if c != idx1 and c not in self.nodes[idx1].children:
                edge_label = self.get_edge_label(idx2, c)
                self.add_edge(idx1, c, edge_label)

        self.nodes[idx1].SWAPPED = False
        self.nodes[idx1].incoming_traces = self.nodes[idx1].incoming_traces | self.nodes[idx2].incoming_traces
        self.remove_node(idx2)

    def get_multi_roots(self):
        multi_roots = []
        for n in self.nodes.keys():
            # root TODO: Detect root with circle
            if self.nodes[n].parents == [] and self.nodes[n].children != []:
                multi_roots.append(n)
        return multi_roots

    def bfs(self, root='r', OUTPUT_NODE_SET=False):
        """if given root and graph is connected, we can do breadth first search"""
        from collections import deque
        visited_nodes = set()
        dep_tuples = []

        queue = deque([root])
        while queue:
            next = queue.popleft()
            if next in visited_nodes:
                continue
            visited_nodes.add(next)
            for child in sorted(self.nodes[next].children):
                if not (next, child) in dep_tuples:
                    if not child in visited_nodes:
                        queue.append(child)
                    dep_tuples.append((next, child))
        return visited_nodes, dep_tuples

    def topologicalSort(self):
        from collections import deque
        stack = deque([])
        visited = set()

        for i in self.nodes:
            if i not in visited:
                self.topologicalSortUtil(i, visited, stack)

        return stack

    def topologicalSortUtil(self, idx, visited, stack):
        visited.add(idx)
        for child in self.nodes[idx].children:
            if child not in visited:
                self.topologicalSortUtil(child, visited, stack)

        stack.appendleft(idx)

    def tuples(self):
        """traverse the graph in index increasing order"""
        graph_tuples = []
        node_set = set()
        for n in sorted(self.nodes.keys()):
            if (self.nodes[n].parents == [] or n not in node_set) and self.nodes[n].children != []:  # root
                visited_nodes, sub_tuples = self.bfs(n, True)
                graph_tuples.extend(
                    [st for st in sub_tuples if st not in graph_tuples])
                node_set.update(visited_nodes)
        return graph_tuples

    def postorder(self, root=0, seq=None):
        """only for dependency trees"""
        if seq is None:
            seq = []
        if self.nodes[root].children == []:
            seq.append(root)

        else:
            for child in self.nodes[root].children:
                if child not in seq:
                    self.postorder(child, seq)
            seq.append(root)
        return seq

    def leaves(self):
        """return all the leaves ordered by their indexes in the sentence"""
        leaves = []
        for nidx in self.nodes:
            if self.nodes[nidx].children == []:
                leaves.append(nidx)
        return sorted(leaves)

    def locInTree(self, idx):
        depth = 0
        candidates = self.leaves()
        while idx not in candidates:
            candidates = sorted(list(
                set([self.nodes[l].parents[0] for l in candidates if self.nodes[l].parents])))
            depth += 1
        assert idx in candidates
        return (candidates.index(idx), depth)

    def path(self, idx):
        """path from root, only for tree structure"""
        path = []
        cur = self.nodes[idx]
        path.insert(0, idx)
        while cur.parents:
            cur = self.nodes[cur.parents[0]]
            if cur.id in path:
                import pdb
                pdb.set_trace()
            path.insert(0, cur.id)

        return path

    def get_path(self, idx1, idx2):
        """path between two nodes, only for tree structure"""
        path1 = self.path(idx1)
        path2 = self.path(idx2)
        direction = '01'
        lenth = len(path1) if len(path1) < len(path2) else len(path2)
        for i in range(lenth):
            if path1[i] != path2[i]:
                break
        if path1[i] == path2[i]:
            if len(path1) > len(path2):
                path = list(reversed(path1[i:]))
                direction = '0'
            else:
                path = path2[i:]
                direction = '1'
        else:
            path = list(reversed(path1[i - 1:])) + path2[i:]

        return path, direction

    def relativePos(self, currentIdx, otherIdx):
        cindex, cdepth = self.locInTree(currentIdx)
        oindex, odepth = self.locInTree(otherIdx)
        return (cindex - oindex, cdepth - odepth)

    def relativePos2(self, currentIdx, otherIdx):
        cpath = self.path(currentIdx)
        opath = self.path(otherIdx)

        # if otherIdx != 0 and otherIdx != currentIdx and \
        #   otherIdx not in self.nodes[currentIdx].parents and \
        #   otherIdx not in self.nodes[currentIdx].children:
        type = None
        if True:
            if len(cpath) > 1 and len(opath) > 1:
                if cpath[-2] == opath[-2]:  # same parent
                    type = 'SP'

            if len(cpath) > 2 and len(opath) > 2:
                # same grand parent not same parent
                if cpath[-3] == opath[-3] and cpath[-2] != opath[-2]:
                    if not type:
                        type = 'SGP'
                    else:
                        raise Exception("Not mutual exclusive child type")
                # current node's parent's brother
                if cpath[-3] == opath[-2] and opath[-1] != cpath[-2]:
                    # return 'PB'
                    if not type:
                        type = 'PB'
                    else:
                        raise Exception("Not mutual exclusive child type")
                # other node's parent's brother
                if opath[-3] == cpath[-2] and cpath[-1] != opath[-2]:
                    # return 'rPB'
                    if not type:
                        type = 'rPB'
                    else:
                        raise Exception("Not mutual exclusive child type")
                if opath[-1] in cpath:  # otherIdx on currentIdx's path
                    # return 'P'+str(len(cpath)-1-cpath.index(opath[-1]))
                    # return 'P'
                    if not type:
                        type = 'P'
                    else:
                        raise Exception("Not mutual exclusive child type")
                if cpath[-1] in opath:  # currentIdx on otherIdx's path
                    # return 'rP'+str(len(opath)-1-opath.index(cpath[-1]))
                    # return 'rP'
                    if not type:
                        type = 'rP'
                    else:
                        raise Exception("Not mutual exclusive child type")

        return type if type else 'O'

    def get_possible_children_unconstrained(self, currentIdx):
        """return all the other nodes in the tree not violated the rules"""
        candidate_children = []
        for otherIdx in self.nodes:
            if otherIdx != 0 and otherIdx != currentIdx and \
                    otherIdx not in self.nodes[currentIdx].children:
                candidate_children.append(otherIdx)

        return candidate_children

    def get_possible_parent_unconstrained(self, currentIdx, currentChildIdx):
        candidate_parents = set([])
        for otherIdx in self.nodes:
            if otherIdx != currentChildIdx and \
               otherIdx not in self.nodes[currentChildIdx].parents:
                candidate_parents.add(otherIdx)
        return candidate_parents

    def get_possible_reentrance_constrained(self, currentIdx, currentChildIdx):
        '''adding siblings, mainly for control verb'''
        # if self.nodes[currentIdx].parents:
        cur_p = self.nodes[currentIdx]
        cur = self.nodes[currentChildIdx]
        result_set = set([])
        if len(cur_p.children) > 1:
            result_set = set([sb for sb in cur_p.children if sb !=
                              currentChildIdx and sb not in cur.parents and sb not in cur.children])
        '''
        cur_p = self.nodes[currentIdx]
        j=0
        while cur_p.parents and j < i:
            cur_gp = cur_p.parents[0]
            result_set.add(cur_gp)
            cur_p = self.nodes[cur_gp]
            j+=1
        '''
        # adding incoming traces
        result_set = result_set.union(set(gov for rel, gov in cur.incoming_traces if gov !=
                                          currentChildIdx and gov in self.nodes and gov not in cur.parents and gov not in cur.children))
        # adding predicate
        if isinstance(currentChildIdx, int) and 'pred' in self.sent[currentChildIdx]:
            result_set = result_set.union(set(prd for prd in self.sent[currentChildIdx]['pred'] if prd !=
                                              currentChildIdx and prd in self.nodes and prd not in cur.parents and prd not in cur.children))
        return result_set

    def get_possible_parent_constrained(self, currentIdx, currentChildIdx, i=2):
        '''promotion: only add ancestors 2 levels up the current node'''
        cur_p = self.nodes[currentIdx]
        cur = self.nodes[currentChildIdx]
        children = sorted(self.nodes[currentIdx].children)
        c = children.index(currentChildIdx)
        candidate_parents = set([])
        visited = set([])

        #candidate_parents.update([sb for sb in cur_p.children if sb != currentChildIdx and sb not in self.nodes[currentChildIdx].parents and sb not in self.nodes[currentChildIdx].children])

        if c > 0:
            left_sp = self.nodes[children[c - 1]]
            candidate_parents.add(children[c - 1])
            visited.add(children[c - 1])
            while left_sp.children:
                ls_r_child = sorted(left_sp.children)[-1]
                if ls_r_child in visited:
                    break
                visited.add(ls_r_child)
                if ls_r_child != currentChildIdx and ls_r_child not in self.nodes[currentChildIdx].children and ls_r_child not in self.nodes[currentChildIdx].parents:
                    candidate_parents.add(ls_r_child)
                left_sp = self.nodes[ls_r_child]

        if c < len(children) - 1:
            right_sp = self.nodes[children[c + 1]]
            candidate_parents.add(children[c + 1])
            visited.add(children[c + 1])
            while right_sp.children:
                rs_l_child = sorted(right_sp.children)[0]
                if rs_l_child in visited:
                    break
                visited.add(rs_l_child)
                if rs_l_child != currentChildIdx and rs_l_child not in self.nodes[currentChildIdx].children and rs_l_child not in self.nodes[currentChildIdx].parents:
                    candidate_parents.add(rs_l_child)
                right_sp = self.nodes[rs_l_child]

        cur_p = self.nodes[currentIdx]
        j = 0
        while cur_p.parents and j < i:
            cur_gp = cur_p.parents[0]
            if cur_gp != currentChildIdx and cur_gp not in self.nodes[currentChildIdx].children and cur_gp not in self.nodes[currentChildIdx].parents:
                candidate_parents.add(cur_gp)
            cur_p = self.nodes[cur_gp]
            j += 1
        sep_nodes = self.multi_roots[:]
        sep_nodes.remove(self.root)
        candidate_parents.update(sep_nodes)

        # if isinstance(currentChildIdx,int) and 'pred' in self.sent[currentChildIdx]:
        #     candidate_parents = candidate_parents.union(set(prd for prd in self.sent[currentChildIdx]['pred'] if prd != currentChildIdx and prd in self.nodes and prd not in cur.parents and prd not in cur.children))
        return candidate_parents

    def get_possible_children(self, currentIdx):
        """only for tree structure, get all the candidate children for current node idx"""
        cpath = self.path(currentIdx)
        possible_children = []
        num_SP = 0
        num_SGP = 0
        num_PB = 0
        num_rPB = 0
        num_P = 0
        num_rP = 0
        num_other = 0
        num_total = 0

        for otherIdx in self.nodes:
            if otherIdx != 0 and otherIdx != currentIdx and \
               otherIdx not in self.nodes[currentIdx].parents and \
               otherIdx not in self.nodes[currentIdx].children:
                num_total += 1
                opath = self.path(otherIdx)
                if len(cpath) > 1 and len(opath) > 1 and cpath[-2] == opath[-2]:
                    possible_children.append('SP' + str(num_SP))
                    num_SP += 1

                if len(cpath) > 2 and len(opath) > 2 and cpath[-3] == opath[-3] and cpath[-2] != opath[-2]:
                    possible_children.append('SGP' + str(num_SGP))
                    num_SGP += 1

                if len(cpath) > 2 and len(opath) > 2 and cpath[-3] == opath[-2] and opath[-1] != cpath[-2]:
                    possible_children.append('PB' + str(num_PB))
                    num_PB += 1

                if len(cpath) > 2 and len(opath) > 2 and cpath[-2] == opath[-3] and cpath[-1] != opath[-2]:
                    possible_children.append('rPB' + str(num_rPB))
                    num_rPB += 1

                if len(cpath) > 0 and len(opath) > 0 and opath[-1] in cpath:
                    possible_children.append('P' + str(num_P))
                    num_P += 1

                if len(cpath) > 0 and len(opath) > 0 and cpath[-1] in opath:
                    possible_children.append('rP' + str(num_rP))
                    num_rP += 1

                assert num_total == len(possible_children)

        return possible_children

    def is_produce_circle(self, currentIdx, node_to_add):
        currentNode = self.nodes[currentIdx]
        stack = [currentIdx]
        while stack:
            next = stack.pop()

            parents = self.nodes[next].parents
            if parents:
                if node_to_add in parents:
                    return True
                else:
                    stack.extend(self.nodes[next].parents)
        return False

    def flipConst(self):
        '''
        since in amr const variable will not have children
        we simply flip the relation when we run into const variable as parent
        '''
        parsed_tuples = self.tuples()
        visited_tuples = set()
        # for parent,child in parsed_tuples:
        while parsed_tuples:
            parent, child = parsed_tuples.pop()
            visited_tuples.add((parent, child))

            if (isinstance(self.nodes[parent].tag, ConstTag) or r'/' in self.nodes[parent].tag) and not isinstance(self.nodes[child].tag, ConstTag):
                for p in self.nodes[parent].parents[:]:
                    if p != child:
                        self.remove_edge(p, parent)
                        self.add_edge(p, child)

                if child in self.nodes[parent].parents:
                    self.remove_edge(parent, child)
                else:

                    self.remove_edge(parent, child)
                    self.add_edge(child, parent)

            elif isinstance(self.nodes[parent].tag, ConstTag) and isinstance(self.nodes[child].tag, ConstTag):
                for p in self.nodes[parent].parents[:]:
                    if p != child:
                        self.add_edge(p, child)

                self.remove_edge(parent, child)

            parsed_tuples = set(self.tuples()) - visited_tuples

    def print_tuples(self, bfs=False):
        """print the dependency graph as tuples"""
        if not self.sent:
            if bfs:
                return '\n'.join("(%s(%s-%s),(%s-%s))" % (self.get_edge_label(g, d), ','.join(w for w in self.nodes[g].words), g, ','.join(t for t in self.nodes[d].words), d) for g, d in self.bfs()[1])
            else:
                return '\n'.join("(%s(%s-%s),(%s-%s))" % (self.get_edge_label(g, d), ','.join(w for w in self.nodes[g].words), g, ','.join(t for t in self.nodes[d].words), d) for g, d in self.tuples())
        else:
            output = ''
            if bfs:
                seq = self.bfs()[1]
            else:
                seq = self.tuples()

            for g, d in seq:
                g_span = ','.join(tok['form'] for tok in self.sent[g:self.nodes[g].end]) if isinstance(
                    g, int) else ','.join(self.nodes[g].words)
                d_span = ','.join(tok['form'] for tok in self.sent[d:self.nodes[d].end]) if isinstance(
                    d, int) else ','.join(self.nodes[d].words)
                output += "(%s(%s-%s:%s),(%s-%s:%s))\n" % (self.get_edge_label(g, d),
                                                           g_span, g, self.nodes[g].tag, d_span, d, self.nodes[d].tag)

            return output

    def print_tuples_dsn(self):
        """print the dependency graph as tuples"""
        output = ''
        seq = self.tuples()

        for g, d in seq:

            g_nset = self.nodes[g].index_set
            d_nset = self.nodes[d].index_set
            g_tag = self.nodes[g].tag
            d_tag = self.nodes[d].tag

            g_span = ','.join(self.sent[idx]['form']
                              for idx in sorted(list(g_nset))) if g_nset else g_tag
            try:
                d_span = ','.join(self.sent[idx]['form']
                                  for idx in sorted(list(d_nset))) if d_nset else d_tag
            except IndexError as e:
                print(str(__file__) + ':' + str(e), file=sys.stderr)
                print(d_nset, self.sent, file=sys.stderr)
                exit(1)
            output += "(%s(%s-%s:%s),(%s-%s:%s))\n" % (self.get_edge_label(g,
                                                                           d), g_span, g, g_tag, d_span, d, d_tag)

        return output.encode('utf8')

    def min_index(self, root):
        '''give smallest index in the subgraph rooted at root'''
        visited = set()
        return self.min_index_util(root, visited)

    def min_index_util(self, r, vset):
        vset.add(r)
        tmp = r
        for c in self.nodes[r].children:
            if c not in vset:
                mc = self.min_index_util(c, vset)
                if mc < tmp:
                    tmp = mc

        return tmp

    def reIndex(self):
        index_list = sorted(self.nodes.keys())
        new_index_list = index_list[:]
        i = len(index_list) - 1

        while i > 0:
            ni = index_list[i]
            if not isinstance(ni, int):
                new_index_list.remove(ni)
                min_desc_index = self.min_index(ni)
                if min_desc_index not in new_index_list:  # leaf abstract concept
                    min_desc_index = self.nodes[min_desc_index].parents[0]
                new_index_list.insert(new_index_list.index(min_desc_index), ni)
            else:
                break

            i -= 1
        return new_index_list

    def print_dep_style_graph(self):
        '''
        output the dependency style collapsed span graph, which can be displayed with DependenSee tool;
        also mark the error
        '''
        tuple_str = ''
        index_list = self.reIndex()
        for g, d in self.tuples():
            span_g = ','.join(tok['form'] for tok in self.sent[g:self.nodes[g].end]) if isinstance(
                g, int) else ','.join(self.nodes[g].words)
            span_d = ','.join(tok['form'] for tok in self.sent[d:self.nodes[d].end]) if isinstance(
                d, int) else ','.join(self.nodes[d].words)
            tag_g = '%s%s' % (self.get_node_tag(g), self.nodes_error_table[g])
            tag_d = '%s%s' % (self.get_node_tag(d), self.nodes_error_table[d])
            edge_label = '%s%s' % (self.get_edge_label(
                g, d), self.edges_error_table[(g, d)])
            tuple_str += "%s(%s-%s:%s, %s-%s:%s)\n" % (edge_label, span_g,
                                                       index_list.index(g), tag_g, span_d, index_list.index(d), tag_d)

        return tuple_str

    def getPGStyleGraph(self, focus=None):

        result = ''
        if focus:
            for g, d in self.tuples():
                if g == focus[0] or g == focus[1]:
                    gwords = '"%s-%d"[blue]' % (
                        ','.join(w for w in self.nodes[g].words), g)
                else:
                    gwords = '"%s-%d"' % (
                        ','.join(w for w in self.nodes[g].words), g)
                if d == focus[0] or d == focus[1]:
                    dwords = '"%s-%d"[blue]' % (
                        ','.join(w for w in self.nodes[d].words), d)
                else:
                    dwords = '"%s-%d"' % (
                        ','.join(w for w in self.nodes[d].words), d)
                if (g, d) == focus:
                    result += '%s ->[red] %s;\n' % (gwords, dwords)
                else:
                    result += '%s -> %s;\n' % (gwords, dwords)
            return result
        else:
            for g, d in self.tuples():
                gwords = ','.join(w for w in self.nodes[g].words)
                dwords = ','.join(w for w in self.nodes[d].words)
                result += '"%s-%d" -> "%s-%d";\n' % (gwords, g, dwords, d)
            return result
