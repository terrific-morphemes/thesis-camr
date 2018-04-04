#!/usr/bin/python
'''
class for different oracles
'''
from __future__ import print_function
# from constants import
import sys
from collections import defaultdict
from common.util import lcsub, StrLiteral, Quantity, is_abstract
from constants import START_ID, START_EDGE, ABT_PREFIX
from constants import NEXT1, NEXT2, REENTRANCE, \
    REATTACH, DELETENODE, REPLACEHEAD, \
    SWAP, MERGE, INFER
from constants import NOT_APPLY


class Oracle():

    def __init__(self, verbose=0):
        self.verbose = verbose
        self.add_child_table = defaultdict(int)
        self.infer_table = defaultdict(int)

    def give_ref_action(self):
        raise NotImplementedError('Not implemented!')


class DynOracle(Oracle):
    '''Using dynamic programming find the optimal action sequence
       for a initial state given gold graph
       The score of one action sequence is defined as 
       parsed graph and gold graph
    '''

    def give_ref_action_seq(self, state, ref_graph):
        pass

    def give_ref_action(self):
        pass


class DetOracleABT(Oracle):
    '''
       deterministic oracle try to infer the unaligned concepts given gold span graph
    '''

    def give_ref_action(self, state, ref_graph):
        self.currentIdx = state.idx
        self.currentChildIdx = state.cidx
        self.currentIdx_p = state.idx
        self.currentChildIdx_p = state.cidx
        self.currentGraph = state.A
        self.refGraph = ref_graph

        return self.give_ref_action_aux()

    def give_ref_action_aux(self):

        def isCorrectReplace(childIdx, node, rgraph):
            for p in node.parents:
                if p in rgraph.nodes and childIdx in rgraph.nodes[p].children:
                    return True
            return False

        if self.currentIdx == START_ID:
            return {'type': NEXT2}, None

        currentNode = self.currentGraph.nodes[self.currentIdx]
        currentChild = self.currentGraph.nodes[self.currentChildIdx] if self.currentChildIdx in self.currentGraph.nodes else None
        #goldNodeSet = [self.refGraph.abt_node_table[k] if k in self.refGraph.abt_node_table else k for k in self.refGraph.nodes]
        goldNodeSet = self.refGraph.nodes
        self.currentIdx = self.currentGraph.abt_node_table.get(
            self.currentIdx, self.currentIdx)  # might be abstract node
        self.currentChildIdx = self.currentGraph.abt_node_table.get(
            self.currentChildIdx, self.currentChildIdx)
        is_contained = self.refGraph.isContained(currentNode)

        result_act_type = None
        result_act_label = None

        if self.currentIdx in goldNodeSet:

            goldNode = self.refGraph.nodes[self.currentIdx]

            if self.currentChildIdx:
                if self.currentChildIdx == START_ID:
                    ret_action = self.try_insert(
                        goldNode, currentNode, goldNodeSet)
                    return ret_action

                if self.currentChildIdx in goldNodeSet:

                    goldChild = self.refGraph.nodes[self.currentChildIdx]

                    if goldChild.contains(currentNode) or goldNode.contains(currentChild):
                        return {'type': MERGE}, None  # merge

                    if self.currentChildIdx in goldNode.children:  # correct
                        ret_action = self.try_reentrance(
                            currentChild, goldChild)
                        return ret_action

                    thead = self.need_swap(
                        self.currentIdx, self.currentIdx_p, self.currentGraph, goldChild, self.refGraph)
                    if thead is not None:  # in goldChild.children:
                        gold_edge = self.refGraph.get_edge_label(
                            self.currentChildIdx, thead)
                        return {'type': SWAP}, gold_edge  # swap

                    ret_action = self.try_attach(goldChild, currentChild)
                    return ret_action

                else:
                    is_contained_child = self.refGraph.isContained(
                        currentChild)
                    if is_contained_child:
                        if is_contained_child == goldNode.id:
                            return {'type': MERGE}, None
                        else:
                            return {'type': NEXT1}, None
                    else:
                        return {'type': NEXT1}, None

            else:  # beta is empty
                gold_tag = goldNode.tag
                # next: done with the current node move to next one
                return {'type': NEXT2, 'tag': gold_tag}, None

        elif is_contained:
            true_currentNode_idx = is_contained
            if self.currentChildIdx:
                if self.currentChildIdx in goldNodeSet:
                    goldChild = self.refGraph.nodes[self.currentChildIdx]
                    # only handles parent-child merge
                    if goldChild.contains(currentNode):
                        return {'type': MERGE}, None
                    else:
                        gold_edge = self.refGraph.edges.get(
                            (true_currentNode_idx, self.currentChildIdx), None)
                        return {'type': NEXT1}, gold_edge
                else:
                    return {'type': NEXT1}, None

            else:
                gold_tag = self.refGraph.nodes[true_currentNode_idx].tag
                return {'type': NEXT2, 'tag': gold_tag}, None

        else:  # current node is not in gold set
            if self.currentChildIdx:
                if self.currentChildIdx == START_ID:
                    return {'type': NEXT1}, None
                if self.currentChildIdx in goldNodeSet:

                    goldChild = self.refGraph.nodes[self.currentChildIdx]
                    # current node's parents are already aligned
                    if isCorrectReplace(self.currentChildIdx, currentNode, self.refGraph) or len(currentNode.children) == 1:
                        return {'type': REPLACEHEAD}, None  # replace head
                    else:
                        ret_action = self.try_attach(goldChild, currentChild)
                        return ret_action
                else:
                    if self.verbose > 1:
                        print >> sys.stderr, "Current child node %s should have been deleted!" % (
                            currentChildIdx)
                    return {'type': NEXT1}, None
            else:
                # here currentNode.children must be empty
                if currentNode.children and self.verbose > 1:
                    print >> sys.stderr, "Unaligned node %s has children" % (
                        currentNode.start)

                return {'type': DELETENODE}, None

        skip = NEXT1 if currentChildIdx else NEXT2
        return {'type': skip}, None

    def try_align_parent(self, abtParentIdx, currentNode, goldNodeSet):
        abtParentTag = self.refGraph.nodes[abtParentIdx].tag

        if re.match('have-.*-role-91', abtParentTag) or abtParentTag.endswith('-quantity')\
           or abtParentTag.endswith('-entity'):  # this is a predefined entity tag
            return False
        tag_form = abtParentTag
        for p in currentNode.parents:
            if isinstance(p, int) and p not in goldNodeSet:
                word_form = self.currentGraph.nodes[p].words[0].lower()
                lcs_len = lcsub(word_form, tag_form)
                if lcs_len > 3:
                    self.refGraph.add_abt_mapping(abtParentIdx, p)
                    self.currentGraph.add_abt_mapping(p, abtParentIdx)
                    return True
        return False

    def getABTParent(self, gn):
        for p in gn.parents:
            # if not isinstance(p,int):
            # if p.startswith('a'):
            if is_abstract(p):
                return p
        return None

    def need_swap(self, idx, idx_p, cg, gc, rg):
        cur_head_p = cg.find_true_head(idx_p)
        try:
            cur_head = cg.abt_node_table[cur_head_p] if cur_head_p in cg.abt_node_table else cur_head_p
        except KeyError:
            print >> sys.stderr, "inferred abt node cur_head_p not in ref graph!"
            return None

        if cur_head in gc.children:
            return cur_head
        cur_grand_head = self.getABTParent(rg.nodes[cur_head])
        # if cur_head.startswith('a') and cur_grand_head in gc.children:
        if is_abstract(cur_head) and cur_grand_head in gc.children:
            return cur_grand_head
        return None

    def try_insert(self, goldNode, currentNode, goldNodeSet):

        def need_infer(gn, cn, gnset, cg, rg):
            for p in cn.parents:
                # correct/merge/swap arc
                if p in gn.parents or gn.contains(cg.nodes[p]) or p in gn.children:
                    return False
            return True

        def is_aligned(abtIdx, rg, cg):
            return abtIdx in rg.abt_node_table and rg.abt_node_table[abtIdx] in cg.nodes

        abtParentIdx = self.getABTParent(goldNode)
        ret_actions = []

        if need_infer(goldNode, currentNode, goldNodeSet,
                      self.currentGraph, self.refGraph) \
            and abtParentIdx \
            and not is_aligned(abtParentIdx,
                               self.refGraph, self.currentGraph):

            gold_tag = self.refGraph.nodes[abtParentIdx].tag
            abt_node_index = ABT_PREFIX + str(self.currentGraph.abt_node_num)
            self.refGraph.add_abt_mapping(abtParentIdx, abt_node_index)
            self.currentGraph.add_abt_mapping(abt_node_index, abtParentIdx)

            return {'type': INFER}, gold_tag

        return {'type': NEXT1}, START_EDGE

    def try_reentrance(self, currentChild, goldChild):
        curr_parents = [self.currentGraph.abt_node_table[cp]
                        if cp in self.currentGraph.abt_node_table else cp for cp in currentChild.parents]
        parents_to_add = [
            p for p in goldChild.parents if p not in curr_parents]

        if parents_to_add:
            pta = parents_to_add[0]
            if pta in self.refGraph.abt_node_table:
                pta = self.refGraph.abt_node_table[pta]
            if pta in self.currentGraph.get_possible_parent_unconstrained(self.currentIdx_p, self.currentChildIdx_p):
                gold_edge = self.refGraph.get_edge_label(
                    parents_to_add[0], self.currentChildIdx)
                return {'type': REENTRANCE, 'parent_to_add': pta}, gold_edge
            else:
                gold_edge = self.refGraph.get_edge_label(
                    self.currentIdx, self.currentChildIdx)
                return {'type': NEXT1}, gold_edge  # next
        else:
            gold_edge = self.refGraph.get_edge_label(
                self.currentIdx, self.currentChildIdx)
            return {'type': NEXT1}, gold_edge  # next

    def try_attach(self, goldChild, currentChild):
        curr_parents = [self.currentGraph.abt_node_table.get(
            cp, cp) for cp in currentChild.parents]
        parents_to_attach = [
            p for p in goldChild.parents if p not in curr_parents]

        if parents_to_attach:
            pta = parents_to_attach[0]
            if pta in self.refGraph.abt_node_table:
                pta = self.refGraph.abt_node_table[pta]
            if pta in self.currentGraph.get_possible_parent_unconstrained(self.currentIdx_p, self.currentChildIdx_p):
                gold_edge = self.refGraph.get_edge_label(
                    parents_to_attach[0], self.currentChildIdx)
                return {'type': REATTACH, 'parent_to_attach': pta}, gold_edge
            else:
                # violates the attachment constraints
                return {'type': NEXT1}, None

        else:
            if self.verbose > 1:
                print >> sys.stderr, "Current child node %s doesn't have parents in gold span graph!" % (
                    self.currentChildIdx)
            return {'type': NEXT1}, None
