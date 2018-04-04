#!/usr/bin/python

# class for different oracles
from constants import *
import sys
from common.util import lcsub, StrLiteral, Quantity


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
            if not isinstance(p, int):
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
        if (not isinstance(cur_head, int) and cur_grand_head in gc.children):
            return cur_grand_head
        return None

    def try_insert_align(self, goldNode, currentNode, goldNodeSet):

        def need_infer(cidx, gn, cn, gnset, cg, rg):
            for p in cn.parents:
                # correct/merge/swap arc
                if p in gn.parents or gn.contains(cg.nodes[p]) or p in gn.children:
                    return False
            return True

        def is_aligned(abtIdx, rg, cg):
            return abtIdx in rg.abt_node_table and rg.abt_node_table[abtIdx] in cg.nodes

        def need_addchild(goldNode):
            unaligned_children = [c for c in goldNode.children if not isinstance(
                c, int) and not c in self.refGraph.abt_node_table]
            if len(unaligned_children) == 0:
                return False, None, None

            add_abt_child = unaligned_children[0]
            add_abt_child_node = self.refGraph.nodes[add_abt_child]
            if len(add_abt_child_node.children) == 0 and (not isinstance(add_abt_child, (StrLiteral, Quantity))):
                # if not isinstance(add_abt_child, (StrLiteral,Quantity)):
                return True, add_abt_child, add_abt_child_node
            else:
                print >> sys.stderr, 'TRY ALIGNING CHILD'
                pass

            return False, None, None

        abtParentIdx = self.getABTParent(goldNode)

       # if abtParentIdx and not is_aligned(abtParentIdx,self.refGraph,self.currentGraph) \
       #         and self.try_align_parent(abtParentIdx, currentNode, goldNodeSet):
       #     # print >> sys.stderr, "TRY ALIGNING HERE!"
       #     return {'type':NEXT1},START_EDGE

        if need_infer(self.currentIdx, goldNode, currentNode, goldNodeSet, self.currentGraph, self.refGraph) \
                and abtParentIdx and not is_aligned(abtParentIdx, self.refGraph, self.currentGraph):
                # and (len(self.refGraph.nodes[abtParentIdx].children) == 1 or not isinstance(self.currentIdx_p, int) or ((goldNode.words[0] in NOMLIST or self.currentGraph.sent[self.currentIdx_p]['lemma'].lower() in NOMLIST) and len(goldNode.words) == 1)):

            gold_tag = self.refGraph.nodes[abtParentIdx].tag
            self.infer_table[gold_tag] += 1
            abt_node_index = ABT_PREFIX + str(self.currentGraph.abt_node_num)
            self.refGraph.add_abt_mapping(abtParentIdx, abt_node_index)
            self.currentGraph.add_abt_mapping(abt_node_index, abtParentIdx)

            return {'type': INFER}, gold_tag

        # is_addchild, add_abt_child, add_abt_child_node = need_addchild(goldNode)
        # if is_addchild:

        #     child_tag = add_abt_child_node.tag
        #     self.add_child_table[child_tag] += 1
        #     child_edge = self.refGraph.get_edge_label(self.currentIdx, add_abt_child)
        #     abt_node_index = ABT_PREFIX+str(self.currentGraph.abt_node_num)
        #     self.refGraph.add_abt_mapping(add_abt_child,abt_node_index)
        #     self.currentGraph.add_abt_mapping(abt_node_index,add_abt_child)
        #     return {'type':ADDCHILD}, child_tag

        return {'type': NEXT1}, START_EDGE

    def try_insert(self, goldNode, currentNode, goldNodeSet):

        def need_infer(cidx, gn, cn, gnset, cg, rg):
            for p in cn.parents:
                # correct/merge/swap arc
                if p in gn.parents or gn.contains(cg.nodes[p]) or p in gn.children:
                    return False
            return True

        def is_aligned(abtIdx, rg, cg):
            return abtIdx in rg.abt_node_table and rg.abt_node_table[abtIdx] in cg.nodes

        def need_addchild(goldNode):
            unaligned_children = [c for c in goldNode.children if not isinstance(
                c, int) and not c in self.refGraph.abt_node_table]
            if len(unaligned_children) == 0:
                return False, None, None

            add_abt_child = unaligned_children[0]
            add_abt_child_node = self.refGraph.nodes[add_abt_child]
            if len(add_abt_child_node.children) == 0:
                return True, add_abt_child, add_abt_child_node

            return False, None, None

        abtParentIdx = self.getABTParent(goldNode)
        ret_actions = []

        if need_infer(self.currentIdx, goldNode, currentNode, goldNodeSet, self.currentGraph, self.refGraph) \
                and abtParentIdx and not is_aligned(abtParentIdx, self.refGraph, self.currentGraph) \
                and (len(self.refGraph.nodes[abtParentIdx].children) == 1 or not isinstance(self.currentIdx_p, int) or ((goldNode.words[0] in NOMLIST or self.currentGraph.sent[self.currentIdx_p]['lemma'].lower() in NOMLIST) and len(goldNode.words) == 1)):

            gold_tag = self.refGraph.nodes[abtParentIdx].tag
            abt_node_index = ABT_PREFIX + str(self.currentGraph.abt_node_num)
            self.refGraph.add_abt_mapping(abtParentIdx, abt_node_index)
            self.currentGraph.add_abt_mapping(abt_node_index, abtParentIdx)

            return {'type': INFER}, gold_tag

        is_addchild, add_abt_child, add_abt_child_node = need_addchild(
            goldNode)
        if is_addchild:
            child_tag = add_abt_child_node.tag
            child_edge = self.refGraph.get_edge_label(
                self.currentIdx, add_abt_child)
            abt_node_index = ABT_PREFIX + str(self.currentGraph.abt_node_num)
            self.refGraph.add_abt_mapping(add_abt_child, abt_node_index)
            self.currentGraph.add_abt_mapping(abt_node_index, add_abt_child)
            return {'type': ADDCHILD}, child_tag

        return {'type': NEXT1}, START_EDGE

    def try_reentrance(self, currentChild, goldChild):
        parents_to_add = [p for p in goldChild.parents if p not in [self.currentGraph.abt_node_table[cp]
                                                                    if cp in self.currentGraph.abt_node_table else cp for cp in currentChild.parents]]

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

        parents_to_attach = [p for p in goldChild.parents if p not in [
            self.currentGraph.abt_node_table[cp] if cp in self.currentGraph.abt_node_table else cp for cp in currentChild.parents]]

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

    def give_ref_action_aux(self):

        def isCorrectReplace(childIdx, node, rgraph):
            for p in node.parents:
                if p in rgraph.nodes and childIdx in rgraph.nodes[p].children:
                    return True
            return False

        if self.currentIdx == START_ID:
            return {'type': NEXT2}, None

        # state.get_current_node()
        currentNode = self.currentGraph.nodes[self.currentIdx]
        # state.get_current_child()
        currentChild = self.currentGraph.nodes[self.currentChildIdx] if self.currentChildIdx in self.currentGraph.nodes else None
        goldNodeSet = [self.refGraph.abt_node_table[k]
                       if k in self.refGraph.abt_node_table else k for k in self.refGraph.nodes]
        result_act_type = None
        result_act_label = None

        if self.currentIdx in goldNodeSet:

            if self.currentIdx not in self.refGraph.nodes:
                # might be abstract node
                self.currentIdx = self.currentGraph.abt_node_table[self.currentIdx]
            goldNode = self.refGraph.nodes[self.currentIdx]

            if self.currentChildIdx:
                if self.currentChildIdx == START_ID:
                    ret_action = self.try_insert_align(
                        goldNode, currentNode, goldNodeSet)
                    return ret_action

                if self.currentChildIdx in goldNodeSet:

                    if self.currentChildIdx not in self.refGraph.nodes:
                        self.currentChildIdx = self.currentGraph.abt_node_table[self.currentChildIdx]

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
                    if goldNode.contains(currentChild):
                        return {'type': MERGE}, None
                    else:
                        return {'type': NEXT1}, None

            else:  # beta is empty
                gold_tag = goldNode.tag
                # next: done with the current node move to next one
                return {'type': NEXT2, 'tag': gold_tag}, None

        elif self.refGraph.isContained(self.currentIdx):

            if self.currentChildIdx:
                if self.currentChildIdx in goldNodeSet:
                    if self.currentChildIdx not in self.refGraph.nodes:
                        self.currentChildIdx = self.currentGraph.abt_node_table[self.currentChildIdx]
                    goldChild = self.refGraph.nodes[self.currentChildIdx]
                    if goldChild.contains(currentNode):
                        return {'type': MERGE}, None
                    else:
                        # todo HERE

                        # parents_to_attach = [p for p in goldChild.parents if p not in currentChild.parents and p in self.currentGraph.get_possible_parent_unconstrained(self.currentIdx,self.currentChildIdx_p)]

                        # if parents_to_attach:
                        #     if self.refGraph.nodes[parents_to_attach[0]].contains(currentNode):
                        #         return {'type':NEXT1},None # delay action for future merge
                        #     else:
                        #         gold_edge = self.refGraph.get_edge_label(parents_to_attach[0],self.currentChildIdx)
                        #         return {'type':REATTACH,'parent_to_attach':parents_to_attach[0]},gold_edge
                        # else:
                        #     return {'type':NEXT1},None
                        return {'type': NEXT1}, None

                else:
                    return {'type': NEXT1}, None

            else:
                return {'type': NEXT2}, None

        else:  # current node is not in gold set
            if self.currentChildIdx:
                if self.currentChildIdx == START_ID:
                    return {'type': NEXT1}, None
                if self.currentChildIdx in goldNodeSet:

                    if self.currentChildIdx not in self.refGraph.nodes:
                        self.currentChildIdx = self.currentGraph.abt_node_table[self.currentChildIdx]
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

    def try_infer_mincost(self, goldNode, currentNode, goldNodeSet):

        def need_infer(cidx, gn, cn, gnset, cg, rg):
            for p in cn.parents:
                # correct/merge/swap arc
                if p in gn.parents or gn.contains(cg.nodes[p]) or p in gn.children:
                    return False
            return True

        def is_aligned(abtIdx, rg, cg):
            return abtIdx in rg.abt_node_table and rg.abt_node_table[abtIdx] in cg.nodes

        abtParentIdx = self.getABTParent(goldNode)
        ret_actions = []

        if need_infer(self.currentIdx, goldNode, currentNode, goldNodeSet, self.currentGraph, self.refGraph) \
                and abtParentIdx and not is_aligned(abtParentIdx, self.refGraph, self.currentGraph):  # this indicates infer could be an option

            gold_tag = self.refGraph.nodes[abtParentIdx].tag
            abt_node_index = ABT_PREFIX + str(self.currentGraph.abt_node_num)

            ret_actions.append(
                ({'type': INFER}, gold_tag, (abt_node_index, abtParentIdx)))
            ret_action.append(({'type': NEXT1}, START_EDGE,
                               (currentNode.parents[0], abtParentIdx)))

        ret_actions.append(({'type': NEXT1}, START_EDGE, None))
        return ret_actions

    def give_ref_action_aux_mincost(self):
        '''
        rather than using heuristic rules to give pseudo-deterministic action,
        we return all possible actions when it's ambiguious
        '''

        def isCorrectReplace(childIdx, node, rgraph):
            for p in node.parents:
                if p in rgraph.nodes and childIdx in rgraph.nodes[p].children:
                    return True
            return False

        if self.currentIdx == START_ID:
            return {'type': NEXT2}, None

        # state.get_current_node()
        currentNode = self.currentGraph.nodes[self.currentIdx]
        # state.get_current_child()
        currentChild = self.currentGraph.nodes[self.currentChildIdx] if self.currentChildIdx in self.currentGraph.nodes else None
        goldNodeSet = [self.refGraph.abt_node_table[k]
                       if k in self.refGraph.abt_node_table else k for k in self.refGraph.nodes]
        result_act_type = None
        result_act_label = None

        if self.currentIdx in goldNodeSet:

            if self.currentIdx not in self.refGraph.nodes:
                # might be abstract node
                self.currentIdx = self.currentGraph.abt_node_table[self.currentIdx]
            goldNode = self.refGraph.nodes[self.currentIdx]

            if self.currentChildIdx:
                if self.currentChildIdx == START_ID:
                    ret_actions = self.try_infer_mincost(
                        goldNode, currentNode, goldNodeSet)
                    return ret_actions

                if self.currentChildIdx in goldNodeSet:

                    if self.currentChildIdx not in self.refGraph.nodes:
                        self.currentChildIdx = self.currentGraph.abt_node_table[self.currentChildIdx]

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
                    if goldNode.contains(currentChild):
                        return {'type': MERGE}, None
                    else:
                        return {'type': NEXT1}, None

            else:  # don't have child
                gold_tag = goldNode.tag
                # next: done with the current node move to next one
                return {'type': NEXT2, 'tag': gold_tag}, None

        elif self.refGraph.isContained(self.currentIdx):

            if self.currentChildIdx:
                if self.currentChildIdx in goldNodeSet:
                    if self.currentChildIdx not in self.refGraph.nodes:
                        self.currentChildIdx = self.currentGraph.abt_node_table[self.currentChildIdx]
                    goldChild = self.refGraph.nodes[self.currentChildIdx]
                    if goldChild.contains(currentNode):
                        return {'type': MERGE}, None
                    else:
                        return {'type': NEXT1}, None
                else:
                    return {'type': NEXT1}, None

            else:
                return {'type': NEXT2}, None

        else:  # current node is not in gold set
            if self.currentChildIdx:
                if self.currentChildIdx == START_ID:
                    return {'type': NEXT1}, None
                if self.currentChildIdx in goldNodeSet:
                    if self.currentChildIdx not in self.refGraph.nodes:
                        self.currentChildIdx = self.currentGraph.abt_node_table[self.currentChildIdx]
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


class DetOracleSC(Oracle):
    '''
       deterministic oracle keeps strong connectivity of the graph
       1) using reattach rather than delete edge
       2) discard ADDCHILD
    '''

    def give_ref_action(self, state, ref_graph):

        def isCorrectReplace(childIdx, node, rgraph):
            for p in node.parents:
                if p in rgraph.nodes and childIdx in rgraph.nodes[p].children:
                    return True
            return False

        currentIdx = state.idx
        currentChildIdx = state.cidx
        currentNode = state.get_current_node()
        currentChild = state.get_current_child()
        currentGraph = state.A
        goldNodeSet = ref_graph.nodes.keys()

        result_act_type = None
        result_act_label = None
        if currentIdx in goldNodeSet:
            goldNode = ref_graph.nodes[currentIdx]
            # for child in currentNode.children:
            if currentChildIdx:
                if currentChildIdx in goldNodeSet:
                    goldChild = ref_graph.nodes[currentChildIdx]
                    if goldChild.contains(currentNode) or goldNode.contains(currentChild):
                        return {'type': MERGE}, None  # merge
                        #result_act_type = {'type':MERGE}
                    if currentIdx in goldChild.children and \
                       currentChildIdx in goldNode.children:
                        if self.verbose > 1:
                            print >> sys.stderr, "Circle detected in gold graph!"
                        gold_edge = ref_graph.get_edge_label(
                            currentIdx, currentChildIdx)
                        return {'type': NEXT1}, gold_edge  # next
                        #result_act_type = {'type':NEXT1}
                        #result_act_label = gold_edge
                    elif currentIdx in goldChild.children:
                        gold_edge = ref_graph.get_edge_label(
                            currentChildIdx, currentIdx)
                        return {'type': SWAP}, gold_edge  # swap
                        #result_act_type = {'type':SWAP}
                    elif currentChildIdx in goldNode.children:  # correct
                        parents_to_add = [p for p in goldChild.parents if p not in currentChild.parents and p in currentGraph.get_possible_reentrance_constrained(
                            currentIdx, currentChildIdx)]
                        #parents_to_add = [p for p in goldChild.parents if p not in currentChild.parents and p in currentGraph.get_possible_parent_unconstrained(currentIdx,currentChildIdx)]
                        if parents_to_add:
                            gold_edge = ref_graph.get_edge_label(
                                parents_to_add[0], currentChildIdx)
                            return {'type': REENTRANCE, 'parent_to_add': parents_to_add[0]}, gold_edge
                        else:
                            gold_edge = ref_graph.get_edge_label(
                                currentIdx, currentChildIdx)
                            return {'type': NEXT1}, gold_edge  # next

                    else:
                        # return {'type':DELETEEDGE} # delete edge
                        #result_act_type = {'type':DELETEEDGE}
                        parents_to_attach = [p for p in goldChild.parents if p not in currentChild.parents and p in currentGraph.get_possible_parent_constrained(
                            currentIdx, currentChildIdx)]
                        #parents_to_attach = [p for p in goldChild.parents if p not in currentChild.parents and p in currentGraph.get_possible_parent_unconstrained(currentIdx,currentChildIdx)]
                        if parents_to_attach:
                            gold_edge = ref_graph.get_edge_label(
                                parents_to_attach[0], currentChildIdx)
                            # if gold_edge == 'x': # not actually gold edge, skip this
                            #    return {'type':NEXT1},None
                            # else:
                            return {'type': REATTACH, 'parent_to_attach': parents_to_attach[0]}, gold_edge
                        else:
                            return {'type': NEXT1}, None
                            # return {'type':REATTACH,'parent_to_attach':None},None
                else:
                    if goldNode.contains(currentChild):
                        return {'type': MERGE}, None
                        #result_act_type = {'type':MERGE}
                    else:
                        # return {'type':DELETEEDGE} # delete edge
                        #result_act_type = {'type':DELETEEDGE}
                        #k = ref_graph.isContained(currentChildIdx)
                        # if k:
                        #    return {'type':REATTACH,'parent_to_attach':k}
                        # else:

                        return {'type': NEXT1}, None
                        # return {'type':REATTACH,'parent_to_attach':None},None

            else:
                # if len(currentNode.children) <= len(goldNode.children) and set(currentNode.children).issubset(set(goldNode.children)):
                #children_to_add = [c for c in goldNode.children if c not in currentNode.children and c in currentGraph.get_possible_children_constrained(currentIdx)]

                # if children_to_add:
                #    child_to_add = children_to_add[0]
                #    gold_edge = ref_graph.get_edge_label(currentIdx,child_to_add)
                #    return {'type':ADDCHILD,'child_to_add':child_to_add,'edge_label':gold_edge}
                # else:
                gold_tag = goldNode.tag
                # next: done with the current node move to next one
                return {'type': NEXT2, 'tag': gold_tag}, None
                # else:
                #    if self.verbose > 2:
                #        print >> sys.stderr, "ERROR: Missing actions, current node's and gold node's children:%s  %s"%(str(currentNode.children), str(goldNode.children))
                #    pass

        elif ref_graph.isContained(currentIdx):
            if currentChildIdx:
                if currentChildIdx in goldNodeSet:
                    goldChild = ref_graph.nodes[currentChildIdx]
                    if goldChild.contains(currentNode):
                        return {'type': MERGE}, None
                    else:
                        parents_to_attach = [p for p in goldChild.parents if p not in currentChild.parents and p in currentGraph.get_possible_parent_constrained(
                            currentIdx, currentChildIdx)]
                        #parents_to_attach = [p for p in goldChild.parents if p not in currentChild.parents and p in currentGraph.get_possible_parent_unconstrained(currentIdx,currentChildIdx)]
                        if parents_to_attach:
                            if ref_graph.nodes[parents_to_attach[0]].contains(currentNode):
                                # delay action for future merge
                                return {'type': NEXT1}, None
                            else:
                                gold_edge = ref_graph.get_edge_label(
                                    parents_to_attach[0], currentChildIdx)
                                # if gold_edge == 'x': # not actually gold edge, skip this
                                #    return {'type':NEXT1},None
                                # else:
                                return {'type': REATTACH, 'parent_to_attach': parents_to_attach[0]}, gold_edge
                        else:
                            return {'type': NEXT1}, None
                            # return {'type':REATTACH,'parent_to_attach':None},None
                else:
                    return {'type': NEXT1}, None
                    # return {'type':REATTACH,'parent_to_attach':None},None
            else:
                return {'type': NEXT2}, None

        else:
            if currentChildIdx:
                #assert len(currentNode.children) == 1
                if currentChildIdx in goldNodeSet:
                    goldChild = ref_graph.nodes[currentChildIdx]
                    # or len(currentNode.children) == 1:
                    if isCorrectReplace(currentChildIdx, currentNode, ref_graph):
                        return {'type': REPLACEHEAD}, None  # replace head
                        #result_act_type = {'type':REPLACEHEAD}
                    else:
                        parents_to_attach = [p for p in goldChild.parents if p not in currentChild.parents and p in currentGraph.get_possible_parent_constrained(
                            currentIdx, currentChildIdx)]
                        #parents_to_attach = [p for p in goldChild.parents if p not in currentChild.parents and p in currentGraph.get_possible_parent_unconstrained(currentIdx,currentChildIdx)]
                        if parents_to_attach:
                            if isCorrectReplace(parents_to_attach[0], currentNode, ref_graph):
                                # delay action for future replace head
                                return {'type': NEXT1}, None
                            else:
                                gold_edge = ref_graph.get_edge_label(
                                    parents_to_attach[0], currentChildIdx)
                                # if gold_edge == 'x': # not actually gold edge, skip this
                                #    return {'type':NEXT1},None
                                # else:
                                return {'type': REATTACH, 'parent_to_attach': parents_to_attach[0]}, gold_edge
                        else:
                            return {'type': NEXT1}, None
                            # return {'type':REATTACH,'parent_to_attach':None},None
                else:
                    # return {'type':DELETEEDGE}
                    #result_act_type = {'type':DELETEEDGE}

                    return {'type': NEXT1}, None
                    # return {'type':REATTACH,'parent_to_attach':None},None
            else:
                # here currentNode.children must be empty
                return {'type': DELETENODE}, None
                #result_act_type = {'type':DELETENODE}

        skip = NEXT1 if currentChildIdx else NEXT2
        return {'type': skip}, None


class DetOracle(Oracle):
    '''
       deterministic oracle keeps weak connectivity of the graph
       1) using delete node rather than reattach
       2) discard ADDCHILD
    '''

    def give_ref_action(self, state, ref_graph):

        def isCorrectReplace(childIdx, node, rgraph):
            for p in node.parents:
                if p in rgraph.nodes and childIdx in rgraph.nodes[p].children:
                    return True
            return False

        currentIdx = state.idx
        currentChildIdx = state.cidx
        currentNode = state.get_current_node()
        currentChild = state.get_current_child()
        currentGraph = state.A
        goldNodeSet = ref_graph.nodes.keys()

        result_act_type = None
        result_act_label = None
        if currentIdx in goldNodeSet:
            goldNode = ref_graph.nodes[currentIdx]
            # for child in currentNode.children:
            if currentChildIdx:
                if currentChildIdx in goldNodeSet:
                    goldChild = ref_graph.nodes[currentChildIdx]
                    if goldChild.contains(currentNode) or goldNode.contains(currentChild):
                        return {'type': MERGE}, None  # merge
                        #result_act_type = {'type':MERGE}
                    if currentIdx in goldChild.children and \
                       currentChildIdx in goldNode.children:
                        if self.verbose > 1:
                            print >> sys.stderr, "Circle detected in gold graph!"
                        gold_edge = ref_graph.get_edge_label(
                            currentIdx, currentChildIdx)
                        return {'type': NEXT1}, gold_edge  # next
                        #result_act_type = {'type':NEXT1}
                        #result_act_label = gold_edge
                    elif currentIdx in goldChild.children:
                        gold_edge = ref_graph.get_edge_label(
                            currentChildIdx, currentIdx)
                        return {'type': SWAP}, gold_edge  # swap
                        #result_act_type = {'type':SWAP}
                    elif currentChildIdx in goldNode.children:  # correct
                        parents_to_add = [p for p in goldChild.parents if p not in currentChild.parents and p in currentGraph.get_possible_reentrance_constrained(
                            currentIdx, currentChildIdx)]
                        #parents_to_add = [p for p in goldChild.parents if p not in currentChild.parents and p in currentGraph.get_possible_parent_unconstrained(currentIdx,currentChildIdx)]
                        if parents_to_add:
                            gold_edge = ref_graph.get_edge_label(
                                parents_to_add[0], currentChildIdx)
                            return {'type': REENTRANCE, 'parent_to_add': parents_to_add[0]}, gold_edge
                        else:
                            gold_edge = ref_graph.get_edge_label(
                                currentIdx, currentChildIdx)
                            return {'type': NEXT1}, gold_edge  # next

                    else:
                        # return {'type':DELETEEDGE} # delete edge
                        #result_act_type = {'type':DELETEEDGE}
                        parents_to_attach = [p for p in goldChild.parents if p not in currentChild.parents and p in currentGraph.get_possible_parent_constrained(
                            currentIdx, currentChildIdx)]
                        #parents_to_attach = [p for p in goldChild.parents if p not in currentChild.parents and p in currentGraph.get_possible_parent_unconstrained(currentIdx,currentChildIdx)]
                        if parents_to_attach:
                            gold_edge = ref_graph.get_edge_label(
                                parents_to_attach[0], currentChildIdx)
                            # if gold_edge == 'x': # not actually gold edge, skip this
                            #    return {'type':NEXT1},None
                            # else:
                            return {'type': REATTACH, 'parent_to_attach': parents_to_attach[0]}, gold_edge
                        else:
                            return {'type': NEXT1}, None
                            # return {'type':REATTACH,'parent_to_attach':None},None
                else:
                    if goldNode.contains(currentChild):
                        return {'type': MERGE}, None
                        #result_act_type = {'type':MERGE}
                    else:
                        # return {'type':DELETEEDGE} # delete edge
                        #result_act_type = {'type':DELETEEDGE}
                        #k = ref_graph.isContained(currentChildIdx)
                        # if k:
                        #    return {'type':REATTACH,'parent_to_attach':k}
                        # else:

                        return {'type': NEXT1}, None
                        # return {'type':REATTACH,'parent_to_attach':None},None

            else:
                # if len(currentNode.children) <= len(goldNode.children) and set(currentNode.children).issubset(set(goldNode.children)):
                #children_to_add = [c for c in goldNode.children if c not in currentNode.children and c in currentGraph.get_possible_children_constrained(currentIdx)]

                # if children_to_add:
                #    child_to_add = children_to_add[0]
                #    gold_edge = ref_graph.get_edge_label(currentIdx,child_to_add)
                #    return {'type':ADDCHILD,'child_to_add':child_to_add,'edge_label':gold_edge}
                # else:
                gold_tag = goldNode.tag
                # next: done with the current node move to next one
                return {'type': NEXT2, 'tag': gold_tag}, None
                # else:
                #    if self.verbose > 2:
                #        print >> sys.stderr, "ERROR: Missing actions, current node's and gold node's children:%s  %s"%(str(currentNode.children), str(goldNode.children))
                #    pass

        elif ref_graph.isContained(currentIdx):
            if currentChildIdx:
                if currentChildIdx in goldNodeSet:
                    goldChild = ref_graph.nodes[currentChildIdx]
                    if goldChild.contains(currentNode):
                        return {'type': MERGE}, None
                    else:
                        parents_to_attach = [p for p in goldChild.parents if p not in currentChild.parents and p in currentGraph.get_possible_parent_constrained(
                            currentIdx, currentChildIdx)]
                        #parents_to_attach = [p for p in goldChild.parents if p not in currentChild.parents and p in currentGraph.get_possible_parent_unconstrained(currentIdx,currentChildIdx)]
                        if parents_to_attach:
                            if ref_graph.nodes[parents_to_attach[0]].contains(currentNode):
                                # delay action for future merge
                                return {'type': NEXT1}, None
                            else:
                                gold_edge = ref_graph.get_edge_label(
                                    parents_to_attach[0], currentChildIdx)
                                # if gold_edge == 'x': # not actually gold edge, skip this
                                #    return {'type':NEXT1},None
                                # else:
                                return {'type': REATTACH, 'parent_to_attach': parents_to_attach[0]}, gold_edge
                        else:
                            return {'type': NEXT1}, None
                            # return {'type':REATTACH,'parent_to_attach':None},None
                else:
                    return {'type': NEXT1}, None
                    # return {'type':REATTACH,'parent_to_attach':None},None
            else:
                return {'type': NEXT2}, None

        else:
            if currentChildIdx:
                #assert len(currentNode.children) == 1
                if currentChildIdx in goldNodeSet:
                    goldChild = ref_graph.nodes[currentChildIdx]
                    # or len(currentNode.children) == 1:
                    if isCorrectReplace(currentChildIdx, currentNode, ref_graph):
                        return {'type': REPLACEHEAD}, None  # replace head
                        #result_act_type = {'type':REPLACEHEAD}
                    else:
                        parents_to_attach = [p for p in goldChild.parents if p not in currentChild.parents and p in currentGraph.get_possible_parent_constrained(
                            currentIdx, currentChildIdx)]
                        #parents_to_attach = [p for p in goldChild.parents if p not in currentChild.parents and p in currentGraph.get_possible_parent_unconstrained(currentIdx,currentChildIdx)]
                        if parents_to_attach:
                            if isCorrectReplace(parents_to_attach[0], currentNode, ref_graph):
                                # delay action for future replace head
                                return {'type': NEXT1}, None
                            else:
                                gold_edge = ref_graph.get_edge_label(
                                    parents_to_attach[0], currentChildIdx)
                                # if gold_edge == 'x': # not actually gold edge, skip this
                                #    return {'type':NEXT1},None
                                # else:
                                return {'type': REATTACH, 'parent_to_attach': parents_to_attach[0]}, gold_edge
                        else:
                            return {'type': NEXT1}, None
                            # return {'type':REATTACH,'parent_to_attach':None},None
                else:
                    # return {'type':DELETEEDGE}
                    #result_act_type = {'type':DELETEEDGE}

                    return {'type': NEXT1}, None
                    # return {'type':REATTACH,'parent_to_attach':None},None
            else:
                # here currentNode.children must be empty
                return {'type': DELETENODE}, None
                #result_act_type = {'type':DELETENODE}

        skip = NEXT1 if currentChildIdx else NEXT2
        return {'type': skip}, None

    ''' 
    give local optimal action based on current state and the gold graph
    
    def give_ref_action_seq(self,state):
        pass

    def give_ref_action(self,state,ref_graph):

        def isCorrectReplace(childIdx,node,rgraph):
            for p in node.parents:
                if p in rgraph.nodes and childIdx in rgraph.nodes[p].children:
                    return True
            return False
            
        currentIdx = state.idx
        currentChildIdx = state.cidx
        currentNode = state.get_current_node()
        currentChild = state.get_current_child()
        currentGraph = state.A
        goldNodeSet = ref_graph.nodes.keys()

        result_act_type = None
        result_act_label = None
        if currentIdx in goldNodeSet:
            goldNode = ref_graph.nodes[currentIdx]
            #for child in currentNode.children: 
            if currentChildIdx:
                if currentChildIdx in goldNodeSet:
                    if ref_graph.nodes[currentChildIdx].contains(currentNode) or ref_graph.nodes[currentIdx].contains(currentChild):
                        return {'type':MERGE} # merge
                        #result_act_type = {'type':MERGE}
                    if currentIdx in ref_graph.nodes[currentChildIdx].children and \
                       currentChildIdx in ref_graph.nodes[currentIdx].children:
                        print >> sys.stderr, "Circle detected in gold graph!"
                        gold_edge = ref_graph.get_edge_label(currentIdx,currentChildIdx)
                        return {'type':NEXT1, 'edge_label':gold_edge} # next
                        #result_act_type = {'type':NEXT1}
                        #result_act_label = gold_edge
                    elif currentIdx in ref_graph.nodes[currentChildIdx].children:
                        return {'type':SWAP} # swap
                        #result_act_type = {'type':SWAP}                        
                    elif currentChildIdx in ref_graph.nodes[currentIdx].children: # correct
                        gold_edge = ref_graph.get_edge_label(currentIdx,currentChildIdx)
                        return {'type':NEXT1, 'edge_label':gold_edge} # next
                        #result_act_type = {'type':NEXT1}
                        #result_act_label = gold_edge
                    else:
                        return {'type':DELETEEDGE} # delete edge
                        #result_act_type = {'type':DELETEEDGE}
                else:
                    if ref_graph.nodes[currentIdx].contains(currentChild):
                        return {'type':MERGE}
                        #result_act_type = {'type':MERGE}
                    else:
                        return {'type':DELETEEDGE} # delete edge
                        #result_act_type = {'type':DELETEEDGE}
            else:
                if set(currentNode.children) == set(goldNode.children):
                    gold_tag = goldNode.tag
                    return {'type':NEXT2, 'tag':gold_tag} # next: done with the current node move to next one
                    #result_act_type = {'type':NEXT2,'tag':gold_tag}
                elif len(currentNode.children) < len(goldNode.children):
                    nodes_to_add = [c for c in goldNode.children if c not in currentNode.children]
                    child_to_add = nodes_to_add[0]
                    if child_to_add != 0 and child_to_add != currentIdx \
                       and child_to_add not in currentNode.children and child_to_add in currentGraph.nodes: 

                        gold_edge = ref_graph.get_edge_label(currentIdx,child_to_add)
                        return {'type':ADDCHILD, 'child_to_add':child_to_add, 'edge_label':gold_edge} # add one child each action
                        #result_act_type = {'type':ADDCHILD, 'child_to_add':nodes_to_add[0]}
                        #result_act_label = gold_edge
                    else:
                        if self.verbose > 2:
                            print >> sys.stderr, "Not a correct link between %s and %s!"%(currentIdx,nodes_to_add[0])
                        return {'type':NEXT2}
                        #result_act_type = {'type':NEXT2}
                else:
                    if self.verbose > 2:
                        print >> sys.stderr, "Missing actions, current node's and gold node's children:%s  %s"%(str(currentNode.children), str(goldNode.children))
                    pass
                    
        elif ref_graph.isContained(currentIdx):
            if currentChildIdx and currentChildIdx in goldNodeSet and \
               ref_graph.nodes[currentChildIdx].contains(currentNode):
                return {'type':MERGE}
                #result_act_type = {'type':MERGE}
            else:
                if currentChildIdx:
                    return {'type':NEXT1}
                    #result_act_type = {'type':NEXT1}
                else:
                    return {'type':NEXT2}
                    #result_act_type = {'type':NEXT2}
        else:
            if currentChildIdx:
                #assert len(currentNode.children) == 1
                if currentChildIdx in goldNodeSet:
                    if (isCorrectReplace(currentChildIdx,currentNode,ref_graph) or len(currentNode.children) == 1):
                        return {'type':REPLACEHEAD} # replace head
                        #result_act_type = {'type':REPLACEHEAD}
                    else:
                        return {'type':NEXT1} #
                        #result_act_type = {'type':NEXT1}
                else:
                    return {'type':DELETEEDGE}
                    #result_act_type = {'type':DELETEEDGE}
            else:
                # here currentNode.children must be empty
                return {'type':DELETENODE} 
                #result_act_type = {'type':DELETENODE}


        skip = NEXT1 if currentChildIdx else NEXT2
        return {'type':skip}
        '''
