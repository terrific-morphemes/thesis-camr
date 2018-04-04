#!/usr/bin/python

"""
aligner for AMR and its corresponding sentence
"""
import re
import sys
from common.util import *
from collections import defaultdict
from span import Span

# class regex_pattern:
#    TempQuantity = '(?P<quant>[0-9]+)(?P<unit>year|month?)s?'


DATE_REL_SET = ['year', 'month', 'day', 'weekday',
                'time', 'decade', 'century', 'season', 'dayperiod']


class SearchException(Exception):
    pass


class Atype:
    WORD_ALIGN = 1
    SPAN_ALIGN = 2
    SPAN_ALIGN2 = 3


class Aligner():

    def __init__(self, verbose=0):
        self.verbose = verbose

    @staticmethod
    def readISIAlignment(amr, ISI_alignment, instance):
        '''
        although isi alignment output is one-to-one amr_concept->english_tok alignment, 
        we try to produce span to concept  
        '''
        alignment = defaultdict(list)
        s2c_alignment = defaultdict(list)  # span to concept mapping
        aligned_seqIDs = set()  # keep record of the already aligned variable
        aligned_en_pos = dict()

        alignment_str_list = sorted(
            ISI_alignment.strip().split(), key=lambda x: int(x.split('-')[0]))
        seqID_dict = dict(
            (pair.split('-')[1], int(pair.split('-')[0])) for pair in alignment_str_list)
        for one_alignment in alignment_str_list:
            en_pos, seqID = one_alignment.split('-')
            if seqID in aligned_seqIDs:
                continue

            en_pos = int(en_pos)
            ctype, incoming, path = amr.get_concept_relation(seqID)
            if ctype == 'r':  # skip relation alignment
                continue
            curr_var = path[-1]

            parent_var, curr_index = incoming

            incoming_edge = amr[parent_var].items(
            )[curr_index][0] if parent_var else 'null'
            parent_cpt = amr.node_to_concepts.get(
                parent_var, parent_var) if parent_var else 'null'
            curr_cpt = amr.node_to_concepts.get(curr_var, curr_var)

            if incoming_edge == 'wiki':  # ignore wiki
                continue

            if en_pos in aligned_en_pos:  # multiple concepts aligned to single english token
                prev_span = aligned_en_pos[en_pos]
                if prev_span.end - prev_span.start > 1 or prev_span.entity_tag == curr_cpt:  # crossing or dup
                    continue
                new_concept_tag = '%s|%s@%s' % (
                    prev_span.entity_tag, incoming_edge, curr_cpt)
                prev_span.set_entity_tag(ETag(new_concept_tag))

                aligned_seqIDs.update(seqID)
                alignment[curr_var].append(prev_span)

            # named entity
            elif parent_cpt == 'name' and len(path) > 4 and path[-4] == 'name' and incoming_edge.startswith('op'):
                pp_var = path[-5]
                concept_tag = '%s|name@name|name' % amr.node_to_concepts.get(
                    pp_var, pp_var)

                children = amr[parent_var].items()
                prefix = seqID.rsplit('.', 1)[0]
                op_seqIDs = [prefix + '.' + str(i + 1)
                             for i in range(len(children))]
                op_seqIDs.append(prefix)
                op_seqIDs.append(prefix.rsplit('.', 1)[0])
                aligned_seqIDs.update(op_seqIDs)

                en_pos_set = [seqID_dict[sid]
                              for sid in op_seqIDs if sid in seqID_dict]

                start = min(en_pos_set) + 1
                end = max(en_pos_set) + 1

                literals = [vs[0] for rel, vs in children]

                # print start, end
                span = Span(
                    start, end + 1, [t['form'] for t in instance.tokens[start:end + 1]], ETag(concept_tag))

                for ep in en_pos_set:
                    aligned_en_pos[ep] = span
                alignment[pp_var].append(span)
                alignment[parent_var].append(span)
                #for v in literals: alignment[v].append(span)
                s2c_alignment[(start, end)].append(pp_var)

            elif parent_cpt == 'date-entity' or (parent_cpt == 'name' and incoming_edge.startswith('op')):
                concept_tag = parent_cpt

                children = amr[parent_var].items()
                prefix = seqID.rsplit('.', 1)[0]
                children_seqIDs = [prefix + '.' +
                                   str(i + 1) for i in range(len(children))]
                aligned_seqIDs.update(children_seqIDs)

                en_pos_set = [seqID_dict[sid]
                              for sid in children_seqIDs if sid in seqID_dict]

                start = min(en_pos_set) + 1
                end = max(en_pos_set) + 1

                children_vars = [vs[0] for rel, vs in children]

                span = Span(
                    start, end + 1, [t['form'] for t in instance.tokens[start:end + 1]], ETag(concept_tag))

                for ep in en_pos_set:
                    aligned_en_pos[ep] = span
                alignment[parent_var].append(span)
                for v in children_vars:
                    alignment[v].append(span)
                s2c_alignment[(start, end)].append(parent_var)

            elif (parent_cpt.endswith('-quantity') and incoming_edge == 'unit') or \
                 (re.match('have-(org|rel)-role-91', parent_cpt) and incoming_edge == 'ARG2'):
                concept_tag = '%s|%s@%s' % (
                    parent_cpt, incoming_edge, curr_cpt)

                aligned_seqIDs.add(seqID)

                start = en_pos + 1
                end = start + 1

                span = Span(
                    start, end + 1, [instance.tokens[start]['form']], ETag(concept_tag))

                aligned_en_pos[en_pos] = span
                alignment[parent_var].append(span)
                alignment[curr_var].append(span)
                s2c_alignment[(start, end)].append(parent_var)

            else:
                concept_tag = curr_cpt

                aligned_seqIDs.add(seqID)

                start = en_pos + 1
                end = start + 1

                span = None
                if curr_var in amr.node_to_concepts:
                    span = Span(
                        start, end, [instance.tokens[start]['form']], concept_tag)
                else:
                    span = Span(
                        start, end, [instance.tokens[start]['form']], ConstTag(concept_tag))

                aligned_en_pos[en_pos] = span
                alignment[curr_var].append(span)
                s2c_alignment[(start, end)].append(concept_tag)

        return alignment, s2c_alignment

    @staticmethod
    def readHMMAlignment(amr, HMM_alignment, instance):
        '''
        although HMM alignment output is one-to-one amr_concept->english_tok alignment, 
        we try to produce span to concept  
        '''
        def general_concept_handler(curr_cpt, seqID, curr_var, en_pos,
                                    aligned_seqIDs, aligned_en_pos,
                                    alignment, s2c_alignment, amr, instance):
            concept_tag = curr_cpt

            aligned_seqIDs.add(seqID)

            start = en_pos + 1
            end = start + 1

            span = None
            if curr_var in amr.node_to_concepts:
                span = Span(
                    start, end, [instance.tokens[start]['form']], concept_tag)
            else:
                span = Span(
                    start, end, [instance.tokens[start]['form']], ConstTag(concept_tag))

            aligned_en_pos[en_pos] = span
            alignment[curr_var].append(span)
            s2c_alignment[(start, end)].append(concept_tag)

        def is_multi_concept_mapping(parent_cpt, incoming_edge, curr_tok):
            return (parent_cpt.endswith('-quantity') and incoming_edge == 'unit') or \
                (re.match('have-(org|rel)-role-91', parent_cpt) and incoming_edge == 'ARG2') or \
                (parent_cpt in ['person', 'thing'] and re.match(
                    'ARG[01]-of', incoming_edge) and not curr_tok.endswith('ed'))

        alignment = defaultdict(list)
        s2c_alignment = defaultdict(list)  # span to concept mapping
        aligned_seqIDs = set()  # keep record of the already aligned variable
        aligned_en_pos = dict()

        alignment_str_list = sorted(
            HMM_alignment.strip().split(), key=lambda x: int(x.split('-')[0]))
        seqID_dict = {}
        for pair in alignment_str_list:
            en_pos, seqID = pair.split('-')
            if seqID in seqID_dict:
                continue
            seqID_dict[seqID] = int(en_pos)

        for one_alignment in alignment_str_list:
            en_pos, seqID = one_alignment.split('-')
            if seqID in aligned_seqIDs:
                continue

            en_pos = int(en_pos)
            curr_tok = instance.tokens[en_pos + 1]['form']
            ctype, incoming, path = amr.get_concept_relation(seqID)
            if ctype == 'r':  # skip relation alignment
                continue
            curr_var = path[-1]

            parent_var, curr_index = incoming

            incoming_edge = amr[parent_var].items(
            )[curr_index][0] if parent_var else 'null'
            parent_cpt = amr.node_to_concepts.get(
                parent_var, parent_var) if parent_var else 'null'
            curr_cpt = amr.node_to_concepts.get(curr_var, curr_var)

            if incoming_edge == 'wiki':  # ignore wiki
                continue

            if en_pos in aligned_en_pos:  # multiple concepts aligned to single english token
                prev_span = aligned_en_pos[en_pos]
                if prev_span.end - prev_span.start > 1 or prev_span.entity_tag == curr_cpt:  # crossing or dup
                    continue
                new_concept_tag = '%s|%s@%s' % (
                    prev_span.entity_tag, incoming_edge, curr_cpt)
                prev_span.set_entity_tag(ETag(new_concept_tag))

                aligned_seqIDs.update(seqID)
                alignment[curr_var].append(prev_span)

            # named entity
            elif parent_cpt == 'name' and len(path) > 4 and path[-4] == 'name' and incoming_edge.startswith('op'):
                pp_var = path[-5]
                concept_tag = '%s|name@name|name' % amr.node_to_concepts.get(
                    pp_var, pp_var)

                children = amr[parent_var].items()
                prefix = seqID.rsplit('.', 1)[0]
                op_seqIDs = [prefix + '.' + str(i + 1)
                             for i in range(len(children))]
                op_seqIDs.append(prefix)
                op_seqIDs.append(prefix.rsplit('.', 1)[0])
                aligned_seqIDs.update(op_seqIDs)

                en_pos_set = [seqID_dict[sid]
                              for sid in op_seqIDs if sid in seqID_dict]

                start = min(en_pos_set) + 1
                end = max(en_pos_set) + 1

                literals = [vs[0] for rel, vs in children]

                # print start, end
                span = Span(
                    start, end + 1, [t['form'] for t in instance.tokens[start:end + 1]], ETag(concept_tag))

                for ep in en_pos_set:
                    aligned_en_pos[ep] = span
                alignment[pp_var] = [span]
                alignment[parent_var].append(span)
                #for v in literals: alignment[v].append(span)
                s2c_alignment[(start, end)].append(pp_var)

            elif parent_cpt == 'name' and incoming_edge.startswith('op'):
                concept_tag = parent_cpt

                prefix = seqID.rsplit('.', 1)[0]
                children_vars = []
                children_seqIDs = []

                # constraint the collapsed children
                for i, pair in enumerate(amr[parent_var].items()):
                    edge, vs = pair
                    v = vs[0]
                    if edge.startswith('op'):
                        children_vars.append(v)
                        children_seqIDs.append(prefix + '.' + str(i + 1))

                #children_seqIDs = [prefix+'.'+str(i+1) for i in range(len(children))]
                aligned_seqIDs.add(prefix)
                aligned_seqIDs.update(children_seqIDs)

                en_pos_set = [seqID_dict[sid]
                              for sid in children_seqIDs if sid in seqID_dict]

                start = min(en_pos_set) + 1
                end = max(en_pos_set) + 1

                #children_vars = [vs[0] for rel,vs in children]

                span = Span(
                    start, end + 1, [t['form'] for t in instance.tokens[start:end + 1]], ETag(concept_tag))

                for ep in en_pos_set:
                    aligned_en_pos[ep] = span
                alignment[parent_var] = [span]
                for v in children_vars:
                    alignment[v].append(span)
                s2c_alignment[(start, end)].append(parent_var)

            elif parent_cpt == 'date-entity':
                concept_tag = parent_cpt

                prefix = seqID.rsplit('.', 1)[0]
                children_vars = []
                children_seqIDs = []

                # constraint the collapsed children
                for i, pair in enumerate(amr[parent_var].items()):
                    edge, vs = pair
                    v = vs[0]
                    if edge in DATE_REL_SET:
                        children_vars.append(v)
                        children_seqIDs.append(prefix + '.' + str(i + 1))

                #children_seqIDs = [prefix+'.'+str(i+1) for i in range(len(children))]
                aligned_seqIDs.add(prefix)
                aligned_seqIDs.update(children_seqIDs)

                en_pos_set = [seqID_dict[sid]
                              for sid in children_seqIDs if sid in seqID_dict]
                if not en_pos_set:
                    # print incoming_edge
                    # print amr.to_amr_string()
                    general_concept_handler(curr_cpt, seqID, curr_var, en_pos,
                                            aligned_seqIDs, aligned_en_pos,
                                            alignment, s2c_alignment, amr, instance)
                    continue

                start = min(en_pos_set) + 1
                end = max(en_pos_set) + 1

                if end - start > 5:  # avoid crossing span
                    end = start

                #children_vars = [vs[0] for rel,vs in children]

                span = Span(
                    start, end + 1, [t['form'] for t in instance.tokens[start:end + 1]], ETag(concept_tag))

                for ep in en_pos_set:
                    aligned_en_pos[ep] = span
                alignment[parent_var] = [span]
                for v in children_vars:
                    alignment[v].append(span)
                s2c_alignment[(start, end)].append(parent_var)

            elif is_multi_concept_mapping(parent_cpt, incoming_edge, curr_tok):

                concept_tag = '%s|%s@%s' % (
                    parent_cpt, incoming_edge, curr_cpt)
                prefix = seqID.rsplit('.', 1)[0]
                aligned_seqIDs.add(prefix)
                aligned_seqIDs.add(seqID)

                start = en_pos + 1
                end = start + 1

                span = Span(
                    start, end, [instance.tokens[start]['form']], ETag(concept_tag))

                aligned_en_pos[en_pos] = span
                alignment[parent_var] = [span]
                alignment[curr_var].append(span)
                s2c_alignment[(start, end)].append(parent_var)

            else:
                general_concept_handler(curr_cpt, seqID, curr_var, en_pos,
                                        aligned_seqIDs, aligned_en_pos,
                                        alignment, s2c_alignment, amr, instance)

        return alignment, s2c_alignment
