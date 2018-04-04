# -*- coding:utf-8 -*-
from __future__ import with_statement
from __future__ import print_function
import codecs
import sys
import argparse
import re
import os
from common.amr_graph import AMRZ
from common.span_graph import SpanGraph
from common.span_graph2 import SpanGraph as SpanGraph2
from common.data import Data
from Aligner import Aligner
from collections import OrderedDict
import subprocess

log = sys.stdout
# SKIP_FNAME='log/skipped_file_list_v5k.txt'
# SKIP_FNAME='log/skipped_file_list_v10k.txt'
# SKIP_CMD_FNAME='scripts/skip-files-10k.sh'


def read_amrz(amr_filepath):
    '''
    read Chinese(zh) AMR
    '''
    comment_list = []
    comment = OrderedDict()
    amr_list = []
    amr_string = ''
    orig_index = 0
    skipped_id_list = []

    print('Reading zh amr:')
    with codecs.open(amr_filepath, 'r', encoding='utf-8') as amrfile:
         # codecs.open(SKIP_FNAME, 'w', encoding='utf8') as skipped_f, \
         # open(SKIP_CMD_FNAME, 'w') as skipped_cmd_f:

        for line in amrfile:

            if line.startswith('#'):
                for m in re.finditer("::([^:\s]+)\s((?<!::).*)", line):
                    # print m.group(1),m.group(2)
                    key = m.group(1)
                    if key in comment and key == 'id':  # skipped sentences
                        orig_index += 1
                        # skipped_id_list.append(orig_index)
                        # print('%d -- %s' % (orig_index, comment[key]), file=skipped_f)

                    comment[key] = m.group(2).strip()

            elif not line.strip():
                if amr_string and comment:
                    orig_index += 1
                    comment_list.append(comment)
                    amr_list.append(amr_string.strip())
                    curr_num = len(amr_list)
                    if curr_num > 0 and curr_num % 1000 == 0:
                        print('%d...' % curr_num, end='')
                        sys.stdout.flush()

                    amr_string = ''
                    comment = {}
            else:
                amr_string += line.strip() + ' '

        if amr_string and comment:
            comment_list.append(comment)
            amr_list.append(amr_string)
        if not amr_string and comment:
            orig_index += 1
            # skipped_id_list.append(orig_index)
            # print('%d -- %s' % (orig_index, comment['id']), file=skipped_f)
        # cmd = 'sed -e \'%s\' $1' % (';'.join(str(sid)+'d' for sid in skipped_id_list))
        # skipped_cmd_f.write(cmd+'\n')

        print('\n')

    return (comment_list, amr_list)


def _write_sentences(file_path, sentences, keep_space=True):
    """
    write out the sentences to file
    """
    print("Writing sentences to file %s" % file_path, file=log)
    with codecs.open(file_path, 'w', encoding='utf-8') as output:
        for sent in sentences:
            if keep_space:
                output.write(sent + '\n')
            else:
                output.write(sent.replace(' ', '') + '\n')


def _write_tok_sentences(file_path, toks, comments=None):
    print("Writing segmented sentences to file %s" % file_path, file=log)
    with codecs.open(file_path, 'w', encoding='utf-8') as output_tok:
        for i, tok in enumerate(toks):
            sent = ' '.join(t.split('_')[1] for t in tok.split())
            #sent = sent.replace(u'\xa0','_')
            if comments:
                output_tok.write("%s %s\n" % (comments[i]['id'], sent))
            else:
                output_tok.write("%s\n" % sent)


def _write_amrs(amr_strings, comments, amr_file, split=None):
    out_amr_file = amr_file + '.amr'
    if split:
        endp1, endp2 = split
        out_dev_amr_file = amr_file + '.dev.amr'
        out_test_amr_file = amr_file + '.test.amr'
        print('Writing amrs to separate files %s, %s, %s' %
              (out_amr_file, out_dev_amr_file, out_test_amr_file), file=log)
        with codecs.open(out_amr_file, 'w', encoding='utf-8') as output, \
                codecs.open(out_dev_amr_file, 'w', encoding='utf-8') as doutput,\
                codecs.open(out_test_amr_file, 'w', encoding='utf-8') as toutput:
            for i, (amr_str, comment) in enumerate(zip(amr_strings, comments)):
                newamr = AMRZ.parse_string(amr_str)
                misc_str = '# %s\n' % (' '.join(('::%s %s') % (k, v) for k, v in comment.items(
                ) if k in ['id', 'date', 'snt-type', 'annotator']))
                tok_str = '# %s\n' % (' '.join(('::%s %s') % (
                    k, v) for k, v in comment.items() if k in ['snt', 'tok']))
                newamr_str = newamr.to_amr_string()
                if i < endp1:
                    doutput.write(misc_str)
                    doutput.write(tok_str)
                    doutput.write(newamr_str)
                    doutput.write('\n\n')
                elif i >= endp1 and i < endp2:
                    toutput.write(misc_str)
                    toutput.write(tok_str)
                    toutput.write(newamr_str)
                    toutput.write('\n\n')

                output.write(misc_str)
                output.write(tok_str)
                output.write(newamr_str)
                output.write('\n\n')
    else:
        print('Writing amrs to file %s' % (out_amr_file), file=log)
        with codecs.open(out_amr_file, 'w', encoding='utf-8') as output:
            for amr_str, comment in zip(amr_strings, comments):
                newamr = AMRZ.parse_string(amr_str)
                output.write('# %s\n' % (' '.join(('::%s %s') % (k, v) for k, v in comment.items(
                ) if k in ['id', 'date', 'snt-type', 'annotator'])))
                output.write('# %s\n' % (' '.join(('::%s %s') % (k, v)
                                                  for k, v in comment.items() if k in ['snt', 'tok'])))

                output.write(newamr.to_amr_string())  # reformat the amr
                output.write('\n\n')


def _word_ner_iter(ner_file):
    with codecs.open(ner_file, encoding='utf8') as f:
        for line in f:
            for wn in line.split():
                yield wn.strip()


def preprocess(input_file, START_SNLP=False, INPUT_AMR='amr', DEBUG_LEVEL=0, ALIGN_FORMAT='gold', split=None, use_gold_dep=False):
    instances = []

    if INPUT_AMR == 'amr':  # input is annotation
        amr_file = input_file

        comments, amr_strings = read_amrz(amr_file)
        #for debugging
        #print(comments[0]['snt'])
        #sents = [c[u'snt'] for c in comments]
        #toks = sents  # [c['wid'] for c in comments]
        sents = list()
        toks = list()

        #Fix for mysterious keyerror problem above
        for comment in comments:
            #print(comment)
            try:
                sents.append(comment['snt'])
                toks.append(comment['snt'])
            except KeyError:
                print("Error with %s" % comment)
        tmp_sent_filename = amr_file + '.sent'  # raw
        tmp_tok_filename = amr_file + '.seg'
        tmp_amr_filename = amr_file + '.amr'  # reformatted amr file

        if not os.path.exists(tmp_sent_filename):
            _write_sentences(tmp_sent_filename, sents, keep_space=False)

        if not os.path.exists(tmp_tok_filename):
            _write_sentences(tmp_tok_filename, toks)

        if not os.path.exists(tmp_amr_filename):
            _write_amrs(amr_strings, comments, amr_file, split=split)

        if START_SNLP:
            print("Start preprocessing ...", file=log)
            subprocess.call('./scripts/corenlp.sh %s' %
                            tmp_tok_filename, shell=True)

        pos_filename = amr_file + '.pos'
        ner_filename = amr_file + '.ner'
        dep_filename = amr_file + \
            '.parse.dep' if not use_gold_dep else amr_file + '.parse.gold.dep'
        pos_file = codecs.open(pos_filename, encoding='utf8')
        ner_gen = _word_ner_iter(ner_filename)
        dep_file = codecs.open(dep_filename, encoding='utf8')

        if ALIGN_FORMAT in ['isi', 'hmm']:
            external_alignment_filename = amr_file + '.%s.align' % ALIGN_FORMAT
            external_align_file = open(external_alignment_filename)
            external_alignments = external_align_file.readlines()
            external_align_file.close()

        word_counter = 0
        Data.reset()  # reset counter
        print('Preprocessing:')
        for i, sent in enumerate(sents):
            if i % 1000 == 0:
                print(str(i) + '...', end='')
                sys.stdout.flush()

            data = Data()

            data.newSen()
            data.addText(sent)
            data.addComment(comments[i])

            tok_list = toks[i].split()

            #pos_line = pos_file.next()
            pos_line = next(pos_file)
            pos_list = [wp for wp in pos_line.strip().split()]

            # adding pos and ne
            for j, word in enumerate(tok_list):

                _, pos = pos_list[j].split('_')
                next_ner = next(ner_gen)
                _, ne = next_ner.rsplit('/', 1)

                data.addToken(word, pos, ne)

            # adding dependency
            #dep_line = dep_file.next().strip()
            dep_line = next(dep_file).strip()
            while dep_line:
                split_entry = re.split(
                    r"\(|(?<=[0-9]),", dep_line[:-1], maxsplit=2)
                assert len(split_entry) == 3, 'Error: %s' % '_'.join(
                    se.encode('utf8') for se in split_entry) + ' Sentence:' + str(i)
                rel, l_lemma, r_lemma = split_entry
                m1 = re.match(r'(?P<lemma>.+)-(?P<index>[0-9]+)', l_lemma)
                assert m1, r_lemma.encode('utf8')
                l_lemma, l_index = m1.group('lemma'), m1.group('index')

                m2 = re.match(r'(?P<lemma>.+)-(?P<index>[0-9]+)', r_lemma)
                assert m2, r_lemma.encode('utf8')
                r_lemma, r_index = m2.group('lemma'), m2.group('index')

                data.addDependency(rel, l_index, r_index)
                # dep_line = dep_file.next().strip()
                dep_line = next(dep_file).strip()
            # add amr graph
            amr = AMRZ.parse_string(amr_strings[i])
            data.addAMR(amr)

            ggraph = None
            # alignment
            if ALIGN_FORMAT in ['isi', 'hmm']:
                alignment_str = external_alignments[i]
                alignment, s2c_alignment = Aligner.readHMMAlignment(
                    amr, alignment_str, data)
                ggraph = SpanGraph2.init_ref_graph_abt(
                    amr, alignment, s2c_alignment, sent=data.tokens)
            else:
                ggraph = SpanGraph.init_ref_graph_abt(
                    amr, data)  # construct gold graph

            if DEBUG_LEVEL > 1:
                print('#Sentence %s -- %s' %
                      (data.sentID, data.annoID), file=sys.stderr)
                if ALIGN_FORMAT in ['isi', 'hmm']:
                    print(ggraph.print_tuples(), file=sys.stderr)
                else:
                    print(ggraph.print_tuples_dsn(), file=sys.stderr)

            data.addGoldGraph(ggraph)
            instances.append(data)

        print('Done.\n')
        pos_file.close()
        dep_file.close()

    elif INPUT_AMR == 'sent':
        sent_fname = input_file  # must ended with .sent
        base_fname = sent_fname.rsplit('.', 1)[0]

        sent_file = codecs.open(sent_fname, encoding='utf8')
        sents = sent_file.readlines()

        tmp_tok_filename = base_fname + '.seg'

        if START_SNLP:
            print("Start preprocessing ...", file=log)
            #script_path = os.path.abspath(__file__).rsplit('/', 1)[0]
            command = './scripts/corenlp.sh %s' % (sent_fname)
            if not os.path.exists(tmp_tok_filename):
                command = './scripts/corenlp.sh -t %s' % (sent_fname)
            print(command, file=log)
            subprocess.call(command, shell=True)

        tok_file = codecs.open(tmp_tok_filename, encoding='utf8')
        toks = tok_file.readlines()

        pos_filename = base_fname + '.pos'
        ner_filename = base_fname + '.ner'
        dep_filename = base_fname + '.parse.dep'
        pos_file = codecs.open(pos_filename, encoding='utf8')
        ner_gen = _word_ner_iter(ner_filename)
        dep_file = codecs.open(dep_filename, encoding='utf8')

        word_counter = 0
        Data.reset()  # reset counter
        print('Preprocessing:')
        for i, sent in enumerate(sents):
            if i % 1000 == 0:
                print(str(i) + '...', end='')
                sys.stdout.flush()

            data = Data()

            data.newSen()
            data.addText(sent.strip())

            tok_list = toks[i].strip().split()

            #pos_line = pos_file.next()
            pos_line = next(pos_file)
            pos_list = [wp for wp in pos_line.strip().split()]

            # adding pos and ne
            for j, word in enumerate(tok_list):

                _, pos = pos_list[j].split('_')
                next_ner = next(ner_gen)
                _, ne = next_ner.rsplit('/', 1)

                data.addToken(word, pos, ne)

            # adding dependency
            #dep_line = dep_file.next().strip()
            dep_line = next(dep_file).strip()
            while dep_line:
                split_entry = re.split(
                    r"\(|(?<=[0-9]),", dep_line[:-1], maxsplit=2)
                assert len(split_entry) == 3, 'Error: %s' % '_'.join(
                    se.encode('utf8') for se in split_entry) + ' Sentence:' + str(i)
                rel, l_lemma, r_lemma = split_entry
                m1 = re.match(r'(?P<lemma>.+)-(?P<index>[0-9]+)', l_lemma)
                assert m1, r_lemma.encode('utf8')
                l_lemma, l_index = m1.group('lemma'), m1.group('index')

                m2 = re.match(r'(?P<lemma>.+)-(?P<index>[0-9]+)', r_lemma)
                assert m2, r_lemma.encode('utf8')
                r_lemma, r_index = m2.group('lemma'), m2.group('index')

                data.addDependency(rel, l_index, r_index)
                #dep_line = dep_file.next().strip()
                dep_line = next(dep_file).strip()
            instances.append(data)

        print('Done.\n')
        sent_file.close()
        tok_file.close()
        pos_file.close()
        dep_file.close()
    else:
        raise Exception('Unknown input format')

    return instances


if __name__ == "__main__":
    import argparse
    opt = argparse.ArgumentParser()
    opt.add_argument("-sp", "--startprep", action='store_true',
                     help='start preprocess')
    opt.add_argument("-af", "--amrfmt", default='amr', help='input amr format')
    opt.add_argument("--alignfmt", default='gold',
                     choices=['isi', 'hmm', 'gold'], help='input amr format')
    opt.add_argument("-dl", "--debuglevel", type=int,
                     default=0, help='input amr format')
    opt.add_argument("-f", "--file", nargs='?', help='input file')
    opt.add_argument("-slt", "--split", default='1264;2541', help='input file')

    args = opt.parse_args()
    split = [int(i) for i in args.split.split(';')] if args.split else None
    instances = preprocess(args.file, START_SNLP=args.startprep,
                           INPUT_AMR=args.amrfmt, DEBUG_LEVEL=args.debuglevel, ALIGN_FORMAT=args.alignfmt, split=split)
    # for inst in instances:
    #    print(inst.to_string())
