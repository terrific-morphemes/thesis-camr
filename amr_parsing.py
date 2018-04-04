#!/usr/bin/python

"""
Interface for the parser:
parse command line 
read in corpus
train and test
"""
from __future__ import print_function

import sys
import codecs
import time
import string
import codecs
import re
import random
import cPickle as pickle
import subprocess
from collections import defaultdict
from preprocess import preprocess
from parser import Parser
from model import Model
# from model2 import Model  # for auto align
import argparse

import constants
from graphstate import GraphState
# from graphstate2 import GraphState  # for auto align
#import matplotlib.pyplot as plt

# reload(sys)
# sys.setdefaultencoding('utf-8')

log = sys.stderr
LOGGED = False
#experiment_log = open('log/experiment.log','a')
experiment_log = sys.stdout


def write_parsed_amr(parsed_amr, instances, output_fn, hand_alignments=None):
    output = codecs.open(output_fn, 'w', encoding='utf8')
    for pamr, inst in zip(parsed_amr, instances):
        if inst.comment:
            output.write('# %s\n' % (' '.join(('::%s %s') % (k, v) for k, v in inst.comment.items(
            ) if k in ['id', 'date', 'snt-type', 'annotator'])))
            output.write('# %s\n' % (' '.join(('::%s %s') % (k, v)
                                              for k, v in inst.comment.items() if k in ['snt', 'tok'])))
            if hand_alignments:
                output.write('# ::alignments %s ::gold\n' %
                             (hand_alignments[inst.comment['id']]))
            # output.write('# %s\n' % (' '.join(('::%s %s')%(k,v) for k,v in inst.comment.items() if k in ['alignments'])))
        else:
            output.write('# ::id %s\n' % (inst.sentID))
            output.write('# ::snt %s\n' % (inst.text))

        try:
            output.write(pamr.to_amr_string())
        except TypeError:
            import pdb
            pdb.set_trace()
        output.write('\n\n')
    output.close()


def write_span_graph(span_graph_pairs, instances, amr_file, suffix='spg'):
    output_d = open(amr_file + '.' + suffix + '.dep', 'w')
    output_p = open(amr_file + '.' + suffix + '.parsed', 'w')
    output_g = open(amr_file + '.' + suffix + '.gold', 'w')

    for i in xrange(len(instances)):
        output_d.write('# id:%s\n%s' %
                       (instances[i].comment['id'], instances[i].printDep()))
        output_p.write('# id:%s\n%s' % (
            instances[i].comment['id'], span_graph_pairs[i][0].print_dep_style_graph()))
        output_g.write('# id:%s\n%s' % (
            instances[i].comment['id'], span_graph_pairs[i][1].print_dep_style_graph()))
        output_p.write('# eval:Unlabeled Precision:%s Recall:%s F1:%s\n' % (
            span_graph_pairs[i][2][0], span_graph_pairs[i][2][1], span_graph_pairs[i][2][2]))
        output_p.write('# eval:Labeled Precision:%s Recall:%s F1:%s\n' % (
            span_graph_pairs[i][2][3], span_graph_pairs[i][2][4], span_graph_pairs[i][2][5]))
        output_p.write('# eval:Tagging Precision:%s Recall:%s\n' %
                       (span_graph_pairs[i][2][6], span_graph_pairs[i][2][7]))
        output_d.write('\n')
        output_p.write('\n')
        output_g.write('\n')

    output_d.close()
    output_p.close()
    output_g.close()


def main():

    arg_parser = argparse.ArgumentParser(
        description="Brandeis transition-based AMR parser 1.0")

    arg_parser.add_argument('-v', '--verbose', type=int,
                            default=0, help='set up verbose level for debug')
    arg_parser.add_argument('-b', '--begin', type=int, default=0,
                            help='specify which sentence to begin the alignment or oracle testing for debug')
    arg_parser.add_argument('-s', '--start_step', type=int, default=0,
                            help='specify which step to begin oracle testing;for debug')
    #arg_parser.add_argument('-i','--input_file',help='the input: preprocessed data instances file for aligner or training')
    arg_parser.add_argument('-d', '--dev', help='development file')
    arg_parser.add_argument('-a', '--add', help='additional training file')
    arg_parser.add_argument(
        '-as', '--actionset', choices=['basic'], default='basic', help='choose different action set')
    arg_parser.add_argument('-m', '--mode', choices=['preprocess', 'test_gold_graph', 'align', 'userGuide', 'oracleGuide', 'train', 'parse',
                                                     'eval'], help="preprocess:generate pos tag, dependency tree, ner\n" "align:do alignment between AMR graph and sentence string")
    arg_parser.add_argument('-dp', '--depparser', choices=['stanford', 'zgparser', 'syntaxnet',
                                                           'clear', 'turbo'], default='zgparser', help='choose the dependency parser')
    arg_parser.add_argument('--coref', action='store_true',
                            help='flag to enable coreference information')
    arg_parser.add_argument('--prop', action='store_true',
                            help='flag to enable semantic role labeling information')
    arg_parser.add_argument('--rne', action='store_true',
                            help='flag to enable rich name entity')
    #arg_parser.add_argument('--verblist',action='store_true',help='flag to enable verbalization list')
    #arg_parser.add_argument('--onto',action='store_true',help='flag to enable charniak parse result trained on ontonotes')
    arg_parser.add_argument('--onto', choices=['onto', 'onto+bolt', 'wsj'],
                            default='wsj', help='choose which charniak parse result trained on ontonotes')
    arg_parser.add_argument('--model', help='specify the model file')
    arg_parser.add_argument('--feat', help='feature template file')
    arg_parser.add_argument('-iter', '--iterations',
                            default=1, type=int, help='training iterations')
    arg_parser.add_argument(
        'amr_file', nargs='?', help='amr annotation file/input sentence file for parsing')
    arg_parser.add_argument(
        '--prpfmt', choices=['xml', 'plain'], default='xml', help='preprocessed file format')
    arg_parser.add_argument(
        '--amrfmt', choices=['sent', 'amr', 'amreval'], default='sent', help='specifying the input file format')
    arg_parser.add_argument(
        '--alignfmt', choices=['gold', 'isi', 'hmm'], default='gold', help='specifying the input alignment source')
    arg_parser.add_argument(
        '--smatcheval', action='store_true', help='give evaluation score using smatch')
    arg_parser.add_argument(
        '-e', '--eval', nargs=2, help='Error Analysis: give parsed AMR file and gold AMR file')
    #arg_parser.add_argument('--section',choices=['proxy','all'],default='all',help='choose section of the corpus. Only works for LDC2014T12 dataset.')
    arg_parser.add_argument('--corpus_version', choices=[
                            '5k', '10k'], default='5k', help='choose which version of chinese amr corpus to train on')

    args = arg_parser.parse_args()

    amr_file = args.amr_file
    instances = None
    train_instance = None
    verbose = args.verbose
    constants.FLAG_COREF = args.coref
    constants.FLAG_PROP = args.prop
    constants.FLAG_RNE = args.rne
    # constants.FLAG_ALIGN=args.alignfmt
    constants.FLAG_ONTO = args.onto
    constants.FLAG_DEPPARSER = args.depparser

    # using corenlp to preprocess the sentences
    if args.mode == 'preprocess':
        instances = preprocess(amr_file, START_SNLP=True,
                               INPUT_AMR=args.amrfmt, DEBUG_LEVEL=verbose, ALIGN_FORMAT=args.alignfmt)
        print("Done preprocessing!")
    # preprocess the JAMR aligned amr
    elif args.mode == 'test_gold_graph':
        instances = preprocess(amr_file, START_SNLP=False,
                               INPUT_AMR=args.amrfmt, DEBUG_LEVEL=verbose)
        gold_amr = []
        for inst in instances:
            GraphState.sent = inst.tokens
            gold_amr.append(GraphState.get_parsed_amr(inst.gold_graph))
        write_parsed_amr(gold_amr, instances, amr_file, 'abt.gold')
        print("Done output AMR!")
    # test user guide actions
    elif args.mode == 'userGuide':
        print('Read in training instances...')
        train_instances = preprocess(amr_file)

        sentID = int(raw_input("Input the sent ID:"))
        amr_parser = Parser()
        amr_parser.testUserGuide(train_instances[sentID])
        sys.exit()

    # test deterministic oracle
    elif args.mode == 'oracleGuide':
        train_instances = preprocess(
            amr_file, START_SNLP=False, INPUT_AMR=args.amrfmt, DEBUG_LEVEL=verbose, ALIGN_FORMAT=args.alignfmt)

        start_step = args.start_step
        begin = args.begin
        amr_parser = Parser(
            oracle_type=constants.DET_T2G_ORACLE_ABT, verbose=args.verbose)
        #ref_graphs = pickle.load(open('./data/ref_graph.p','rb'))
        n_correct_total = .0
        n_parsed_total = .0
        n_gold_total = .0
        pseudo_gold_amr = []
        n_correct_tag_total = .0
        n_parsed_tag_total = 0.
        n_gold_tag_total = .0
        concept_counter = defaultdict(int)
        act_counter = defaultdict(int)

        gold_amr = []

        # each time we get number of samples to look at
        sample_size = 500  # len(train_instances)
        sample_instances = train_instances[begin:begin + sample_size]
        # print "shuffling training instances"
        # random.shuffle(train_instances)

        print('Running oracle:')
        for i, instance in enumerate(sample_instances):

            state = amr_parser.testOracleGuide(
                instance, start_step, act_counter=act_counter)

            n_correct_arc, n1, n_parsed_arc, n_gold_arc, n_correct_tag, n_parsed_tag, n_gold_tag = state.evaluate()

            assert n_correct_arc == n1

            n_correct_total += n_correct_arc
            n_parsed_total += n_parsed_arc
            n_gold_total += n_gold_arc
            p = n_correct_arc / n_parsed_arc if n_parsed_arc else .0
            r = n_correct_arc / n_gold_arc if n_gold_arc else .0
            indicator = 'PROBLEM!' if p < 0.5 else ''
            if args.verbose > 2:
                print >> sys.stderr, "Precision: %s Recall: %s  %s\n" % (
                    p, r, indicator)
            n_correct_tag_total += n_correct_tag
            n_parsed_tag_total += n_parsed_tag
            n_gold_tag_total += n_gold_tag
            p1 = n_correct_tag / n_parsed_tag if n_parsed_tag else .0
            r1 = n_correct_tag / n_gold_tag if n_gold_tag else .0
            if args.verbose > 2:
                print >> sys.stderr, "Tagging Precision:%s Recall:%s" % (
                    p1, r1)

            if args.verbose > 2:
                print(state.A.print_tuples_dsn(), file=sys.stderr)
            pamr = GraphState.get_parsed_amr(state.A)
            pseudo_gold_amr.append(pamr)

        print('Done.')
        print(act_counter)
        pt = n_correct_total / n_parsed_total if n_parsed_total != .0 else .0
        rt = n_correct_total / n_gold_total if n_gold_total != .0 else .0
        ft = 2 * pt * rt / (pt + rt) if pt + rt != .0 else .0
        pseudo_gold_fname = amr_file + '.pseudo-gold'
        write_parsed_amr(pseudo_gold_amr, sample_instances, pseudo_gold_fname)
        print("Total Accuracy: %s, Recall: %s, F-1: %s" % (pt, rt, ft))

        tp = n_correct_tag_total / n_parsed_tag_total if n_parsed_tag_total != .0 else .0
        tr = n_correct_tag_total / n_gold_tag_total if n_gold_tag_total != .0 else .0
        print("Tagging Precision:%s Recall:%s" % (tp, tr))

    elif args.mode == 'train':  # training
        print("Parser Config:")
        print("Incorporate Coref Information: %s" % (constants.FLAG_COREF))
        print("Incorporate SRL Information: %s" % (constants.FLAG_PROP))
        print("Dependency parser used: %s" % (constants.FLAG_DEPPARSER))

        all_instances = preprocess(amr_file, START_SNLP=False, INPUT_AMR=args.amrfmt,
                                   DEBUG_LEVEL=args.verbose, ALIGN_FORMAT=args.alignfmt)
        if args.dev:  # have separate dev set
            train_instances = all_instances
            dev_instances = preprocess(args.dev, START_SNLP=False, INPUT_AMR=args.amrfmt,
                                       DEBUG_LEVEL=args.verbose, ALIGN_FORMAT=args.alignfmt)
        elif args.corpus_version == '5k':
            train_instances = all_instances[907:]
            dev_instances = all_instances[:414]
            test_instances = all_instances[414:907]
        elif args.corpus_version == '10k':
            train_instances = all_instances[2541:]
            dev_instances = all_instances[:1264]
            test_instances = all_instances[1264:2541]

        dev_fname = amr_file + '.dev.amr'
        test_fname = amr_file + '.test.amr'

        feat_template = args.feat if args.feat else None
        model = Model(elog=experiment_log)
        parser = Parser(model=model, oracle_type=constants.DET_T2G_ORACLE_ABT,
                        action_type=args.actionset, verbose=args.verbose, elog=experiment_log)
        model.setup(action_type=args.actionset, instances=train_instances,
                    parser=parser, feature_templates_file=feat_template)

        print("BEGIN TRAINING!", file=experiment_log)
        best_fscore = 0.0
        best_pscore = 0.0
        best_rscore = 0.0
        best_model = None
        best_iter = 1
        for iter in xrange(1, args.iterations + 1):
            print("shuffling training instances", file=experiment_log)
            random.shuffle(train_instances)

            print("Iteration:%d" % iter, file=experiment_log)
            begin_updates = parser.perceptron.get_num_updates()
            parser.parse_corpus_train(train_instances)
            parser.perceptron.average_weight()
            # model.save_model(args.model+'.%s.m'%(str(iter)))

            print("Result on develop set:", file=experiment_log)
            _, parsed_amr = parser.parse_corpus_test(dev_instances)
            parsed_suffix = args.model.split(
                '.', 2)[-1] + '.' + str(iter) + '.parsed'
            parsed_filename = dev_fname + '.' + parsed_suffix
            write_parsed_amr(parsed_amr, dev_instances, parsed_filename)
            if args.smatcheval:
                smatch_path = "./smatch_2.0.2/smatch.py"
                python_path = 'python'
                options = '--pr -f'
                command = '%s %s %s %s %s' % (
                    python_path, smatch_path, options, parsed_filename, dev_fname)

                print('Evaluation using command: ' + command)
                print('Saving model')
                model.save_model(args.model + '.m')
                # print subprocess.check_output(command, stderr=subprocess.STDOUT, shell=True)
                eval_output = subprocess.check_output(
                    command, stderr=subprocess.STDOUT, shell=True)
                print(eval_output)
                #print('eval output: ' + eval_output.split('\n'))
                if "Error" in eval_output.split('\n')[0]:
                    eval_output = '\n'.join(eval_output.split('\n')[2:])
                print(eval_output)
                pscore = float(eval_output.split('\n')[
                               0].split(':')[1].rstrip())
                rscore = float(eval_output.split('\n')[
                               1].split(':')[1].rstrip())
                fscore = float(eval_output.split('\n')[
                               2].split(':')[1].rstrip())
                if fscore > best_fscore:
                    best_model = model
                    print("Saving model to {}".format(args.model + '.m'))
                    best_model.save_model(args.model + '.m')
                    best_iter = iter
                    best_fscore = fscore
                    best_pscore = pscore
                    best_rscore = rscore

        if best_model is not None:
            print("Best result on iteration %d:\n Precision: %f\n Recall: %f\n F-score: %f" %
                  (best_iter, best_pscore, best_rscore, best_fscore), file=experiment_log)

        print("DONE TRAINING!", file=experiment_log)

    elif args.mode == 'parse':  # actual parsing
        test_instances = preprocess(
            amr_file, START_SNLP=False, INPUT_AMR=args.amrfmt, DEBUG_LEVEL=args.verbose)

        print("Loading model: %s" % args.model, file=experiment_log)
        model = Model.load_model(args.model)
        parser = Parser(model=model, oracle_type=constants.DET_T2G_ORACLE_ABT,
                        action_type=args.actionset, verbose=args.verbose, elog=experiment_log)
        print("BEGIN PARSING", file=experiment_log)
        span_graph_pairs, results = parser.parse_corpus_test(
            test_instances, EVAL=True)
        parsed_filename = '%s.%s.parsed' % (amr_file, args.model.rsplit(
            '.', 2)[-2])  # here needs specific model name format
        write_parsed_amr(results, test_instances, parsed_filename)
        ################
        # for eval     #
        ################

        print("DONE PARSING", file=experiment_log)
        if args.smatcheval:
            smatch_path = "./smatch_2.0.2/smatch.py"
            python_path = 'python'
            options = '--pr -f'
            gold_amr_file = amr_file + '.amr'
            command = '%s %s %s %s %s' % (
                python_path, smatch_path, options, parsed_filename, gold_amr_file)

            print('Evaluation using command: ' + (command))
            print(subprocess.check_output(
                command, stderr=subprocess.STDOUT, shell=True))

        # plt.hist(results)
        # plt.savefig('result.png')
    else:
        arg_parser.print_help()


if __name__ == "__main__":
    main()
