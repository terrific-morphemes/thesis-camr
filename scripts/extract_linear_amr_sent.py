'''
extract linearized Chinese amr and tokenized sentences 
'''
import sys, os

CURDIR=os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(CURDIR))

from preprocess import read_amrz as readAMR
from common.amr_graph import AMRZ as AMR
from common.util import StrLiteral, Quantity, Polarity
from codecs import open as open
import re

# en_stopwords_f=open(os.path.join(os.path.dirname(CURDIR),'resources/ENG_stopword.txt'))
# ENG_STOP_SET=set(line.strip() for line in en_stopwords_f)
# en_stopwords_f.close()

def filter1(node, amr=None):
    '''
    ignore reentrancy
    '''
    iswiki = node.trace == 'wiki'
    return (node.firsthit or isinstance(node.node_label, (StrLiteral, Quantity, Polarity))) and (not iswiki)

def filter2(node, amr):
    '''
    ignore reentrancy and name and source of relation :name 
    '''
    isfirsthit = node.firsthit or isinstance(node.node_label, (StrLiteral, Quantity, Polarity))
    isname = node.trace == 'name' and amr.node_to_concepts.get(node.node_label, node.node_label) == 'name'
    isnamesource = 'name' in amr[node.node_label]
    iswiki = node.trace == 'wiki'
    return isfirsthit and (not isname) and (not isnamesource) and (not iswiki)

def filter3(en_str):
    return en_str.lower() not in ENG_STOP_SET

def filter4(en_str):
    return True

def get_amr_tuples(amr, linearized_amr_nodes):
    gpos_mapping = dict((n.seqID,(i+1,n)) for i,n in enumerate(linearized_amr_nodes))
    tuple_list = []
    
    for node in linearized_amr_nodes:
        curr = node.node_label
        curr_seqID = node.seqID
        curr_cpt = amr.node_to_concepts.get(curr, curr)
        curr_id = gpos_mapping[curr_seqID][0]
        if node.parent is None: # root
            tuple_list.append(('root', 0, 'ROOT', gpos_mapping[curr_seqID][0], curr_cpt))
        else:
            parent_seqID = curr_seqID.rsplit('.',1)[0]
            while parent_seqID not in gpos_mapping and parent_seqID != '1': # root or 
                parent_seqID = parent_seqID.rsplit('.',1)[0]
            if parent_seqID == '1' and parent_seqID not in gpos_mapping:
                rel = 'root'
                parent_id = 0
                parent_cpt = 'ROOT'
            else:
                rel = node.trace
                parent_id = gpos_mapping[parent_seqID][0]
                parent = gpos_mapping[parent_seqID][1].node_label
                parent_cpt = amr.node_to_concepts.get(parent, parent)
            #rel = rel.replace('-','_')
            tuple_list.append((rel, parent_id, parent_cpt, curr_id, curr_cpt))

    return tuple_list
    
    
def extract_snt(tok_amr_file, amr_filter_func=filter2, zh_filter_func=filter4):
    '''
    [deprecated] input: tokenized aligned amr file [SemEval format]
    [deprecated] handle encoding exception xa0
    '''
    def get_clean_form(amr, n):
        clean_form = amr.node_to_concepts.get(n.node_label,n.node_label).split('~')[0].replace(' ','_')
        if clean_form.startswith('"') and clean_form.endswith('"'):
            return clean_form[1:-1]
        return clean_form

    comment_list, amr_list = readAMR(tok_amr_file)
    linearized_amr_list = []
    tok_sent_list = []

    output_snt_file = tok_amr_file + '.snt'

    output_snt_file_zh = output_snt_file + '.zh'
    output_snt_file_amr = output_snt_file + '.amr'
    output_snt_file_amrgpos = output_snt_file + '.amrgpos' # graph position 
    output_snt_file_zhspos = output_snt_file + '.zhspos' # original sentence position 
    output_snt_file_amrlvl = output_snt_file + '.amrlvl'

    output_snt_file_amrdep = output_snt_file + '.amrdep' # tuple dump in dependency style

    with open(output_snt_file_zh, 'w', encoding='utf8') as zhf, \
         open(output_snt_file_amr, 'w', encoding='utf8') as amrf, \
         open(output_snt_file_zhspos, 'w', encoding='utf8') as zhsposf, \
         open(output_snt_file_amrgpos, 'w', encoding='utf8') as amrgposf, \
         open(output_snt_file_amrlvl, 'w', encoding='utf8') as amrlvlf, \
        open(output_snt_file_amrdep, 'w', encoding='utf8') as amrdepf:

        for comment, amr_str in zip(comment_list, amr_list):
            sentID = comment['id']

            toks = comment['snt']
            # toks = toks.replace(u'\xa0','_')

            amr = AMR.parse_string(amr_str)
            tok_sent = [w for w in toks.split() if zh_filter_func(w)] 
            spos_zh = [str(i) for i,w in enumerate(toks.split()) if zh_filter_func(w)] 

            nlist, _ = amr.dfs()
            linearized_amr = [n for n in nlist if amr_filter_func(n, amr)]
            gpos_amr = [n.seqID for n in linearized_amr]
            lvl_amr = [n.depth for n in linearized_amr]

            amr_tuples = get_amr_tuples(amr, linearized_amr)

            zhf.write(' '.join(tok_sent)+'\n')
            zhsposf.write(' '.join(spos_zh)+'\n')
            amrf.write(' '.join(get_clean_form(amr, n) for n in linearized_amr)+'\n')
            amrgposf.write(' '.join(gpos_amr)+'\n')
            amrlvlf.write(' '.join(str(d) for d in lvl_amr)+'\n')
            amrdepf.write('\n'.join('%s(%s-%d, %s-%d)' % (rel, pcpt, pid, ccpt, cid) for rel, pid, pcpt, cid, ccpt in amr_tuples)+'\n\n')
            


def normalize_amr(amr_line):
    new_amr_line = []
    patterns = re.compile('(?P<predicate>([^-\d]+)-\d+)|(?P<number>[\d,-]+)')
    for word in amr_line.split():
        match = patterns.match(word)
        if match:
            mtype = match.lastgroup
        else:
            mtype = "NONE"
        if mtype == "predicate":
            new_word = match.group(2).lower()
        elif mtype == "number":
            new_word = re.sub('\d','0', word)
        else:
            new_word = word.lower()
        new_word=new_word[:4]
        new_amr_line.append(new_word)
    return ' '.join(new_amr_line)

def normalize_en(en_line):
    new_en_line = []
    patterns = re.compile('(?P<number>[\d,-]+)')
    for word in en_line.split():
        match = patterns.match(word)
        if match:
            mtype = match.lastgroup
        else:
            mtype = "NONE"
        if mtype == "number":
            new_word = re.sub('\d','0', word)
        else:
            new_word = word.lower()
        new_word=new_word[:4]
        new_en_line.append(new_word)
    return ' '.join(new_en_line)

def merge_en_amr(amr_file, en_file, normalize=False):
    '''
    merge english and amr sequence files;
    fast_align format
    '''

    output_file = en_file.rsplit('.' ,1)[0] + '.fast_align.txt.amr-en'
    with open(amr_file, encoding='utf8') as amrf, \
         open(en_file, encoding='utf8') as enf,\
         open(output_file, 'w', encoding='utf8') as outf:
        for amr_line, en_line in zip(amrf, enf):
            if normalize:
                outf.write('%s ||| %s\n' % (normalize_amr(amr_line.strip()), normalize_en(en_line.strip())))
            else:
                outf.write('%s ||| %s\n' % (amr_line.strip(), en_line.strip()))

def extract_plain(tok_amr_file):
    '''
    prepare for running isi aligner
    input: tokenized amr file
    '''
    basefilename = tok_amr_file.rsplit('.',2)[0]
    output_plain_en = basefilename + '.en'
    output_plain_amr = basefilename + '.amr'

    comment_list, amr_list = readAMR(tok_amr_file)

    with open(output_plain_en, 'w', encoding='utf8') as output_en, \
         open(output_plain_amr, 'w', encoding='utf8') as output_amr:
        output_en.write('\n'.join(comment['tok'].strip() for comment in comment_list)+'\n')
        output_amr.write('\n'.join(amr_str.strip() for amr_str in amr_list)+'\n')

    


if __name__  == '__main__':
    mode = sys.argv[1]
    if mode == 'separate':
        if len(sys.argv) != 3:
            print 'Usage: \npython extract_linear_amr_sent.py %s [tok_aligned_file]' % mode
            exit(0)

        extract_snt(sys.argv[2])

    elif mode == 'joint':
        if len(sys.argv) != 4:
            print 'Usage: \npython extract_linear_amr_sent.py %s [amr_file] [en_file]' % mode
            exit(0)
        merge_en_amr(sys.argv[2], sys.argv[3], normalize=True)
    
    elif mode == 'plain':
        extract_plain(sys.argv[2])
    
