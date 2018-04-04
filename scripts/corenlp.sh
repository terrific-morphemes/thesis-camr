DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
BASEDIR="$( dirname $DIR)"
INPUT=$1
SEGINPUT=$INPUT

CORENLP='/home/j/llc/cwang24/Tools/stanford-corenlp-full-2016-10-31'
#CORENLP = '/home/j/llc/cwang24/Tools/stanford-corenlp-full-2015-04-20'
if [ "$#" -eq 2 ] && [ "$1" == "-t" ]; then
    echo 'Input is not segmented. Segmenting ...'
    INPUT=$2
    $CORENLP/segment.sh ctb $INPUT UTF-8 0 2>$BASEDIR/log/corenlp-seg.log 1>${INPUT%.sent}.seg
    SEGINPUT=${INPUT%.sent}.seg
fi

echo 'POS tagging ...'
$CORENLP/postag.sh $SEGINPUT ${SEGINPUT%.seg}.pos.tmp
sed 's/#/_/g' ${SEGINPUT%.seg}.pos.tmp > ${SEGINPUT%.seg}.pos
rm ${SEGINPUT%.seg}.pos.tmp

echo 'NER tagging ...'
$CORENLP/ner.sh $SEGINPUT ${SEGINPUT%.seg}.ner

 echo 'Dependency parsing ...'
 $CORENLP/dparse.sh $SEGINPUT
 ZGPARSER='/home/j/llc/cwang24/Tools/zgwang-parser/2014-06-16SystemRelease'
 $ZGPARSER/parse.sh ${SEGINPUT%.seg}.pos ${SEGINPUT%.seg}.parse

 java -cp "${CORENLP}/*" edu.stanford.nlp.trees.international.pennchinese.ChineseGrammaticalStructure -treeFile ${SEGINPUT%.seg}.parse > ${SEGINPUT%.seg}.parse.dep


