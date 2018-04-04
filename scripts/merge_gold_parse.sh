DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
BASEDIR="$( dirname $DIR)"
CTB8_PATH="/home/j/clp/chinese/corpora/ctb8.0-berkeley-repaired/"

TREE_FILE=$BASEDIR/data/camrnew/ctb8_gold.trees
touch $TREE_FILE

for f in {000..558}
do
    cat $CTB8_PATH/chtb_5$f.seg.bps >> $TREE_FILE
done
    
