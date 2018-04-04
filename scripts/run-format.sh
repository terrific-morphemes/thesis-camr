# remove BOM
awk '{ gsub(/\xef\xbb\xbf/,""); print }' data/camrnew/CAMR10040.txt > amr_zh_10k.txt.orig

# fix format
sed -E 's/(op[0-9]*) x[0-9]*\/([^ )]*)/\1 "\2"/g' data/camrnew/amr_zh_10k.txt.orig > data/camrnew/amr_zh_10.txt
