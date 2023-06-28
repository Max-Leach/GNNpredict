#! /bin/bash

mkdir incorrect
for i in *.log;do freq_real=$(grep " Frequencies -- " $i | head -n 1 | awk '{print $3}');freq=$(grep " Frequencies -- " $i | head -n 1 | awk '{print $3}' | cut -f 1-1 -d .);if [ $freq -lt 0 ];then echo "$i $freq_real" >> negative_freqs;mv $i incorrect/;mv ${i%log}xyz incorrect/;fi;done

for i in *.xyz;do obabel -i xyz $i -o sdf -O ${i%.xyz}_i.sdf;done
mv *.xyz mols/
for i in *.log;do python2.7 gooutcoord.py $i;done
for i in *.xyz;do obabel -i xyz $i -o sdf -O ${i%.xyz}_o.sdf;done
cp *.log mols/
rm -r *.Identifier

for i in *.sdf; do
        python3 changebonds.py ${i} >> temp
        mv temp ${i}
done

