#! /bin/bash

#add all input xyz's and output log files within a molecules subdirectory and execute this script
#outputs: 
#incorrect molecules in the incorrect directory
#cmp file containing structural matches between input and output
#inch_diff file containing inchi checks between output molecule and input SMILES

mkdir incorrect
mkdir sdfs

for i in molecules/*.log;do freq_real=$(grep " Frequencies -- " $i | head -n 1 | awk '{print $3}');freq=$(grep " Frequencies -- " $i | head -n 1 | awk '{print $3}' | cut -f 1-1 -d .);if [ $freq -lt 0 ];then echo "$i $freq_real" >> negative_freqs;mv $i incorrect/;mv ${i%log}xyz incorrect/;fi;done

for i in molecules/*.xyz;do
        pre=${i#molecules/};
        obabel -i xyz $i -o sdf -O sdfs/${pre%.xyz}_i.sdf;
done

for i in molecules/*.log;do
        pre=${i#molecules/};
        python2.7 gooutcoord.py $i;done
        obabel -i xyz $i -o sdf -O sdfs/${pre%.xyz}_o.sdf;
done
rm *.xyz

for i in sdfs/*.sdf; do
        python3 changebonds.py ${i} >> temp
        mv temp ${i}
done

#move the xyz's
for i in sdfs/*_i.sdf; do
        j=${i#molecules/}
	python3 compare.py ${i} ${i%_i.sdf}_o.sdf molecules/${j%_i.sdf}.log cmp
done

python cross-check.py
