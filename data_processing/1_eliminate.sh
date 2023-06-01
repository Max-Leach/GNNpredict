#! /bin/bash

list=("." "+" "-" "He" "Li" "Be" "Ne" "Na" "Mg" "Al" "Ar" "Ca" "Ti" "Cr" "Mn" "Fe" "Ni" "Cu" "Zn" "Ga" "Ge" "As" "Se" "Br" "Kr" "Rb" "Sr" "Zr" "Mo" "Tc" "Ru" "Rh" "Pd" "Ag" "Cd" "Te" "Xe" "Ba" "La" "Hf" "Ta" "Re" "Ir" "Pt" "Au" "Hg" "Tl" "Bi" "At" "Rn" "Fr" "Ra" "Ac" "Rf" "Db" "Sg" "Bh" "Hs" "Mt" "Ds" "Rg" "Nh" "Fl" "Mc" "Lv" "Ts" "Og" "Ce" "Pr" "Nd" "Pm" "Sm" "Eu" "Gd" "Tb" "Dy" "Ho" "Er" "Tm" "Yb" "Lu" "Th" "Pa" "Pu" "Am" "Cm" "Bk" "Es" "Fm" "Md" "Lr" "K" "V" "Y" "W" "U")

grep -iv "\." CID-SMILES > minus-._CID-SMILES
echo "Done ."

for i in ${!list[@]}; do
	if [ ${list[$i]} != "U" ]
	then
		grep -iv "${list[$i+1]}" minus-${list[$i]}_CID-SMILES > minus-${list[$i+1]}_CID-SMILES
		echo "Done ${list[$i+1]}"
		rm minus-${list[$i]}_CID-SMILES
	fi
done

