import sys
import re

start = 0
end = 0
c = 0

filename = sys.argv[1]
name = filename.replace('.log',"")
newfile = str(name) + ".xyz"

# Open the original file in read mode
# Create a new file with writing rights

openold = open(filename,"r")
opennew = open(newfile,"w")


# Read the entire original file
rline = openold.readlines()

for i in range (len(rline)):
    if "Input orientation:" in rline[i]:
        start = i
    if "Charge =" in rline[i]:
        qm = rline[i].split()
        q = qm[2]
        mul = qm[5]

for m in range (start + 5, len(rline)):
    if "---" in rline[m]:
        end = m
        break

#Total no. of atoms, charge and multiplicity
for line in rline[start+5 : end]:
    c = c+1

print >>opennew, "%s" % (c)
print >>opennew, "%s %s" % (q,mul)

# Conversion section
for line in rline[start+5 : end] :
    words = line.split()
    word1 = int(words[1])
    word3 = str(words[3])
    if word1 == 1 :
        word1 = "H "
    elif word1 == 5 :
        word1 = "B "
    elif word1 == 6 :
        word1 = "C "
    elif word1 == 7 :
        word1 = "N "
    elif word1 == 8 :
        word1 = "O "
    elif word1 == 9:
        word1 = "F "
    elif word1 == 14:
        word1 = "Si "
    elif word1 == 15:
        word1 = "P "
    elif word1 == 16:
        word1 = "S "
    elif word1 == 17:
        word1 = "Cl "
    print >>opennew, "%s%s" % (word1,line[30:-1])
opennew.close()
openold.close()
