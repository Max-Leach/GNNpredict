import sys

f1 = open(sys.argv[1], 'r')
lines = f1.readlines()
for n, i in enumerate(lines):
    if len(i) == 70:
        print(lines[n][:49] + ' 0' + lines[n][51:], end='')
    elif len(i) == 22:
        print(lines[n][:7] + ' 1' + lines[n][9:], end='')
    else:
        print(lines[n], end='')
