#! /bin/bash

for i in *_i.sdf; do
	python3 compare.py ${i} ${i%_i.sdf}_o.sdf ${i%_i.sdf}.log cmp
done
