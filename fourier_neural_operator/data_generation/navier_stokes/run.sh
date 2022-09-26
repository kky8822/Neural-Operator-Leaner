for j in {0..119}
do
	echo "Iter " $j
      	python ns_2d.py 1e-2 $j > ns_1e-2_$j.log &
	sleep 1
      	python ns_2d.py 1e-3 $j > ns_1e-3_$j.log &
	sleep 1
      	python ns_2d.py 1e-4 $j > ns_1e-4_$j.log
done
