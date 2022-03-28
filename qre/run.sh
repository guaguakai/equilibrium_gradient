for BUDGET in 1
do
	for SEED in {1..1}
	do
		python3 main.py --n=3 --method=knitro --constrained --forward-iterations=1000 --actions=10 --budget=$BUDGET --seed=$SEED --init
	done
done
