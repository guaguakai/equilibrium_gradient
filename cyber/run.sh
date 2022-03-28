for RISK in 0.1
do
	for SEED in {1..1}
	do
		python3 main.py --n=10 --method=knitro --constrained --forward-iterations=1000 --risk=$RISK --seed=$SEED --init
	done
done
