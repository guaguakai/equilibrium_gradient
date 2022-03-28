for BUDGET in 0.2 # 0.4 0.6 0.8 1
do
	for SEED in {1..1}
	do
		python3 main.py --agents=5 --method=knitro --constrained --forward-iterations=1000 --targets=100 --actions=50 --budget=$BUDGET --seed=$SEED --init
	done
done
