for rewire in c2a
do
  for seed in 1
  do
    for model in gvp
    do
      for p in 0.95 0.9 0.85 0.75 0.6 0.5 0.4 0.25 0.1 0
      do
        python evaluate_carbon_rewirings.py --model $model --rewire $rewire --p $p --seed $seed
      done
    done
  done
done

for rewire in c2c
do
  for seed in 1
  do
    for model in gvp
    do
      for p in 0.95 0.9 0.85 0.75 0.6 0.5 0.4 0.25 0.1 0
      do
        python evaluate_carbon_rewirings.py --model $model --rewire $rewire --p $p --seed $seed
      done
    done
  done
done