for rewire in c2a c2c
do
  for seed in 1  2 3
  do
    for model in gvp dime
    do
      for p in 0.5 0.25 0
      do
        python evaluate_carbon_rewirings.py --model $model --rewire $rewire --p $p --seed $seed
      done
    done
  done
done