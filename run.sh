for rewire in no
do
  for seed in 1  2 3
  do
    for model in gvp dime
    do
      python evaluate_rewirings.py --model $model --rewire $rewire --p 0 --k 0 --seed $seed
    done
  done
done

for rewire in k0hop
do
  for seed in 1  2 3
  do
    for model in gvp dime
    do
      for p in 0.5 0.25 0
      do
        for k in 2 3 4
        do
            python evaluate_rewirings.py --model $model --rewire $rewire --p $p --k $k --seed $seed
        done
      done
    done
  done
done