for rewire in k0hop
do
  for seed in 1
  do
    for model in gvp
    do
      for p in 0.85 0.9
      do
        for k in 2 3 4 5 6 7 8
        do
            python evaluate_rewirings.py --model $model --rewire $rewire --p $p --k $k --seed $seed
        done
      done
    done
  done
done