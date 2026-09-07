topic_attrs=(world sports business science)

for at in "${topic_attrs[@]}"; do
  python rl_train.py \
    --attr "$at" \
    --task topic
done


sentiment_attr=(pos neg)

for at in "${sentiment_attr[@]}"; do
  python rl_train.py \
    --attr "$at" \
    --task sentiment
done


python rl_train.py \
  --attr nontoxic \
  --task detoxification


