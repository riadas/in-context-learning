import sys
from generate_train_test_data import * 

num_bits = sys.argv[1]
num_examples_mode = int(sys.argv[2])
num_prompts = int(sys.argv[3])
error_prob = float(sys.argv[4])

if error_prob == 0:
  error_prob = int(error_prob)

if num_bits != "mixed":
  num_bits = int(num_bits)

if num_examples_mode == 1:
  num_examples = 5
elif num_examples_mode == 2:
  if num_bits == 3:
    num_examples = 6
  elif num_bits == 5:
    num_examples = 8
  elif num_bits == 7:
    num_examples = 8
  elif num_bits == "mixed":
    num_examples = 8

mixed_pos_and_neg = (num_examples - 2, 2)

print("mixed_pos_and_neg")
print(mixed_pos_and_neg)

gen_train_and_test_data(0.8, num_formulas=num_prompts, num_vectors_per_formula=num_examples, feature_dim=num_bits, min_clauses=1, mixed_pos_and_neg=mixed_pos_and_neg, format_labels=True, error_prob=error_prob)
