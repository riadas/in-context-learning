from generative_model import execute, gen_DNF, gen_input, num_non_constant_clauses
import random 
import os
'''
Generate dataset of prompts (labeled feature vectors + one unlabeled query vector for one formula) and values (label for query vector).
@param num_formulas: number of distinct formulas (prompts)
@param num_vectors_per_formula: number of feature vectors per formula (prompt)
@param feature_dim: size of feature vector 
@param min_clauses: min number of non-constant (not True/False) clauses in formula
@param mixed_pos_and_neg: Tuple of (num_pos::Int, num_neg::Int) where sum of tuple values equals num_vectors_per_formula, and num_pos is
       number of positive (True-labeled) vectors in prompt and num_neg is numbe of negative (False-labeled) vectors in prompt
@param format_labeled: converts True/False vector labels to [1, 0]/[0, 1] if set to true

Output: list of formulas and corresponding list of prompts + query labels 
> formulas, prompts = gen_feature_vectors_and_labels(num_formulas=1000,
                                                     num_vectors_per_formula=5,
                                                     feature_dim=5,
                                                     min_clauses=1,
                                                     mixed_pos_and_neg=(3,2))
'''
def gen_feature_vectors_and_labels(num_formulas, num_vectors_per_formula, feature_dim_unused, min_clauses=-1, mixed_pos_and_neg=None, format_labels=False, error_prob=0):
  formulas = []
  prompts = []
  # num_vectors_per_formula = random.sample(list(range(4, 8)), 1)[0]
  # mixed_pos_and_neg = (num_vectors_per_formula - 2, 2)
  while len(formulas) != num_formulas:
    if feature_dim_unused == "mixed":
      feature_dim = random.sample(list(range(4, 8)), 1)[0]
    else:
      feature_dim = feature_dim_unused

    formula = gen_DNF(feature_dim)
    if min_clauses != -1: # ensure that formula has minimum number of non-constant clauses
      while (False if min_clauses == -1 else (num_non_constant_clauses(formula) < min_clauses)): # or num_non_constant_clauses(formula) > 5)
        formula = gen_DNF(feature_dim)

    labeled_vectors = []
    all_vectors = list(map(lambda i: int_to_bit_vector(i, feature_dim), range(2**feature_dim)))
    if mixed_pos_and_neg is None:
      sampled_vectors = random.sample(all_vectors, num_vectors_per_formula)
    else:
      all_labels = list(map(lambda i: execute(formula, int_to_bit_vector(i, feature_dim)), range(2**feature_dim)))
      pos_vectors = list(map(lambda j: int_to_bit_vector(j, feature_dim), [i for i, x in enumerate(all_labels) if x == 1]))
      neg_vectors = list(map(lambda j: int_to_bit_vector(j, feature_dim), [i for i, x in enumerate(all_labels) if x == 0]))

      if len(pos_vectors) == 0 or len(neg_vectors) == 0:
        continue # formula is either True or False, so we ignore it

      # num_pos = int(num_vectors_per_formula * (len(pos_vectors)/2**feature_dim))
      # num_neg = 2**feature_dim - num_pos 
      p, n = mixed_pos_and_neg
      if p > len(pos_vectors):
        num_pos = len(pos_vectors)
        num_neg = num_vectors_per_formula - num_pos
      elif n > len(neg_vectors):
        num_neg = len(neg_vectors)
        num_pos = num_vectors_per_formula - num_neg
      else:
        num_pos = p
        num_neg = n

      sampled_vectors = random.sample(pos_vectors, num_pos) + random.sample(neg_vectors, num_neg)
      random.shuffle(sampled_vectors)

    for v in sampled_vectors:
      labeled_vectors.append(v)
      label = execute(formula, v)

      if error_prob != 0:
        p = random.uniform(0, 1)
        if p < error_prob:
          label = not label

      if format_labels:
        label = format_label(label, feature_dim)
      labeled_vectors.append(label)

    # generate query that has not appeared in labeled_vectors 
    unseen_ints = list(filter(lambda i: not int_to_bit_vector(i, feature_dim) in sampled_vectors, range(2**feature_dim)))
    query = int_to_bit_vector(random.sample(unseen_ints, 1)[0], feature_dim)
    query_label = execute(formula, query)

    if format_labels:
      query_label = format_label(query_label, feature_dim)

    labeled_vectors.append(query)

    prompt = (labeled_vectors, query_label)
    # if not prompts in prompts:
    prompts.append(prompt)
    formulas.append(formula)

    if len(formulas) % 500 == 0:
      print("num formulas: "+str(len(formulas)))

  return (formulas, prompts)


def bit_vector_to_int(bit_vector):
  l = len(bit_vector)
  val = 0
  for i in range(l):
    val += bit_vector[i] * 2**(l - i - 1) 
  return val

def int_to_bit_vector(num, dim):
  vec = list(map(lambda x : int(x), list(bin(num))[2:]))
  leading_zeros = [0 for i in range(dim - len(vec))]
  vec = leading_zeros + vec 
  return vec

def format_label(label, dim):
  return int(label)
  # if label:
  #   return [1, 0]
  # else:
  #   return [0, 1]

def gen_train_and_test_data(train_split, num_formulas, num_vectors_per_formula, feature_dim, min_clauses=-1, mixed_pos_and_neg=None, format_labels=False, error_prob=0):
  formulas, prompts = gen_feature_vectors_and_labels(num_formulas, num_vectors_per_formula, feature_dim, min_clauses, mixed_pos_and_neg, format_labels, error_prob)

  train_size = int(num_formulas * train_split)

  train_prompts = prompts[:train_size]
  test_prompts = prompts[train_size:]

  train_formulas = formulas[:train_size]
  test_formulas = formulas[train_size:]

  data_dir = "../data/text_generation/ERROR_PROB_" + str(error_prob)
  if not os.path.isdir(data_dir):
    os.mkdir(data_dir)
  data_dir = data_dir + "/params_num_formulas_" + str(num_formulas) + "_num_bits_" + str(feature_dim) + "_num_clauses_" + str(min_clauses) + "_num_examples_" + str(num_vectors_per_formula)

  if not os.path.isdir(data_dir):
    os.mkdir(data_dir)
    
  with open(data_dir + "/training_data.txt", "w+") as train_file:
    new_lines = []
    for prompt in train_prompts:
      line = str(prompt[0])[1:-1] + "," + str(prompt[1]) + "\n"
      new_line = " " + line.replace("], 1", "]: True").replace("], 0", "]: False").replace(",0", ": False").replace(",1", ": True").replace("[1", "[ 1").replace("[0", "[ 0")      
      new_lines.append(new_line)
      print("line: " + line)
      print("new_line: " + new_line)    
    train_file.write("\n".join(new_lines))

  with open(data_dir + "/validation_data.txt", "w+") as train_file:
    for prompt in test_prompts:
      line = str(prompt[0])[1:-1] + "," + str(prompt[1]) + "\n"
      new_line = " " + line.replace("], 1", "]: True").replace("], 0", "]: False").replace(",0", ": False").replace(",1", ": True").replace("[1", "[ 1").replace("[0", "[ 0")      
      new_lines.append(new_line)    
    train_file.write("\n".join(new_lines))

  # formula files (for reference)
  with open(data_dir + "/training_formulas.txt", "w+") as train_file:
    for formula in train_formulas:
      train_file.write(formula + "\n")

  with open(data_dir + "/validation_formulas.txt", "w+") as val_file:
    for formula in test_formulas:
      val_file.write(formula + "\n")


def gen_train_and_test_data_for_classifier(train_val_test_split, num_formulas, num_vectors_per_formula, feature_dim, min_clauses=-1, mixed_pos_and_neg=None, format_labels=False):
  formulas, prompts = gen_feature_vectors_and_labels(num_formulas, num_vectors_per_formula, feature_dim, min_clauses, mixed_pos_and_neg, format_labels)

  train_split, val_split, _ = train_val_test_split 

  train_size = int(num_formulas * train_split)
  val_size = int(num_formulas * val_split)
  # test_size = num_formulas - train_size - val_size

  train_prompts = prompts[:train_size]
  val_prompts = prompts[train_size:train_size + val_size]
  # test_prompts = prompts[train_size + val_size:]

  train_formulas = formulas[:train_size]
  val_formulas = formulas[train_size:train_size + val_size]

  # train/val files
  data_dir = "../data/classification/distinct_formulas/params_num_formulas_" + str(num_formulas) + "_num_bits_" + str(feature_dim) + "_num_clauses_" + str(min_clauses) + "_num_examples_" + str(num_vectors_per_formula)
  if not os.path.isdir(data_dir):
    os.mkdir(data_dir)

  with open(data_dir + "/training_data_classifier.txt", "w+") as train_file:
    for prompt in train_prompts:
      formatted_prompt = str(prompt[0])[1:-1] + "," + str(prompt[1]) + "\n"
      train_file.write(formatted_prompt)

  with open(data_dir + "/validation_data_classifier.txt", "w+") as val_file:
    for prompt in val_prompts:
      formatted_prompt = str(prompt[0])[1:-1] + "," + str(prompt[1]) + "\n"
      val_file.write(formatted_prompt)

  # formula files (for reference)
  with open(data_dir + "/training_formulas_classifier.txt", "w+") as train_file:
    for formula in train_formulas:
      train_file.write(formula + "\n")

  with open(data_dir + "/validation_formulas_classifier.txt", "w+") as val_file:
    for formula in val_formulas:
      val_file.write(formula + "\n")


  # with open("../data/test_data_classifier.txt", "w+") as test_file:
  #   for prompt in test_prompts:
  #     formatted_prompt = str(prompt[0])[1:-1] + "," + str(prompt[1]) + "\n"
  #     test_file.write(formatted_prompt)

  # with open("../data/test_data.txt", "w+") as test_file:
  #   for prompt in test_prompts:
  #     formatted_prompt = 


# file_paths = [
#               "../data/text_generation/old_mixed_examples_params_num_formulas_50000_num_bits_7_num_clauses_1_num_examples_5/",
# ]

# file_names = ["training_data.txt", "validation_data.txt"]

# for file_path in file_paths:
#   print(file_path)
#   for file_name in file_names:
#     path = file_path + file_name
#     with open(path, "r+") as f:
#       text = f.read()
#       lines = list(filter(lambda x: x != "", text.split("\n")))
#       new_lines = []
#       for line in lines: 
#         new_line = line.replace("], True", "]: True").replace("], False", "]: False").replace(",False", ": False").replace(",True", ": True")
#         new_lines.append(new_line)
#       f.seek(0)
#       f.write("\n".join(new_lines))
#       f.truncate()

