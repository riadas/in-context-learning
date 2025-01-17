import random 
import os
import sys
from generative_model import execute, gen_DNF, gen_input, num_non_constant_clauses, prettify

existing_dataset_directory = sys.argv[0]
pretty = sys.argv[1] # none, missing-and-or, all

if existing_dataset_directory[-1] != "/":
  existing_dataset_directory = existing_dataset_directory + "/"

training_data_path = existing_dataset_directory + "training_data.txt"
training_formulas_path = existing_dataset_directory + "training_formulas.txt"
validation_data_path = existing_dataset_directory + "validation_data.txt"
validation_formulas_path = existing_dataset_directory + "validation_formulas.txt"

with open(training_data_path, "r") as f:
  training_sentences = list(filter(lambda x: len(x) > 5, f.read().split("\n")))

with open(training_formulas_path, "r") as f:
  training_formulas = list(filter(lambda x: len(x) > 5, f.read().split("\n")))

with open(validation_data_path, "r") as f:
  validation_sentences = list(filter(lambda x: len(x) > 5, f.read().split("\n")))

with open(validation_formulas_path, "r") as f:
  validation_formulas = list(filter(lambda x: len(x) > 5, f.read().split("\n")))

hyp_generation_training_data_path = existing_dataset_directory + "hyp_generation_training_data.txt"
hyp_generation_validation_data_path = existing_dataset_directory + "hyp_generation_validation_data.txt"

with open(hyp_generation_training_data_path, "w") as f:
  for i in range(len(training_sentences)):
    sentence = training_sentences[i]
    formula = training_formulas[i]
    
    if pretty == "missing-and-or":
      pretty_formula = prettify(formula, change_and_or=False)
    elif pretty == "all":
      pretty_formula = prettify(formula, change_and_or=True)
    else:
      pretty_formula = formula

    prompts = sentence.split(", ")
    index = random.sample(list(range(1, len(prompts))), 1)[0]
    prompts = prompts[:index] + [pretty_formula] + prompts[index:]
    formatted_prompt = ", ".join(prompts) + "\n"
    f.write(formatted_prompt)

with open(hyp_generation_validation_data_path, "w") as f:
  for i in range(len(training_sentences)):
    sentence = validation_sentences[i]
    formula = validation_formulas[i]
    
    if pretty == "missing-and-or":
      pretty_formula = prettify(formula, change_and_or=False)
    elif pretty == "all":
      pretty_formula = prettify(formula, change_and_or=True)
    else:
      pretty_formula = formula

    prompts = sentence.split(", ")
    index = random.sample(list(range(1, len(prompts))), 1)[0]
    prompts = prompts[:index] + [pretty_formula] + prompts[index:]
    formatted_prompt = ", ".join(prompts) + "\n"
    f.write(formatted_prompt)