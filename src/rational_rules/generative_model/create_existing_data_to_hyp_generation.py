import random 
import os
import sys
from generative_model import execute, gen_DNF, gen_input, num_non_constant_clauses, prettify

existing_dataset_directory = sys.argv[1]
#pretty = sys.argv[2] # none, missing-and-or, all

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

if len(validation_sentences) > len(training_sentences):
  validation_formulas = training_formulas + validation_formulas

print("training_sentences length: " + str(len(training_sentences)))
print("training_formulas length: " + str(len(training_formulas)))
print("validation_sentences length: " + str(len(validation_sentences)))
print("validation_formulas length: " + str(len(validation_formulas)))

#hyp_generation_training_data_path = existing_dataset_directory + "hyp_generation_training_data_mode " + pretty + ".txt"
#hyp_generation_validation_data_path = existing_dataset_directory + "hyp_generation_validation_data_mode " + pretty + ".txt"



for pretty in ["all", "missing-and-or", "none"]:
  hyp_generation_training_data_path = existing_dataset_directory + "hyp_generation_training_data_mode " + pretty + ".txt"
  hyp_generation_validation_data_path = existing_dataset_directory + "hyp_generation_validation_data_mode " + pretty + ".txt"
  with open(hyp_generation_training_data_path, "w") as f:
    for i in range(len(training_sentences)):
      sentence = training_sentences[i]
      formula = training_formulas[i]

      prompts = sentence.split(", [")
      index = random.sample(list(range(1, len(prompts))), 1)[0]
 
      if pretty == "missing-and-or":
        pretty_formula = prettify(formula, change_and_or=False)
      elif pretty == "all":
        pretty_formula = prettify(formula, change_and_or=True)
      else:
        pretty_formula = formula

      #prompts = sentence.split(", [")
      #index = random.sample(list(range(1, len(prompts))), 1)[0]
      prompts = prompts[:index] + ["hyp: " + pretty_formula] + prompts[index:]
      formatted_prompt = (", ".join(prompts) + "\n").replace("[hyp", "hyp")
      f.write(formatted_prompt)

  with open(hyp_generation_validation_data_path, "w") as f:
    for i in range(len(validation_sentences)):
      sentence = validation_sentences[i]
      formula = validation_formulas[i]
    
      prompts = sentence.split(", [")
      index = random.sample(list(range(1, len(prompts))), 1)[0]

      for pretty in ["all", "missing-and-or", "none"]:
        if pretty == "missing-and-or":
          pretty_formula = prettify(formula, change_and_or=False)
        elif pretty == "all":
          pretty_formula = prettify(formula, change_and_or=True)
        else:
          pretty_formula = formula

        #prompts = sentence.split(", [")
        #index = random.sample(list(range(1, len(prompts), 1)), 1)[0]
        prompts = prompts[:index] + ["hyp: " + pretty_formula] + prompts[index:]
        formatted_prompt = (", [".join(prompts) + "\n").replace("[hyp", "hyp")
        f.write(formatted_prompt)
