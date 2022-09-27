from random import randint 

# ----- FORMULA EVALUATION -----

'''Evaluate Boolean formula on feature vector input'''
def execute(formula, input):
  if "f_" in formula:
    formatted_formula = unprettify(formula)
  else: 
    formatted_formula = formula

  eval_string = "(lambda x : " + formatted_formula + ")(" + str(input) + ")"
  return eval(eval_string)

'''Convert math formatting to Python formatting'''
def unprettify(formula):
  return formula.replace("(x)", "]").replace("f_", "x[").replace("∧", "and").replace("∨", "or")

# ----- GENERATIVE MODEL: GENERATE FORMULAS + INPUTS -----

'''Generate DNF formula according to Rational Rules (RR) CFG, where input feature vectors have length num_features'''
def gen_DNF(num_features, pretty=False):
  return gen_D(num_features, pretty)

'''D production in RR CFG'''
def gen_D(num_features, pretty=False):
  choice = randint(0,1)

  if choice == 0: 
    if not pretty:
      return "(" + gen_C(num_features, pretty) + " or " + gen_D(num_features, pretty) + ")" 
    else:
      return "(" + gen_C(num_features, pretty) + " ∨ " + gen_D(num_features, pretty) + ")"
  else: 
    return "False"

'''C production in RR CFG'''
def gen_C(num_features, pretty=False):
  choice = randint(0,1)

  if choice == 0: 
    if not pretty:
      return "(" + gen_comparison(num_features, pretty) + " and " + gen_C(num_features, pretty) + ")" 
    else:
      return "(" + gen_comparison(num_features, pretty) + " ∧ " + gen_C(num_features, pretty) + ")"
  else: 
    return "True"

'''Comparison production in RR CFG'''
def gen_comparison(num_features, pretty=False):
  feature_index = randint(0, num_features - 1)
  bit = randint(0, 1)
  if not pretty:
    return "(x[" + str(feature_index) + "] == " + str(bit) + ")"
  else:
    return "(f_" + str(feature_index) + "(x) == " + str(bit) + ")"

'''Generate random input feature vector of length num_features'''
def gen_input(num_features):
  return [randint(0,1) for i in range(num_features)]

# ----- UTILS -----

def num_clauses(formula):
  if "(x)" in formula:
    formatted_formula = unprettify(formula)
  else:
    formatted_formula = formula 

  return formatted_formula.count("or") + formatted_formula.count("and") + 1

def num_non_constant_clauses(formula):
  return num_clauses(formula) - formula.count("True") - formula.count("False")