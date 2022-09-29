from generate_train_test_data import *

"""
#E (number of examples): RR uses 5 to 8 examples for 7-feature vectors -> have this value depend on #F?
#F (number of features): n -> 3 through 10
#C (number of clauses): 2n distinct comparison clauses; 2^(2n-1) different formulas -> 1 through 5
"""

def generate_curriculum(num_examples_range, num_features_range, num_clauses_range):
  pass