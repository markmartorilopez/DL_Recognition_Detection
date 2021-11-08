import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, '/model/')
from network import create_model_with_kfold_cross_validation

sys.path.insert(1,'/test/')
from testing import test_generality_crossValidation_over_test_set


if __name__ == '__main__':
  info_string, models = create_model_with_kfold_cross_validation()
  test_generality_crossValidation_over_test_set(info_string, models)
