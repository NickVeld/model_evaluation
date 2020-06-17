import model_evaluation

print('Generated files:'
      + str(model_evaluation.main([
          '--output-filename', 'total',
          '--predictions-list', '*',
          '--metrics-list', 'MAE', 'MASE', 'MALE',
          '--whole-country',
          '--horizons-list', '1', '5', '30',
      ], rel_path_to_this_script_dir='.')))
