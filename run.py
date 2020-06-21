import model_evaluation

print('Generated files:'
      + str(model_evaluation.main([
          '--output-filename', 'total',
          '--predictions-list', '*',
          '--predictions-filter', 'SIR:SIRp,SIRDp,SIRFp', 'CUS:CUSa',  #noIq2.5,CUSc70q2.5',
          '--metrics-list', 'MAE', 'MASE', 'MALE',
          '--whole-country',
          '--horizons-list', '1', '5', '30',
          '--date-selector', '2020-04-21',
          '--compare-diff-with-actual',
      ], rel_path_to_this_script_dir='.')))
