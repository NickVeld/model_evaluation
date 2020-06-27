import importlib
import inspect
import os
import sys

thismodule = sys.modules[__name__]
for module_file in os.listdir('metrics'):
    if (module_file.startswith('_') or len(module_file) < 2
            or not(module_file.endswith(".py")) or module_file == '..'):
        continue
    module_file = module_file[:module_file.rfind('.')]
    try:
        module_obj = importlib.import_module('metrics.' + module_file)
    except ValueError:
        continue
    for class_obj in inspect.getmembers(sys.modules['metrics.' + module_file],
                                        lambda x: (inspect.isclass(x)
                                                   or inspect.isfunction(x))):
        setattr(thismodule, *class_obj)


del importlib, inspect, os, sys

if __name__ == '__main__':
    print(dir())
