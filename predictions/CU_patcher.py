import os, re

fnp = re.compile(r'CU-select_(?P<c>\d{2})?(?P<t>contact|nointerv)_quantile(?P<q>[0-9.]+\d).csv')

for fn in os.listdir():
    fres = fnp.search(fn)
    if not(fres):
        continue
    if fres.group('t') == 'contact':
        newfn = 'CUs_c{}q{}.csv'.format(fres.group('c'), fres.group('q'))
    else:
        newfn = 'CUs_NoIq{}.csv'.format(fres.group('q'))
    
    os.rename(fn, newfn)

    # Read in the file
    with open(newfn, 'r') as file :
        filedata = file.read()

    # Replace the target string
    filedata = filedata.replace('04/22/2020', '04/21/2020')

    # Write the file out again
    with open(newfn, 'w') as file:
        file.write(filedata)