import re #python regular expression matching module

script = re.sub(r'\n# In\[.*\]:\n\n', '', open('Kc.py').read())

with open('Kc_new.py','w') as fh:
    fh.write(script)
