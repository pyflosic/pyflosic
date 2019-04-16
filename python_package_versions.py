import importlib

modules = ['ase','h5py','numba','scipy','numpy','pyscf']

for m in modules:
  try: 
    globals()[m] = importlib.import_module(m)
    try:
        print('%s %s' %(globals()[m].name,globals()[m].version))
    except: print('%s %s' %(globals()[m].__name__,globals()[m].__version__))
  except: print('%s not installed!' %(m))
