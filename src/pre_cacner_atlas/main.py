import sys, importlib
print("Exec:", sys.executable)
np = importlib.import_module("numpy")
pd = importlib.import_module("pandas")
print("numpy:", np.__version__, "pandas:", pd.__version__)
