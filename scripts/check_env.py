import sys, importlib.util
print("Exec:", sys.executable)
for m in ("numpy","pandas","ipykernel"):
    spec = importlib.util.find_spec(m)
    print(f"{m}:","installed" if spec else "not installed")
