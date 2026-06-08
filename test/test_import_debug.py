import sys
import chisurf as cs
print("DEBUG: cs imported")
import chisurf.core.fitting
print("DEBUG: cs.core.fitting imported")
import chisurf.core.fitting.fit
print("DEBUG: cs.core.fitting.fit imported")
import chisurf.core.fitting.parameter
print("DEBUG: cs.core.fitting.parameter imported")

if __name__ == "__main__":
    test_imports()
    print("ALL OK")
