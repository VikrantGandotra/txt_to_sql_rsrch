"""Analysis 1: aggregate and per-request prompt volume versus execution accuracy."""
import sys
from analysis.forge_analysis import main

if __name__ == "__main__":
    main(["--analysis", "tokens", *sys.argv[1:]])
