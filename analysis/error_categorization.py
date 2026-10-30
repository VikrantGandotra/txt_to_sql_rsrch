"""Analysis 2: paired EX transitions and SQL errors in degraded questions."""
import sys
from analysis.forge_analysis import main

if __name__ == "__main__":
    main(["--analysis", "errors", *sys.argv[1:]])
