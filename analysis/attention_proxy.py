"""Analysis 3: descriptive prompt-length proxies versus column-name errors."""
import sys
from analysis.forge_analysis import main

if __name__ == "__main__":
    main(["--analysis", "proxy", *sys.argv[1:]])
