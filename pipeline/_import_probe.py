import sys

sys.path.insert(0, sys.argv[1])
__import__(sys.argv[2])
