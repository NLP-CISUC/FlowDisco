from graphviz import Source
import sys

def main():
    args = sys.argv[1:]
    #print(args[0])
    s = Source.from_file(args[0])
    s.view()
    # args is a list of the command line args

if __name__ == "__main__":
    main()



