import sys

from plataformateste import graphs


def main():
    # args is a list of the command line args
    args = sys.argv[1:]
    # print(args[0])

    graphs.show_file(args[0])


if __name__ == "__main__":
    main()
