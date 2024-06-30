import sys
import time


def do_work(n):
    time.sleep(n)
    print('I just did some hard work for {}s!'.format(n))


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Please provide one integer argument', file=sys.stderr)
    try:
        seconds = int(sys.argv[1])
        do_work(seconds)
    except Exception as e:
        print(e)