import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--tgt_time', type=str, default=None)
args = parser.parse_args()
tgt_time = time.strptime(args.tgt_time, '%Y-%m-%d %X')
while True:
    current_time = time.localtime()
    if current_time >= tgt_time:
        print('its high noon!')
        break
    print('not the right time, currently {}'.format(current_time))
    time.sleep(30)