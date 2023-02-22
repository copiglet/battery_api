import os
import sys
import argparse

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--si', type=int, default=0)

    opt = parser.parse_args()

    save_img = 0
    print(opt.si)
    print(type(opt.si))
    if opt.si == int(0):
        save_img = 0
    else:
        save_img = 1
    print(save_img)
    cmd = "python detect.py --save_img {0}".format(save_img)
    os.system(cmd)

