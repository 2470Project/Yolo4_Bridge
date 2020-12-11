import os
import  subprocess
with open('./data/test2.txt') as f:
    for l in f.readlines():
        img = l.strip() + '.jpg'
        img_path = os.path.join('./data/images', img)
        if not os.path.exists(img_path):
            continue
        bashCommand = 'python detect.py --weights ./checkpoints/bridge-416 --dont_show true --images ' + img_path
        process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
        output, error = process.communicate()