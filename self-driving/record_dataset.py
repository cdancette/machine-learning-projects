import gym
import numpy as np
import imageio
import os
import sys
from pyglet.window import key

from model import LEFT, RIGHT, GO, ACTIONS, action_to_id 

samples_each_classes = 200

if __name__=='__main__':

    if len(sys.argv) < 2:
        sys.exit("Usage : python record_dataset.py path")
    
    folder = sys.argv[1]
    images = os.path.join(folder, "images")
    labels = os.path.join(folder, "labels.txt")
    os.makedirs(images, exist_ok=True)

    a = np.array([0.0, 0.0, 0.0])

    def key_press(k, mod):
        global restart
        if k==key.LEFT:  a[0] = -1.0
        if k==key.RIGHT: a[0] = +1.0
        if k==key.UP:    a[1] = +1.0
        if k==key.DOWN:  a[2] = +0.8   # set 1.0 for wheels to block to zero rotation

    def key_release(k, mod):
        if k==key.LEFT  and a[0]==-1.0: a[0] = 0
        if k==key.RIGHT and a[0]==+1.0: a[0] = 0
        if k==key.UP:    a[1] = 0
        if k==key.DOWN:  a[2] = 0

    env = gym.make('CarRacing-v0').env
    env.reset()

    env.viewer.window.on_key_press = key_press
    env.viewer.window.on_key_release = key_release
    env.reset()
    for i in range(100):
        # initialisation
        env.step([0, 0, 0])
        env.render()
    
    file_labels = open(labels, 'w')
    samples_saved = {a: 0 for a in ACTIONS}

    i = 0
    while True:
        s, r, done, info = env.step(a)
        action_id = action_to_id(a)
        if samples_saved[action_id] < samples_each_classes:
            samples_saved[action_id] += 1
            samples_each_classes
            imageio.imwrite(os.path.join(folder, 'images', 'img-%s.jpg' % i ), s)
            file_labels.write('%s %s\n' % ('img-%s.jpg' % i, action_id))
            file_labels.flush()
            i += 1
            print(samples_saved)
        env.render()
    env.close()
