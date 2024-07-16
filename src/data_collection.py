import os
import matplotlib.pyplot as plt
import rospy
from pynput import keyboard
from cloud_proxy_three import CloudProxy

class DataCollection:

    def __init__(self, obj_set_id=0):
        self.CloudProxy = CloudProxy()
        self.obj_set_id = obj_set_id
        self.fp = f'/home/master_oogway/panda_ws/src/helping_hands_panda/data/{obj_set_id}'
        self.img_id = 0

    def save_image(self):
        fp = self.fp
        img_bob = self.CloudProxy.getRGBImage('bob')
        img_kevin = self.CloudProxy.getRGBImage('kevin')
        img_stuart = self.CloudProxy.getRGBImage('stuart')

        bob_fp = os.path.join(fp, f'bob/{self.img_id}.jpg')
        kevin_fp = os.path.join(fp, f'kevin/{self.img_id}.jpg')
        stuart_fp = os.path.join(fp, f'stuart/{self.img_id}.jpg')

        if not os.path.exists(os.path.dirname(bob_fp)):
            os.makedirs(os.path.dirname(bob_fp))
        if not os.path.exists(os.path.dirname(kevin_fp)):
            os.makedirs(os.path.dirname(kevin_fp))
        if not os.path.exists(os.path.dirname(stuart_fp)):
            os.makedirs(os.path.dirname(stuart_fp))
        
        plt.imsave(bob_fp, img_bob)
        plt.imsave(kevin_fp, img_kevin)
        plt.imsave(stuart_fp, img_stuart)
        print(f'Images saved: {self.img_id}')
        self.img_id += 1

def on_press(key):
    global dc
    if key == keyboard.Key.space:
        dc.save_image()
    if key == keyboard.Key.esc:
        return False

def main():
    rospy.init_node('test')
    global dc
    dc = DataCollection()
    print("Press SPACE to save images. Press ESC to exit.")
    with keyboard.Listener(on_press=on_press) as listener:
        listener.join()

if __name__ == '__main__':
    main()