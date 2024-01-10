import numpy as np

down_left = [-0.27917353, -0.31692874]
upper_left = [-0.61820,  -0.31932 ]
down_right = [-0.26832, 0.16188 ]
upper_right = [-0.60830454, 0.15945354]
UPPER_XY = -0.61820
DOWN_XY = -0.26832
LEFT_XY = -0.31932
RIGHT_XY = 0.16188
UPPERDOWN_XY = down_right[0] - upper_left[0]
LEFTRIGHT_XY = down_right[1] - down_left[1]
Z_MIN_ROBOT = -0.55862
X_MIN_ROBOT = upper_left[0]
X_MAX_ROBOT = down_left[0]
Y_MIN_ROBOT = down_left[1]
Y_MAX_ROBOT = down_right[1]

# pixel_x_reso = 208
# pixel_y_reso = 288
_x_reso = DOWN_XY - UPPER_XY
_y_reso = RIGHT_XY - LEFT_XY
pixel_x_reso = 206
pixel_y_reso = 284

def xy2pixel(xy):
    # upper left corner of the action space is the pixel origin
    x, y = xy
    # pixel_x = (pixel_x_reso/ _x_reso) * (x - UPPER_XY)
    # pixel_y = (pixel_y_reso/ _y_reso) * (y - LEFT_XY)
    pixel_x = (x-X_MIN_ROBOT) / (X_MAX_ROBOT-X_MIN_ROBOT) * pixel_x_reso
    pixel_y = (y - Y_MIN_ROBOT) / (Y_MAX_ROBOT - Y_MIN_ROBOT) * pixel_y_reso
    pixel = np.array([pixel_x, pixel_y]).astype(int)
    return pixel

def pixel2xy(pixel):
    pixel_x, pixel_y = pixel
    # x = UPPER_XY + UPPERDOWN_XY * pixel_x / pixel_x_reso
    # y = LEFT_XY + LEFTRIGHT_XY * pixel_y / pixel_y_reso
    # x = (_x_reso / pixel_x_reso) * pixel_x +  UPPER_XY
    # y = (_y_reso / pixel_y_reso) * pixel_y + LEFT_XY
    x = pixel_x/pixel_x_reso*(X_MAX_ROBOT-X_MIN_ROBOT)+X_MIN_ROBOT
    y = pixel_y / pixel_y_reso * (Y_MAX_ROBOT - Y_MIN_ROBOT) + Y_MIN_ROBOT
    xy = np.array([x, y])
    return xy

def pixel2xyz(pixel):
    x,y = pixel2xy(pixel)
    z = -0.1511397
    return (x,y,z)


def getGraspZ(pixel, depthmap):
    # take pixel_xy and output z in robot coordinate
    Z_MIN_DEPTH_PIXEL = 1.051
    height = Z_MIN_DEPTH_PIXEL - depthmap[pixel]/1000
    return Z_MIN_ROBOT + height

if __name__ == '__main__':
    print(1)