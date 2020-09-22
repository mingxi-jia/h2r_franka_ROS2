import numpy as np

obj_dict = {
    1: 'blue-red-blue',
    2: 'blue',
    3: 'brown-yellow',
    4: 'red-brown',
    5: 'green-blue-green',
    6: 'yellow',
    7: 'green-red',
    8: 'green',
    9: 'orange-yellow-orange',
    10: 'orange',
    11: 'red',
    12: 'green-blue',
    13: 'purple-green',
    14: 'orange-blue'
}

brick_dict = {
    1: 'normal',
    2: 'lego',
    3: 'box',
    4: 'train'
}

for i in range(40):
    obj_idx = np.random.choice(list(obj_dict.keys()), 2, False)
    brick_idx = np.random.choice(list(brick_dict.keys()), 1, False)
    print('OBJ: {}:{}, {}:{}; BRICK: {}:{}'.format(obj_idx[0], obj_dict[obj_idx[0]], obj_idx[1], obj_dict[obj_idx[1]], brick_idx[0], brick_dict[brick_idx[0]]))
