import numpy as np
from classes.models import alexnet, alexnet_v2, inception_v3
from collections import Counter
import pandas as pd
from random import shuffle
import os


cull_length = True

WIDTH = 250  # Image Width
HEIGHT = 250  # Image Height
LR = 1e-3  # Learning Rate
EPOCHS = 8
MODEL_NAME = 'pywebdevelopment_{0}_{1}_{2}_epochs.model'.format(LR, 'alexnet_v2', EPOCHS)

train_data = np.load(r'S:\Python Development\PyWebDevelopment\pywebdevelopment\training_data\training_data_1.npy')
print(len(train_data))

df = pd.DataFrame(train_data)
# print(df.head)
print(Counter(df[1].apply(str)))

# Balance Data
mod_header = []
mod_item = []
mod_item_expanded = []
mod_create_item = []

header_arr = [1, 0, 0, 0]
item_arr = [0, 1, 0, 0]
item_expanded_arr = [0, 0, 1, 0]
create_item_arr = [0, 0, 0, 1]

test = []
num_h = 1
num_i = 1
num_i_e = 1
num_c_i = 1

duplicates = 0
test_num = 1

# shuffle(train_data)

for data in train_data:
    img = data[0]
    style = data[1]

    if style == 'module_header':
        mod_header.append([img, header_arr])
        if num_h <= test_num:
            test.append([img, header_arr])
            num_h = num_h + 1
    elif style == 'module_item':
        mod_item.append([img, item_arr])
        if num_i <= test_num:
            test.append([img, item_arr])
            num_i = num_i + 1
    elif style == 'module_content':
        mod_item_expanded.append([img, item_expanded_arr])
        if num_i_e <= test_num:
            test.append([img, item_expanded_arr])
            num_i_e = num_i_e + 1
    elif style == 'module_create_item':
        mod_create_item.append([img, create_item_arr])
    if num_c_i <= test_num:
        test.append([img, create_item_arr])
        num_c_i = num_c_i + 1

# Get shortest length of data
# Gets the numerical value of the shortest list supplied
max_num = min(len(mod_header), len(mod_item), len(mod_item_expanded), len(mod_create_item))
print("Max per type: %s" % max_num)

if cull_length:
    final_data = mod_header[:max_num] + mod_item[:max_num] + mod_item_expanded[:max_num] + mod_create_item[:max_num]
else:
    final_data = mod_header + mod_item + mod_item_expanded + mod_create_item


data_len = len(final_data)
print(data_len)

# Train Data to model
train = final_data

print('Creating Duplicates...')
for i in range(0, duplicates):
    train = train + train
    # test = test + test

# train = train[:-50]
shuffle(train)
# test = train[-300:]
shuffle(test)
for i in test:
    print(i[1])
print(len(train))
print(len(test))

X = np.array([i[0] for i in train]).reshape(-1, WIDTH, HEIGHT, 1)
Y = [i[1] for i in train]

test_x = np.array([i[0] for i in test]).reshape(-1, WIDTH, HEIGHT, 1)
test_y = [i[1] for i in test]

# model = alexnet(WIDTH, HEIGHT, LR, output=7)
# model = inception_v3(WIDTH, HEIGHT, LR, output=7)
model = alexnet_v2(WIDTH, HEIGHT, LR, output=4)

try:
    model.fit({'input': X}, {'targets': Y}, n_epoch=100,
              validation_set=({'input': test_x}, {'targets': test_y}),
              batch_size=10, snapshot_step=500, show_metric=True, run_id=MODEL_NAME)
    '''
    model.fit({'input': X}, {'targets': Y}, n_epoch=500, validation_set=0.1,
              batch_size=10, snapshot_step=500, show_metric=True, run_id=MODEL_NAME)
    '''
except Exception as e:
    print(e)

model.save('{0}/model/{1}'.format(os.getcwd(), MODEL_NAME))

# tensorboard --logdir=foo:"S:\Python Development\PyWebDevelopment\pywebdevelopment\log"
