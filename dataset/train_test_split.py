import json
import numpy as np
import random

random.seed(2023)
with open('./split/landslide.json', 'r') as lsf, open('./split/non_landslide.json', 'r') as nlsf:
    landslide_info = json.load(lsf)
    non_landslide_info = json.load(nlsf)

print(len(landslide_info))
print(len(non_landslide_info))
random.shuffle(landslide_info)
random.shuffle(non_landslide_info)

landslide_info = np.array(landslide_info)
non_landslide_info = np.array(non_landslide_info)

test_landslide, landslide_rest = np.split(landslide_info, [500])
test_non_landslide, non_landslide_rest = np.split(non_landslide_info, [500])
print(test_landslide.shape, landslide_rest.shape)
print(test_non_landslide.shape, non_landslide_rest.shape)

test_set = list(np.concatenate([test_landslide, test_non_landslide]))
print(len(test_set))
with open('./split/test.json', 'w') as testf:
    json.dump(test_set, testf, indent='\n')

train_set = {}
label_train_landslide, unlabel_train_landslide, rest_landslide = np.split(landslide_rest, [500, 1000])
label_train_non_landslide, unlabel_train_non_landslide, rest_non_landslide = np.split(non_landslide_rest, [500, 1000])

labeled_train_set = list(np.concatenate([label_train_landslide, label_train_non_landslide]))
unlabeled_train_set = list(np.concatenate([unlabel_train_landslide, unlabel_train_non_landslide]))

print(len(labeled_train_set))
print(len(unlabeled_train_set))
print(rest_landslide.shape)
print(rest_non_landslide.shape)

train_set['labeled'] = labeled_train_set
train_set['unlabeled'] = unlabeled_train_set

with open('./split/train.json', 'w') as trainf:
    json.dump(train_set, trainf, indent='\n')







