import sys 
sys.path.append("../")
import pandas as pd 

def load_imagenet():
    path = "../data/imagenet_classes.txt"
    df = pd.read_csv(path, header=None)
    classes = df.values.squeeze() 
    imagenet_class_to_ix = {}  
    ix_to_imagenet_class = {}
    for ix, clas in enumerate(classes):
        imagenet_class_to_ix[clas] = ix 
        ix_to_imagenet_class[ix] = clas 

    return imagenet_class_to_ix, ix_to_imagenet_class


# Imagenet classes hierarchy 
# https://observablehq.com/@mbostock/imagenet-hierarchy
def load_imagenet_hierarchy_dicts():
    path = "../data/image_net_hierarchy.csv"
    f = open(path, "r") 
    l2_to_l1 = {} 
    l3_to_l2 = {} 
    for line in f: 
        line = line.replace("\n", "")
        # line = line.replace(" ", "")
        line = line.split(", ")
        line = [s.strip() for s in line]
        if line[0] == "2":
            l2_to_l1[line[1]] = line[2:]
        elif line[0] == "3":
            l3_to_l2[line[1]] = line[2:] 
    l3_to_l1 = {}  # 26
    for l3_class in l3_to_l2.keys():
        l2_classes = l3_to_l2[l3_class]
        all_l1_classes = []
        for l2_class in l2_classes:
            all_l1_classes = all_l1_classes + l2_to_l1[l2_class]
        l3_to_l1[l3_class] = all_l1_classes

    return l2_to_l1, l3_to_l1


def get_imagenet_sub_classes(optimal_class):
    l2_to_l1, l3_to_l1 = load_imagenet_hierarchy_dicts() 
    if optimal_class in l2_to_l1.keys():
        optimal_sub_classes = l2_to_l1[optimal_class]
        optimal_class_level = 2
    elif optimal_class in l3_to_l1.keys():
        optimal_sub_classes = l3_to_l1[optimal_class]
        optimal_class_level = 3
    else:
        optimal_sub_classes = []
        optimal_class_level = 1
    
    return optimal_class_level, optimal_sub_classes
