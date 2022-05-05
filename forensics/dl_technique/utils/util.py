import numpy as np
from tqdm import tqdm
from glob import glob

def save_list_to_file(list, file_name: str):
    if isinstance(list, np.ndarray):
        list = list.tolist()
    assert type(list) == 'list', "Type of iteration should be a list!"
    with open('result/' + file_name, 'w') as f:
        f.write(','.join([str(ele) for ele in list]))
        f.write('\n')
            
        