import os, sys
import os.path as osp
from os.path import join
from tracemalloc import Statistic

from sklearn.metrics import d2_tweedie_score
root_dir = osp.dirname(__file__)

from glob import glob
from tqdm import tqdm
import random
import shutil
from typing import List, Dict
import argparse
import json

random.seed(0)

def log_dataset_statistic(dataset_path: str, dataset_name: str, statistic_dir: str):
    dsets = [join(dataset_path, 'train'), join(dataset_path, 'test'), join(dataset_path, 'val')]
    for dset in dsets:
        print(dset)
        with open(join(statistic_dir, "{}_{}.txt".format(dataset_name, osp.basename(dset))), 'w') as f:
            img_paths = glob(join(dset, "*/*"))
            for img_path in tqdm(img_paths):
                info = img_path.split("/")
                img_name = info[-1]
                cls = info[-2]
                phase = info[-3]
                saved_path = "{}/{}\n".format(cls, img_name)
                f.write(saved_path)
    

def statisticize_dataset(dataset_path: str) -> Dict[str, Dict[str, int]]:
    train_set = join(dataset_path, 'train')
    test_set = join(dataset_path, 'test')
    val_set = join(dataset_path, 'val')
    
    dsets = [train_set, test_set, val_set]

    statistic = {
        train_set: {},
        test_set: {},
        val_set: {}
    }
    for dset in dsets:
        for cls in ['0_real', '1_df', '1_f2f', '1_fs', '1_nt', '1_fake']:
            if not osp.exists(join(dset, cls)):
                continue
            num_samples = len(os.listdir(join(dset, cls)))
            if num_samples:
                statistic[dset][cls] = num_samples
    print(json.dumps(statistic, indent=4))
    return statistic

def get_image_from_txt_file(txt_file: str, head_path: str):
    dset = []
    with open(txt_file, 'r') as f:
        lines = f.readlines()
        for line in tqdm(lines):
            line = line.strip()
            if line == '':
                continue
            dset.append(head_path + '/' + line)
    return dset
    
def check_synchronization_between_servers(dataset_path: str, train_set: List[str], val_set: List[str], test_set: List[str]) -> bool:
    cur_train_set = glob(join(dataset_path, 'train', '*/*'))
    cur_test_set = glob(join(dataset_path, 'test', '*/*'))
    cur_sets = [cur_train_set, cur_test_set]
    
    txt_train_set = train_set.extend(val_set)
    txt_test_set = test_set
    txt_sets = [txt_train_set, txt_test_set]
    
    assert len(txt_train_set) == len(cur_train_set), 'Correct! Train length is matched.'
    assert len(txt_test_set) == len(cur_test_set), 'May be test dataset of this benchmark: {} - is removed!'.format(find_dataset_name(dataset_path))
    
    for i in range(2):
        cur_set = cur_sets[i]
        txt_set = txt_sets[i]
        phase = 'train' if 'train' in cur_set[0] else 'test'
        print("Check phase {} in dataset {}".format(phase, find_dataset_name(dataset_path)))
         
        cur_dict = {path: 0 for path in cur_set}
        for path in tqdm(txt_set):
            try:
                if cur_dict[path] == 0:
                    cur_dict[path] = 1
            except:
                cur_dict[path] = -1
        
        cnt_lack = 0
        for path, v in cur_dict.items():
            if v == -1:
                print("{} | in txt-file not exists in current-device!".format(path))
            if v == 0:
                if phase == 'train':
                    print("{} | in current-device not exists in txt_file!".format(path))
                cnt_lack += 1
        print('Lack {} image'.format(cnt_lack))
                
def make_dataset_from_txt_file(dataset_path: str, train_file: str, test_file: str, val_file: str, check_sync=True, sync=False):
    txt_train_set = get_image_from_txt_file(train_file, join(dataset_path, 'train'))
    txt_val_set = get_image_from_txt_file(val_file, join(dataset_path, 'train'))
    txt_test_set = get_image_from_txt_file(test_file, join(dataset_path, 'test'))
    if check_sync:
        check_synchronization_between_servers(dataset_path, txt_train_set, txt_val_set, txt_test_set)
        
    # Make dataset
    if sync:
        # Move val
        val_dir = join(dataset_path, 'val')
        if not osp.exists(val_dir):
            os.mkdir(val_dir)
        for img_path in txt_val_set:
            shutil.move(img_path, val_dir)
            
        # Delete test:
        cur_test_set = glob(join(dataset_path, 'test', '*/*'))
        cur_test_dict = {path: 0 for path in cur_test_set}
        redunt_dir = join(dataset_path, 'redunt_test')
        if not osp.exists(redunt_dir):
            os.mkdir(redunt_dir)
        for path in tqdm(txt_test_set):
            try:
                if cur_test_dict[path] == 0:
                    cur_test_dict[path] = 1
            except:
                cur_test_dict[path] = -1
        
        for path, v in cur_test_dict.items():
            if v == 0:
                shutil.move(path, redunt_dir)
    
def make_validation_set(dataset_path: str, num_real: int, num_fake: int):
    """ Function for some dataset that maintains number of test samples. eg: Celeb-DF, UADFV, df_timit
    Args:
        dataset_path (str): path to dataset
        num_real (int): number of real validation samples that want to make
        num_fake (int): number of fake validation samples that want to make
    """
    train_set = join(dataset_path, 'train')
    val_set = join(dataset_path, 'val')
    if not osp.exists(val_set):
        os.mkdir(val_set)
        
    clses = os.listdir(train_set)
    for cls in clses:
        print(cls)
        val_dir = join(val_set, cls)
        if not osp.exists(val_dir):  
            os.mkdir(val_dir)
        imgs = glob(join(train_set, cls, '*'))
        random.shuffle(imgs)
        num_samples = num_real if 'real' in cls else num_fake
        for _ in tqdm(range(num_samples)):
            img = imgs.pop()
            shutil.move(img, val_dir)
        
    ### CHECK ###
    for cls in os.listdir(val_set):
        print(join(val_set, cls), end=' - ')
        print("Number samples: ", len(os.listdir(join(val_set, cls))))
    
def delete_test_set(dataset_path: str, num_real: int, num_fake: int):
    """ Function to delete some samples for some dataset. eg: dfdc, df_in_the_wild, ff
    Args:
        dataset_path (str): path to dataset
        num_real (int): number of real test samples that want to delete
        num_fake (int): number of fake test samples that want to delete
    """
    test_set = join(dataset_path, 'test')
    for cls in os.listdir(test_set):
        print(cls)
        imgs = glob(join(test_set, cls, '*'))
        random.shuffle(imgs)
        num_samples = num_real if 'real' in cls else num_fake
        for _ in tqdm(range(num_samples)):
            img = imgs.pop()
            os.remove(img)
        
    ### CHECK ###
    for cls in os.listdir(test_set):
        print(join(test_set, cls), end=' - ')
        print("Number samples: ", len(os.listdir(join(test_set, cls))))

def aggregate_fake_ff_set(ff_path: str):
    """ Aggregate all the samples in component of ff dataset
    Args:
        ff_path (str): path to ff dataset
    """
    train_ff = join(ff_path, 'train')
    val_ff = join(ff_path, 'val')
    test_ff = join(ff_path, 'test')
    
    ff_set = [train_ff, val_ff, test_ff]
    for dset in ff_set:
        print("* ", dset)
        fake_agg_dir = join(dset, '1_fake')
        if not osp.exists(fake_agg_dir):
            os.mkdir(fake_agg_dir)
        for fake_type in ['1_df', '1_f2f', '1_fs', '1_nt']:
            fake_type_dir = join(dset, fake_type)
            print('   === ', fake_type)
            for img_name in tqdm(os.listdir(fake_type_dir)):
                img_path = join(fake_type_dir, img_name)
                new_path = join(fake_type_dir, fake_type.replace('1_', '') + '_' + img_name)
                os.rename(img_path, new_path)
                shutil.move(new_path, fake_agg_dir)

def find_dataset_name(dataset_path: str):
    if 'ff' in dataset_path:
        return 'ff'
    if 'UADFV' in dataset_path:
        return 'uadfv'
    if 'df_in_the_wild' in dataset_path:
        return 'wild'
    if 'dfdc' in dataset_path:
        return 'dfdc'
    if 'df_timit' in dataset_path:
        return 'timit'
    if 'Celeb-DF' in dataset_path:
        return 'celeb'
         
def parse_args():
    parser = argparse.ArgumentParser(description="Filter noise image by another face detection module")
    parser.add_argument("--dataset_path", type=str, required=True, help="path to dataset")
    parser.add_argument("--make_validation_set", type=int, default=0, help="make validation set from train set (maintain test samples) or not")
    parser.add_argument("--num_real_val", type=int, default=0)
    parser.add_argument("--num_fake_val", type=int, default=0)
    parser.add_argument("--delete_test_set", type=int, default=0, help="Delete some samples in test set")
    parser.add_argument("--num_real_test", type=int, default=0)
    parser.add_argument("--num_fake_test", type=int, default=0)
    parser.add_argument("--agg_fake_ff_set", type=int, default=0, help="Aggregate fake samples of ff dataset")
    return parser.parse_args()
    
if __name__ == '__main__':
    args = parse_args()
    if args.make_validation_set:
        make_validation_set(args.dataset_path, int(args.num_real_val), int(args.num_fake_val))
    if args.delete_test_set:
        delete_test_set(args.dataset_path, int(args.num_real_test), int(args.num_fake_test))
    if args.agg_fake_ff_set:
        aggregate_fake_ff_set(args.dataset_path)
        
    statisticize_dataset(args.dataset_path)
    log_dataset_statistic(args.dataset_path, find_dataset_name(args.dataset_path), "/mnt/disk1/phucnp/Graduation_Thesis/review/forensics/preprocess_data/data_statistic")