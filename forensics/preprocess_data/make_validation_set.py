import os, sys
import os.path as osp

from sklearn.metrics import d2_tweedie_score
root_dir = osp.dirname(__file__)

from glob import glob
from tqdm import tqdm
import random
import shutil
import auxiliary

random.seed(0)


######################################################
############ dfdc dataset
######################################################
def move_image_in_dfdc_dataset():
    # Folder
    train_folder = "../../../../2_Deep_Learning/Dataset/facial_forgery/dfdc/images/train"
    val_folder = train_folder.replace("train", "val")
    real_folder = osp.join(val_folder, "0_real")
    fake_folder = osp.join(val_folder, "1_df")
    # Make val dir:
    if not osp.exists(val_folder):
        os.mkdir(val_folder)
    if not osp.exists(real_folder):
        os.mkdir(real_folder)
    if not osp.exists(fake_folder):
        os.mkdir(fake_folder)
    # Move data from train to val:
    real_train_data = glob(osp.join(train_folder, "0_real/*"))
    fake_train_data = glob(osp.join(train_folder, "1_df/*"))

    # Shuffle image
    random.shuffle(real_train_data)
    random.shuffle(fake_train_data)
    # Number of images need to be moved
    num_real_images = 35000
    num_fake_images = 105000
    print("Move real images...")
    for i in tqdm(range(num_real_images)):
        real_image = real_train_data.pop()
        shutil.move(real_image, real_folder)

    print("Move fake images...")
    for i in tqdm(range(num_fake_images)):
        fake_image = fake_train_data.pop()
        shutil.move(fake_image, fake_folder)
    
    # Check images:
    a = len(os.listdir(osp.join(train_folder, "0_real")))
    b = len(os.listdir(osp.join(train_folder, "1_df")))
    c = len(os.listdir(real_folder))
    d = len(os.listdir(fake_folder))
    print("Done move files!")
    print("Number of real train: ", a)
    print("Number of fake train: ", b)
    print("Number of real val: ", c)
    print("Number of fake val: ", d)

def inverse():
    # Folder
    train_folder = "../../../../2_Deep_Learning/Dataset/facial_forgery/dfdc/images/train"
    val_folder = "../../../../2_Deep_Learning/Dataset/facial_forgery/dfdc/images/val"
    train_real_folder = osp.join(train_folder, "0_real")
    train_fake_folder = osp.join(train_folder, "1_df")
    val_real_folder = osp.join(val_folder, "0_real")
    val_fake_folder = osp.join(val_folder, "1_df")

    val_real_files = os.listdir(val_real_folder)
    for file in val_real_files:
        fpath = osp.join(val_real_folder, file)
        shutil.move(fpath, train_real_folder)

    val_fake_files = os.listdir(val_fake_folder)
    for file in val_fake_files:
        fpath = osp.join(val_fake_folder, file)
        shutil.move(fpath, train_fake_folder)

    # Check number of images:
    a = len(os.listdir(train_real_folder))
    b = len(os.listdir(train_fake_folder))
    print(a, b, a+b)



######################################################
############ df_in_the_wild dataset
######################################################
def get_images_from_file(file: str):
    fnames = []
    with open(osp.join(root_dir, file), "r") as f:
        lines = f.readlines()
        for line in lines:
            fname = line.strip()
            fnames.append(fname)
    return fnames

def check_correct_images_in_file(train_file: str, val_file: str, test_file: str,\
                                 train_dir="../../../../2_Deep_Learning/Dataset/facial_forgery/df_in_the_wild/image_jpg/train",\
                                 test_dir="../../../../2_Deep_Learning/Dataset/facial_forgery/df_in_the_wild/image_jpg/test"):
    # Get basename of images
    train_fnames = get_images_from_file(train_file)
    val_fnames = get_images_from_file(val_file)
    test_fnames = get_images_from_file(test_file)

    # Get absolute path of images:
    train_fpath = [osp.join(train_dir, fname) for fname in train_fnames]
    val_fpath = [osp.join(train_dir, fname) for fname in val_fnames]
    train_fpath.extend(val_fnames)
    test_fpath = [osp.join(test_dir, fname) for fname in test_fnames]

    # Get path of images in my device :
    train_images = glob(osp.join(train_dir, "*/*"))
    test_images = glob(osp.join(test_dir, "*/*"))

    # Check length:
    assert len(train_fpath) == len(train_images), "Train set incorrect!"
    assert len(test_fpath) == len(test_images), "Test set incorrect!"
    print("Train set: ", len(train_fpath))
    print("Test set: ", len(test_fpath))

    # Fix delimiter:
    train_fpath = [f.replace("\\", "/") for f in train_fpath]
    test_fpath = [f.replace("\\", "/") for f in test_fpath]
    train_images = [f.replace("\\", "/") for f in train_images]
    test_images = [f.replace("\\", "/") for f in test_images]

    # Check content:
    corrected = True
    for img_path in tqdm(train_fpath):
        if img_path not in train_images:
            print("Unexpected file <{}>!".format(img_path))
            corrected = False

    for img_path in tqdm(test_fpath):
        if img_path not in test_images:
            print("Unexpected file <{}>!".format(img_path))
            corrected = False
    return corrected

def move_image_from_txt_file(val_file: str, train_dir="../../../../2_Deep_Learning/Dataset/facial_forgery/df_in_the_wild/image_jpg/train",\
                                            val_dir="../../../../2_Deep_Learning/Dataset/facial_forgery/df_in_the_wild/image_jpg/val"):
    # Get path of moved images
    val_fnames = get_images_from_file(val_file)
    val_fpath = [osp.join(train_dir, fname) for fname in val_fnames]
    
    for img_path in tqdm(val_fpath):
        img_type = "0_real" if "0_real" in img_path else "1_df"
        shutil.move(img_path, osp.join(val_dir, img_type))


corrected = True
# corrected = check_correct_images_in_file("auxiliary/train_img.txt", "./auxiliary/val_img.txt", "./auxiliary/test_img.txt")
# if corrected:
#     move_image_from_txt_file("auxiliary/val_img.txt")
# inverse()
move_image_in_dfdc_dataset()
