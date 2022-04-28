from ast import Return
import os, sys
import os.path as osp
from os.path import join

from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from tqdm import tqdm

import json

from typing import Dict, List
import shutil

class GoogleDriveAPI(object):
    """
        Class for using some API to manipulate with google drive, like: 
            * upload file/folder to gdrive
            * download file/folder to device
            * delete a file/folder in gdrive
            * find a file/folder in gdrive
            ...
    """
    def __init__(self, root_id: str, method: str, display_tree=False):
        """
        Args:
            root_id (str): id of root directory 
            method (str): the approach method to connect google drive, in list: ['pydrive']. If method is pydrive, you must use API Console to create a project on 
                        Google Cloud Platform, create credentials with Desktop Application in Application Type and authenticate to your google drive. 
                        The tutorials come from: https://codingshiksha.com/tutorials/python-upload-files-to-google-drive-using-pydrive-library-full-tutorial/
            display_tree (bool): display hierachy of root directory if set to True.
        """
        self.method = method
        self.drive = None

        # Define drive instance
        if method == 'pydrive':
            gauth = GoogleAuth()
            gauth.CommandLineAuth()
            self.drive = GoogleDrive(gauth)
            
        # Define root folder
        self.root_id = root_id
        try:
            self.root = self.get_file(root_id)
        except:
            self.root = None
        # Traverse all files in root:
        if display_tree:
            self.display_hierachical_tree_structure(self.root_id)
        self.paths = self.traverse_folder(self.root_id)
    
    def get_file(self, id: str):
        """Return GoogleDriveFile from an id.
        """
        return self.drive.CreateFile({'id': id})
        
        
    def is_directory(self, id: str):
        """_summary_ Check a file is a directory or not. Returns: Boolean
        """
        file = self.get_file(id)
        return file['mimeType'] == 'application/vnd.google-apps.folder'
    
    def is_file(self, id: str):
        """_summary_ Check a file is a file or not. Returns: Boolean
        """
        file = self.get_file(id)
        return file['mimeType'] != 'application/vnd.google-apps.folder'
    
    def list_files(self, folder_id: str):
        """_summary_: return all files in a folder having id <folder_id>
        Returns:
            file_list(
                List[
                    GoogleDriveFile({
                        'id': '<file_id>',
                        'alternateLink': <full link of this file/folder>,
                        'title': <name of file/folder>,
                        'parents': <parent folder of this file/folder> [
                            'id': '<parent_id>',
                            ...
                        ],
                        'ownerNames': List[str], like: ['Phúc Nguyễn Phi', ...]
                    })
                ]
            )
        """
        file_list = None
        if self.method == 'pydrive':
            file_list = self.drive.ListFile({'q': "'{}' in parents and trashed=false".format(folder_id)}).GetList()
        return file_list

    def create_file(self, file_name: str, content: str, dst_id: str, overwrite=True):
        """_summary_ Create a file with content in destination directory
        """
        assert self.is_directory(dst_id), "Error! Destination should be a folder"
        dst_files = self.find_file_in_current_folder(file_name, dst_id, verbose=True)
        while len(dst_files):
            name, ext = osp.splitext(osp.basename(file_name))
            file_name = name + ' - copy' + ext
            dst_files = self.find_file_in_current_folder(file_name, dst_id, verbose=True)
            
        file = self.drive.CreateFile(metatdata={
            'parents': [{
                'id': dst_id
            }],
            'title': file_name
        })
        file.SetContentString(content)
        file.Upload()
        
    def create_folder(self, folder_name: str, dst_id: str):
        """_summary_ Create a folder in destination directory
        """
        assert self.is_directory(dst_id), "Error! Destination should be a folder"
        folder = self.drive.CreateFile(metadata={
            'parents': [{
                'id': dst_id
            }],
            'title': folder_name,
            'mimeType': 'application/vnd.google-apps.folder'
        })
        folder.Upload()
    
    def delete_file(self, id: str, permanent=True):
        """_summary_ Delete a file/folder
        """
        file = self.get_file(id=id)
        if not permanent:
            file.Trash()
        else:
            file.Delete()
            
    def upload_file_to_drive(self, file_path: str, dst_id="", overwrite=True):
        """Upload a file from this device to drive. If not have destination folder, default sets root directory.
        """
        # Delete existing file before overwrite:
        dst_id = self.root_id if dst_id == "" else dst_id
        assert self.is_directory(dst_id), "Destination should be a folder."
        assert osp.exists(file_path), "Source not exist."
        
        if overwrite:
            file_list = self.list_files(folder_id=dst_id)
            try:
                for file in tqdm(file_list):
                    if file['title'] == osp.basename(file_path):
                        print("Found existing files in gdrive.")
                        file.Trash()
            except:
                pass
        # Upload file to drive:
        file_name = osp.basename(file_path)
        exist = True if len(self.find_file_in_current_folder(file_name, dst_id)) else False
        if not overwrite and exist:
            name, ext = osp.splitext(file_name)
            file_name = name + ' - copy' + ext
            files = self.find_file_in_current_folder(file_name, dst_id, verbose=False)
            while len(files):
                name, ext = osp.splitext(file_name)
                file_name = name + ' - copy' + ext
                files = self.find_file_in_current_folder(file_name, dst_id, verbose=False)
        
        f = self.drive.CreateFile({
            'title': file_name,
            'parents': [{
                'id': dst_id
            }]
        })
        f.SetContentFile(filename=file_path)
        f.Upload()
    
    def upload_folder_to_drive(self, folder_path: str, dest_id: str, overwrite=True, merge=False):
        """_summary_ upload a folder to gdrive
        Args:
            folder_path (str): path to folder
            dst_id (str): id of folder in gdrive
            overwrite (bool, optional): If overwrite, folder in drive will be replace base on "merge" parameter:
                If merge=True, 2 folder will be merge, 2 files have the same name will be added 'copy' in one of their name. 
                If merge=False, uploaded folder will completely replace current folder in gdrive. 
                If not overwrite, another with the name is concatenated with 'copy' will be created.
                Defaults to False.
        """
        assert self.is_directory(id=dest_id), "Error! Destination should be a folder."
        assert osp.isdir(folder_path), "Error! Source should be a folder"
        
        # Create a folder in destination dir:
        dst_subfolder = [d['title'] for d in self.list_files(dest_id) if self.is_directory(d['id'])]
        exist = True if osp.basename(folder_path) in dst_subfolder else False
        new_name = osp.basename(folder_path)
        if exist and not overwrite:
            new_name = osp.basename(folder_path) + ' - copy'
            
        def upload(src_folder: str, dst_id: str):
            """_summary_ Upload all files in a src folder to destionation dir.
            """
            print("overwrite", overwrite)
            print("merge", merge)
            src_files = os.listdir(src_folder)
            dst_files = self.list_files(dst_id)
            dst_file_name = [f['title'] for f in dst_files]
            
            for src_file in src_files:
                src_path = join(src_folder, src_file)
                if osp.isfile(src_path):
                    overwrite_ = True if 
                    self.upload_file_to_drive(file_path=src_path, folder_id=dst_id, overwrite=overwrite_)
        
        pass

    def download_file_to_device(self, file_path: str, dest_dir: str, overwrite=True):
        """_summary_ download a file with name <file_name> to destination directory.
        """
        assert osp.isdir(dest_dir), "Destination should be a folder."
        files = self.find_file(file_path, verbose=True)
        assert len(files) == 1, "{} files found!".format("No" if len(files) == 0 else len(files))
        assert self.is_file(files[0][0]), "Source should be a file."
        
        # Check destination contains file has the same name or not.
        file_name = osp.basename(file_path)
        dest_path = join(dest_dir, file_name)
        exist = osp.exists(dest_path)
        if overwrite and exist:
            os.remove(dest_path)
            
        # Redefine the name if file existed and overwrite is set to False.
        if not overwrite and exist:
            name, ext = osp.splitext(file_name)
            file_name = name + ' - copy' + ext
            while osp.exists(join(dest_dir, file_name)):
                name, ext = osp.splitext(osp.basename(file_path))
                file_name = name + ' - copy' + ext  

        dest_path = join(dest_dir, dest_name)
        file = self.get_file(id=files[0][0])
        file.GetContentFile(filename=dest_name)
        shutil.move(src=dest_name, dst=dest_path)
    
    def download_folder_to_device(self, folder_name: str, dest_dir: str, overwrite=True):
        pass

    def traverse_folder(self, folder_id: str):
        """_summary_ Traverse a file/folder and returns a dictionary with KEY is file path and VALUE is a List contains those files'id.
        """
    
        def traverse(cur_id: str, cur_folder_name: str, traverse_dict: Dict[str, List[str]]):
            traverse_dict[cur_id] = cur_folder_name    
            if self.is_directory(cur_id):
                list_file = self.list_files(folder_id=cur_id)
                for file in list_file:
                    file_name = cur_folder_name + '/' + file['title']
                    traverse(file['id'], file_name, traverse_dict)
                    
        result = {}
        folder = self.get_file(folder_id)
        traverse(folder['id'], folder['title'], result)
        return result
           
    def find_file(self, file_path: str, verbose=False):
        """_summary_ Find a file with specific path on hierachical structure with root has id <folder_id>
            Returns: List[str]: list of file_ids
        """
        result = []
        for id, path in self.paths.items():
            if (file_path in path) and (osp.basename(file_path) == osp.basename(path)) and all(sub in path.split('/') for sub in file_path.split('/')):
                result.append((id, path))
                
        if verbose:
            if len(result) == 0:
                print("File not found!")
            else:
                print("{} {} found: ".format(len(result), "file" if len(result) == 1 else "files"))
                for r in result:
                    print("   {} with id={}".format(r[1], r[0]))
        print(result)
        
    def find_file_in_current_folder(self, file_name: str, folder_id: str, verbose=False):
        """_summary_ Find all files with name <file_name> in a folder
        """
        files = self.drive.ListFile({'q': "'{}' in parents and title='{}' and trashed=false".format(folder_id, file_name)}).GetList()
        if verbose:
            if len(files) == 0:
                print("File not found!")
            else:
                print("{} {} found. ".format(len(files), "file" if len(files) == 1 else "files"))  
        
    def display_hierachical_tree_structure(self, folder_id: str, indent=4):
        """_summary_ Display structure of a folder. 
        """
        def display(id: str, n_tab: int):
            tabs = "".join([" "] * indent * n_tab)
            file = self.get_file(id)
            print(tabs + file['title'])
            if self.is_directory(id):
                list_file = self.list_files(folder_id=id)
                for file in list_file:
                    display(file['id'], n_tab + 1)  
        display(id=folder_id, n_tab=0)
        
    def name_copy_file(self, name: str, ext: str):
        return name + ' - copy' + ext
            
if __name__ == '__main__':
    # gdrive = GoogleDriveAPI(root_id="1P2Tm9ZQqR5CKTYXVPxcHHEC0VyVrJgFK", method='pydrive', display_tree=False)
    # gdrive.upload_file_to_drive(folder_id="1P2Tm9ZQqR5CKTYXVPxcHHEC0VyVrJgFK", file_path="test/template.pptx", overwrite=True)
    # gdrive.find_file