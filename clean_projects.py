import os
import shutil

def delete_folder_recursively(path):
    """"
    Delete folder if it was created before timestemp ts
    """
    if os.path.exists(path):
        shutil.rmtree(path)

def get_sub_directory_paths_of_folder(path):
    """
    Get sub directory paths of folder
    """
    return [os.path.join(path, name) for name in os.listdir(path) if os.path.isdir(os.path.join(path, name))]



project_folders = get_sub_directory_paths_of_folder("mlruns")
#print(project_folders)
for project_folder in project_folders:
    run_folders = get_sub_directory_paths_of_folder(project_folder)
    for run_folder in run_folders:
        task_file_path = os.path.join(run_folder, "tags", "task")
        #print(task_file_path)
        if not os.path.exists(task_file_path):
            delete_folder_recursively(run_folder)
            continue
        f = open(task_file_path, "r")
        task = f.read()
        f.close()
        if task == "rl_model_checking":
            #print("Hello")
            delete_folder_recursively(run_folder)
