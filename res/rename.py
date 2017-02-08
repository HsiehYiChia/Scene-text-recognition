import os, sys

def rename_folder_file(folder, new_folder):
    i = 1
    for file in os.listdir(folder):
        os.rename(folder+file, new_folder+"%d.jpg" % i)
        i+=1
        print (file)

rename_folder_file("pos3/", "tmp1/")