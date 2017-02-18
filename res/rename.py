import os, sys
import random
import string

def rename_folder_file(folder):
    s = string.ascii_lowercase+string.digits
    random_string = ''.join(random.sample(s,10))
    
    i = 1
    for file in os.listdir(folder):
        os.rename(folder+file, folder+random_string+"_%d.jpg" % i)
        i+=1
    print ("move as temporary files")
    
    
    i = 1
    for file in os.listdir(folder):
        os.rename(folder+file, folder+"%d.jpg" % i)
        i+=1
    print ("move as refined files")

        
rename_folder_file("neg4/")