import os
import zipfile
 
local_zip = "D:\PRATIK\MACHINE LEARNING\Tensorflow\Google AIML Course\horse-or-human.zip"
zip_ref = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall('horse-or-human')
zip_ref.close()

# Directory with our training horse pictures
train_horse_dir = os.path.join('horse-or-human/horses')
 
# Directory with our training human pictures
train_human_dir = os.path.join('horse-or-human/humans')

train_horse_names = os.listdir(train_horse_dir)
print(train_horse_names[:10])
train_human_names = os.listdir(train_human_dir)
print(train_human_names[:10])

print('total training horse images:', len(os.listdir(train_horse_dir)))
print('total training human images:', len(os.listdir(train_human_dir)))