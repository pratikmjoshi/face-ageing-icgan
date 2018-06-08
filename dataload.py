import os
import cv2
import numpy as np


images = []
ages = []
labels = []
buckets = [0,20,40,60,200]

PATH = './wiki_crop/'
for files in os.listdir(PATH):
    for image in os.listdir(os.path.join(PATH,files)):
        path = PATH + files + '/' + image
        img = cv2.imread(path,1)
        if((img.shape[0]>100) | (img.shape[1]>100)):
            #img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            img = img/255.
            img = cv2.resize(img,(64,64))
            images.append(img)
            _,dob,dtaken = image.split('_')
            dtaken = dtaken.split('.')[0]
            age = int(dtaken) - int(dob.split('-')[0])
            ages.append(age)
        if(len(images)>800):
            break
    if(len(images)>800):
        break

for age in ages:
    a=0
    for i in range(len(buckets)-1):
        if ((age>=buckets[i]) & (age<buckets[i+1])):
            a=i
    labels.append(a)

print("Done")

images = np.array(images)
ages = np.array(ages)
labels = np.array(labels)
print(images.shape,ages.shape,labels.shape)

np.save('images.npy',images)
np.save('ages.npy',ages)
np.save('labels.npy',labels)
