import os
import progressbar
import numpy as np
import pandas as pd
from keras.preprocessing import image
from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering

# easy switch between directories for HPC and local set
current_dir = '/WAVE/projects/COEN-281-Wi20/data/traffic/'
#current_dir = os.getcwd() + '/traffic-small/' 
X_train = os.listdir(current_dir)

feature_list = []

# init variables and progress bar
count = 0
widgets = [progressbar.Percentage(), progressbar.Bar()]
bar = progressbar.ProgressBar(widgets=widgets).start()


# init agglo, PCA, resNet50 instance
link = 'average'
aff = 'cosine'

agglo = AgglomerativeClustering(n_clusters=14,linkage=link ,affinity=aff)

transformer = PCA(n_components=100)
model_resNet50 = ResNet50(weights='imagenet', include_top=False)

print("feature extraction")

# cycle through all test files
for i in range(1, 100001):

    # open image and run filtering
    img_rgb = image.load_img(current_dir + str(i).zfill(6) + '.jpg', target_size=(224, 224), color_mode='rgb')

    img_data = image.img_to_array(img_rgb)
    img_data = np.expand_dims(img_data, axis=0)
    img_data = preprocess_input(img_data)
    resNet50_feature_np = model_resNet50.predict(img_data)
    resNet50_feature_np = resNet50_feature_np.flatten()
    feature_res = resNet50_feature_np.tolist()
    feature_list.append(feature_res)     

    # increment count for progress bar and update
    count = (i/len(X_train)*100)
    bar.update(int(count))

# finish progress bar
bar.finish()

print("fitting")

transformer.fit(feature_list)

print('transforming')

X_train_pca = transformer.transform(feature_list)
feature_list.clear()

print("Building the model.")

agglo.fit(X_train_pca)

print('gettings predictions')

labels = agglo.labels_

# init variables and progress bar

print("length of predictions=" + str(len(labels)))
print('writing to file. . .')

# write predictions to a file
f = open("pred_" + str(link) + "_" + str(aff) + ".txt", "w")

# init variables and progress bar
count = 0
bar = progressbar.ProgressBar(widgets=widgets).start()

# go through all predictions and write to a file
for i in range (0, len(labels)):
    ans = labels[i]
    if ans == 0:
    	ans = 14
    f.write(str(ans) + "\n")
    count = (i/len(labels)*100)
    bar.update(int(count))

# close file and finish progress bar
f.close()
bar.finish()

print("Writing predictions to file complete.")




