import matplotlib.pyplot as plt
import os
import PIL.Image as Image
import numpy as np

# def inverse_normalize(batch_image,batch_label = None,isShow = False,isSave = True, 
#                         saveFile="savefig.png",mean=(0.485, 0.456, 0.406),std=(0.229, 0.224, 0.225)):

#     fig = plt.figure(figsize=(batch_image.size(0) * 2.5, 5))
#     for i in range(batch_image.size(0)):
#         img = batch_image[i].numpy().transpose(1, 2, 0)
        
#         if (mean is not None) and (std is not None):
#             img = (img * std + mean) * 255
#         else:  #如果只是经过了ToTensor()
#             img = img * 255

#         if isShow or isSave:
#             ax = fig.add_subplot(1, batch_image.size(0), i+1, xticks=[], yticks=[])
            
            
if __name__ == "__main__":
    path = 'results/males_model/test_360/images/'
    imgs = os.listdir(path)
    batch = []
    batch_size = 20
    plt.figure(figsize=(batch_size * 2.5, 5))
    for i in range(batch_size):
        img = imgs[i]
        img = Image.open(path+img)
        img = np.array(img)
        # img = np.transpose(img,(1,2,0))
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = (img * std + mean) * 255
        batch.append(img)
        plt.subplot(1,batch_size,i+1)
        plt.imshow(img.astype(np.uint8))
    
        