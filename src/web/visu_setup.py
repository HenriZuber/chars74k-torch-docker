
import sys, os
import pickle
from PIL import Image
import matplotlib
matplotlib.use('pdf')
import numpy as np
import seaborn as sns # pylint: disable=import-error
import torch  # pylint: disable=import-error

sys.path.append(os.path.abspath("/opt/code/src"))
import load_sets

USE_GPU = True
DTYPE = torch.float32
if USE_GPU and torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    # print("using cuda")
else:
    DEVICE = torch.device("cpu")
    print("using cpu")

PIXEL_SIZE = 64
CLASSES = [
    "0",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
    "A",
    "B",
    "C",
    "D",
    "E",
    "F",
    "G",
    "H",
    "I",
    "J",
    "K",
    "L",
    "M",
    "N",
    "O",
    "P",
    "Q",
    "R",
    "S",
    "T",
    "U",
    "V",
    "W",
    "X",
    "Y",
    "Z",
    "a",
    "b",
    "c",
    "d",
    "e",
    "f",
    "g",
    "h",
    "i",
    "j",
    "k",
    "l",
    "m",
    "n",
    "o",
    "p",
    "q",
    "r",
    "s",
    "t",
    "u",
    "v",
    "w",
    "x",
    "y",
    "z",
]


model = torch.load("/opt/code/src/model_de_test_nouvelle_archi.pt")
model.eval()
model.to(DEVICE)
model.cuda()
 
SETS = load_sets.get_datasets()

image_dataset = SETS['fat_dataset']
confusion_font_dataset = SETS['font_dataset']
confusion_image_dataset = SETS['image_dataset']
confusion_small_font_dataset = SETS['small_fontset']
confusion_small_test_dataset = SETS['small_testset']


def new_images(size=10):
    image_numbers = np.random.randint(0, len(image_dataset), size=size)
    images = []
    results = []
    for i in image_numbers:
        image, _ = image_dataset[i]
        images.append(image)

    for j, img in enumerate(images):
        ime = np.reshape(img, (64, 64))
        ime = ime.T
        ime = Image.fromarray(ime, mode="L")
        ime.save("/opt/code/src/web/static/image{}.png".format(j))
        img = torch.tensor(img, dtype=DTYPE)
        img = img.to(DEVICE)
        img.unsqueeze_(0)
        results.append(model(img))

    idx = []
    for h, res in enumerate(results):
        res = res.cpu()
        res = res.detach().numpy()
        list_res = np.argsort(res)
        list_res = list_res[0]
        classes_and_scores = []
        for g in range(len(list_res)):
            classes_and_scores.append(
                str(CLASSES[list_res[g]]) + ":" + str(res[0][list_res[g]])
            )
        classes_and_scores = classes_and_scores[::-1]
        classes_and_scores = classes_and_scores[:5]
        idx.append(["/static/image{}.png".format(h), classes_and_scores])
        # idx.append([os.path.join(app.config['UPLOAD_FOLDER'], 'image{}.png'.format(h)), np.argsort(res)[-5:]])
    
    
    return idx


def matrice_confusion(dataset):
    mat_conf = np.zeros((len(CLASSES), len(CLASSES)))
    for h in range(len(dataset)):
        image, classe = dataset[h]
        image = torch.tensor(image, dtype=DTYPE)
        image = image.to(DEVICE)
        image.unsqueeze_(0)
        result = model(image)
        result = result.cpu()
        result = result.detach().numpy()
        result = np.argsort(result)
        result = result[0]
        id_max = result[-1]
        mat_conf[classe][id_max]+=1
        

    for i in range(len(CLASSES)):
        somme_ligne = np.sum(mat_conf[i])
        for j in range(len(CLASSES)):
            if somme_ligne != 0:
                mat_conf[i][j]=mat_conf[i][j]/somme_ligne
    
    sns.set()
    os.remove("/opt/code/src/web/static/mat_conf.png")
    ax = sns.heatmap(mat_conf,square=True,xticklabels=CLASSES,yticklabels=CLASSES)
    axe = ax.get_figure()
    axe.savefig("/opt/code/src/web/static/mat_conf.png")
    return  

def new_full_data(dataset):
    with open('/opt/code/src/web/idx.txt',"wb") as file_to_use:
        matrice_confusion(dataset)
        pickle.dump(new_images(),file_to_use) 

if __name__ == '__main__':
    new_full_data(confusion_font_dataset)
