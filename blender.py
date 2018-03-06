from PIL import Image
import os

def read_data(path):
    imgs = [s for s in os.listdir(path) if s.endswith('jpg') or
            s.endswith('jpeg')]
    r_imgs = []
    for img in imgs:
        r_imgs.append(Image.open(os.path.join(path, img)))
    return r_imgs

bases = read_data('.')
for i in range(1, len(bases)):
    bases[i] = Image.blend(bases[i - 1], bases[i], 1.0 / (i + 1))
bases[-1].show()