import numpy as np
import os
import math
import sys
import argparse
from tqdm import tqdm
from PIL import Image, ImageDraw
from delaunay2D import Delaunay2D
from matplotlib import pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--width', default=600, type=int)
parser.add_argument('--height', default=600, type=int)
parser.add_argument('--path', default='./presidents', type=str)

def read_data(path):
    imgs = [s for s in os.listdir(path) if s.endswith('jpg') or
            s.endswith('jpeg')]
    r_imgs = []
    r_cfgs = []
    for img in imgs:
        #r_imgs.append(np.array(Image.open(os.path.join(path, img))) / 255.0)
        r_imgs.append(Image.open(os.path.join(path, img)))
        txt = img + '.txt'
        f = open(os.path.join(path, txt))
        poi = []
        for l in f:
            x, y = l.split()
            poi.append([int(x), int(y)])
        r_cfgs.append(np.array(poi).astype(float))
    return r_imgs, r_cfgs

def warp_points(pois, xform):
    mat, sft = xform
    ret = [mat.dot(p.T).T + sft for p in pois]
    return ret

def get_seg_xform(src, dst):
    def dot(a, b):
        return a[0] * b[0] + a[1] * b[1]
    def cross(a, b):
        return a[0] * b[1] - a[1] * b[0]
    def vec_len(vec):
        return np.sqrt(dot(vec, vec))

    v_src = src[1] - src[0]
    v_dst = dst[1] - dst[0]
    l_src = vec_len(v_src)
    l_dst = vec_len(v_dst)
    s = cross(v_dst, v_src) / l_dst / l_src
    c = dot(v_dst, v_src) / l_dst / l_src
    assert(abs(s * s + c * c - 1) < 0.0001)
    rot_mat = np.array([[c, s], [-s, c]])

    r = l_dst / l_src
    scale_mat = np.array([r, 0, 0, r]).reshape(2, 2)

    mat = scale_mat.dot(rot_mat)
    src = warp_points(src, (mat, np.zeros((2, ))))
    sft = dst[0] - src[0]
    return (mat, sft)

def get_AFFINE_data(mat, sft):
    return (mat[0][0], mat[0][1], sft[0], mat[1][0], mat[1][1], sft[1])

def get_tri_xform(src, dst):
    src.astype(np.float)
    dst.astype(np.float)
    src = np.linalg.inv(np.concatenate((src.T, np.ones((1, 3)))))
    dst = np.concatenate((dst.T, np.ones((1, 3)))).dot(src)
    dst = dst.reshape((-1))
    return tuple(dst[:6])

def draw_points(pois, col):
    x = [p[0] for p in pois]
    y = [p[1] for p in pois]
    plt.scatter(x, y, c=col)

def draw_tri(pois, tri):
    canvas = Image.new('RGB', (600, 600))
    pois = pois.astype(int)
    draw = ImageDraw.Draw(canvas)
    for t in tri:
        draw.polygon([(pois[i][0], pois[i][1]) for i in t])
    canvas.show()

def add_tri(base, img, tri):
    mask = Image.new('RGBA', img.size, (255, 255, 255, 255))
    draw = ImageDraw.Draw(mask)
    tri = tri.astype(int)
    draw.polygon(list(map(tuple, list(tri))), fill=(255, 255, 255, 0), outline=(255, 255, 255, 0))
    return Image.composite(base, img, mask)

if __name__ == '__main__':
    args = parser.parse_args()
    imgs, cfgs = read_data(args.path)
    w, h = args.width, args.height
    eye_dst = np.array([ [np.int(0.3 * w ), np.int(h / 3)], [np.int(0.7 * w ), np.int(h / 3)] ])
    bound = np.array([(0,0), (w/2,0), (w-1,0), (w-1,h/2), ( w-1, h-1 ), ( w/2, h-1 ), (0, h-1), (0,h/2) ]).astype(np.float)
    
    tot = len(imgs)
    p_num = len(cfgs[0])
    avg_base = np.array([(0.0, 0.0)] * p_num)
    norm_imgs = []
    for i in range(tot):
        cfg = cfgs[i]
        eye_src = [cfg[36], cfg[45]]
        xform = get_seg_xform(eye_src, eye_dst)
        img = imgs[i].transform(imgs[i].size, Image.AFFINE, get_AFFINE_data(*get_seg_xform(eye_dst, eye_src)), Image.BICUBIC).crop((0, 0, w, h))
        norm_imgs.append(img)
        pois = warp_points(cfg, xform)
        # draw_points(pois, col[i])
        avg_base += pois
        cfgs[i] = np.concatenate((pois, bound))

    avg_base /= tot
    avg_base = np.concatenate((avg_base, bound))

    # draw_tri(avg_base, tris)
    

    # draw_points(avg_base, 'r')
    # plt.show()
    
    dt = Delaunay2D()
    for p in avg_base:
        dt.addPoint(p)
    tris = dt.exportTriangles()
    bases = []
    for i in tqdm(range(tot)):
        cfg = cfgs[i]
        norm_img = norm_imgs[i]
        base = Image.new('RGBA', norm_img.size)
        for tri in tris:
            src_tri = np.array([cfg[t] for t in tri])
            dst_tri = np.array([avg_base[t] for t in tri])
            xform = get_tri_xform(dst_tri, src_tri)
            img = norm_img.transform(norm_img.size, Image.AFFINE, xform, Image.BICUBIC)
            base = add_tri(base, img, dst_tri)
        base.show()
        base.save(str(i) + '.jpg')
        bases.append(base)

    for i in range(1, len(bases)):
        bases[i] = Image.blend(bases[i - 1], bases[i], 1.0 / (i + 1))
    bases[-1].show()
    bases[-1].save('result.jpg')
    

