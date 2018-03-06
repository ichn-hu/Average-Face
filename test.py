# from af import *
# from PIL import Image
# def dot(a, b):
#     return a[0] * b[0] + a[1] * b[1]
# def cross(a, b):
#     return a[0] * b[1] - a[1] * b[0]
# def vec_len(vec):
#     return np.sqrt(dot(vec, vec))

# def same_direction(a, b):
#     return np.abs(cross(a[1] - a[0], b[1] - b[0])) < 0.00001

# def same(a, b):
#     a = a[1] - a[0]
#     b = b[1] - b[0]
#     return np.abs(vec_len(b - a)) < 0.00001


# def test_get_similarity_xform():
#     np.random.seed(1926)
#     for i in range(10):
#         a = np.array([[0, 0], [1, 1]])
#         b = np.array([[0, 0], [0, 1]])
#         a = np.random.randn(2, 2)
#         b = np.random.randn(2, 2)
#         mat, sft = get_similarity_xform(a, b)
#         #print(mat)
#         a = mat.dot(a.T).T
#         assert(same_direction(a, b))
#         assert(same(a, b))
#         print(a[1] - a[0], b[1] - b[0])

# data = get_AFFINE_data(np.array([[0, 0], [1, 1]]), np.array([[1000, -1000], [1000 + 0.2, -1000]]))
# img = Image.open("presidents/barak-obama.jpg")
# img.transform((2000, 2000), Image.AFFINE, data, Image.BICUBIC).show()
from PIL import Image, ImageDraw

im = Image.new('RGB', (100, 100))

draw = ImageDraw.Draw(im)
draw.line((0, 0) + im.size)
draw.line((0, im.size[1], im.size[0], 0))
im.show()