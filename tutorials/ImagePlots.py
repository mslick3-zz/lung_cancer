from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage import measure

def plot_histogram_scan(image, path=None):
    """
    Plot a histogram of the pixel values in a full 3d scan or 2d image slice
    :param image: a numpy array of the image to plot
    :param path: the path to save the image. if none, image will render in window
    :return: none
    """
    plt.hist(image.flatten(), bins=80, color='c')
    plt.xlabel("Hounsfield Units (HU)")
    plt.ylabel("Frequency")
    if path:
        plt.savefig(path)
        plt.clf()
    else:
        plt.show()

def plot_2d_slice(image, path=None):
    """
    Plots a 2d slice of the image
    :param image: a 2d numpy array of the image to plot
    :param path: the path to save the image. if none, image will render in window
    :return: none
    """
    plt.imshow(image, cmap=plt.cm.gray)
    if path:
        plt.savefig(path)
        plt.clf()
    else:
        plt.show()

def plot_grid_scan(scan, path=None):
    """
    Plot slices of scan in a 2d grid
    :source: https://www.kaggle.com/arnavkj95/data-science-bowl-2017/candidate-generation-and-luna16-preprocessing
    :param scan: list of 2d numpy arrays of pixels representing image slices
    :return: none
    """
    f, plots = plt.subplots(int(scan.shape[0] / 20) + 1, 4, figsize=(25, 25))
    for i in range(0, scan.shape[0], 5):
        plots[int(i / 20), int((i % 20) / 5)].axis('off')
        plots[int(i / 20), int((i % 20) / 5)].imshow(scan[i], cmap=plt.cm.bone)

    if path:
        plt.savefig(path)
        plt.clf()
    else:
        plt.show()

def plot_3d_scan(image, threshold=.2, path=None):
    """
    Plots a 3d view of a scan.  WARNING: This is pretty slow
    :param image: 3d numpy array of pixel values for the scan
    :return: none
    """
    # Position the scan upright,
    # so the head of the patient would be at the top facing the camera
    p = image.transpose(2, 1, 0)
    p = p[:, :, ::-1]

    verts, faces = measure.marching_cubes(p, threshold)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Fancy indexing: `verts[faces]` to generate a collection of triangles
    mesh = Poly3DCollection(verts[faces], alpha=0.1)
    face_color = [0.5, 0.5, 1]
    mesh.set_facecolor(face_color)
    ax.add_collection3d(mesh)

    ax.set_xlim(0, p.shape[0])
    ax.set_ylim(0, p.shape[1])
    ax.set_zlim(0, p.shape[2])

    if path:
        plt.savefig(path)
        plt.clf()
    else:
        plt.show()