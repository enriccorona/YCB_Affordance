import numpy as np

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection

# LOADING FUNCTIONS:

def fast_load_obj(file_obj, **kwargs):
    """
    Code slightly adapted from trimesh (https://github.com/mikedh/trimesh)
    and taken from ObMan dataset (https://github.com/hassony2/obman) 
    Thanks to Michael Dawson-Haggerty for this great library !
    loads an ascii wavefront obj file_obj into kwargs
    for the trimesh constructor.

    vertices with the same position but different normals or uvs
    are split into multiple vertices.

    colors are discarded.

    parameters
    ----------
    file_obj : file object
                   containing a wavefront file

    returns
    ----------
    loaded : dict
                kwargs for trimesh constructor
    """

    # make sure text is utf-8 with only \n newlines
    text = file_obj.read()
    if hasattr(text, 'decode'):
        text = text.decode('utf-8')
    text = text.replace('\r\n', '\n').replace('\r', '\n') + ' \n'

    meshes = []
    def append_mesh():
        # append kwargs for a trimesh constructor
        # to our list of meshes
        if len(current['f']) > 0:
            # get vertices as clean numpy array
            vertices = np.array(
                current['v'], dtype=np.float64).reshape((-1, 3))
            # do the same for faces
            faces = np.array(current['f'], dtype=np.int64).reshape((-1, 3))

            # get keys and values of remap as numpy arrays
            # we are going to try to preserve the order as
            # much as possible by sorting by remap key
            keys, values = (np.array(list(remap.keys())),
                            np.array(list(remap.values())))
            # new order of vertices
            vert_order = values[keys.argsort()]
            # we need to mask to preserve index relationship
            # between faces and vertices
            face_order = np.zeros(len(vertices), dtype=np.int64)
            face_order[vert_order] = np.arange(len(vertices), dtype=np.int64)

            # apply the ordering and put into kwarg dict
            loaded = {
                'vertices': vertices[vert_order],
                'faces': face_order[faces],
                'metadata': {}
            }

            # build face groups information
            # faces didn't move around so we don't have to reindex
            if len(current['g']) > 0:
                face_groups = np.zeros(len(current['f']) // 3, dtype=np.int64)
                for idx, start_f in current['g']:
                    face_groups[start_f:] = idx
                loaded['metadata']['face_groups'] = face_groups

            # we're done, append the loaded mesh kwarg dict
            meshes.append(loaded)

    attribs = {k: [] for k in ['v']}
    current = {k: [] for k in ['v', 'f', 'g']}
    # remap vertex indexes {str key: int index}
    remap = {}
    next_idx = 0
    group_idx = 0

    for line in text.split("\n"):
        line_split = line.strip().split()
        if len(line_split) < 2:
            continue
        if line_split[0] in attribs:
            # v, vt, or vn
            # vertex, vertex texture, or vertex normal
            # only parse 3 values, ignore colors
            attribs[line_split[0]].append([float(x) for x in line_split[1:4]])
        elif line_split[0] == 'f':
            # a face
            ft = line_split[1:]
            if len(ft) == 4:
                # hasty triangulation of quad
                ft = [ft[0], ft[1], ft[2], ft[2], ft[3], ft[0]]
            for f in ft:
                # loop through each vertex reference of a face
                # we are reshaping later into (n,3)
                if f not in remap:
                    remap[f] = next_idx
                    next_idx += 1
                    # faces are "vertex index"/"vertex texture"/"vertex normal"
                    # you are allowed to leave a value blank, which .split
                    # will handle by nicely maintaining the index
                    f_split = f.split('/')
                    current['v'].append(attribs['v'][int(f_split[0]) - 1])
                current['f'].append(remap[f])
        elif line_split[0] == 'o':
            # defining a new object
            append_mesh()
            # reset current to empty lists
            current = {k: [] for k in current.keys()}
            remap = {}
            next_idx = 0
            group_idx = 0

        elif line_split[0] == 'g':
            # defining a new group
            group_idx += 1
            current['g'].append((group_idx, len(current['f']) // 3))

    if next_idx > 0:
        append_mesh()

    return meshes



# PLOT FUNCTIONS:

def plot_hand_w_object(obj_verts, obj_faces, hand_verts, hand_faces, flip=True):
    """
    Functions taken from the ObMan dataset repo (https://github.com/hassony2/obman)
    """
    colors = ['r']*len(hand_faces) + ['b']*len(obj_faces)

    frames = []
    fig = plt.figure()
    fig.subplots_adjust(0.05, 0.05, 0.95, 0.95)  # get rid of margins
    ax = fig.add_subplot(111, projection='3d')
    ax.axis('off')

    verts = hand_verts
    add_group_meshs(ax, np.concatenate((verts, obj_verts)), np.concatenate((hand_faces, obj_faces + 778)), alpha=1, c=colors)
    cam_equal_aspect_3d(ax, obj_verts, flip_x=flip, flip_y=flip)
    ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('z')

    pressed_keyboard = False
    while not pressed_keyboard:
        pressed_keyboard = plt.waitforbuttonpress()
    
    plt.close(fig)

    return

def plot_scene_w_grasps(list_obj_verts, list_obj_faces, list_obj_handverts, list_obj_handfaces, plane_parameters):
    fig = plt.figure()
    fig.subplots_adjust(0.05, 0.05, 0.95, 0.95)  # get rid of margins

    ax = fig.add_subplot(111, projection='3d')
    ax.axis('off')

    # We will convert this into a single mesh, and then use add_group_meshs to plot it in 3D
    allverts = np.zeros((0,3))
    allfaces = np.zeros((0,3))
    colors = []
    for i in range(len(list_obj_verts)):
        allfaces = np.concatenate((allfaces, list_obj_faces[i]+len(allverts)))
        allverts = np.concatenate((allverts, list_obj_verts[i]))
        colors = np.concatenate((colors, ['r']*len(list_obj_faces[i])))

    for i in range(len(list_obj_handverts)):
        allfaces = np.concatenate((allfaces, list_obj_handfaces[i]+len(allverts)))
        allverts = np.concatenate((allverts, list_obj_handverts[i]))
        colors = np.concatenate((colors, ['b']*len(list_obj_handfaces[i])))

    allfaces = np.int32(allfaces)
    print(np.max(allfaces))
    print(np.shape(allverts))
    add_group_meshs(ax, allverts, allfaces, alpha=1, c=colors)

    cam_equal_aspect_3d(ax, np.concatenate(list_obj_verts, 0), flip_z=True)
    ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('z')

    # Show plane too:
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    step = 0.05
    border = 0.0 #step
    X, Y = np.meshgrid(np.arange(xlim[0]-border, xlim[1]+border, step),
               np.arange(ylim[0]-border, ylim[1]+border, step))
    Z = np.zeros(X.shape)
    for r in range(X.shape[0]):
      for c in range(X.shape[1]):
        Z[r, c] = (-plane_parameters[0] * X[r, c] - plane_parameters[1] * Y[r, c] + plane_parameters[3])/plane_parameters[2]
    ax.plot_wireframe(X, Y, Z, color='r')

    pressed_keyboard = False
    while not pressed_keyboard:
        pressed_keyboard = plt.waitforbuttonpress()

    plt.close(fig)


def add_mesh(ax, verts, faces, alpha=0.1, c='b'):
    mesh = Poly3DCollection(verts[faces], alpha=alpha)
    if c == 'b':
        face_color = (141 / 255, 184 / 255, 226 / 255)
    elif c == 'r':
        face_color = (226 / 255, 184 / 255, 141 / 255)
    edge_color = (50 / 255, 50 / 255, 50 / 255)
    mesh.set_edgecolor(edge_color)
    mesh.set_facecolor(face_color)
    ax.add_collection3d(mesh)

def add_group_meshs(ax, verts, faces, alpha=0.1, c='b'):
    mesh = Poly3DCollection(verts[faces], alpha=alpha)
    face_color = []
    for i in range(len(c)):
        if c[i] == 'b':
            face_color.append((141 / 255, 184 / 255, 226 / 255))
        elif c[i] == 'r':
            face_color.append((226 / 255, 184 / 255, 141 / 255))
    edge_color = (50 / 255, 50 / 255, 50 / 255)
    mesh.set_edgecolor(edge_color)
    mesh.set_facecolor(face_color)
    ax.add_collection3d(mesh)


def cam_equal_aspect_3d(ax, verts, flip_x=False, flip_y=False, flip_z=False):
    """
    Centers view on cuboid containing hand and flips y and z axis
    and fixes azimuth
    """
    extents = np.stack([verts.min(0), verts.max(0)], axis=1)
    sz = extents[:, 1] - extents[:, 0]
    centers = np.mean(extents, axis=1)
    maxsize = max(abs(sz))
    r = maxsize / 2
    if flip_x:
        ax.set_xlim(centers[0] + r, centers[0] - r)
    else:
        ax.set_xlim(centers[0] - r, centers[0] + r)
    # Invert y and z axis
    if flip_y:
        ax.set_ylim(centers[1] + r, centers[1] - r)
    else:
        ax.set_ylim(centers[1] - r, centers[1] + r)

    if flip_z:
        ax.set_zlim(centers[2] + r, centers[2] - r)
    else:
        ax.set_zlim(centers[2] - r, centers[2] + r)


