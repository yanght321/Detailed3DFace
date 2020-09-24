import pygame
from OpenGL.GL import *
from OpenGL.GLU import *
import pickle
import numpy as np


class MeshRenderer():
    def __init__(self):
        self.tset = Textureset()

    def render(self, verts_raw, tris, viewport, img_path, out_path):
        verts = verts_raw.copy()
        pygame.init()
        srf = pygame.display.set_mode(viewport, pygame.OPENGL | pygame.DOUBLEBUF)
        draw_2side = False

        mat_shininess = [64]
        global_ambient = [0.3, 0.3, 0.3, 0.05]
        light0_ambient = [0, 0, 0, 0]
        light0_diffuse = [0.55, 0.55, 0.55, 0.55]

        light1_diffuse = [-0.01, -0.01, -0.03, -0.03]
        glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
        glMaterialfv(GL_FRONT_AND_BACK, GL_SHININESS, mat_shininess)
        glLightfv(GL_LIGHT0, GL_AMBIENT, light0_ambient)
        glLightfv(GL_LIGHT0, GL_DIFFUSE, light0_diffuse)
        glLightfv(GL_LIGHT1, GL_DIFFUSE, light1_diffuse)
        glLightModelfv(GL_LIGHT_MODEL_AMBIENT, global_ambient)
        glLightModeli(GL_LIGHT_MODEL_LOCAL_VIEWER, GL_FALSE)
        glLightModeli(GL_LIGHT_MODEL_TWO_SIDE, draw_2side)
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glEnable(GL_LIGHT1)
        glEnable(GL_COLOR_MATERIAL)
        glEnable(GL_NORMALIZE)

        verts[:, 1] = verts[:, 1] - viewport[1] / 2
        verts[:, 0] = verts[:, 0] - viewport[0] / 2

        self.tset.load(img_path, size=(viewport[0], viewport[1]))

        fooimage = GL_Image(self.tset, img_path)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glDisable(GL_DEPTH_TEST)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluOrtho2D(0, viewport[0], 0, viewport[1])
        glMatrixMode(GL_MODELVIEW)

        # set up texturing
        glEnable(GL_TEXTURE_2D)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        fooimage.draw((0, 0))
        glDisable(GL_TEXTURE_2D)

        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glOrtho(-viewport[0] / 2, viewport[0] / 2, -viewport[1] / 2, viewport[1] / 2, 0.001, 3000000)
        glEnable(GL_DEPTH_TEST)
        glMatrixMode(GL_MODELVIEW)

        glLoadIdentity()

        glFrontFace(GL_CCW)

        verts[:, 2] = verts[:, 2] - 3000
        normals = np.zeros(verts.shape)
        tri_verts = verts[tris]
        n0 = np.cross(tri_verts[::, 1] - tri_verts[::, 0], tri_verts[::, 2] - tri_verts[::, 0])
        n0 = n0 / np.linalg.norm(n0, axis=1)[:, np.newaxis]
        for i in range(tris.shape[0]):
            normals[tris[i, 0]] += n0[i]
            normals[tris[i, 1]] += n0[i]
            normals[tris[i, 2]] += n0[i]
        normals = normals / np.linalg.norm(normals, axis=1)[:, np.newaxis]

        glEnableClientState(GL_VERTEX_ARRAY)
        glEnableClientState(GL_NORMAL_ARRAY)

        glVertexPointer(3, GL_FLOAT, 0, verts)
        glNormalPointer(GL_FLOAT, 0, normals)

        vertex_index = tris.ravel()
        glDrawElements(GL_TRIANGLES, len(vertex_index), GL_UNSIGNED_INT, vertex_index)

        glDisableClientState(GL_VERTEX_ARRAY)
        glDisableClientState(GL_NORMAL_ARRAY)

        pygame.display.flip()
        pygame.image.save(srf, out_path)
        pygame.quit()


class GL_Texture:
    def __init__(self, texname=None, size=(512, 512)):
        # filename = os.path.join('data/dataset/imgs/', texname)
        filename = texname

        self.texture, self.width, self.height = self._loadImage(filename)
        self.displaylist = self._createTexDL(self.texture, size[0], size[1])

    def __del__(self):
        if self.texture != None:
            self._delTexture(self.texture)
            self.texture = None
        if self.displaylist != None:
            self._delDL(self.displaylist)
            self.displaylist = None

    def _delDL(self, list):
        glDeleteLists(list, 1)

    def _loadImage(self, image):
        textureSurface = pygame.image.load(image)

        textureData = pygame.image.tostring(textureSurface, "RGBA", 1)

        width = textureSurface.get_width()
        height = textureSurface.get_height()

        texture = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, texture)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA,
                     GL_UNSIGNED_BYTE, textureData)

        return texture, width, height

    def _delTexture(self, texture):
        glDeleteTextures(texture)

    def _createTexDL(self, texture, width, height):
        newList = glGenLists(1)
        glNewList(newList, GL_COMPILE)
        glBindTexture(GL_TEXTURE_2D, texture)
        glBegin(GL_QUADS)

        # Bottom Left Of The Texture and Quad
        glTexCoord2f(0, 0)
        glVertex2f(0, 0)

        # Top Left Of The Texture and Quad
        glTexCoord2f(0, 1)
        glVertex2f(0, height)

        # Top Right Of The Texture and Quad
        glTexCoord2f(1, 1)
        glVertex2f(width, height)

        # Bottom Right Of The Texture and Quad
        glTexCoord2f(1, 0)
        glVertex2f(width, 0)
        glEnd()
        glEndList()

        return newList

    def __repr__(self):
        return self.texture.__repr__()


class Textureset:
    """Texturesets contain and name textures."""

    def __init__(self):
        self.textures = {}

    def load(self, texname=None, size=(512, 512)):
        self.textures[texname] = GL_Texture(texname, size)

    def set(self, texname, data):
        self.textures[texname] = data

    def delete(self, texname):
        del self.textures[texname]

    def __del__(self):
        self.textures.clear()
        del self.textures

    def get(self, name):
        return self.textures[name]


class GL_Image:
    def __init__(self, texset, texname):
        self.texture = texset.get(texname)
        self.width = self.texture.width
        self.height = self.texture.height
        self.abspos = None
        self.relpos = None
        self.color = (1, 1, 1, 1)
        self.rotation = 0
        self.rotationCenter = None

    def draw(self, abspos=None, relpos=None, width=None, height=None,
             color=None, rotation=None, rotationCenter=None):
        if color == None:
            color = self.color

        glColor4fv(color)

        if abspos:
            glLoadIdentity()
            glTranslate(abspos[0], abspos[1], 0)
        elif relpos:
            glTranslate(relpos[0], relpos[1], 0)

        if rotation == None:
            rotation = self.rotation

        if rotation != 0:
            if rotationCenter == None:
                rotationCenter = (self.width / 2, self.height / 2)
            glTranslate(rotationCenter[0], rotationCenter[1], 0)
            glRotate(rotation, 0, 0, -1)
            glTranslate(-rotationCenter[0], -rotationCenter[1], 0)

        if width or height:
            if not width:
                width = self.width
            elif not height:
                height = self.height

            glScalef(width / (self.width * 1.0), height / (self.height * 1.0), 1.0)

        glCallList(self.texture.displaylist)

        if rotation != 0:  # reverse
            glTranslate(rotationCenter[0], rotationCenter[1], 0)
            glRotate(-rotation, 0, 0, -1)
            glTranslate(-rotationCenter[0], -rotationCenter[1], 0)
