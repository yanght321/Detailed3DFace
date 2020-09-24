import numpy as np
import pickle
import dlib
import cv2
from scipy.optimize import least_squares
from scipy.optimize import minimize


class BilinearModel():
    def __init__(self, model_dir):
        self.num_lm = 68

        self.lm_index = np.zeros(self.num_lm, dtype=int)
        with open(f'{model_dir}/index_68.txt', 'r') as f:
            lines = f.readlines()
            for i in range(self.num_lm):
                line = lines[i]
                values = line.split()
                self.lm_index[i] = int(values[0])

        with open(f'{model_dir}/faces.pkl', 'rb') as f:
            self.texcoords, self.faces = pickle.load(f)
        with open(f'{model_dir}/front_verts_indices.pkl', 'rb') as f:
            self.front_verts_indices = pickle.load(f)
        with open(f'{model_dir}/front_texcoords.pkl', 'rb') as f:
            self.front_texcoords = pickle.load(f)
        with open(f'{model_dir}/front_faces.pkl', 'rb') as f:
            self.front_faces = pickle.load(f)

        with open(f'{model_dir}/id_mean.pkl', 'rb') as f:
            self.id_mean = pickle.load(f)
        with open(f'{model_dir}/id_var.pkl', 'rb') as f:
            self.id_var = pickle.load(f)
        with open(f'{model_dir}/exp_GMM.pkl', 'rb') as f:
            self.exp_gmm = pickle.load(f)
        with open(f'{model_dir}/front_faces.pkl', 'rb') as f:
            self.front_faces = pickle.load(f)

        with open(f'{model_dir}/contour_line_68.pkl', 'rb') as f:
            self.contour_line_right, self.contour_line_left = pickle.load(f)
        self.core_tensor = np.load(f'{model_dir}/core_847_50_52.npy')
        self.factors_id = np.load(f'{model_dir}/factors_id_847_50_52.npy')

        self.core_tensor = self.core_tensor.transpose((2, 1, 0))
        for i in range(51):
            self.core_tensor[:, i + 1, :] = self.core_tensor[:, i + 1, :] - self.core_tensor[:, 0, :]

        with open(f'{model_dir}/front_face_indices.pkl', 'rb') as f:
            self.front_face_indices = pickle.load(f)

        self.detector = dlib.get_frontal_face_detector()
        self.points = dlib.shape_predictor('./dlib/shape_predictor_68_face_landmarks.dat')

        # for render
        tris = []
        self.vert_texcoords = np.zeros((len(self.front_verts_indices), 2))
        for face in self.front_faces:
            vertices, normals, texture_coords, material = face
            tris.append([vertices[0] - 1, vertices[1] - 1, vertices[2] - 1])
            for i in range(len(vertices)):
                self.vert_texcoords[vertices[i] - 1] = self.front_texcoords[texture_coords[i] - 1]
        self.tris = np.array(tris)

    def fit_image(self, img):
        w = img.shape[1]
        h = img.shape[0]
        scale = max(w, h) / 512
        img_small = cv2.resize(img, (int(w / scale), int(h / scale)))
        faces = self.detector(img_small, 1)
        pts = self.points(img_small, faces[0])
        lm_pos = np.zeros((68, 2), dtype=int)
        for i in range(68):
            lm_pos[i, :] = (pts.part(i).x, pts.part(i).y)
        lm_pos = lm_pos * scale
        lm_pos[:, 1] = h - lm_pos[:, 1]

        id = self.factors_id[0]
        exp = np.zeros(52)
        exp[0] = 1
        rot_vector = np.array([0, 0.1, 0], dtype=np.double)
        trans = np.array([0, 0])
        scale = 1

        mesh_vertices = self.core_tensor.dot(id).dot(exp).reshape((-1, 3))
        verts_img = self.project(mesh_vertices, rot_vector, scale, trans)
        lm_index = self.lm_index

        for optimize_loop in range(4):

            vertices_mean = np.mean(verts_img[lm_index], axis=0)
            vertices_2d = verts_img[lm_index] - vertices_mean
            lm_index_full = np.zeros(self.num_lm * 3, dtype=int)
            for i in range(self.num_lm * 3):
                lm_index_full[i] = lm_index[i // 3] * 3 + i % 3

            lm_mean = np.mean(lm_pos, axis=0)
            lm = lm_pos - lm_mean
            scale = np.sum(np.linalg.norm(lm, axis=1)) / np.sum(np.linalg.norm(vertices_2d, axis=1))
            trans = lm_mean - vertices_mean * scale

            lm_core_tensor = self.core_tensor[lm_index_full]

            lm_pos_3D = lm_core_tensor.dot(id).dot(exp).reshape((-1, 3))
            scale, trans, rot_vector = self._optimize_rigid_pos(scale, trans, rot_vector, lm_pos_3D, lm_pos)
            id = self._optimize_identity(scale, trans, rot_vector, id, exp, lm_core_tensor, lm_pos, prior_weight=1)
            exp = self._optimize_expression(scale, trans, rot_vector, id, exp, lm_core_tensor, lm_pos, prior_weight=1)

            mesh_vertices = self.core_tensor.dot(id).dot(exp).reshape((-1, 3))
            verts_img = self.project(mesh_vertices, rot_vector, scale, trans)

            lm_index = self._update_3d_lm_index(verts_img, lm_index)

        return (rot_vector, scale, trans), mesh_vertices

    def _update_3d_lm_index(self, points_proj, lm_index):
        updated_lm_index = list(lm_index)
        modify_key_right = range(9, 17)
        modify_key_left = range(0, 8)

        # get the outest point on the contour line
        for i in range(len(modify_key_right)):
            if len(self.contour_line_right[i]) != 0:
                max_ind = np.argmax(points_proj[self.contour_line_right[i], 0])
                updated_lm_index[modify_key_right[i]] = self.contour_line_right[i][max_ind]

        for i in range(len(modify_key_left)):
            if len(self.contour_line_left[i]) != 0:
                min_ind = np.argmin(points_proj[self.contour_line_left[i], 0])
                updated_lm_index[modify_key_left[i]] = self.contour_line_left[i][min_ind]

        bottom_cand = [11789, 1804, 11792, 5007, 11421, 1681, 11410, 5000, 11423, 3248, 11427, 1687, 15212, 6204, 15216,
                       2851]
        updated_lm_index[8] = bottom_cand[np.argmin((points_proj[bottom_cand, 1]))]

        return updated_lm_index

    def project(self, points, rot_vec, scale, trans, keepz=False):
        points_proj = self._rotate(points, rot_vec.reshape(1, 3))
        points_proj = points_proj * scale
        if keepz:
            points_proj[:, 0:2] = points_proj[:, 0:2] + trans
        else:
            points_proj = points_proj[:, 0:2] + trans
        return points_proj

    def _rotate(self, points, rot_vec):
        """Rotate points by given rotation vectors.
        Rodrigues' rotation formula is used.
        """
        theta = np.linalg.norm(rot_vec)
        with np.errstate(invalid='ignore'):
            v = rot_vec / theta
            v = np.nan_to_num(v)
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)

        return cos_theta * points + sin_theta * np.cross(v, points) + (points.dot(v.T) * (1 - cos_theta)).dot(v)

    def _optimize_rigid_pos(self, scale, trans, rot_vector, lm_pos_3D, lm_pos):
        lm_pos_3D = lm_pos_3D.ravel()
        lm_pos = lm_pos.ravel()
        params = np.hstack((scale, trans, rot_vector))
        result = least_squares(self._compute_res_rigid, params, verbose=0, x_scale='jac', ftol=1e-5, method='lm',
                               args=(lm_pos_3D, lm_pos))
        return result.x[0], result.x[1:3], result.x[3:6]

    def _compute_res_id(self, id, id_matrix, scale, trans, rot_vector, lm_pos, prior_weight):
        id_matrix = id_matrix.reshape(-1, id.shape[0])
        lm_pos_3D = id_matrix.dot(id).reshape((-1, 3))
        lm_proj = self.project(lm_pos_3D, rot_vector, scale, trans).ravel()
        return np.linalg.norm(lm_proj - lm_pos) ** 2 / scale / scale + prior_weight * (id - self.id_mean).dot(
            np.diag(1 / self.id_var)).dot(np.transpose([id - self.id_mean]))

    def _optimize_identity(self, scale, trans, rot_vector, id, exp, lm_core_tensor, lm_pos, prior_weight=20):
        id_matrix = np.tensordot(lm_core_tensor, exp, axes=([1], [0])).ravel()
        lm_pos = lm_pos.ravel()
        result = minimize(self._compute_res_id, id, method='L-BFGS-B',
                          args=(id_matrix, scale, trans, rot_vector, lm_pos, prior_weight), options={'maxiter': 100})
        return result.x

    def _compute_res_exp(self, exp, exp_matrix, scale, trans, rot_vector, lm_pos, prior_weight):
        exp_matrix = exp_matrix.reshape(-1, exp.shape[0] + 1)
        exp_full = np.ones(52)
        exp_full[1:52] = exp
        lm_pos_3D = exp_matrix.dot(exp_full).reshape((-1, 3))
        lm_proj = self.project(lm_pos_3D, rot_vector, scale, trans).ravel()

        return np.linalg.norm(lm_proj - lm_pos) ** 2 / scale / scale - prior_weight * \
               self.exp_gmm.score_samples(exp.reshape(1, -1))[0]

    def _optimize_expression(self, scale, trans, rot_vector, id, exp, lm_core_tensor, lm_pos, prior_weight=0.02):
        exp_matrix = np.dot(lm_core_tensor, id).ravel()
        lm_pos = lm_pos.ravel()
        bounds = []
        for i in range(exp.shape[0] - 1):
            bounds.append((0, 1))
        result = minimize(self._compute_res_exp, exp[1:52], method='L-BFGS-B', bounds=bounds,
                          args=(exp_matrix, scale, trans, rot_vector, lm_pos, prior_weight), options={'maxiter': 100})
        exp_full = np.ones(52)
        exp_full[1:52] = result.x
        return exp_full

    def _compute_res_rigid(self, params, lm_pos_3D, lm_pos):
        lm_pos_3D = lm_pos_3D.reshape(-1, 3)
        lm_proj = self.project(lm_pos_3D, params[3:6], params[0], params[1:3])
        return lm_proj.ravel() - lm_pos

    def get_texture(self, img, verts_img):
        h, w, _ = img.shape

        texture = np.zeros((4096, 4096, 3))

        for face_idx in self.front_face_indices:
            face = self.faces[int(face_idx)]
            ver_indices, n_indices, tc, material = face

            if max(abs(self.texcoords[tc[0] - 1][0] - self.texcoords[tc[1] - 1][0]),
                   abs(self.texcoords[tc[0] - 1][0] - self.texcoords[tc[2] - 1][0]),
                   abs(self.texcoords[tc[1] - 1][0] - self.texcoords[tc[2] - 1][0]),
                   abs(self.texcoords[tc[0] - 1][1] - self.texcoords[tc[1] - 1][1]),
                   abs(self.texcoords[tc[0] - 1][1] - self.texcoords[tc[2] - 1][1]),
                   abs(self.texcoords[tc[1] - 1][1] - self.texcoords[tc[2] - 1][1])) > 0.3:
                continue

            tri1 = np.float32([[[(h - int(verts_img[ver_indices[0] - 1, 1])),
                                 int(verts_img[ver_indices[0] - 1, 0])],
                                [(h - int(verts_img[ver_indices[1] - 1, 1])),
                                 int(verts_img[ver_indices[1] - 1, 0])],
                                [(h - int(verts_img[ver_indices[2] - 1, 1])),
                                 int(verts_img[ver_indices[2] - 1, 0])]]])
            tri2 = np.float32(
                [[[4096 - self.texcoords[tc[0] - 1][1] * 4096, self.texcoords[tc[0] - 1][0] * 4096],
                  [4096 - self.texcoords[tc[1] - 1][1] * 4096, self.texcoords[tc[1] - 1][0] * 4096],
                  [4096 - self.texcoords[tc[2] - 1][1] * 4096, self.texcoords[tc[2] - 1][0] * 4096]]])
            r1 = cv2.boundingRect(tri1)
            r2 = cv2.boundingRect(tri2)

            tri1Cropped = []
            tri2Cropped = []

            for i in range(0, 3):
                tri1Cropped.append(((tri1[0][i][1] - r1[1]), (tri1[0][i][0] - r1[0])))
                tri2Cropped.append(((tri2[0][i][1] - r2[1]), (tri2[0][i][0] - r2[0])))

            # Apply warpImage to small rectangular patches
            img1Cropped = img[r1[0]:r1[0] + r1[2], r1[1]:r1[1] + r1[3]]
            warpMat = cv2.getAffineTransform(np.float32(tri1Cropped), np.float32(tri2Cropped))

            # Get mask by filling triangle
            mask = np.zeros((r2[2], r2[3], 3), dtype=np.float32)
            cv2.fillConvexPoly(mask, np.int32(tri2Cropped), (1.0, 1.0, 1.0), 16, 0)

            # Apply the Affine Transform just found to the src image
            img2Cropped = cv2.warpAffine(img1Cropped, warpMat, (r2[3], r2[2]), None, flags=cv2.INTER_LINEAR,
                                         borderMode=cv2.BORDER_REFLECT_101)

            # Apply mask to cropped region
            img2Cropped = img2Cropped * mask

            # Copy triangular region of the rectangular patch to the output image
            texture[r2[0]:r2[0] + r2[2], r2[1]:r2[1] + r2[3]] = texture[r2[0]:r2[0] + r2[2],
                                                                r2[1]:r2[1] + r2[3]] * ((1.0, 1.0, 1.0) - mask)

            texture[r2[0]:r2[0] + r2[2], r2[1]:r2[1] + r2[3]] = texture[r2[0]:r2[0] + r2[2],
                                                                r2[1]:r2[1] + r2[3]] + img2Cropped

        return texture

    def save_obj(self, path, verts, texture_name=None, front=False):
        if front:
            verts = verts[self.front_verts_indices]
            faces = self.front_faces
            texcoords = self.front_texcoords
        else:
            faces = self.faces
            texcoords = self.texcoords
        with open(path, 'w') as f:
            if texture_name is not None:
                f.write('mtllib ./%s.mtl\n' % path.split('/')[-1])
            for i in range(len(verts)):
                f.write("v %.6f %.6f %.6f\n" % (verts[i][0], verts[i][1], verts[i][2]))
            for i in range(len(texcoords)):
                f.write("vt %.6f %.6f\n" % (texcoords[i][0], texcoords[i][1]))
            if texture_name is not None:
                f.write('usemtl material_0\n')
            for face in faces:
                face_vertices, face_normals, face_texture_coords, material = face
                f.write("f %d/%d %d/%d %d/%d\n" % (
                    face_vertices[0], face_texture_coords[0], face_vertices[1], face_texture_coords[1],
                    face_vertices[2],
                    face_texture_coords[2]))
        if texture_name is not None:
            with open(path + '.mtl', 'w') as f:
                f.write('newmtl material_0\nKa 0.200000 0.200000 0.200000\nKd 0.000000 0.000000 0.000000\n')
                f.write('Ks 1.000000 1.000000 1.000000\nTr 0.000000\nillum 2\nNs 0.000000\nmap_Kd %s' % texture_name)
