# -*- coding: utf-8 -*-
from __future__ import absolute_import
import numpy as np
import glob
import os
import cv2
import random
import copy

from pysixd_stuff.pysixd import transform
from pysixd_stuff.pysixd import view_sampler
from utils import lazy_property

verbose=False
class Dataset(object):
    def __init__(self, dataset_path, fg_path_format, codebook_path_format, list_objs, **kw):
        self.shape = (int(kw['h']), int(kw['w']), int(kw['c']))
        self.inshape = (int(kw['h']), int(kw['w']), int(kw['ci']))
        self.outshape = (int(kw['h']), int(kw['w']), int(kw['co']))

        self.num_imgs_per_obj = int(kw['noof_training_imgs'])
        self.list_objs=copy.copy(list_objs)
        print(self.list_objs)
        self.num_objs=len(self.list_objs)
        self.dataset_path = dataset_path
        self.fg_path_format = fg_path_format
        self.codebook_path_format = codebook_path_format

        self.bg_img_paths = glob.glob(kw['background_images_glob'])
        self.noof_bg_imgs = int(kw['noof_bg_imgs'])

        self._kw = kw
        self.train_x = np.empty((self.num_imgs_per_obj*self.num_objs,) + self.shape, dtype=np.uint8)
        self.mask_x = np.empty((self.num_imgs_per_obj*self.num_objs,) + self.shape[:2], dtype=bool)
        self.noof_obj_pixels = np.empty((self.num_imgs_per_obj*self.num_objs,), dtype=bool)

        self.train_y = np.empty((self.num_imgs_per_obj*self.num_objs,) + self.shape, dtype=np.uint8)
        self.mask_y = np.empty((self.num_imgs_per_obj*self.num_objs,) + self.shape[:2], dtype=bool)
        self.matrix_rot_y = np.empty((self.num_imgs_per_obj*self.num_objs,) + (3, 3), dtype=np.float32)

        self.embedding_size=None# = 8020
        self.codebook_bgr=None# = np.empty((self.embedding_size*self.num_objs,) + self.shape, dtype= np.uint8)
        self.codebook_quaternion=None# = np.zeros(shape=(self.embedding_size, 4), dtype=np.float32)
        self.knn_rot_embedding_indices = None
        self.knn_rot_embedding_barycentric_coord=None

        self.bg_imgs = np.empty((self.noof_bg_imgs,) + self.shape, dtype=np.uint8)
        if np.float(eval(self._kw['realistic_occlusion'])):
            self.random_syn_masks

    def load_training_images(self):
        for iid,oid in enumerate(self.list_objs):
            print('Loading training images of object', oid)
            current_config_hash = self.fg_path_format.format(oid)  # hashlib.md5(str(args.items('Dataset')+args.items('Paths'))).hexdigest()
            current_file_name = os.path.join(self.dataset_path, current_config_hash + '.npz')
            if os.path.exists(current_file_name):
                training_data = np.load(current_file_name)
                self.train_x[iid*self.num_imgs_per_obj:(iid+1)*self.num_imgs_per_obj] = training_data['bgr_x'].astype(np.uint8)
                self.mask_x[iid*self.num_imgs_per_obj:(iid+1)*self.num_imgs_per_obj] = training_data['mask_x']
                self.train_y[iid*self.num_imgs_per_obj:(iid+1)*self.num_imgs_per_obj] = training_data['bgr_y'].astype(np.uint8)#(training_data['bgr_y']*255.).astype(np.uint8)
                self.mask_y[iid*self.num_imgs_per_obj:(iid+1)*self.num_imgs_per_obj] = training_data['mask_y']
                self.matrix_rot_y[iid*self.num_imgs_per_obj:(iid+1)*self.num_imgs_per_obj] = training_data['matrix_rot_y']
                if verbose:
                    print('check rotation: ', self.matrix_rot_y[(iid+1)*self.num_imgs_per_obj-1])
                    cv2.imshow('trainx',self.train_x[(iid+1)*self.num_imgs_per_obj-1])
                    cv2.imshow('trainy',self.train_y[(iid+1)*self.num_imgs_per_obj-1])
                    cv2.imshow('maskx',self.mask_x[(iid+1)*self.num_imgs_per_obj-1].astype(np.uint8)*255)
                    cv2.imshow('masky',self.mask_y[(iid+1)*self.num_imgs_per_obj-1].astype(np.uint8)*255)
                    cv2.waitKey()
            else:
                print('ERROR - Cannot find fg images in:',current_file_name)
                exit(0)
        self.noof_obj_pixels = np.count_nonzero(self.mask_x == 0, axis=(1, 2))
        print('Loaded %s training images' % len(self.train_x))


    def load_embedding_images(self):
        self.codebook_bgr=np.empty((self.embedding_size*self.num_objs,) + self.shape, dtype= np.uint8)
        for iid,oid in enumerate(self.list_objs):
            print('Loading embedding images of object', oid)
            mpath_embedding_imgs = os.path.join(self.dataset_path, self.codebook_path_format.format(oid), 'imgs', '{:05d}.png')
            for i in range(0, self.embedding_size):
                self.codebook_bgr[iid*self.embedding_size+i] = cv2.imread(mpath_embedding_imgs.format(i)).astype(np.uint8)
            if verbose:
                cv2.imshow('codebook',self.codebook_bgr[(iid+1)*self.embedding_size-1])
                cv2.waitKey()
        print('Loaded {0} codebook images'.format(self.embedding_size*self.num_objs))

    def load_bg_images(self):
        current_config_hash = 'prepared_bg_imgs'
        current_file_name = os.path.join(self.dataset_path, current_config_hash + '.npy')
        if os.path.exists(current_file_name):
            self.bg_imgs = np.load(current_file_name)
        else:
            file_list = self.bg_img_paths[:self.noof_bg_imgs]
            from random import shuffle
            shuffle(file_list)
            for j, fname in enumerate(file_list):
                print('loading bg img %s/%s' % (j, self.noof_bg_imgs))
                bgr = cv2.imread(fname)
                H, W = bgr.shape[:2]
                y_anchor = int(np.random.rand() * (H - self.shape[0]))
                x_anchor = int(np.random.rand() * (W - self.shape[1]))
                # bgr = cv2.resize(bgr, self.shape[:2])
                bgr = bgr[y_anchor:y_anchor + self.shape[0], x_anchor:x_anchor + self.shape[1], :]
                if bgr.shape[0] != self.shape[0] or bgr.shape[1] != self.shape[1]:
                    continue
                if self.shape[2] == 1:
                    bgr = cv2.cvtColor(np.uint8(bgr), cv2.COLOR_BGR2GRAY)[:, :, np.newaxis]
                self.bg_imgs[j] = bgr
            np.save(current_file_name, self.bg_imgs)
        print('loaded %s bg images' % self.noof_bg_imgs)

    def extract_square_patch(self, scene_img, bb_xywh, pad_factor, resize=(128, 128), interpolation=cv2.INTER_NEAREST, black_borders=False):
        x, y, w, h = np.array(bb_xywh).astype(np.int32)
        size = int(np.maximum(h, w) * pad_factor)

        left = np.maximum(x + w / 2 - size / 2, 0)
        right = np.minimum(x + w / 2 + size / 2, scene_img.shape[1])
        top = np.maximum(y + h / 2 - size / 2, 0)
        bottom = np.minimum(y + h / 2 + size / 2, scene_img.shape[0])

        scene_crop = scene_img[top:bottom, left:right].copy()

        if black_borders:
            scene_crop[:(y - top), :] = 0
            scene_crop[(y + h - top):, :] = 0
            scene_crop[:, :(x - left)] = 0
            scene_crop[:, (x + w - left):] = 0

        scene_crop = cv2.resize(scene_crop, resize, interpolation=interpolation)
        return scene_crop

    @lazy_property
    def _aug(self):
        from imgaug.augmenters import Sequential, SomeOf, OneOf, Sometimes, WithColorspace, WithChannels, \
            Noop, Lambda, AssertLambda, AssertShape, Scale, CropAndPad, \
            Pad, Crop, Fliplr, Flipud, Superpixels, ChangeColorspace, PerspectiveTransform, \
            Grayscale, GaussianBlur, AverageBlur, MedianBlur, Convolve, \
            Sharpen, Emboss, EdgeDetect, DirectedEdgeDetect, Add, AddElementwise, \
            AdditiveGaussianNoise, Multiply, MultiplyElementwise, Dropout, \
            CoarseDropout, Invert, ContrastNormalization, Affine, PiecewiseAffine, \
            ElasticTransformation
        return eval(self._kw['code'])

    @lazy_property
    def _aug_occl(self):
        from imgaug.augmenters import Sequential, SomeOf, OneOf, Sometimes, WithColorspace, WithChannels, \
            Noop, Lambda, AssertLambda, AssertShape, Scale, CropAndPad, \
            Pad, Crop, Fliplr, Flipud, Superpixels, ChangeColorspace, PerspectiveTransform, \
            Grayscale, GaussianBlur, AverageBlur, MedianBlur, Convolve, \
            Sharpen, Emboss, EdgeDetect, DirectedEdgeDetect, Add, AddElementwise, \
            AdditiveGaussianNoise, Multiply, MultiplyElementwise, Dropout, \
            CoarseDropout, Invert, ContrastNormalization, Affine, PiecewiseAffine, \
            ElasticTransformation
        return Sequential([Sometimes(0.7, CoarseDropout(p=0.4, size_percent=0.01))])

    @lazy_property
    def random_syn_masks(self):
        import bitarray
        workspace_path = os.environ.get('AE_WORKSPACE_PATH')

        random_syn_masks = bitarray.bitarray()
        with open(os.path.join(workspace_path, 'random_tless_masks/arbitrary_syn_masks_1000.bin'), 'r') as fh:
            random_syn_masks.fromfile(fh)
        occlusion_masks = np.fromstring(random_syn_masks.unpack(), dtype=np.bool)
        occlusion_masks = occlusion_masks.reshape(-1, 224, 224, 1).astype(np.float32)
        print(occlusion_masks.shape)

        occlusion_masks = np.array(
            [cv2.resize(mask, (self.shape[0], self.shape[1]), interpolation=cv2.INTER_NEAREST) for mask in
             occlusion_masks])
        return occlusion_masks

    def augment_occlusion_mask(self, masks, verbose=False, min_trans=0.2, max_trans=0.7, max_occl=0.25, min_occl=0.0):
        new_masks = np.zeros_like(masks, dtype=np.bool)
        occl_masks_batch = self.random_syn_masks[np.random.choice(len(self.random_syn_masks), len(masks))]
        for idx, mask in enumerate(masks):
            occl_mask = occl_masks_batch[idx]
            while True:
                trans_x = int(np.random.choice([-1, 1]) * (np.random.rand() * (max_trans - min_trans) + min_trans) *
                              occl_mask.shape[0])
                trans_y = int(np.random.choice([-1, 1]) * (np.random.rand() * (max_trans - min_trans) + min_trans) *
                              occl_mask.shape[1])
                M = np.float32([[1, 0, trans_x], [0, 1, trans_y]])

                transl_occl_mask = cv2.warpAffine(occl_mask, M, (occl_mask.shape[0], occl_mask.shape[1]))

                overlap_matrix = np.invert(mask.astype(np.bool)) * transl_occl_mask.astype(np.bool)
                overlap = len(overlap_matrix[overlap_matrix == True]) / float(len(mask[mask == 0]))

                if overlap < max_occl and overlap > min_occl:
                    new_masks[idx, ...] = np.logical_xor(mask.astype(np.bool), overlap_matrix)
                    if verbose:
                        print('overlap is ', overlap)
                    break

        return new_masks

    def augment_squares(self, masks, rand_idcs, max_occl=0.25):
        new_masks = np.invert(masks)

        idcs = np.arange(len(masks))
        while len(idcs) > 0:
            new_masks[idcs] = self._aug_occl.augment_images(np.invert(masks[idcs]))
            new_noof_obj_pixels = np.count_nonzero(new_masks, axis=(1, 2))
            idcs = np.where(new_noof_obj_pixels / self.noof_obj_pixels[rand_idcs].astype(np.float32) < 1 - max_occl)[0]
            print(idcs)
        return np.invert(new_masks)

    def batch(self, batch_size, batchx_clean=False,stack_codebook=False):
        rand_idcs = np.random.choice(self.num_objs*self.num_imgs_per_obj, batch_size, replace=False)
        assert self.noof_bg_imgs > 0
        rand_idcs_bg = np.random.choice(self.noof_bg_imgs, batch_size, replace=False)
        batch_x = self.train_y[rand_idcs] if batchx_clean else self.train_x[rand_idcs]
        masks = self.mask_y[rand_idcs] if batchx_clean else self.mask_x[rand_idcs]
        batch_y = self.train_y[rand_idcs]

        pose_label=self.knn_rot_embedding_indices[rand_idcs, 0].reshape((batch_size,1))#index
        obj_label= (rand_idcs//self.num_imgs_per_obj).reshape((batch_size,1))

        codebook_img_ids=obj_label.flatten()*self.embedding_size+pose_label.flatten()
        if stack_codebook:
            pose_label=codebook_img_ids.reshape((batch_size,1))
        pose_img=self.codebook_bgr[codebook_img_ids]

        if not batchx_clean:
            rand_vocs = self.bg_imgs[rand_idcs_bg]
            if eval(self._kw['realistic_occlusion']):
                masks = self.augment_occlusion_mask(masks.copy(), max_occl=np.float(self._kw['realistic_occlusion']))
            if eval(self._kw['square_occlusion']):
                masks = self.augment_squares(masks.copy(), rand_idcs, max_occl=np.float(self._kw['square_occlusion']))

            batch_x[masks] = rand_vocs[masks]
            # needs uint8
            batch_x = self._aug.augment_images(batch_x)

        if self.inshape[-1] != 3:
            edge_x = np.empty((batch_size,) + self.shape[:-1] + (1,), dtype=np.float32)
            for i in range(0,batch_size):
                p1 = random.randint(30, 100)  # (50,150)
                p2 = random.random() * 0.8 + 1.2
                edge_x[i] = (cv2.Canny(batch_x[i], p1, p1 * p2)).astype(np.float32)[:, :, np.newaxis]
                if verbose:
                    if i==0:
                        print('Dataset: Obj Label',obj_label.shape,'Pose Label',pose_label.shape, 'Pose Img',pose_img.shape)
                    cv2.imshow('batch_x_{:02d}_{:04d}'.format(self.list_objs[obj_label[i,0]],pose_label[i,0]),batch_x[i])
                    cv2.imshow('batch_x_pose_label',pose_img[i])
                    cv2.waitKey()
                    cv2.destroyAllWindows()

            if self.inshape[-1] == 1:
                batch_x = edge_x
            else:  # if self.inshape[-1]==4:
                batch_x = np.concatenate((batch_x, edge_x), axis=-1)


        weight_y = np.zeros((batch_size,) + self.shape[:-1] + (1,), dtype=np.float32)
        if self.outshape[-1] != 3:
            edge_y = np.empty((batch_size,) + self.shape[:-1] + (1,), dtype=np.float32)
            for i in range(0,batch_size):
                edge_y[i] = (cv2.Canny(batch_y[i], 50, 150)).astype(np.float32)[:, :, np.newaxis]
                cnt_edgey = float(np.count_nonzero(edge_y[i]))
                weight_y[i] = np.where(edge_y[i] == 0, cnt_edgey / (self.shape[0] * self.shape[1]),
                                       1 - cnt_edgey / (self.shape[0] * self.shape[1]))

            if self.outshape[-1] == 1:
                batch_y = edge_y
            else:  # outshape[-1]==4
                batch_y = np.concatenate((batch_y, edge_y), axis=-1)

        _batch_x, _batch_y = batch_x, batch_y

        batch_x = (_batch_x / 255.).astype(np.float32)
        batch_y = (_batch_y / 255.).astype(np.float32)
        return (batch_x, batch_y, weight_y, obj_label, pose_label, pose_img)

    def load_codebook_rotation(self, path_codebook_rotation):
        print('Load Codebook Rotation from',path_codebook_rotation)
        rot_matrix = np.load(os.path.join(self.dataset_path,path_codebook_rotation))['rots'].astype(np.float32)
        self.embedding_size=rot_matrix.shape[0]
        self.codebook_quaternion=np.zeros(shape=(self.embedding_size, 4), dtype=np.float32)

        for i in range(0, self.embedding_size):
            crot = np.eye(4, dtype=np.float32)
            crot[:3, :3] = rot_matrix[i]
            self.codebook_quaternion[i] = transform.quaternion_from_matrix(crot, False)
        print('Codebook Rotation Shape: ',self.codebook_quaternion.shape)

    def compute_knn_rot_embedding_indices(self, knn, use_probability=False):
        query_size = self.matrix_rot_y.shape[0]
        query_quaternion = np.zeros(shape=(query_size, 4), dtype=np.float32)
        print('Query Quaternion Shape',query_quaternion.shape)
        for i in range(0, query_size):
            crot = np.eye(4, dtype=np.float32)
            crot[:3, :3] = self.matrix_rot_y[i]
            query_quaternion[i] = transform.quaternion_from_matrix(crot, False)

        dot_query_embed = -np.fabs(np.dot(query_quaternion, self.codebook_quaternion.transpose()))
        print(dot_query_embed.max(), dot_query_embed.min())
        self.knn_rot_embedding_indices = np.argsort(dot_query_embed, axis=-1)[:, :knn]

        if use_probability:
            print('Use Barycentric Coordinate')
            self.knn_rot_embedding_barycentric_coord=np.zeros(shape=(query_size,self.codebook_quaternion.shape[0]),dtype=np.float32)
            for i in range(0,query_size):
                denominator=self.codebook_quaternion[self.knn_rot_embedding_indices[i].flatten()]
                for j in range(0,knn):
                    if np.dot(query_quaternion[i], denominator[j])<0:
                        denominator[j]=-denominator[j]
                denominator=np.concatenate((np.ones(shape=(5, 1), dtype=self.codebook_quaternion.dtype),denominator),axis=1)

                det_denominator=np.linalg.det(denominator)
                for j in range(0,knn):
                    nominator=denominator.copy()
                    nominator[j,1:5]=query_quaternion[i]
                    self.knn_rot_embedding_barycentric_coord[i,self.knn_rot_embedding_indices[i,j]]=np.linalg.det(nominator)/det_denominator

            #print(i, 'possibility', self.knn_rot_embedding_barycentric_coord[i,self.knn_rot_embedding_indices[i]],self.knn_rot_embedding_barycentric_coord[i],np.sum(self.knn_rot_embedding_barycentric_coord[i]))
            self.knn_rot_embedding_barycentric_coord[self.knn_rot_embedding_barycentric_coord<0]=0
            #print('possibility 2', self.knn_rot_embedding_barycentric_coord[i, self.knn_rot_embedding_indices[i]],np.sum(self.knn_rot_embedding_barycentric_coord[i]))
            print((self.knn_rot_embedding_barycentric_coord<0).any())
            self.knn_rot_embedding_barycentric_coord=self.knn_rot_embedding_barycentric_coord/np.sum(self.knn_rot_embedding_barycentric_coord,axis=1,keepdims=True)
            print(np.sum(self.knn_rot_embedding_barycentric_coord,axis=1,keepdims=True).max(),np.sum(self.knn_rot_embedding_barycentric_coord,axis=1,keepdims=True).min())
            #print('possibility 3', self.knn_rot_embedding_barycentric_coord[i, self.knn_rot_embedding_indices[i]],np.sum(self.knn_rot_embedding_barycentric_coord[i]))

