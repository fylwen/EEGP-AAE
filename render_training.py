# -*- coding: utf-8 -*-
# This is to sample rotations to train the Autoencoder.
# For each sampled rotation, its BGR image(foreground only) for Decoder GT and Encoder input, the foreground masks, and the rotation of decoder GT is generated and preserved.

import os
import progressbar

import data_utils
from pysixd_stuff.pysixd import inout,transform,view_sampler
from imgaug.augmenters import *

verbose=True
class Model(object):
    def __init__(self, dir_dataset,path_model, saved_file_name, num_per_batch):
        self.shape_c3 = (128,128,3)
        self.num_per_batch= num_per_batch
        self.dir_dataset=dir_dataset#path to save rendered training data
        self.path_model=path_model#path of mesh
        self.saved_file_name=saved_file_name

        self.bgr_x = np.empty((self.num_per_batch,) + self.shape_c3, dtype=np.uint8)
        self.bgr_y = np.empty((self.num_per_batch,) + self.shape_c3, dtype=np.uint8)

        self.mask_x = np.empty((self.num_per_batch,) + self.shape_c3[:2], dtype=bool)
        self.mask_y = np.empty((self.num_per_batch,)+self.shape_c3[:2],dtype=bool)

        self.matrix_rot_y=np.empty((self.num_per_batch,)+(3,3),dtype=np.float32)


    def combine_rendered_batches(self,num_batches):
        num_total=self.num_per_batch*num_batches
        self.bgr_x = np.empty((num_total,) + self.shape_c3, dtype=np.uint8)
        self.bgr_y = np.empty((num_total,) + self.shape_c3, dtype=np.uint8)

        self.mask_x = np.empty((num_total,) + self.shape_c3[:2], dtype=bool)
        self.mask_y = np.empty((num_total,)+self.shape_c3[:2],dtype=bool)

        self.matrix_rot_y=np.empty((num_total,)+(3,3),dtype=np.float32)
        self.noof_obj_pixels = np.empty((num_total,), dtype=np.uint8)
        print('Size',self.bgr_x.shape)

        for i in range(0,num_batches):
            current_file_name=os.path.join(self.dir_dataset, self.saved_file_name + '{0}.npz'.format(i))
            training_data = np.load(current_file_name)
            self.bgr_x[i*self.num_per_batch:(i+1)*self.num_per_batch]=training_data['bgr_x'].astype(np.uint8)
            self.bgr_y[i*self.num_per_batch:(i+1)*self.num_per_batch]=training_data['bgr_y'].astype(np.uint8)

            self.mask_y[i*self.num_per_batch:(i+1)*self.num_per_batch] = training_data['mask_y'].astype(bool)
            self.mask_x[i*self.num_per_batch:(i+1)*self.num_per_batch] = training_data['mask_x'].astype(bool)
            self.matrix_rot_y[i*self.num_per_batch:(i+1)*self.num_per_batch]=training_data['matrix_rot_y'].astype(np.float32)

            if verbose:
                vis_mask_x = np.where(self.mask_x[(i+1)*self.num_per_batch-1],0,255).astype(np.uint8)
                vis_mask_y = np.where(self.mask_y[(i+1)*self.num_per_batch-1],0,255).astype(np.uint8)
                cv2.imshow('bgr_x', self.bgr_x[(i+1)*self.num_per_batch-1])
                cv2.imshow('mask_x', vis_mask_x)
                cv2.imshow('mask_y', vis_mask_y)
                cv2.imshow('bgr_y', self.bgr_y[(i+1)*self.num_per_batch-1])
                cv2.waitKey()
                print('rot_y:',self.matrix_rot_y[(i+1)*self.num_per_batch-1])
        current_file_name = os.path.join(self.dir_dataset, self.saved_file_name + '.npz')
        np.savez(current_file_name, bgr_x=self.bgr_x,bgr_y=self.bgr_y, mask_x=self.mask_x, mask_y=self.mask_y, matrix_rot_y=self.matrix_rot_y)

    def load_training_images(self):
        current_file_name = os.path.join(self.dir_dataset, self.saved_file_name + '.npz')
        training_data = np.load(current_file_name)
        self.bgr_x = training_data['bgr_x'].astype(np.uint8)
        self.mask_x = training_data['mask_x']

        self.bgr_y = training_data['bgr_y'].astype(np.uint8)
        self.mask_y = training_data['mask_y']
        self.matrix_rot_y=training_data['matrix_rot_y'].astype(np.float32)

        print('Size',self.bgr_x.shape)

        if verbose:
            vis_mask_x = np.where(self.mask_x[-1],0,255).astype(np.uint8)
            vis_mask_y = np.where(self.mask_y[-1],0,255).astype(np.uint8)

            cv2.imshow('bgr_x',self.bgr_x[-1])
            cv2.imshow('mask_x',vis_mask_x)
            cv2.imshow('mask_y',vis_mask_y)
            cv2.imshow('bgr_y',(self.bgr_y[-1]).astype(np.uint8))
            cv2.waitKey()
            print('check rot: ',self.matrix_rot_y[-1])

    def render_batch_training_images(self,render_dims,cam_K,batch_id,depth_scale=1.,texture_img=None):
        from pysixd_stuff.pysixd import renderer_vt
        current_file_name = os.path.join(self.dir_dataset, self.saved_file_name + '{0}.npz'.format(batch_id))
        H, W = self.shape_c3[0],self.shape_c3[1]
        K = np.array(cam_K).reshape(3, 3)
        clip_near = float(10)
        clip_far = float(10000)
        pad_factor = float(1.2)
        crop_offset_sigma = float(20)
        t = np.array([0, 0, float(700)])

        bar = progressbar.ProgressBar(
            maxval=self.num_per_batch,
            widgets=[' [', progressbar.Timer(), ' | ',
                     progressbar.Counter('%0{}d / {}'.format(len(str(self.num_per_batch)), self.num_per_batch)), ' ] ', progressbar.Bar(), ' (',
                     progressbar.ETA(), ') ']
        )
        bar.start()

        model = inout.load_ply(self.path_model)

        model['pts']*=depth_scale

        for i in np.arange(self.num_per_batch):
            bar.update(i)

            R = transform.random_rotation_matrix()[:3, :3]
            im_size = (render_dims[0], render_dims[1])
            rgb_x, depth_x = renderer_vt.render_phong(model, im_size, K.copy(), R, t, clip_near=clip_near,
                                                  clip_far=clip_far,texture=texture_img, mode='rgb+depth', random_light=True)
            rgb_y, depth_y = renderer_vt.render_phong(model, im_size, K.copy(), R, t, clip_near=clip_near,
                                                  clip_far=clip_far,texture=texture_img, mode='rgb+depth', random_light=False)


            #cv2.imshow('rgbx',rgb_x)
            #cv2.imshow('rgby',rgb_y)
            #cv2.waitKey()

            bgr_x = rgb_x.copy()
            bgr_y = rgb_y.copy()
            for cc in range(0, 3):
                bgr_x[:,:,cc]=rgb_x[:,:,2-cc]
                bgr_y[:,:,cc]=rgb_y[:,:,2-cc]

            ys, xs = np.nonzero(depth_x > 0)
            try:
                obj_bb = view_sampler.calc_2d_bbox(xs, ys, render_dims)
            except ValueError as e:
                print('Object in Rendering not visible. Have you scaled the vertices to mm?')
                break

            x, y, w, h = obj_bb

            rand_trans_x = np.random.uniform(-crop_offset_sigma, crop_offset_sigma)
            rand_trans_y = np.random.uniform(-crop_offset_sigma, crop_offset_sigma)

            size = int(np.maximum(h, w) * pad_factor)
            left = int(x + w / 2 - size / 2 + rand_trans_x)
            right = int(x + w / 2 + size / 2 + rand_trans_x)
            top = int(y + h / 2 - size / 2 + rand_trans_y)
            bottom = int(y + h / 2 + size / 2 + rand_trans_y)


            bgr_x = bgr_x[top:bottom, left:right]
            depth_x = depth_x[top:bottom, left:right]
            bgr_x = cv2.resize(bgr_x, (W, H), interpolation=cv2.INTER_NEAREST)
            depth_x = cv2.resize(depth_x, (W, H), interpolation=cv2.INTER_NEAREST)

            mask_x = depth_x == 0.

            ys, xs = np.nonzero(depth_y > 0)
            obj_bb = view_sampler.calc_2d_bbox(xs, ys, render_dims)

            bgr_y = data_utils.extract_square_patch(bgr_y, obj_bb, pad_factor, resize=(W, H), interpolation=cv2.INTER_NEAREST)
            depth_y = data_utils.extract_square_patch(depth_y, obj_bb, pad_factor, resize=(W, H), interpolation=cv2.INTER_NEAREST)
            mask_y = depth_y==0


            self.bgr_x[i] = bgr_x
            self.mask_x[i] = mask_x
            self.bgr_y[i]= bgr_y
            self.mask_y[i] = mask_y
            self.matrix_rot_y[i]=R


            if i%100==0:
                path_out_dir=os.path.join(self.dir_dataset,'imgs')
                if not os.path.exists(path_out_dir):
                    os.makedirs(path_out_dir)
                cv2.imwrite(os.path.join(path_out_dir,'{0}_{1}_x_bgr.png'.format(batch_id,i)),self.bgr_x[i])
                cv2.imwrite(os.path.join(path_out_dir,'{0}_{1}_y_bgr.png'.format(batch_id,i)),self.bgr_y[i])
        bar.finish()
        np.savez(current_file_name, bgr_x=self.bgr_x, mask_x=self.mask_x, bgr_y=self.bgr_y, mask_y=self.mask_y, matrix_rot_y=self.matrix_rot_y)


if __name__=='__main__':
    #Rendered by batch, with batch size=50
    model_id=int(sys.argv[1])
    bid=int(sys.argv[2])
    render_model=Model(dir_dataset='./ws/tmp_datasets/{:02d}'.format(model_id),
                       path_model='./ws/meshes/obj_{:02d}.ply'.format(model_id),
                       saved_file_name='prepared_training_data_{:02d}_subdiv'.format(model_id),
                       num_per_batch=50)
    path_texture=None
    texture_img_rgb=None
    if path_texture:
        texture_img_bgr=cv2.imread(path_texture['{:02d}'.format(model_id)])
        texture_img_rgb=texture_img_bgr[:,:,2::-1]
    if True:
        render_model.render_batch_training_images(render_dims=(720,540),
                                                  cam_K=[1075.65, 0, 720 / 2, 0, 1073.90, 540 / 2, 0, 0, 1],
                                                  batch_id=bid,depth_scale=1.,texture_img=texture_img_rgb)
    else:
        render_model.combine_rendered_batches(400)#Combine all generated data into one .npz file
        render_model.load_training_images()#This step is for double check
