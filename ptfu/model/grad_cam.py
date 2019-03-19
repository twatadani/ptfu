''' grad_cam.py: Grad-Cam (Selvaraju et al 2016) を実装する '''

import tensorflow as tf

class GradCam:
    ''' GradCam '''

    def __init__(self, fully_connected_tensor, last_conv_tensor):
        ''' fully_connected_tensor: softmax直前のrank 1 tensor
        last_conv_tensor: back propagationの到達点となる最終convolutionのoutput '''
        self.fc = fully_connected_tensor
        self.lastconv = last_conv_tensor

        # 2回定義しないように
        self.defined = False

        self.define_gradcam()

        return


    def define_gradcam(self):
        ''' grad-camの各tensorを定義する '''

        if self.defined == True:
            return

        fcshape = self.fc.shape # (?, nclasses)のはず
        nclasses = fcshape[1].value

        with tf.name_scope('grad_cam'):
            argmax = tf.argmax(self.fc, axis=1)
            #print('argmax shape:', argmax.shape)
            #one_hot = tf.sparse_to_dense(sparse_indices = argmax,
            #output_shape = [argmax.shape[0].value, nclasses],
                                         #sparse_values = 1.0,
                                         #name = 'grad_cam_onehot')
            one_hot = tf.one_hot(indices = argmax,
                                 depth = nclasses,
                                 on_value = 1.0,
                                 off_value = 0.0,
                                 dtype = self.fc.dtype,
                                 name = 'grad_cam_onehot')
            #print('one_hot shape:', one_hot.shape)
            #print('self.fc shape:', self.fc.shape)
            mul = one_hot * self.fc
            #print('mul.shape:', mul.shape)
            loss = tf.reduce_mean(mul,
                                  name = 'grad_cam_loss')
            #print('loss shape:', loss.shape)
            grads = tf.gradients(ys = loss,
                                 xs = self.lastconv,
                                 name = 'grad_cam_grads')[0]
            norm_grads = tf.div(grads, tf.sqrt(tf.reduce_mean(tf.square(grads))) + 1e-6,
                                name = 'grad_cam_normgrads') # 論文のαに相当
            #print('norm_grads shape:', norm_grads.shape)
            weights = tf.reduce_mean(norm_grads, axis=(1, 2),
                                     name = 'grad_cam_weights') # (?, channels)
            #print('weights shape:', weights.shape)
            nchannles = weights.shape[1].value
            gradcams = []
            for channel in range(nchannles):
                channelwise_w = weights[:, channel] # (?,)
                #print('channelwise_w.shape:', channelwise_w.shape)
                #assert channelwise_w.shape[1] == 1
                # self.lastconv.shape == (?, x, y, nchannels)

                exp1 = tf.expand_dims(channelwise_w, axis=1) #(?, 1)
                catlist1 = [exp1 for _ in range(self.lastconv.shape[1].value)]
                cat1 = tf.concat(catlist1, axis=1)

                exp2 = tf.expand_dims(exp1, axis=2) #(?, x, 1)
                catlist2 = [exp2 for _ in range(self.lastconv.shape[2].value)]
                cat2 = tf.concat(catlist2, axis=2) # (?, x, y)
                #print('cat2 shape:', cat2.shape)

                channelwise_lastconv = self.lastconv[:, :, :, channel] #(?, x, y)
                #print('channelwise_lastconv shape:', channelwise_lastconv.shape)
                #indices = tf.zeros_like(channelwise_lastconv, dtype=tf.int32) # (?, x, y)
                #print('indices shape:', indices.shape)
                #w_expanded = tf.batch_gather(exp2, indices)
                #print('w_expanded shape:', w_expanded.shape)

                channelwise_gradcam = cat2 * channelwise_lastconv # (?, x, y)
                #print('channelwise_gradcam shape:', channelwise_gradcam.shape)

                # broadcastingでこの計算でいける? →いけなかった
                #channelwise_gradcam = channelwise_w * self.lastconv[:, :, :, channel] #(?, x, y)
                #assert channelwise_gradcam.shape[1] == 1
                
                gradcams.append(tf.expand_dims(channelwise_gradcam,
                                               axis = 3))
            pre_relu = tf.reduce_mean(tf.concat(gradcams, axis=3), # (?, x, y, nchannels)
                                    axis=-1) #(?, x, y)
            self.grad_cam = tf.nn.relu(pre_relu,
                                       name = 'grad_cam')
        self.defined = True
        return

    @staticmethod
    def gradcam2heatmap(ndarray, size):
        ''' grad-camのndarrayからRGB heatmapのPIL Imageに変換する。
        sizeは拡大後の最終サイズ '''
        from PIL import Image
        import ptfu.functions as f
        # 0-1にremapする
        rgb = f.create_colormap(ndarray) # uint8 (x, y, 3)
        img = Image.fromarray(rgb, mode='RGB')
        resized = img.resize(size, resample=Image.BICUBIC)
        return resized


name = 'grad_cam'
                                         
            
            
