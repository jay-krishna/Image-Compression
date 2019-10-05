import numpy as np
import cv2
# from chroma_subsampling.chroma_subsampling import ChromaSubsampler
class ChromaSubsampler(object):
    def __init__(self, a, b, J=4):
        super(ChromaSubsampler, self).__init__()
        self._J = J
        if a == 4 and b == 4:
            (col_step, row_step) = (1, 1)
        elif a == 4 and b == 0:
            (col_step, row_step) = (1, 2)
        elif a == 2 and b == 2:
            (col_step, row_step) = (2, 1)
        elif a == 2 and b == 0:
            (col_step, row_step) = (2, 2)
        elif a == 1 and b == 1:
            (col_step, row_step) = (4, 1)
        elif a == 1 and b == 0:
            (col_step, row_step) = (4, 2)
        else:
            raise ValueError('Invalid value for argument "a" or "b"')
        self._col_step = col_step
        self._row_step = row_step

    def encode(self, image):
        encoded = image[::self._col_step, ::self._row_step]
        return encoded

    def decode(self, encoded):
        decoded_0 = np.repeat(encoded, self._col_step, axis=0)
        decoded = np.repeat(decoded_0, self._row_step, axis=1)
        return decoded


def build_ycrcb_test(J, a, b):
    def test(image):
        image_ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCR_CB)
        (y, cr, cb) = (image_ycrcb[:,:,0], image_ycrcb[:,:,1], image_ycrcb[:,:,2])
        sampler = ChromaSubsampler(J=J, a=a, b=b)
        cr_encoded = sampler.encode(cr)
        cb_encoded = sampler.encode(cb)
        print(cr_encoded.shape)
        print(cb_encoded.shape)
        encoded_size = y.nbytes + cr_encoded.nbytes + cb_encoded.nbytes

        # temp_ycrcb = np.dstack((y, cr_encoded, cb_encoded))
        # temp_bgr = cv2.cvtColor(temp_ycrcb,  cv2.COLOR_YCR_CB2BGR)
        # test_name = '{}:{}:{}'.format(J, a, b)
        # cv2.imwrite( test_name+'.jpg', temp_bgr) 


        cr_decoded = sampler.decode(cr_encoded)
        cb_decoded = sampler.decode(cb_encoded)
        print(cr_decoded.shape)
        print(cb_decoded.shape)
        print(y.shape)
        print("Decoded")
        decoded_ycrcb = np.dstack((y, cr_decoded, cb_decoded))
        decoded_bgr = cv2.cvtColor(decoded_ycrcb,  cv2.COLOR_YCR_CB2BGR)
        return (decoded_bgr, encoded_size)
    return test


def build_rgb_test(J, a, b):
    def test(image):
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        (rc, gc, bc) = (image_rgb[:,:,0], image_rgb[:,:,1], image_rgb[:,:,2])
        sampler = ChromaSubsampler(J=J, a=a, b=b)
        r_encoded = sampler.encode(rc)
        g_encoded = sampler.encode(gc)
        b_encoded = sampler.encode(bc)
        encoded_size = r_encoded.nbytes + g_encoded.nbytes + b_encoded.nbytes
        r_decoded = sampler.decode(r_encoded)
        g_decoded = sampler.decode(g_encoded)
        b_decoded = sampler.decode(b_encoded)

        print("Sizes")
        print(r_decoded.shape)
        print(g_decoded.shape)
        print(b_decoded.shape)

        decoded_rgb = np.dstack((r_decoded, g_decoded, b_decoded))
        print(decoded_rgb.shape)
        decoded_bgr = cv2.cvtColor(decoded_rgb,  cv2.COLOR_RGB2BGR)
        return (decoded_bgr, encoded_size)
    return test


if __name__ == '__main__':

    image=cv2.imread('/home/singular/img/3.jpg')
    image = cv2.resize(image, (256, 256))
    image_size = image.nbytes
    print('Original size (bytes): {}'.format(image_size))
    # image = cv2.resize(image, (256, 256))
    image_size = image.nbytes

    # ratios = [(4, 4, 4), (4, 4, 0), (4, 2, 2), (4, 2, 0), (4, 1, 1), (4, 1, 0)]
    ratios = [(4,2,2)]
    for (J, a, b) in ratios:
        ycrcb_test = build_ycrcb_test(J, a, b)
        rgb_test = build_rgb_test(J, a, b)
        (ycrcb_result, ycrcb_enconded_size) = ycrcb_test(image)
        (rgb_result, rgb_enconded_size) = rgb_test(image)
        ycrcb_enconded_ratio = ycrcb_enconded_size / image_size
        rgb_enconded_ratio = rgb_enconded_size / image_size
        result = np.hstack((image, ycrcb_result, rgb_result))
        test_name = '{}:{}:{}'.format(J, a, b)
        print(image.shape)
        print(ycrcb_result.shape)
        print(rgb_result.shape)
        cv2.imwrite('/home/singular/img/8.jpg',ycrcb_result)
        # cv2.imshow(test_name, result)
        print('TEST FOR {} -----'.format(test_name))
        print('YCrCb Encoded size (bytes): {} ({}%)'.format(ycrcb_enconded_size, ycrcb_enconded_ratio * 100))
        print('RGB Encoded size (bytes): {} ({}%)'.format(rgb_enconded_size, rgb_enconded_ratio * 100))
        print('')
        # char = cv2.waitKey(0)
  
