import matplotlib.pyplot as plt
import cv2
import numpy as np
import sys, getopt

VID_NAME = str(sys.argv[1])

def get_final_image(flatten_vec, new_flat_0, new_flat_1, new_flat_2, og_shape):
    new_flatten_vec = np.zeros((len(flatten_vec), len(flatten_vec.T)*3))
    for i in range(len(new_flat_0.T)):
        new_flatten_vec[:,i+2*i] = new_flat_0[:,i]
        new_flatten_vec[:,i+1+2*i] = new_flat_1[:,i]
        new_flatten_vec[:,i+2+2*i] = new_flat_2[:,i]

    vec12 = np.matrix(new_flatten_vec[0])
    final_arr = np.asarray(vec12).reshape(og_shape)
    return final_arr

def main():
    for t in range(3):
        counter = 0
        video_capture = cv2.VideoCapture(VID_NAME)
        success, img = video_capture.read()
        og_shape = img.shape
        if t == 0:
            first_img = img
            print("Video Resolution: ", og_shape[0:2])
        arr = img[:,:,t]
        flatten_vec = arr.ravel()
        while success:
            video_capture.set(cv2.CAP_PROP_POS_MSEC, (500*counter))
            success, img = video_capture.read()
            if not success:
                break
            arr = img[:,:,t]
            flatten_vec = np.vstack([flatten_vec, arr.ravel()])
            counter += 1
        print(counter + 1, "Frames taken from Video")
        print("Calculations Ongoing...")
        SVD = np.linalg.svd(flatten_vec.T, full_matrices=False)
        u, s, v = SVD
        temp_arr = np.zeros((len(u), len(v)))
        temp_arr += s[0] * np.outer(u.T[0], v[0])
        if t == 2:
            new_flat_2 = temp_arr.T
        elif t == 1:
            new_flat_1 = temp_arr.T
        elif t == 0:
            new_flat_0 = temp_arr.T 

    final_arr = get_final_image(flatten_vec, new_flat_0, new_flat_1, new_flat_2, og_shape)
    cv2.imwrite(VID_NAME[0:4] + '_bg.jpg', final_arr)

if __name__ == "__main__":
    main()
