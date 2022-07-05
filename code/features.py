import cv2 as cv
import numpy as np
from scipy.signal import convolve2d
from sklearn.metrics import mean_squared_error
import math

# ======== Feature Detector ======== #
def get_corner_response(Sxx, Syy, Sxy, k=0.04):  # k should be between 0.04-0.06
    det = Sxx * Syy - Sxy * Sxy
    trace = Sxx + Syy
    return det - k * (trace ** 2)

def harris_corner_detector(src, num_features, k_size = 11, anms='SSC'):
    gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)

    # use Sobel or Scharr operator to calculate image derivatives
    # when kernel size = 3, scharr operator gives a more accurate estimation
    if k_size == 3:
        scharr_x = np.array([[-3,  0,  3],
                             [-10, 0, 10],
                             [-3,  0,  3]])
        scharr_y = np.transpose(scharr_x)

        scharr_Ix = convolve2d(gray, scharr_x, mode='same', boundary='symm')
        scharr_Iy = convolve2d(gray, scharr_y, mode='same', boundary='symm')
        Ix = cv.convertScaleAbs(scharr_Ix)  # convert to CV_8U
        Iy = cv.convertScaleAbs(scharr_Iy)

    else:
        Ix = cv.Sobel(gray, 6, 1, 0, ksize=k_size)
        Iy = cv.Sobel(gray, 6, 0, 1, ksize=k_size)
    
    # calculate product of derivatives
    Ixx = Ix ** 2
    Iyy = Iy ** 2
    Ixy = Ix * Iy

    # apply Gaussian filter
    gauss_kernel = np.array([[1, 2, 1],
                            [2, 4, 2], 
                            [1, 2, 1]])/16
    Sxx = cv.filter2D(Ixx, -1, gauss_kernel)
    Syy = cv.filter2D(Iyy, -1, gauss_kernel)
    Sxy = cv.filter2D(Ixy, -1, gauss_kernel)

    # get corner response of each pixel
    corner_response = get_corner_response(Sxx, Syy, Sxy)

    r_threshold = 10
    y, x = np.where(corner_response >= r_threshold)
    valid_response = corner_response[y, x]

    # sort by response score in descending order
    loc_score = np.stack([y, x, valid_response], axis=0)
    sorted = loc_score[:, (-loc_score[2]).argsort()]
    sorted_keypoints = np.transpose(sorted[:2])  # shape: (num_valid_response, 2), num_valid_response pairs of (y, x)
    sorted_response = sorted[2]

    # apply Adaptive Non-Maximal Suppression to get well distributed feature points
    if anms == 'original':
        features = np.transpose(ANMS(sorted_keypoints, sorted_response)[:num_features])
    elif anms == 'SSC':
        selected = ANMS_SSC(sorted_keypoints, num_features, src.shape[1], src.shape[0])[:num_features]
        features = np.transpose(sorted_keypoints[selected])
    else:  # don't perform ANMS
        features = np.transpose(sorted_keypoints[:num_features])

    return features

# adaptive non-maximal suppression -- suppression via square covering
# by Bailo et al.(2018) "Efficient adaptive non-maximal suppression algorithms for homogeneous spatial keypoint distribution"

# RETURN: index of keypoints (shape: (y, x)) to choose
def ANMS_SSC(keypoints, target_num, w, h):

    # initialize binary search boundaries
    discriminant = 4*w + 4*target_num + 4*h*target_num + h**2 + w**2 - 2*w*h + 4*w*h*target_num
    exp1 = h + w + 2 * target_num
    exp2 = math.sqrt(discriminant)
    exp3 = 2 * (target_num - 1)

    sol1 = -float(exp1 + exp2) / exp3
    sol2 = -float(exp1 - exp2) / exp3

    high = sol1 if (sol1 > sol2) else sol2
    low = math.floor(0.5 * math.sqrt(keypoints.shape[0] / target_num))

    # binary search for ANMS keypoints
    complete = False
    prev_width = -1
    result = []
    result_lst = []
    
    while not complete:
        width = low + (high - low) / 2
        if width == prev_width or low > high:
            result_lst = result
            break

        grid_size = width / 2
        num_cell_w = int(math.floor(w / grid_size))
        num_cell_h = int(math.floor(h / grid_size))
        covered = np.zeros((num_cell_w + 1, num_cell_h + 1))

        # pick the next highest response that is not inside previous cells
        for i in range(keypoints.shape[0]):
            row = int(math.floor(keypoints[i, 0] / grid_size))
            col = int(math.floor(keypoints[i, 1] / grid_size))
            if not covered[col, row]:
                result.append(i)

                # update covered area
                row_min = row - 2
                row_max = row + 2
                col_min = col - 2
                col_max = col + 2

                if row_min < 0:
                    row_min = 0
                if row_max > num_cell_h:
                    row_max = num_cell_h
                if col_min < 0:
                    col_min = 0
                if col_max > num_cell_w:
                    col_max = num_cell_w

                covered[col_min:col_max + 1, row_min:row_max + 1] = 1


        if len(result) == target_num:
            result_lst = result
            complete = True
        elif len(result) < target_num:
            high = width - 1
        else:
            low = width + 1
    
    return result_lst

# adaptive non-maximal suppression
def ANMS(keypoints, response, robust_coef = 1.11):
    num_valid_response = response.shape[0]
    radius = np.zeros(num_valid_response)
    radius[0] = np.inf

    for i in range(1, num_valid_response):
        y = keypoints[i, 0]
        x = keypoints[i, 1]
        neighbor_response = response[:i] * robust_coef
        candidate_idx = np.where(neighbor_response > response[i])

        delta_y = keypoints[candidate_idx, 0] - y
        delta_x = keypoints[candidate_idx, 1] - x
        radius[i] = np.min(np.sqrt(delta_x ** 2 + delta_y ** 2))

    sorted_idx = (-radius).argsort()

    return keypoints[sorted_idx, :]

# ======== Feature Descriptor ======== #
def feature_desciptor(img, feat_lst):
    # sub pixel refinement
    # orientation assignment by blurred gradient
    src = cv.GaussianBlur(img, (3, 3), 4.5, 4.5)
    gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)

    feat_x = feat_lst[1].astype(int)
    feat_y = feat_lst[0].astype(int)

    Ix = cv.Sobel(gray, cv.CV_64F, 1, 0, ksize=3)
    Iy = cv.Sobel(gray, cv.CV_64F, 0, 1, ksize=3)
    vec_len = np.sqrt(Ix ** 2 + Iy ** 2 + 0.0001)
    cos = (Ix / vec_len)[feat_y, feat_x]
    sin = (Iy / vec_len)[feat_y, feat_x]

    # 8x8 intensity patch as desciptor
    rows = img.shape[0]
    cols = img.shape[1]
    patch_lst = []
    wavelet_lst = []
    for i in range(len(feat_x)):
        pt_x = feat_x[i]
        pt_y = feat_y[i]
        pt_cos = cos[i]
        pt_sin = sin[i]

        if pt_y-20 < 0 or pt_y+20 > rows or pt_x-20 < 0 or pt_x+20 > cols:
            wavelet_lst.append(np.zeros(3))
            patch_lst.append(np.zeros((8,8)))  # black patch
            continue

        # rotation matrix, centering at (pt_x, pt_y)
        trans_mat = np.float64([[pt_cos,  pt_sin, (1-pt_cos) * pt_x - pt_sin * pt_y],
                                [-pt_sin, pt_cos, pt_sin * pt_x + (1-pt_cos) * pt_y]])
        
        # patch = cv.warpAffine(gray, trans_mat, (cols, rows))[pt_y-20:pt_y+20:5, pt_x-20:pt_x+20:5]
        patch = np.average(img[pt_y-20:pt_y+20:5, pt_x-20:pt_x+20:5], axis=2)
        i_mean = np.mean(patch)
        i_std = np.std(patch) + 0.0001
        patch = (patch - i_mean) / i_std

        patch_lst.append(patch)
        wavelet_lst.append(get_wavelet(patch))

    return patch_lst, wavelet_lst

def get_wavelet(patch):
    wavelet = np.zeros((3))
    wavelet[0] = np.sum(patch[:, 4:]) - np.sum(patch[:, :4])  # right - left
    wavelet[1] = np.sum(patch[:4, :]) - np.sum(patch[4:, :])  # up - down
    wavelet[2] = np.sum(patch[:4, 4:]) + np.sum(patch[4:, :4]) - np.sum(patch[:4, :4]) - np.sum(patch[4:, 4:]) 
    # top-right + bottom-left - top-left - bottom-right
    return wavelet

# ======== Feature Matching ======== #
def brute_force_matching(src1, src2, thres_ratio=0.8):
    # find the best match and second best match
    # accept the best match only if the best match is much better than second best

    matches = []
    for i in range(len(src1)):
        mse = []
        for j in range(len(src2)):
            mse.append(mean_squared_error(src1[i], src2[j]))
        order = np.argsort(np.array(mse))
        dist_ratio = mse[order[0]] / mse[order[1]]
        if dist_ratio < thres_ratio:
            matches.append([i, order[0]])

    return np.transpose(matches)

# ======== Main Function ======== #
def find_keypoint(img_lst, downsample=None, num_level=1):
    if downsample is None:
        if img_lst[0].shape[0] > 2000:
            downsample = 4
        elif img_lst[0].shape[0] > 1000:
            downsample = 2
        else:
            downsample = 1
    
    # Image downsampling
    img_dwnspl_lst = []
    for img in img_lst:
        img_dwnspl_lst.append(img[::downsample, ::downsample, :])
    
    # Image Pyramid
    print("Detecting keypoints ...")
    feature_lst = []
    pyramid_lst = []

    for i in range(len(img_dwnspl_lst)):
        img_pyramid = []
        pyramid_feat_lst = []

        prev_level = img_dwnspl_lst[i]
        feat = (harris_corner_detector(prev_level, 250, k_size = 11, anms='SSC'))  # k_size=11
        img_pyramid.append(prev_level)
        pyramid_feat_lst.append(feat)

        for lvl in range(1, num_level):
            prev_level = cv.GaussianBlur(prev_level, (3, 3), 1, 1)
            new_level = prev_level[::2, ::2]
            img_pyramid.append(new_level)
            pyramid_feat_lst.append(harris_corner_detector(new_level, 250, k_size = 11, anms='SSC'))
            prev_level = new_level

        pyramid_lst.append(img_pyramid)
        feature_lst.append(pyramid_feat_lst)

    # Feature descriptor
    patch_lst = []
    for i in range(len(img_lst)):
        img_patch = []
        for lvl in range(num_level):
            patch, _ = feature_desciptor(img_dwnspl_lst[i], feature_lst[i][lvl])
            img_patch.append(patch)
        patch_lst.append(img_patch)
        
    patch_lst = np.array(patch_lst)

    # Feature matching
    print("Matching keypoints ...")

    w = img_lst[0].shape[1]
    half_w = w // 2 // downsample
    points1_lst = []
    points2_lst = []
    for lvl in range(num_level):
        for i in range(len(img_lst)):
            left_ft_idx = np.argwhere(feature_lst[i][lvl][1] <= half_w).flatten()
            right_ft_idx = np.argwhere(feature_lst[i][lvl][1] > half_w).flatten()
            
            if i == 0:
                prev_right_idx = right_ft_idx.copy()
            else:
                feat1 = patch_lst[i-1, lvl, prev_right_idx]
                feat2 = patch_lst[i, lvl, left_ft_idx]
                
                matches = brute_force_matching(feat1, feat2, thres_ratio=0.5)
                prev_match_idx = prev_right_idx[matches[0]]
                curr_match_idx = left_ft_idx[matches[1]]
                # np.save(dataset + '/msop_results/img' + str(i - 1).zfill(1) + '_points1.npy', np.transpose(feature_lst[i-1][lvl][:, prev_match_idx]) * 4)
                # np.save(dataset + '/msop_results/img' + str(i - 1).zfill(1) + '_points2.npy', np.transpose(feature_lst[i][lvl][:, curr_match_idx]) * 4)
                points1_lst.append(np.transpose(feature_lst[i-1][lvl][:, prev_match_idx]) * downsample)
                points2_lst.append(np.transpose(feature_lst[i][lvl][:, curr_match_idx]) * downsample)

                # import matplotlib.pyplot as plt
                # plt.figure(figsize=(24, 20))
                # plt.subplot(1, 2, 1)
                # plt.imshow(img_lst[0])
                # plt.subplot(122)
                # plt.imshow(img_lst[1])

                # img0_points1 = np.transpose(feature_lst[i-1][lvl][:, prev_match_idx]) * downsample
                # img0_points2 = np.transpose(feature_lst[i][lvl][:, curr_match_idx]) * downsample
                # for idx in range(20):
                #     plt.subplot(121)
                #     plt.plot(img0_points1[idx, 1], img0_points1[idx, 0], '+')

                #     plt.subplot(122)
                #     plt.plot(img0_points2[idx, 1], img0_points2[idx, 0], '+')
                # plt.show()
                prev_right_idx = right_ft_idx.copy()
    
    return [points1_lst, points2_lst]