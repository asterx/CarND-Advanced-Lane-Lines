# coding=utf-8
import cv2
import glob
import numpy as np
from utils.utils import not_none

DEFAULT_COLOR = (255, 0, 0) # RED

objp = np.zeros((9*6, 3), np.float32)
objp[:,:2] = np.mgrid[0:9, 0:6].T.reshape(-1,2)


def undistort(img, mtx, dist):
    return cv2.undistort(img, mtx, dist, None, mtx)


def get_img_size(img):
    return img.shape[1::-1]


def get_img_corners(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(img, (9, 6), None)
    if ret:
        return corners


def get_mtx_dist(path):
    imgs = [ cv2.imread(p) for p in glob.glob(path) ]
    img_size = get_img_size(imgs[0])
    imgpoints = list(filter(not_none, [ get_img_corners(img) for img in imgs ]))
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        [ objp ] * len(imgpoints),
        imgpoints,
        img_size,
        None,
        None
    )
    return mtx, dist


def get_m_minv(src, dst):
    src_pts = np.float32(src)
    dst_pts = np.float32(dst)
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    Minv = cv2.getPerspectiveTransform(dst_pts, src_pts)
    return M, Minv


def prepare(path, mtx, dist):
    return undistort(cv2.imread(path), mtx, dist)


def warp(img, M):
    return cv2.warpPerspective(img, M, get_img_size(img), flags=cv2.INTER_LINEAR)


def unwarp(img, Minv):
    return cv2.warpPerspective(img, Minv, get_img_size(img), flags=cv2.INTER_LINEAR)


def process(img, y_thresholded = 200, u_v_thresholded = 30):
    yuv_img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)

    y = yuv_img[:,:,0]
    u = yuv_img[:,:,1]
    v = yuv_img[:,:,2]

    y_binary = np.zeros_like(y)
    y_binary[y > y_thresholded] = 1

    u_v_diff = u.astype(np.int16) - v.astype(np.int16)

    u_v_diff_binary = np.zeros_like(u_v_diff)
    u_v_diff_binary[u_v_diff > u_v_thresholded] = 1
    u_v_diff_binary = np.uint8(u_v_diff_binary)

    result = np.zeros_like(y)
    result[((u_v_diff_binary == 1) | (y_binary == 1))] = 1

    return result


def region_mask(img, left_margin = 300, right_margin = 300):
    masked_img = np.copy(img)
    masked_img[:, 0:left_margin] = 0
    masked_img[:, get_img_size(img)[0]-right_margin:] = 0
    return masked_img


def get_histogram(img):
    return np.sum(img[get_img_size(img)[1]//2:, :], axis=0)


def dstack(img):
    return np.dstack((img,) * 3) * 255


def find_lane_lines(img, xm_per_px, ym_per_px, num_windows = 12, margin = 80, minpix = 50, draw = True, color = DEFAULT_COLOR):
    warped = dstack(img)
    img_size = get_img_size(img)
    histogram = get_histogram(img)
    img_middle = img_size[0]//2
    window_height = img_size[1]//num_windows
    result_img = None

    # Calculate window dimensions
    L_W_C = np.argmax(histogram[:img_middle])
    R_W_C = np.argmax(histogram[img_middle:]) + img_middle
    L_L_INDS, R_L_INDS = [], []

    n_zero_y, n_zero_x = list(map(np.array, img.nonzero()))

    if draw:
        result_img = dstack(img)

    for i in range(num_windows):
        # Calculate boundaries
        L_L_L_B = L_W_C - margin
        L_L_R_B = L_W_C + margin
        R_L_L_B = R_W_C - margin
        R_L_R_B = R_W_C + margin
        U_B = img_size[1] - i * window_height
        L_B = U_B - window_height

        # Draw rectangle
        if draw:
            cv2.rectangle(result_img, (L_L_L_B, L_B), (L_L_R_B, U_B), (0,255,0), 3)
            cv2.rectangle(result_img, (R_L_L_B, L_B), (R_L_R_B, U_B),(0,255,0), 3)

        L_W_INDS = ((n_zero_x > L_L_L_B) & (n_zero_x < L_L_R_B) & (n_zero_y > L_B) & (n_zero_y < U_B)).nonzero()[0]
        R_W_INDS = ((n_zero_x > R_L_L_B) & (n_zero_x < R_L_R_B) & (n_zero_y > L_B) & (n_zero_y < U_B)).nonzero()[0]

        L_L_INDS.append(L_W_INDS)
        R_L_INDS.append(R_W_INDS)

        if len(L_W_INDS) > minpix:
            L_W_C = np.mean(n_zero_x[L_W_INDS]).astype(np.int)
        if len(R_W_INDS) > minpix:
            R_W_C = np.mean(n_zero_x[R_W_INDS]).astype(np.int)

    L_L_INDS = np.concatenate(L_L_INDS)
    R_L_INDS = np.concatenate(R_L_INDS)
    L_L_X = n_zero_x[L_L_INDS]
    L_L_Y = n_zero_y[L_L_INDS]
    R_L_X = n_zero_x[R_L_INDS]
    R_L_Y = n_zero_y[R_L_INDS]

    if draw:
        result_img[L_L_Y, L_L_X] = result_img[R_L_Y, R_L_X] = color

    # Polynoms
    L_L_CF_PX = np.polyfit(L_L_Y, L_L_X, 2)
    R_L_CF_PX = np.polyfit(R_L_Y, R_L_X, 2)
    L_L_CF_M = np.polyfit(L_L_Y * ym_per_px, L_L_X * xm_per_px, 2)
    R_L_CF_M = np.polyfit(R_L_Y * ym_per_px, R_L_X * xm_per_px, 2)

    return result_img, L_L_CF_PX, R_L_CF_PX, L_L_CF_M, R_L_CF_M


def fill_lane(img, Minv, L_L_CF_PX, R_L_CF_PX):
    img_size = get_img_size(img)
    fill_image = np.zeros_like(img)
    plot_y = np.linspace(0, img_size[1], img_size[1]+1)
    left_plot_x = L_L_CF_PX[0] * plot_y**2 + L_L_CF_PX[1] * plot_y + L_L_CF_PX[2]
    right_plot_x = R_L_CF_PX[0] * plot_y**2 + R_L_CF_PX[1] * plot_y + R_L_CF_PX[2]

    left_poly_points = np.vstack((left_plot_x, plot_y)).T
    right_poly_points = np.flipud(np.vstack((right_plot_x, plot_y)).T)

    poly_points = np.vstack((left_poly_points, right_poly_points))
    poly_points = np.expand_dims(poly_points, axis=0).astype(np.int_)

    cv2.fillPoly(fill_image, poly_points, (0, 255, 0))

    fill_image = unwarp(fill_image, Minv)
    return cv2.addWeighted(img, 1, fill_image, 0.3, 0)


def calculate_curve_radius(L_L_CF_M, R_L_CF_M, img_size, ym_per_px):
    y = img_size[1] * ym_per_px
    return int((
        (1 + (2 * L_L_CF_M[0] * y + L_L_CF_M[1])**2)**1.5 / np.absolute(2 * L_L_CF_M[0]) +
        (1 + (2 * R_L_CF_M[0] * y + R_L_CF_M[1])**2)**1.5 / np.absolute(2 * R_L_CF_M[0])
    ) // 2)


def calculate_off_center(L_L_CF_PX, R_L_CF_PX, img_size, xm_per_px):
    y = img_size[1]
    leftx = L_L_CF_PX[0] * y**2 + L_L_CF_PX[1] * y + L_L_CF_PX[2]
    rightx = R_L_CF_PX[0] * y**2 + R_L_CF_PX[1] * y + R_L_CF_PX[2]
    return (img_size[0]/2 - (leftx + rightx) / 2) * xm_per_px


def stats(img, L_L_CF_M, R_L_CF_M, L_L_CF_PX, R_L_CF_PX, xm_per_px, ym_per_px, color = DEFAULT_COLOR):
    img_size = get_img_size(img)
    radius = calculate_curve_radius(L_L_CF_M, R_L_CF_M, img_size, ym_per_px)
    off = calculate_off_center(L_L_CF_PX, R_L_CF_PX, img_size, xm_per_px)
    curve_radius_text = 'Curve radius: {}m'.format(radius)
    off_center_text = 'Off center: {:.2f}m'.format(off)
    cv2.putText(img, curve_radius_text, (100, 100), cv2.FONT_HERSHEY_PLAIN, 3, color, 3)
    cv2.putText(img, off_center_text, (100, 200), cv2.FONT_HERSHEY_PLAIN, 3, color, 3)
    return img
