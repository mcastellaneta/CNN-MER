from pyflow import pyflow
import numpy as np
import cv2
import constants as const
import pandas as pd


def pol2cart(rho, phi):
    # Convert polar coordinates to cartesian coordinates for computation of optical strain
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return (x, y)


def computeStrain(u, v):
    # Compute os , setting t=1 to maximize the sensitivity of ME
    u_x = u - pd.DataFrame(u).shift(1, axis=1)
    v_y = v - pd.DataFrame(v).shift(1, axis=0)
    u_y = u - pd.DataFrame(u).shift(1, axis=0)
    v_x = v - pd.DataFrame(v).shift(1, axis=1)
    os = np.array(
        np.sqrt(
            (u_x**2).fillna(0) +
            (v_y**2).fillna(0) +
            1/2 * (u_y.fillna(0) + v_x.fillna(0)) ** 2
        )
    )
    return os


def calc_flow_pyflow(prvs, next, py_parameters, logger=None):

    """ Calculate optical flow with Pyflow.

        Args:
          prvs: First frame
          next: Second frame

        Returns:
          Optic flow of the two frames
    """

    (pyflow_alpha, pyflow_ratio, pyflow_minWidth,
     pyflow_nOuterFPIterations, pyflow_nInnerFPIterations,
     pyflow_nSORIterations, pyflow_colType) = py_parameters.values()

    if logger:
        logger.info(f'''
        Pyflow OF parameters:
            alpha:{pyflow_alpha},
            ratio:{pyflow_ratio},
            minWidth:{pyflow_minWidth},
            nOuterFPIterations:{pyflow_nOuterFPIterations},
            nInnerFPIterations:{pyflow_nInnerFPIterations},
            nSORIterations:{pyflow_nSORIterations},
            colType:{pyflow_colType}
            ''')

    im1 = prvs.astype(float) / 255.
    im2 = next.astype(float) / 255.


    u, v, im2W = pyflow.coarse2fine_flow(
        im1, im2, pyflow_alpha, pyflow_ratio, pyflow_minWidth,
        pyflow_nOuterFPIterations, pyflow_nInnerFPIterations,
        pyflow_nSORIterations, pyflow_colType
    )

    flow_py = np.concatenate((u[..., None], v[..., None]), axis=2)


    # hsv_py = np.zeros_like(face)
    hsv_py = np.zeros((next.shape[0], next.shape[1], 3)).astype(np.uint8)
    hsv_py[..., 1] = 255


    mag_py, ang_py = cv2.cartToPolar(flow_py[..., 0], flow_py[..., 1])
    hsv_py[..., 0] = ang_py * 180 / np.pi / 2
    hsv_py[..., 2] = cv2.normalize(mag_py, None, 0, 255, cv2.NORM_MINMAX)
    bgr_py = cv2.cvtColor(hsv_py, cv2.COLOR_HSV2BGR)
    gr1_py = cv2.cvtColor(bgr_py, cv2.COLOR_BGR2GRAY)

    return gr1_py


def calc_flow_farneback(prvs, next, parameters, flow=None, logger=None):

    """ Calculate optical flow with Farneback method.

    Args:
      prvs: First frame
      next: Second frame

    Returns:
      Optic flow of the two frames
    """

    (WINSIZE, PYR_SCALE, POLY_SIGMA,
     POLY_N, LEVELS, ITERATIONS, FLAGS) = parameters.values()

    if logger:
        logger.info(f'''
        Farneback OF parameters:
            pyr_scale:{PYR_SCALE},
            levels:{LEVELS},
            winsize:{WINSIZE},
            iterations:{ITERATIONS},
            poly_n:{POLY_N},
            poly_sigma:{POLY_SIGMA},
            flags:{FLAGS}
        ''')

    flow = cv2.calcOpticalFlowFarneback(
        prev=prvs,
        next=next,
        flow=flow,
        pyr_scale=PYR_SCALE,
        levels=LEVELS,
        winsize=WINSIZE,
        iterations=ITERATIONS,
        poly_n=POLY_N,
        poly_sigma=POLY_SIGMA,
        flags=FLAGS
    )
    # cv2.imshow('Draw', draw_flow(prvs, flow))
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])

    hsv = np.zeros((next.shape[0], next.shape[1], 3)).astype(np.uint8)
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)

    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    return bgr


def calc_flow_TVL1(prvs, next, parameters, logger=None):
    (warps, useInitialFlow, theta, tau, scaleStep, outerIterations,
     nscales, medianFiltering, lambda_, innerIterations,
     gamma, epsilon) = parameters.values()

    optical_flow = cv2.optflow.DualTVL1OpticalFlow_create(
            tau, lambda_, theta, nscales, warps,
            epsilon, innerIterations,
            outerIterations, scaleStep,
            gamma, medianFiltering, useInitialFlow
        )

    flow = optical_flow.calc(prvs, next, None)
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    u, v = pol2cart(mag, ang)
    os = computeStrain(u, v)

    final = np.zeros((next.shape[0], next.shape[1], 3)).astype(np.float32)
    final[..., 0] = u
    final[..., 1] = v
    final[..., 2] = os

    for channel in range(3):
        final[..., channel] = cv2.normalize(
            final[..., channel], None, alpha=0, beta=255,
            norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1
        )
    final = final.astype(np.uint8)
    return final


def draw_flow(img, flow, step=16):
    # https://www.youtube.com/watch?v=WrlH5hHv0gE
    # https://github.com/niconielsen32/ComputerVision/blob/master/opticalFlow/denseOpticalFlow.py
    h, w = img.shape[:2]
    y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2, -1).astype(int)
    fx, fy = flow[y, x].T

    lines = np.vstack([x, y, x-fx, y-fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)

    img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.polylines(img_bgr, lines, 0, (0, 255, 0))

    for (x1, y1), (_x2, _y2) in lines:
        cv2.circle(img_bgr, (x1, y1), 1, (0, 255, 0), -1)

    return img_bgr