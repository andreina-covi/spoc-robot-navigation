import argparse
import cv2
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description="Online evaluation")
    parser.add_argument("--coordinates", type=tuple, default=(5.85, 2.55))
    parser.add_argument("--image_path", type=str)
    parser.add_argument("--filename", type=str)
    parser.add_argument("--color", type=tuple, default=(0, 0, 255))
    args = parser.parse_args()
    return args

def draw_point(args):
    radius = 1
    thickness = -1
    image = cv2.imread(args.image_path)
    cv2.circle(image, args.coordinates, radius, args.color, thickness)
    cv2.imwrite(args.filename, image)

if __name__ == "__main__":
    # args = parse_args()
    # # coordinates = (5.85, 0.9, 2.55)
    # draw_point(args)

    # Define the camera matrix
    fx = 800
    fy = 800
    cx = 640
    cy = 480
    camera_matrix = np.array([[fx, 0, cx],
                            [0, fy, cy],
                            [0, 0, 1]], np.float32)

    # Define the distortion coefficients
    dist_coeffs = np.zeros((5, 1), np.float32)

    # Define the 3D point in the world coordinate system
    x, y, z = 10, 20, 30
    points_3d = np.array([[[x, y, z]]], np.float32)

    # Define the rotation and translation vectors
    rvec = np.zeros((3, 1), np.float32)
    tvec = np.zeros((3, 1), np.float32)

    # Map the 3D point to 2D point
    points_2d, _ = cv2.projectPoints(points_3d,
                                    rvec, tvec,
                                    camera_matrix,
                                    dist_coeffs)

    # Display the 2D point
    print("2D Point:", points_2d)