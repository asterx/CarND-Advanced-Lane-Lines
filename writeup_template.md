## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./examples/undistort_output.png "Undistorted"
[image2]: ./test_images/test1.jpg "Road Transformed"
[image3]: ./examples/binary_combo_example.jpg "Binary Example"
[image4]: ./examples/warped_straight_lines.jpg "Warp Example"
[image5]: ./examples/color_fit_lines.jpg "Fit Visual"
[image6]: ./examples/example_output.jpg "Output"
[video1]: ./project_video.mp4 "Video"


### Camera Calibration
First step is camera calibration. Irocess is pretty straightforward and looks like this:
1. read multiple calibration images
2. finds all chessboard corners using cv2 API method findChessboardCorners
3. get distortion coefficients using cv2 API method calibrateCamera
See my implementation at ipynb file or here [lines].

Distortion coefficients found on previous steps are using for images undistorting.
Example:
[image]


### Perspective transform
Second step is perspective transform, which allows us to see images as they seen from above.
[describe steps]
[set correct values]

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 585, 460      | 320, 0        | 
| 203, 720      | 320, 720      |
| 1127, 720     | 960, 720      |
| 695, 460      | 960, 0        |

[image before and after]


### Identifying lanes
Third step is identifying the white and yellow lanes. To achieve this we should apply several technickes described below.
1. image converts to YUV colorspace
2. y-channel is thresholded for all values above 200. This allows us to identify white lines
3. difference between u-channel and v-channel is thresholded for all values above 30. This allows us to identify yellow lines
Implementation could be found here [lines]
[image before and after]


### Warping
Next step is to warp images using warping matrix (M found on the 1st step).
Implementation could be found here [lines]
[images]


### Locating Lane Lines using histogram
[images]


### Lane Lines on actuall image


### Filling in the Lane


### Calculating Position and Radius of Curvature


### Pipeline


### Result (video)
[youtube link]
https://youtu.be/WF4PpEcjDYs
You can also find it [here](./project_video.mp4).

---

### Discussion
