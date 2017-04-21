import numpy as np
import cv2


class RelativeRectangle:
    """
    2D Rectangle defined by top-left and bottom-right corners.
    NOTICE: COORDS ARE EXPRESSED IN TERMS OF PERCENTAGE OF SCREEN W AND H

    Parameters
    ----------
    x_min : float
        x coordinate of top-left corner.
    y_min : float
        y coordinate of top-left corner.
    x_max : float
        x coordinate of bottom-right corner.
    y_min : float
        y coordinate of bottom-right corner.
    """

    def __init__(self, x_min, y_min, x_max, y_max):
        self.x_min = x_min
        self.y_min = y_min
        self.x_max = x_max
        self.y_max = y_max

        self.x_side = self.x_max - self.x_min
        self.y_side = self.y_max - self.y_min

    def draw(self, frame, color=255, thickness=1):
        """
        Draw Rectangle on a given frame.

        Notice: while this function does not return anything, original image `frame` is modified.

        Parameters
        ----------
        frame : 2D / 3D np.array
            The image on which the rectangle is drawn.
        color : tuple, optional
            Color used to draw the rectangle (default = 255)
        thickness : int, optional
            Line thickness used t draw the rectangle (default = 1)

        Returns
        -------
        None
        """

        h, w = frame.shape[:2]

        # convert back from relative coordinates to frame coordinates
        x_min = int(self.x_min * w)
        y_min = int(self.y_min * h)
        x_max = int(self.x_max * w)
        y_max = int(self.y_max * h)

        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, thickness)

    def get_mask(self, mask_shape):
        """
        Return the foreground mask of the rectangle.

        Parameters
        ----------
        mask_shape : tuple
            Tuple (height, width) that defines the shape of the fg_mask

        Returns
        -------
        fg_mask : ndarray
            Foreground mask of this RelativeRectangle

        RelativeRectangle has relative coordinates, so the fg_mask shape must be passed as parameter.
        """
        h, w = mask_shape

        fg_mask = np.zeros(shape=(h, w), dtype=np.float32)

        # convert back from relative coordinates to frame coordinates
        x_min = int(self.x_min * w)
        y_min = int(self.y_min * h)
        x_max = int(self.x_max * w)
        y_max = int(self.y_max * h)

        fg_mask[y_min:y_max, x_min:x_max] = 1.

        return fg_mask

    @property
    def tl_corner(self):
        """
        Coordinates of the top-left corner of rectangle (as float32).

        Returns
        -------
        tl_corner : float32 tuple
        """
        return tuple(map(np.float32, (self.x_min, self.y_min)))

    @property
    def br_corner(self):
        """
        Coordinates of the bottom-right corner of rectangle.

        Returns
        -------
        br_corner : float32 tuple
        """
        return tuple(map(np.float32, (self.x_max, self.y_max)))

    @property
    def coords(self):
        """
        Coordinates (x_min, y_min, x_max, y_max) which define the Rectangle.

        Returns
        -------
        coordinates : float32 tuple
        """
        return tuple(map(np.float32, (self.x_min, self.y_min, self.x_max, self.y_max)))

    @property
    def area(self):
        """
        Get the area of Rectangle

        Returns
        -------
        area : float32
        """
        return np.float32(self.x_side * self.y_side)


def imagenet_mean_bgr(frame_bgr, op='subtract'):
    """
    Add or subtract ImageNet mean pixel value from a given BGR frame.
    """
    imagenet_mean_BGR = np.array([123.68, 116.779, 103.939])

    frame_bgr = np.float32(frame_bgr)

    for c in range(0, 3):
        if op == 'subtract': frame_bgr[:, :, c] -= imagenet_mean_BGR[c]
        elif op == 'add':    frame_bgr[:, :, c] += imagenet_mean_BGR[c]

    return frame_bgr


def stitch_together(input_images, layout, resize_dim=None, off_x=None, off_y=None, bg_color=(0, 0, 0)):
    """
    Stitch together N input images into a bigger frame, using a grid layout.
    Input images can be either color or grayscale, but must all have the same size.
    :param input_images: list of input images
    :param layout: grid layout expressed (rows, cols) of the stitch
    :param resize_dim: if not None, stitch is resized to this size
    :param off_x: offset between stitched images along x axis
    :param off_y: offset between stitched images along y axis
    :param bg_color: color used for background
    :return: stitch of input images
    """

    if len(set([img.shape for img in input_images])) > 1:
        raise ValueError('All images must have the same shape')

    # determine if input images are color (3 channels) or grayscale (single channel)
    if len(input_images[0].shape) == 2:
        mode = 'grayscale'
        img_h, img_w = input_images[0].shape
    elif len(input_images[0].shape) == 3:
        mode = 'color'
        img_h, img_w, img_c = input_images[0].shape
    else:
        raise ValueError('Unknown shape for input images')

    # if no offset is provided, set to 10% of image size
    if off_x is None:
        off_x = img_w // 10
    if off_y is None:
        off_y = img_h // 10

    # create stitch mask
    rows, cols = layout
    stitch_h = rows * img_h + (rows + 1) * off_y
    stitch_w = cols * img_w + (cols + 1) * off_x
    if mode == 'color':
        bg_color = np.array(bg_color)[None, None, :]  # cast to ndarray add singleton dimensions
        stitch = np.uint8(np.repeat(np.repeat(bg_color, stitch_h, axis=0), stitch_w, axis=1))
    elif mode == 'grayscale':
        stitch = np.zeros(shape=(stitch_h, stitch_w), dtype=np.uint8)

    for r in range(0, rows):
        for c in range(0, cols):

            list_idx =  r * cols + c

            if list_idx < len(input_images):
                if mode == 'color':
                    stitch[ r * (off_y + img_h) + off_y: r*(off_y+img_h) + off_y + img_h,
                            c * (off_x + img_w) + off_x: c * (off_x + img_w) + off_x + img_w,
                            :] = input_images[list_idx]
                elif mode == 'grayscale':
                    stitch[ r * (off_y + img_h) + off_y: r*(off_y+img_h) + off_y + img_h,
                            c * (off_x + img_w) + off_x: c * (off_x + img_w) + off_x + img_w]\
                        = input_images[list_idx]

    if resize_dim:
        stitch = cv2.resize(stitch, dsize=(resize_dim[::-1]))

    return stitch


def show_prediction(frontal_image, birdeye_image, x_coords, y_coords, y_pred_coords):
    """
    Display network prediction.

    Parameters
    ----------

    frontal_image : ndarray
        Frame taken from dashboard camera view
    birdeye_image : ndarray
        Frame taken from bird's eye view
    x_coords : list
        Coords of vehicle in the frontal view
    y_coords : list
        Coords of vehicle in the bird's eye view (GT)
    y_pred_coords : list
        Coords of vehicle in the bird's eye view (pred)

    Returns
    -------
    None
    """
    birdeye_image_pred = birdeye_image.copy()
    birdeye_image_true = birdeye_image.copy()

    bbox_frontal = RelativeRectangle(*[x_coords[j] for j in range(0, 4)])

    # cast back from tanh range (-1, 1) to (0, 1)
    bbox_pred = RelativeRectangle(*[((y_pred_coords[j] * 0.5) + 0.5) for j in range(0, 4)])
    bbox_true = RelativeRectangle(*[((y_coords[j] * 0.5) + 0.5) for j in range(0, 4)])

    # draw bounding boxes
    bbox_frontal.draw(frontal_image, color=(0, 0, 255), thickness=6)
    bbox_pred.draw(birdeye_image_pred, color=(0, 0, 255), thickness=6)
    bbox_true.draw(birdeye_image_true, color=(0, 0, 255), thickness=6)

    # stitch frames for showing
    stitch = stitch_together([frontal_image, birdeye_image_pred, birdeye_image_true],
                             layout=(1, 3), resize_dim=(300, 1800))
    cv2.imshow('Dashboard view | Birdeye Prediction | Birdeye GT', stitch)
    cv2.waitKey(0)