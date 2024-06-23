import cv2
import matplotlib.pyplot as plt
import numpy as np

# Show a image with matplotlib


def pltDisplay(image, cmap='rainbow', ax=None):
    # Show image

    image = np.asarray(image)  # convert image as an numpy.ndarray
    # img = np.squeeze(img)  # remove single-dimensional entry from array shape
    img_channels = image.shape[-1]

    if len(image.shape) == 2 or img_channels == 1:
        # grayscale
        image = np.squeeze(image)
        cmap = 'gray'
    else:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # opencv reads and displays an image as BGR whereas matplotlib uses RGB

    plt.set_cmap(cmap)

    if ax is None:
        plt.imshow(image, cmap=cmap, vmin=0, vmax=255)
        plt.show()
    else:
        ax.imshow(image, cmap=cmap, vmin=0, vmax=255)


# Show a image with opencv

def cv2Display(inp_img, win_name='Window'):
    cv2.imshow(win_name, inp_img)
    # waits for user to press any key
    # (this is necessary to avoid Python kernel form crashing)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        return
    cv2.destroyAllWindows()