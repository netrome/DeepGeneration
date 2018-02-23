from skimage.feature import canny
from skimage.transform import hough_ellipse


def fit_ellipse(img):
    edges = canny(img, sigma=3.0)
    try:
        result = hough_ellipse(edges, min_size=3, max_size=40)
    except ValueError:
        return 128, 128  # Blang yields average prediction
    result.sort(order="accumulator")

    best = list(result[-1])
    yc, xc, a, b = [x for x in best[1:5]]

    return yc, xc