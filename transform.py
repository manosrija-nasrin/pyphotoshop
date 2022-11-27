from image import Image
import numpy as np


def brighten(image, factor):
    # when we brighten, we just want to make each channel higher by some amount
    # factor is a value > 0, how much you want to brighten the image by (< 1 = darken, > 1 = brighten)
    # get x, y pixels of image, # channels
    x_pixels, y_pixels, num_channels = image.array.shape
    new_im = Image(x_pixels=x_pixels, y_pixels=y_pixels,
                   num_channels=num_channels)

    # vectorized implementation
    new_im.array = image.array * factor

    return new_im


def adjust_contrast(image, factor, mid):
    # adjust the contrast by increasing the difference from the user-defined midpoint by factor amount
    # higher factor, more contrast
    x_pixels, y_pixels, num_channels = image.array.shape
    new_im = Image(x_pixels=x_pixels, y_pixels=y_pixels,
                   num_channels=num_channels)

    # vectorized implementation
    new_im.array = (image.array - mid) * factor + mid

    return new_im


def blur(image, kernel_size):
    # kernel size is the number of pixels to take into account when applying the blur
    # (ie kernel_size = 3 would be neighbors to the left/right, top/bottom, and diagonals)
    # kernel size should always be an *odd* number
    x_pixels, y_pixels, num_channels = image.array.shape
    new_im = Image(x_pixels=x_pixels, y_pixels=y_pixels,
                   num_channels=num_channels)

    # how many pixels on one side we need to look at
    neighbour_range = kernel_size // 2

    # we'll be using memoization for faster execution

    # naive implementation
    # for kernel_size = 3, time = 2.3298606872558594 seconds
    # for kernel_size = 15, time = 29.354379177093506 seconds

    # memoized implementation
    # for kernel_size = 3, time = 0.9197995662689209 seconds
    # for kernel_size = 15, time = 0.8251416683197021 seconds

    row_total = np.zeros((x_pixels, y_pixels, num_channels))

    # adding along x - axis per channel
    for c in range(num_channels):
        for x in range(x_pixels):
            for y in range(y_pixels):
                left = y - neighbour_range
                right = y + neighbour_range
                if y == 0:  # summing for first row only
                    for y_i in range(max(left, 0), min(y_pixels - 1, right) + 1):
                        row_total[x, y, c] += image.array[x, y_i, c]
                    continue
                elif left <= 0 and right < y_pixels:
                    row_total[x, y, c] = (row_total[x, y - 1, c] +
                                          image.array[x, right, c])
                elif left > 0 and right < y_pixels:
                    row_total[x, y, c] = (row_total[x, y - 1, c] -
                                          image.array[x, left - 1, c] + image.array[x, right, c])
                elif left > 0 and right >= y_pixels:
                    row_total[x, y, c] = (row_total[x, y - 1, c] -
                                          image.array[x, left - 1, c])

    # adding along y - axis per channel
    for c in range(num_channels):
        for y in range(y_pixels):
            for x in range(x_pixels):
                top = x - neighbour_range
                bottom = x + neighbour_range
                if x == 0:   # summing for first column only
                    for x_i in range(max(top, 0), min(x_pixels - 1, bottom) + 1):
                        new_im.array[x, y, c] += row_total[x_i, y, c]
                    continue
                elif top <= 0 and bottom < x_pixels:
                    new_im.array[x, y, c] = (
                        new_im.array[x - 1, y, c] + row_total[bottom, y, c])
                elif top > 0 and bottom < x_pixels:
                    new_im.array[x, y, c] = (new_im.array[x - 1, y, c] -
                                             row_total[top - 1, y, c] +
                                             row_total[bottom, y, c])
                elif top > 0 and bottom >= x_pixels:
                    new_im.array[x, y, c] = (new_im.array[x - 1,
                                                          y, c] - row_total[top - 1, y, c])

    # averaging the pixel values
    new_im.array = new_im.array / (kernel_size ** 2)

    return new_im


def apply_kernel(image, kernel):
    # the kernel should be a 2D array that represents the kernel we'll use!
    # for the sake of simiplicity of this implementation, let's assume that the kernel is SQUARE
    # for example the sobel x kernel (detecting horizontal edges) is as follows:
    # [1 0 -1]
    # [2 0 -2]
    # [1 0 -1]
    x_pixels, y_pixels, num_channels = image.array.shape
    new_im = Image(x_pixels=x_pixels, y_pixels=y_pixels,
                   num_channels=num_channels)

    kernel_size = kernel.shape[0]
    # how many pixels on one side we need to look at
    neighbour_range = kernel_size // 2

    # naive implementation
    # time taken to apply kernel: 3.0327587127685547 seconds

    for x in range(x_pixels):
        for y in range(y_pixels):
            for c in range(num_channels):
                total = 0
                for x_i in range(max(0, (x - neighbour_range)), min((x + neighbour_range), (x_pixels - 1)) + 1):
                    for y_i in range(max(0, (y - neighbour_range)), min((y + neighbour_range), (y_pixels - 1)) + 1):
                        x_k = x_i + neighbour_range - x
                        y_k = y_i + neighbour_range - y
                        total += image.array[x_i, y_i, c] * kernel[x_k, y_k]

                new_im.array[x, y, c] = total

    return new_im


def combine_images(image1, image2):
    # let's combine two images using the squared sum of squares: value = sqrt(value_1**2, value_2**2)
    # size of image1 and image2 MUST be the same
    x_pixels, y_pixels, num_channels = image1.array.shape
    new_im = Image(x_pixels=x_pixels, y_pixels=y_pixels,
                   num_channels=num_channels)

    # vectorized implementation
    new_im.array = (image1.array ** 2 + image2.array ** 2) ** 0.5

    return new_im


if __name__ == '__main__':
    lake = Image(filename='lake.png')
    city = Image(filename='city.png')

    # increase the brightness
    incr_brightness = brighten(lake, 2)
    incr_brightness.write_image("brightened.png")

    # decrease the brightness
    decr_brightness = brighten(lake, 0.4)
    decr_brightness.write_image("darkened.png")

    # increase contrast
    incr_contrast = adjust_contrast(lake, 1.5, 0.4)
    incr_contrast.write_image("increased_contrast.png")

    # decrease contrast
    decr_contrast = adjust_contrast(lake, 0.6, 0.4)
    decr_contrast.write_image("decreased_contrast.png")

    # blur with kernel = 3
    blur_3 = blur(city, 3)
    blur_3.write_image("blurred_3.png")

    # blur with kernel = 15
    blur_15 = blur(city, 15)
    blur_15.write_image("blurred_15.png")

    # let's apply a sobel edge detection kernel on the x and y axis
    sobel_x = apply_kernel(city, np.array(
        [[1, 2, 1],
         [0, 0, 0],
         [-1, -2, -1]]
    ))

    sobel_x.write_image('edge_x.png')
    sobel_y = apply_kernel(city, np.array(
        [[1, 0, -1],
         [2, 0, -2],
         [1, 0, -1]]
    ))
    sobel_y.write_image('edge_y.png')

    # let's combine these and make an edge detector!
    sobel_xy = combine_images(sobel_x, sobel_y)
    sobel_xy.write_image("edge_xy.png")
