# PyPhotoshop

Python implementation of image filters. Adjust brightness and contrast, add blur, and detect edges!

### Status

Finished.

### Usage

In order to download this code, either click the green button at the top right and download as ZIP, or clone the repository. You will need to `pip install -r requirements.txt` (or use `pip3` if you are getting a module not found error).

In the folder, you will find these files:

- image.py: contains the `Image` class that will read and write the images using the PNG `Writer` and `Reader`
- png.py: pure Python PNG `Reader` and `Writer` classes
- transform.py: implemented image filter functions (adjust brightness, adjust contrast, edge detection using Sobel-Feldman kernel)

### Notes

- used memoization for `blur` filter. space complexity O(x_pixels \* y_pixels \* num_channels).

### Credits

- [12 Beginner Python Projects - Coding Course](https://youtu.be/8ext9G7xspg?t=7534) by freeCodeCamp ([Kylie Ying](https://www.youtube.com/ycubed))
- [Tutorial on Kylie Ying's channel](https://youtu.be/4ifdUQmZqhM)
- Python PNG `Reader` and `Writer` classes from Johann C. Rocholl ([png.py](/png.py))
