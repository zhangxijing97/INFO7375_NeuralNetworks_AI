# Quiz 4

# Bounding Box Technique in Object Localization and Detection

A bounding box is a rectangular coordinate system used to define a geometric area on an image.<br>

Coordinates of a bounding box are encoded with four values in pixels: [x_min, y_min, x_max, y_max].<br>
x_min and y_min are coordinates of the top-left corner of the bounding box.<br>
x_max and y_max are coordinates of bottom-right corner of the bounding box.<br>

Image/Object localization is a regression problem where the output is x and y coordinates around the object of interest to draw bounding boxes.<br>
Input: an image<br>
Output: x_min, y_min, x_max, y_max, of an object of interest<br>

Image localization could be done with regular CNN vision algorithms.<br>
These algorithms predict classes with discrete numbers.<br>
In object localization, the algorithm predicts a set of 4 numbers, namely, x_min, y_min, x_max, and y_max to draw a bounding box around an object of interest.<br>