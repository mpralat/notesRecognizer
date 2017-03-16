# notesRecognizer

## Installing dependencies
You can install all of the needed dependencies with pip. In the project folder, run:

```
pip install -r requirements.txt
```

## How it works

### Preprocessing the photo

Given the input image, this script runs several image processing functions to find and recognize the pitch of the musical notes. The input image should be a photo of a white musical sheet. The whole sheet should be visible. 

Several effects are applied: converting to grayscale, GaussianBlur and Canny filter for edge detection.
Afterwards, we find contours on the preprocessed image and choose the biggest contour with 4 corners as the sheet contour.
The last step of this stage is adjusting the contour and getting the clean image of the sheet only.

![Detecting the sheet](https://github.com/mpralat/notesRecognizer/blob/master/processed_examples/1.jpg)
>Detecting the musical sheet

### Detecting the main elements

Using the Hough Transform, we detect the lines in the picture and then the whole staffs. 
![Detecting the lines](https://github.com/mpralat/notesRecognizer/blob/master/processed_examples/6lines_line.jpg)
>Detecting the lines

The next step is to detect the positions of the notes. We detect the blobs with given parameters.
To correctly find the pitch of each note, we also recognize the clef. Using the Hu moments, we have been able to determine, whether the violin or the bass clef had been used. 

### Computing the pitch

Knowing both the position of each note and the kind of clef that's been used, we get the pitch of each note.
![Getting the pitch](https://github.com/mpralat/notesRecognizer/blob/master/processed_examples/9_with_pitch.jpg)
>Getting the pitch of the notes

The result of each stage is saved to an image in the output folder.

## How to run
```
python main.py -i PATH_TO_IMAGE
```
