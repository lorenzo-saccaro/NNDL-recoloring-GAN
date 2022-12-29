import os
from PIL import Image
import tqdm

# TODO: decide if we want to delete the images that are not RGB
# Set the directory you want to search for grayscale images
directory = 'C:\\Users\\loren\\Datasets\\coco2017\\train2017'

# Initialize a counter for the number of grayscale images
gray_count = 0
# Initialize variables to store the lowest width and height
lowest_width = float('inf')
lowest_height = float('inf')

# Iterate through all the files in the directory
for file in tqdm.tqdm(os.listdir(directory)):
    # Check if the file is a valid image
    try:
        img = Image.open(os.path.join(directory, file))
        # Check if the image is grayscale
        if img.mode == 'L':
            # Increment the counter for grayscale images
            gray_count += 1
        # Update the lowest width and height if necessary
        if img.width < lowest_width:
            lowest_width = img.width
        if img.height < lowest_height:
            lowest_height = img.height
    except:
        pass

# Print the number of grayscale images and the lowest width and height found
print(f'Number of grayscale images: {gray_count}')
print(f'Lowest width: {lowest_width}')
print(f'Lowest height: {lowest_height}')
