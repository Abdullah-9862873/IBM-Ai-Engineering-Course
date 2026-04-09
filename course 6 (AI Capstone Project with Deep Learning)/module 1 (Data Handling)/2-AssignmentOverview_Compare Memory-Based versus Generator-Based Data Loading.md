# Assignment Overview: Compare Memory-Based versus Generator-Based Data Loading

**Estimated reading time: 2 minutes**

Welcome to the hands-on lab for working with satellite imagery and comparing memory-based versus generator-based data loading.

This set of instructions will guide you through all of the targeted tasks in the provided notebook. By following these steps, you'll gain crucial experience manipulating images in Python, understanding directory handling, and visualizing remote sensing data for deep learning projects.

## Dataset Structure

You'll start by downloading the dataset and exploring the two main folders contained within:
- **class_0_non_agri**: Contains images of non-agricultural land
- **class_1_agri**: Contains images representing agricultural land

Each class folder includes numerous images.

## Lab Tasks

### Step 1: Explore Directory Structure

Your first step is to use Python's standard library (`os`) to list, sort, and handle file paths for these categories. You will use `os.listdir()` to obtain file names within the `class_0_non_agri` directory.

### Step 2: Load and Display Images

Then, you will open and display the first image in `non_agri_images`. Your first exercise would be to look at the image dimensions of a single image in `non_agri_images`.

### Step 3: Memory-Based Loading

Using memory-based loading, you will read all images into a list. This approach loads all images into memory at once, which can be faster but requires more RAM.

### Step 4: Generator-Based (Lazy) Loading

To compare with lazy loading, you will display the first four non-agricultural images. Similarly, you will create the list of all agricultural images and calculate their number in your next exercise.

### Step 5: Agricultural Images

Finally, you will end the lab by displaying the first four images of the agricultural land.

## Tips

If you get errors opening images, re-check your directory paths and ensure you're using the correct full absolute or relative paths.

## Learning Outcomes

By following these steps, you'll gain confidence in:
- File handling
- Image loading
- Visualization with common Python tools for deep learning

Complete all the code and questions to finish the lab successfully. You will need to download and save the finished lab on your computer for final evaluation at the end of this course. Good luck!

________________________
