# Module 1 Quiz: Data Handling

Answer the following questions by selecting the correct option.

---

### Question 1

At TechVision Inc., AI Engineer Sarah is tasked with optimizing the data pipeline for a new image classification project. She needs to decide whether to use bulk loading or sequential loading. How should she approach this decision, considering memory usage and training speed?

- [ ] Choose bulk loading to ensure both low memory usage and slow training speed.
- [ ] Use bulk loading to minimize memory usage and maximize training speed.
- [ ] Sequential loading is preferred for faster training speeds in all scenarios.
- [x] Select sequential loading for lower memory usage but potentially slower training speed.

---

### Question 2

At DeepAI Solutions, Engineer Tom is implementing data augmentation using Keras. What is a key consideration he should keep in mind when applying these techniques?

- [x] Augmentation techniques should be relevant and enhance the model's ability to generalize.
- [ ] Ensure that the augmented data does not exceed the original dataset size.
- [ ] Keras automatically handles all augmentation processes without any user input.
- [ ] Data augmentation should be applied after the model training is complete.

---

### Question 3

At InnovateAI, data scientist Emily is using PyTorch to develop a flexible image pipeline. What is an important factor she should consider when implementing the Dataset and DataLoader classes?

- [x] DataLoader can be used to manage memory usage by controlling batch size and data shuffling.
- [ ] The DataLoader class automatically determines the optimal batch size.
- [ ] The Dataset class is responsible for the model's training speed.
- [ ] The Dataset class should be used for data augmentation only.

---

### Question 4

Why is `os.path.join(dir_non_agri, non_agri_images)` preferred over string concatenation for file paths?

- [ ] It automatically downloads the file.
- [ ] It converts relative paths to absolute paths.
- [x] It ensures platform-independent path separators.
- [ ] It validates that the file exists.

---

### Question 5

Fill in the blank:

The generator reshuffles indices at the start of _________.

- [ ] Validation only
- [ ] Each batch
- [ ] The script
- [x] Each epoch

---

### Question 6

Consider the following code snippet:

```python
def imshow(img): 
    img = img / 2 + 0.5
    npimg = img.numpy() 
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
```

The line `img = img / 2 + 0.5` assumes the tensor was previously normalized by dividing by 0.5 after subtracting 0.5. What original normalization call matches that assumption?

- [ ] Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
- [ ] No normalization was applied
- [ ] Normalize(mean=[0.0], std=[1.0])
- [x] Normalize(mean=[0.5], std=[2.0])

---

### Question 7

How does `datasets.ImageFolder` determine class labels?

- [ ] Hashing file names
- [ ] Reading a CSV file called labels.csv
- [x] The alphabetical order of immediate subfolder names
- [ ] EXIF metadata inside each image

---

## Answer Key

| Question | Answer | Explanation |
|----------|--------|--------------|
| Question 1 | Select sequential loading for lower memory usage but potentially slower training speed. | Generator-based (sequential) loading uses less memory but may be slower than bulk loading. |
| Question 2 | Augmentation techniques should be relevant and enhance the model's ability to generalize. | Data augmentation should help the model generalize better to unseen data. |
| Question 3 | DataLoader can be used to manage memory usage by controlling batch size and data shuffling. | DataLoader provides control over batch size and shuffling. |
| Question 4 | It ensures platform-independent path separators. | `os.path.join()` handles path separators across platforms. |
| Question 5 | Each epoch | The DataLoader reshuffles at every epoch. |
| Question 6 | Normalize(mean=[0.5], std=[2.0]) | To reverse normalize: img / 0.5 + 0.5 = img / 2 + 0.5 |
| Question 7 | The alphabetical order of immediate subfolder names | ImageFolder assigns labels based on subfolder names alphabetically. |

---

**Total Points: 7 points (1 point each)**

Complete all tasks in the lab to answer questions correctly.