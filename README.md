# A simple library for unsupervised Polarimetric SAR change detection

## Introduction
This is a simple library for unsupervised Polarimetric SAR change detection. The main function is the `**unsupervised_CD**` function in `unsupervised_CD.py`.

The workflow of change detection is:
1. read C3 (covaraince matrix) of the PolSAR data
2. perform boxcar smoothing to ensure non-negative definteness of C3 matrix
3. calculate the pixel-wise distance between two PolSAR images. Availabel distance metrics: `Bartlett distance, revised Wishart distance, symmetric revised Wishart distance`
4. perform [generalized histogram thresholding algorithm](https://arxiv.org/abs/2007.07350) to segment the change and unchange pixels

## Example
Edit in the `if __name__ == '__main__'` part of unsupervised_change_detection.py like below
```python
if __name__ == '__main__':
    fa = r'path/to/C3/folder_A'
    fb = r'path/to/C3/folder_B'
    gt = r'path/to/groundtruth/file'
    save_path = r'path/to/save/result/images'
    
    confusion_matrix = unsupervised_CD(fa, fb, save_path, gt,
                        is_print=True, distance_type='Bartlett')

    print(f'confusion_matrix:\n{confusion_matrix}')
```

The change detection result will be like that below:
<!-- <center> -->
<!-- ![time A](images/PauliRGB_1.bmp) -->
| <img src="images/PauliRGB_1.bmp" width="150"> | <img src="images/PauliRGB_2.bmp" width="150"> | <img src="images/result.png" width="150"> |
| :-------------------------: | :---------------------------: | :-------: |
|time A | time B | change detection result |
<!-- </center> -->

## Requirements
This repo need [my another library](https://github.com/yoyoyoohh/mylib)

## License
This project is released under the [Apache 2.0 license](LICENSE).