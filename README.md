Yukihiro Fujita 'CUDA at Scale for the Enterprise' final project

The CUDA program in this project removes the blue component of the image and saves the file.
The input images are 24-bit, but are converted to 8-bit at the loading.
The only part performed by the GPU kernel is the removal of the blue component.

repository URL
https://github.com/tacores/CUDAatScaleForTheEnterpriseCourseProjectTemplate

Input files
{"01.jpg", "02.jpg", "03.jpg", "04.jpg", "05.jpg", "06.jpg", "07.jpg", "08.jpg", "09.jpg", "10.jpg"}

Output files
{"01_o.jpg", "02_o.jpg", "03_o.jpg", "04_o.jpg", "05_o.jpg", "06_o.jpg", "07_o.jpg", "08_o.jpg", "09_o.jpg", "10_o.jpg"}
