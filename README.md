# Coursera GPU Programming final project

## repository URL
https://github.com/tacores/CUDAatScaleForTheEnterpriseCourseProjectTemplate

## Description
The CUDA program in this project removes the blue component of the image and saves the file.
The only part performed by the GPU kernel is the removal of the blue component. (cut_blue_element())

| Before  | After |
| ------------- | ------------- |
| <img src="https://github.com/tacores/CUDAatScaleForTheEnterpriseCourseProjectTemplate/blob/main/09.jpg" width="300">  | <img src="https://github.com/tacores/CUDAatScaleForTheEnterpriseCourseProjectTemplate/blob/main/09_o.jpg" width="300">  |
| <img src="https://github.com/tacores/CUDAatScaleForTheEnterpriseCourseProjectTemplate/blob/main/02.jpg" width="300">  | <img src="https://github.com/tacores/CUDAatScaleForTheEnterpriseCourseProjectTemplate/blob/main/02_o.jpg" width="300">  |
| <img src="https://github.com/tacores/CUDAatScaleForTheEnterpriseCourseProjectTemplate/blob/main/10.jpg" width="300">  | <img src="https://github.com/tacores/CUDAatScaleForTheEnterpriseCourseProjectTemplate/blob/main/10_o.jpg" width="300">  |

## Demo Video
https://youtu.be/hc9iqgg_iSc

### Input
{"01.jpg", "02.jpg", "03.jpg", "04.jpg", "05.jpg", "06.jpg", "07.jpg", "08.jpg", "09.jpg", "10.jpg"}
### Output
{"01_o.jpg", "02_o.jpg", "03_o.jpg", "04_o.jpg", "05_o.jpg", "06_o.jpg", "07_o.jpg", "08_o.jpg", "09_o.jpg", "10_o.jpg"}


## Environment Requirement

1. Visual Studio 2022 is installed
2. NVIDIA GPU Computing Toolkit is installed

## How to Run

1.
`git clone https://github.com/tacores/CUDAatScaleForTheEnterpriseCourseProjectTemplate.git tacores_project`

2. Open blueCutFilter_vs2022.sln via Visual Studio 2022.

3. `build`

4. list image files before execution (10 jpg files)

`Open Command prompt`

5.
`cd bin\win64\Debug`

6.
`cp data bin\win64\Debug`

7. execute. no parameter because file names are hold in hard coding string array
`./blueCutFilter.exe`
