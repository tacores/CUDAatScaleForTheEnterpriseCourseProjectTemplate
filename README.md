'CUDA at Scale for the Enterprise' final project

(repository URL)
https://github.com/tacores/CUDAatScaleForTheEnterpriseCourseProjectTemplate

(Description)
The CUDA program in this project removes the blue component of the image and saves the file.
The input images are 24-bit, but are converted to 8-bit at the loading time.
The only part performed by the GPU kernel is the removal of the blue component. (cut_blue_element())

The 8-bit pixels consist of 3 bits for Red and Green and 2 bits for Blue.
The following operations are then performed to remove the Blue component (last 2bits).
d_dst[idx] = d_img[idx] & 0xFC; // 1111 1100

Since loadImage() and saveImage() did not support ImageCPU_8u_C3, I defined them on blueCutFilter.cu file.


(Input)
{"01.jpg", "02.jpg", "03.jpg", "04.jpg", "05.jpg", "06.jpg", "07.jpg", "08.jpg", "09.jpg", "10.jpg"}
(Output)
{"01_o.jpg", "02_o.jpg", "03_o.jpg", "04_o.jpg", "05_o.jpg", "06_o.jpg", "07_o.jpg", "08_o.jpg", "09_o.jpg", "10_o.jpg"}

(How to Run)
1. It is assumed that it's run on the Coursera Lab and there is a Common directory in the project directory.
cd /home/coder/project

2.
git clone https://github.com/tacores/CUDAatScaleForTheEnterpriseCourseProjectTemplate.git tacores_project

3.
cd tacores_project

4.
make build

5. list image files before execution (10 jpg files)
ls ./data

6. execute. no parameter because file names are hold in hard coding string array
./blueCutFilter.exe

7. list image files after execution (NN_o.jpg files should be created)
ls ./data



(Logs when I run)
coder@dfc53a0a04f4:~/project$ cd /home/coder/project
coder@dfc53a0a04f4:~/project$ git clone https://github.com/tacores/CUDAatScaleForTheEnterpriseCourseProjectTemplate.git tacores_project
Cloning into 'tacores_project'...
remote: Enumerating objects: 33, done.
remote: Counting objects: 100% (33/33), done.
remote: Compressing objects: 100% (26/26), done.
remote: Total 33 (delta 4), reused 19 (delta 3), pack-reused 0
Unpacking objects: 100% (33/33), 1.76 MiB | 1.42 MiB/s, done.
coder@dfc53a0a04f4:~/project$ cd tacores_project
coder@dfc53a0a04f4:~/project/tacores_project$ make build
/usr/local/cuda/bin/nvcc -ccbin g++ -I../Common -I../Common/UtilNPP  -m64    --threads 0 -gencode arch=compute_35,code=compute_35 -o blueCutFilter.o -c src/blueCutFilter.cu
nvcc warning : The 'compute_35', 'compute_37', 'compute_50', 'sm_35', 'sm_37' and 'sm_50' architectures are deprecated, and may be removed in a future release (Use -Wno-deprecated-gpu-targets to suppress warning).
/usr/local/cuda/bin/nvcc -ccbin g++   -m64      -gencode arch=compute_35,code=compute_35 -o blueCutFilter.exe blueCutFilter.o  -lnppisu_static -lnppif_static -lnppc_static -lculibos -lfreeimage
nvcc warning : The 'compute_35', 'compute_37', 'compute_50', 'sm_35', 'sm_37' and 'sm_50' architectures are deprecated, and may be removed in a future release (Use -Wno-deprecated-gpu-targets to suppress warning).
cp blueCutFilter.exe ./bin/
coder@dfc53a0a04f4:~/project/tacores_project$ ls ./data
01.jpg  02.jpg  03.jpg  04.jpg  05.jpg  06.jpg  07.jpg  08.jpg  09.jpg  10.jpg
coder@dfc53a0a04f4:~/project/tacores_project$ ./blueCutFilter.exe
opened: <data/01.jpg> successfully!
Saved image: data/01_o.jpg
opened: <data/02.jpg> successfully!
Saved image: data/02_o.jpg
opened: <data/03.jpg> successfully!
Saved image: data/03_o.jpg
opened: <data/04.jpg> successfully!
Saved image: data/04_o.jpg
opened: <data/05.jpg> successfully!
Saved image: data/05_o.jpg
opened: <data/06.jpg> successfully!
Saved image: data/06_o.jpg
opened: <data/07.jpg> successfully!
Saved image: data/07_o.jpg
opened: <data/08.jpg> successfully!
Saved image: data/08_o.jpg
opened: <data/09.jpg> successfully!
Saved image: data/09_o.jpg
opened: <data/10.jpg> successfully!
Saved image: data/10_o.jpg
coder@dfc53a0a04f4:~/project/tacores_project$ ls ./data
01.jpg  01_o.jpg  02.jpg  02_o.jpg  03.jpg  03_o.jpg  04.jpg  04_o.jpg  05.jpg  05_o.jpg  06.jpg  06_o.jpg  07.jpg  07_o.jpg  08.jpg  08_o.jpg  09.jpg  09_o.jpg  10.jpg  10_o.jpg
coder@dfc53a0a04f4:~/project/tacores_project$ 


