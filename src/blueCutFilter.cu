#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
#define WINDOWS_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#pragma warning(disable : 4819)
#endif

#include <Exceptions.h>
#include <FreeImage.h>
#include "device_launch_parameters.h"

#include <string.h>
#include <fstream>
#include <iostream>

#include <cuda_runtime.h>
#include <npp.h>

#include <helper_cuda.h>
#include <helper_string.h>

std::string inputFileNames[] = {"01.jpg", "02.jpg", "03.jpg", "04.jpg", "05.jpg", "06.jpg", "07.jpg", "08.jpg", "09.jpg", "10.jpg"};
std::string outputFileNames[] = {"01_o.jpg", "02_o.jpg", "03_o.jpg", "04_o.jpg", "05_o.jpg", "06_o.jpg", "07_o.jpg", "08_o.jpg", "09_o.jpg", "10_o.jpg"};

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

FIBITMAP*
loadImage(const std::string &rFileName)
{
  FREE_IMAGE_FORMAT eFormat = FreeImage_GetFileType(rFileName.c_str());
  std::cout << "eFormat : " << eFormat << std::endl;

  // no signature? try to guess the file format from the file extension
  if (eFormat == FIF_UNKNOWN)
  {
      eFormat = FreeImage_GetFIFFromFilename(rFileName.c_str());
  }

  NPP_ASSERT(eFormat != FIF_UNKNOWN);
  // check that the plugin has reading capabilities ...
  FIBITMAP *pBitmap = FreeImage_Load(eFormat, rFileName.c_str());

  NPP_ASSERT(pBitmap != 0);
  std::cout << "src width : " << FreeImage_GetWidth(pBitmap) << std::endl;
  std::cout << "src height : " << FreeImage_GetHeight(pBitmap) << std::endl;
  return pBitmap;
}

// Device Code
// Set blue element of RGB pixel to Zero.
__global__ void cut_blue_element(BYTE* d_dst, int dstPitch, BYTE* d_img, int srcPitch) {
    int block_id = blockIdx.z * (gridDim.x * gridDim.y)
        + blockIdx.y * (gridDim.x)
        + blockIdx.x;
    int idx = block_id * (blockDim.x * blockDim.y * blockDim.z)
        + (threadIdx.z * (blockDim.x * blockDim.y))
        + (threadIdx.y * blockDim.x)
        + threadIdx.x;

  if (idx % 3 == 0) {   // blue element of RGB
      d_dst[idx] = 0;
  }
  else {    // red or green element of RGB
      d_dst[idx] = d_img[idx];
  }
}

int main(int argc, char *argv[])
{
  try
  {
    for (int i = 0; i < 10; ++i) {
      // src image file name
      std::string sFilename = "data/" + inputFileNames[i];

      // if we specify the filename at the command line, then we only test
      // sFilename[0].
      int file_errors = 0;
      std::ifstream infile(sFilename.data(), std::ifstream::in);

      if (infile.good())
      {
        std::cout << "opened: <" << sFilename.data()
                  << "> successfully!" << std::endl;
        file_errors = 0;
        infile.close();
      }
      else
      {
        std::cout << "unable to open: <" << sFilename.data() << ">"
                  << std::endl;
        file_errors++;
        infile.close();
      }

      if (file_errors > 0)
      {
        exit(EXIT_FAILURE);
      }

      std::string sResultFilename = "data/" + outputFileNames[i];

      // load source image from disk
      FIBITMAP*  srcBitmap = loadImage(sFilename);

      int height = FreeImage_GetHeight(srcBitmap);
      int width = FreeImage_GetWidth(srcBitmap);
      int pitch = FreeImage_GetPitch(srcBitmap);
      int bpp = FreeImage_GetBPP(srcBitmap);

      size_t size = height * pitch;
      // device memory
      BYTE* d_src = NULL;
      BYTE* d_dst = NULL;

      cudaError_t err = cudaMalloc(&d_src, size);
      if (err != cudaSuccess)
      {
          fprintf(stderr, "Failed to allocate device vector d_src (error code %s)!\n", cudaGetErrorString(err));
          exit(EXIT_FAILURE);
      }
      err = cudaMalloc(&d_dst, size);
      if (err != cudaSuccess)
      {
          fprintf(stderr, "Failed to allocate device vector d_dst (error code %s)!\n", cudaGetErrorString(err));
          exit(EXIT_FAILURE);
      }

      // copy host to device
      BYTE* h_src = FreeImage_GetBits(srcBitmap);
      err = cudaMemcpy(d_src, h_src, size, cudaMemcpyHostToDevice);

      // execute kernel code
      dim3 grdDim(width, 3, 1); // 3bits per piexel
      dim3 blkDim(height, 1, 1);
      cut_blue_element<<<grdDim, blkDim>>>(d_dst, pitch, d_src, pitch);

      gpuErrchk(cudaPeekAtLastError());
      gpuErrchk(cudaDeviceSynchronize());

      FIBITMAP* pNewBitmap = FreeImage_Allocate(width, height, bpp);

      // copy device to host
      err = cudaMemcpy(FreeImage_GetBits(pNewBitmap), d_dst, size, cudaMemcpyDeviceToHost);

      // Save Image file
      FreeImage_Save(FIF_JPEG, pNewBitmap, sResultFilename.c_str(), 0);
      std::cout << "Saved image: " << sResultFilename << std::endl;

      // Free memory
      err = cudaFree(d_dst);
      if (err != cudaSuccess)
      {
          fprintf(stderr, "Failed to free device vector d_dst (error code %s)!\n", cudaGetErrorString(err));
          exit(EXIT_FAILURE);
      }
      err = cudaFree(d_src);
      if (err != cudaSuccess)
      {
          fprintf(stderr, "Failed to free device vector d_dst (error code %s)!\n", cudaGetErrorString(err));
          exit(EXIT_FAILURE);
      }
    }
  }
  catch (npp::Exception &rException)
  {
    std::cerr << "Program error! The following exception occurred: \n";
    std::cerr << rException << std::endl;
    std::cerr << "Aborting." << std::endl;

    exit(EXIT_FAILURE);
  }
  catch (...)
  {
    std::cerr << "Program error! An unknow type of exception occurred. \n";
    std::cerr << "Aborting." << std::endl;

    exit(EXIT_FAILURE);
  }

  return 0;
}
