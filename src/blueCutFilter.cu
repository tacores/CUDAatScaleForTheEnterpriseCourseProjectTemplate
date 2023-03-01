#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
#define WINDOWS_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#pragma warning(disable : 4819)
#endif

#include <Exceptions.h>
#include <ImageIO.h>
#include <ImagesCPU.h>
#include <ImagesNPP.h>

#include <string.h>
#include <fstream>
#include <iostream>

#include <cuda_runtime.h>
#include <npp.h>

#include <helper_cuda.h>
#include <helper_string.h>

std::string inputFileNames[] = {"01.jpg", "02.jpg", "03.jpg", "04.jpg", "05.jpg", "06.jpg", "07.jpg", "08.jpg", "09.jpg", "10.jpg"};
std::string outputFileNames[] = {"01_o.jpg", "02_o.jpg", "03_o.jpg", "04_o.jpg", "05_o.jpg", "06_o.jpg", "07_o.jpg", "08_o.jpg", "09_o.jpg", "10_o.jpg"};

void
loadImage(const std::string &rFileName, npp::ImageCPU_8u_C3 &rImage)
{
  // set your own FreeImage error handler
  FreeImage_SetOutputMessage(FreeImageErrorHandler);

  FREE_IMAGE_FORMAT eFormat = FreeImage_GetFileType(rFileName.c_str());

  // no signature? try to guess the file format from the file extension
  if (eFormat == FIF_UNKNOWN)
  {
      eFormat = FreeImage_GetFIFFromFilename(rFileName.c_str());
  }

  NPP_ASSERT(eFormat != FIF_UNKNOWN);
  // check that the plugin has reading capabilities ...
  FIBITMAP *pBitmap;

  if (FreeImage_FIFSupportsReading(eFormat))
  {
      FIBITMAP *pTempBitmap = FreeImage_Load(eFormat, rFileName.c_str());
      pBitmap = FreeImage_ConvertTo8Bits(pTempBitmap);
  }

  NPP_ASSERT(pBitmap != 0);
  // make sure this is an 8-bit single channel image
  //NPP_ASSERT(FreeImage_GetColorType(pBitmap) == FIC_MINISBLACK);
  //std::cout << FreeImage_GetBPP(pBitmap) << std::endl;
  NPP_ASSERT(FreeImage_GetBPP(pBitmap) == 8);

  // create an ImageCPU to receive the loaded image data
  npp::ImageCPU_8u_C3 oImage(FreeImage_GetWidth(pBitmap), FreeImage_GetHeight(pBitmap));

  // Copy the FreeImage data into the new ImageCPU
  unsigned int nSrcPitch = FreeImage_GetPitch(pBitmap);
  const Npp8u *pSrcLine = FreeImage_GetBits(pBitmap) + nSrcPitch * (FreeImage_GetHeight(pBitmap) -1);
  Npp8u *pDstLine = oImage.data();
  unsigned int nDstPitch = oImage.pitch();

  for (size_t iLine = 0; iLine < oImage.height(); ++iLine)
  {
      memcpy(pDstLine, pSrcLine, oImage.width() * sizeof(Npp8u));
      pSrcLine -= nSrcPitch;
      pDstLine += nDstPitch;
  }

  // swap the user given image with our result image, effecively
  // moving our newly loaded image data into the user provided shell
  oImage.swap(rImage);
}

// Save an RGB image to disk.
void
saveImage(const std::string &rFileName, const npp::ImageCPU_8u_C3 &rImage)
{
    // create the result image storage using FreeImage so we can easily
    // save
    FIBITMAP *pResultBitmap = FreeImage_Allocate(rImage.width(), rImage.height(), 8 /* bits per pixel */);
    NPP_ASSERT_NOT_NULL(pResultBitmap);
    unsigned int nDstPitch   = FreeImage_GetPitch(pResultBitmap);
    Npp8u *pDstLine = FreeImage_GetBits(pResultBitmap) + nDstPitch * (rImage.height()-1);
    const Npp8u *pSrcLine = rImage.data();
    unsigned int nSrcPitch = rImage.pitch();

    for (size_t iLine = 0; iLine < rImage.height(); ++iLine)
    {
        memcpy(pDstLine, pSrcLine, rImage.width() * sizeof(Npp8u));
        pSrcLine += nSrcPitch;
        pDstLine -= nDstPitch;
    }

    // now save the result image
    bool bSuccess;
    bSuccess = FreeImage_Save(FIF_PGM, pResultBitmap, rFileName.c_str(), 0) == TRUE;
    NPP_ASSERT_MSG(bSuccess, "Failed to save result image.");
}

__global__ void cut_blue_element(Npp8u* d_dst, int dstPitch, Npp8u* d_img, int srcPitch) {
  int idx = threadIdx.y * srcPitch + threadIdx.x;
  d_dst[idx] = d_img[idx] & 0xFC; // 1111 1100
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

      // declare a host image object for an 8-bit grayscale image
      npp::ImageCPU_8u_C3 oHostSrc;
      // load gray-scale image from disk
      loadImage(sFilename, oHostSrc);
      int width = (int)oHostSrc.width();
      int height = (int)oHostSrc.height();

      // declare a device image and copy construct from the host image,
      // i.e. upload host to device
      npp::ImageNPP_8u_C3 oDeviceSrc(oHostSrc);
      npp::ImageNPP_8u_C3 oDeviceDst(width, height);

      dim3 grdDim(1, 1, 1);
    	dim3 blkDim(width, height, 1);

      Npp8u *pSrc = oDeviceSrc.data();
      unsigned int nSrcPitch = oDeviceSrc.pitch();

      Npp8u *pDst = oDeviceDst.data();
      unsigned int nDstPitch = oDeviceDst.pitch();

      cut_blue_element<<<grdDim, blkDim>>>(pDst, nDstPitch, pSrc, nSrcPitch);

      // declare a host image for the result
      npp::ImageCPU_8u_C3 oHostDst(oDeviceDst.size());
      // and copy the device result data into it
      oDeviceDst.copyTo(oHostDst.data(), oHostDst.pitch());

      saveImage(sResultFilename, oHostDst);
      std::cout << "Saved image: " << sResultFilename << std::endl;

      nppiFree(oDeviceSrc.data());
      nppiFree(oDeviceDst.data());
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
    return -1;
  }

  return 0;
}
