#include "Blink_C_wrapper.h"  // Relative path to SDK header.
#include "ImageGen.h"
#include "math.h"
#include <afxwin.h> 

// ------------------------- Blink_SDK_example --------------------------------
// Simple example using the Blink_SDK DLL to send a sequence of phase targets
// to a single SLM.
// To run the example, ensure that Blink_SDK.dll is in the same directory as
// the Blink_SDK_example.exe.
// ----------------------------------------------------------------------------
int main()
{
  int board_number;
  // Construct a Blink_SDK instance
  unsigned int bits_per_pixel = 12U;   //12U is used for 1920x1152 SLM, 8U used for the small 512x512
  bool         is_nematic_type = true;
  bool         RAM_write_enable = true;
  bool         use_GPU_if_available = true;
  unsigned int n_boards_found = 0U;
  int         constructed_okay = true;
  bool ExternalTrigger = false;
  bool FlipImmediate = false; //only supported on the 1k
  
  //Both pulse options can be false, but only one can be true. You either generate a pulse when the new image begins loading to the SLM
  //or every 1.184 ms on SLM refresh boundaries, or if both are false no output pulse is generated.
  bool OutputPulseImageFlip = false;
  bool OutputPulseImageRefresh = false; //only supported on 1920x1152, FW rev 1.8. 

  //if bits per pixel is wrong, the lower level code will figure out
  //what it should be and construct properly.
  Create_SDK(bits_per_pixel, &n_boards_found, &constructed_okay, is_nematic_type, RAM_write_enable, use_GPU_if_available, 10U, 0);

  // return of 1 means okay
  if (constructed_okay != 1)
    ::AfxMessageBox(Get_last_error_message());

  // return of 0 means okay, return -1 means error
  if (n_boards_found > 0)
  {
	  board_number = 1;
	  int height = Get_image_height(board_number);
	  int width = Get_image_width(board_number);
	  int depth = Get_image_depth(board_number); //bits per pixel
	  int Bytes = depth/8;
    int ImgSize = height*width*Bytes;
	
      //***you should replace *_linearVoltage.LUT with your custom LUT file***
	  //but for now open a generic LUT that linearly maps input graylevels to output voltages
	  //***Using *_linearVoltage.LUT does NOT give a linear phase response***
	  if((width == 512)&&(depth == 8))
		  Load_LUT_file(board_number, "C:\\Program Files\\Meadowlark Optics\\Blink OverDrive Plus\\LUT Files\\512x512_linearVoltage.LUT");
	  if((width == 512)&&(depth == 16))
		  Load_LUT_file(board_number, "C:\\Program Files\\Meadowlark Optics\\Blink OverDrive Plus\\LUT Files\\512x512_16bit_linearVoltage.LUT");	  
	  if(width == 1920)
		  Load_LUT_file(board_number, "C:\\Program Files\\Meadowlark Optics\\Blink OverDrive Plus\\LUT Files\\1920x1152_linearVoltage.LUT");
	  if(width == 1024)
		  Load_LUT_file(board_number, "C:\\Program Files\\Meadowlark Optics\\Blink OverDrive Plus\\LUT Files\\1024x1024_linearVoltage.LUT");
	

	  //to keep the example generic a blank wavefront correction is used, you can replace this with your real wavefront correction
	  unsigned char* WFC = new unsigned char[ImgSize];
	  memset(WFC, 0, ImgSize);

	  // Create two vectors to hold values for two SLM images with opposite ramps.
	  unsigned char* ImageOne = new unsigned char[ImgSize];
	  unsigned char* ImageTwo = new unsigned char[ImgSize];
	  memset(ImageOne, 0, ImgSize);
	  memset(ImageTwo, 0, ImgSize);
	
	  //start the SLM with a blank image
	  Write_image(board_number, ImageOne, ImgSize, ExternalTrigger, FlipImmediate, OutputPulseImageFlip, OutputPulseImageRefresh, 5000);
	  ImageWriteComplete(board_number, 5000);	
	
	  // Generate phase gradients
	  int VortexCharge = 5;
	  bool RGB = false;
	  bool fork = false;
	  Generate_LG(ImageOne, WFC, width, height, depth, VortexCharge, width / 2.0, height / 2.0, fork, RGB);
	  VortexCharge = 3;
	  Generate_LG(ImageTwo, WFC, width, height, depth, VortexCharge, width / 2.0, height / 2.0, fork, RGB);

	  for (int i = 0; i < 10; i++)
	  {
		  //write image returns on DMA complete, ImageWriteComplete returns when the hardware
		  //image buffer is ready to receive the next image. Breaking this into two functions is 
		  //useful for external triggers. It is safe to apply a trigger when Write_image is complete
		  //and it is safe to write a new image when ImageWriteComplete returns
		  Write_image(board_number, ImageOne, ImgSize, ExternalTrigger, FlipImmediate, OutputPulseImageFlip, OutputPulseImageRefresh, 5000);
		  ImageWriteComplete(board_number, 5000);
		  Sleep(500);

		  Write_image(board_number, ImageTwo, ImgSize, ExternalTrigger, FlipImmediate, OutputPulseImageFlip, OutputPulseImageRefresh, 5000);
		  ImageWriteComplete(board_number, 5000);
		  Sleep(500);
	  }

	  delete[]ImageOne;
	  delete[]ImageTwo;
	  delete[]WFC;

	  SLM_power(false);
	  Delete_SDK();
	  return EXIT_SUCCESS;
  }

  return EXIT_FAILURE;
}