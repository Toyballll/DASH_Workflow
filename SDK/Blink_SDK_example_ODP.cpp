#include "Blink_C_wrapper.h"  // Relative path to SDK header.
#include "ImageGen.h"
#include "math.h"
#include <Windows.h>

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

  //if bits per pixel is wrong, the lower level code will figure out
  //what it should be and construct properly.
  //SLM_lut.txt is a generic ODP regional LUT, this should be replaced with your custom calibration
  Create_SDK(bits_per_pixel, &n_boards_found,
    &constructed_okay, is_nematic_type, RAM_write_enable,
    use_GPU_if_available, 10U, "SLM_lut.txt");

  if (n_boards_found == 1)
  {
    board_number = 1;
    int height = Get_image_height(board_number);
	int width = Get_image_width(board_number);

	// Create two vectors to hold values for two SLM images with opposite ramps.
	unsigned char* ImageOne = new unsigned char[width*height];
	unsigned char* ImageTwo = new unsigned char[width*height];
	// Generate phase gradients
	int VortexCharge = 5;
	Generate_LG(ImageOne, width, height, VortexCharge, width / 2.0, height / 2.0, false);
	VortexCharge = 3;
	Generate_LG(ImageTwo, width, height, VortexCharge, width / 2.0, height / 2.0, false);

	//set the basic SLM parameters
	Set_true_frames(5);
	// A linear LUT must be loaded to the controller for OverDrive Plus
	Load_linear_LUT(board_number);
	//Turn the SLM power on
	SLM_power(true);

	bool ExternalTrigger = false;
	bool OutputPulseImageFlip = false;
	for (int i = 0; i < 5; i++)
	{
		Write_overdrive_image(board_number, ImageOne, ExternalTrigger, OutputPulseImageFlip, 5000);
		Sleep(500);

		Write_overdrive_image(board_number, ImageTwo, ExternalTrigger, OutputPulseImageFlip, 5000);
		Sleep(500);
	}

	delete[]ImageOne;
	delete[]ImageTwo;

	SLM_power(false);
 	return EXIT_SUCCESS;
  }

  return EXIT_FAILURE;
}
