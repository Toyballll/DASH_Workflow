# # Example usage of Blink_C_wrapper.dll
# # Meadowlark Optics Spatial Light Modulators
# # September 12 2019
#
# import os
# import numpy
# from ctypes import *
# from scipy import misc
# from time import sleep
#
# # Load the DLL
# # Blink_C_wrapper.dll, Blink_SDK.dll, ImageGen.dll, FreeImage.dll and wdapi1021.dll
# # should all be located in the same directory as the program referencing the
# # library
# cdll.LoadLibrary("Blink_C_wrapper")
# slm_lib = CDLL("Blink_C_wrapper")
#
# # Open the image generation library
# cdll.LoadLibrary("ImageGen")
# image_lib = CDLL("ImageGen")
#
# # Basic parameters for calling Create_SDK
# bit_depth = c_uint(12)
# num_boards_found = c_uint(0)
# constructed_okay = c_bool(0)
# is_nematic_type = c_bool(1)
# RAM_write_enable = c_bool(1)
# use_GPU = c_bool(1)
# max_transients = c_uint(20)
# board_number = c_uint(1)
# wait_For_Trigger = c_uint(0)
# OutputPulseImageFlip = c_uint(0)
# timeout_ms = c_uint(5000)
# center_x = c_float(256)
# center_y = c_float(256)
# VortexCharge = c_uint(3)
#
#
# # Call the Create_SDK constructor
#
# # SLM_lut.txt is a generic ODP regional LUT, this should be replaced with your custom calibration
# slm_lib.Create_SDK(bit_depth, byref(num_boards_found),
#                    byref(constructed_okay), is_nematic_type,
#                    RAM_write_enable, use_GPU, max_transients, "SLM_lut.txt")
#
# if constructed_okay.value == 0:
#     slm_lib.Get_last_error_message.restype = c_char_p
#     print slm_lib.Get_last_error_message()
#
# if num_boards_found.value == 1:
#     print "Blink SDK was successfully constructed"
#     print "Found %s SLM controller(s)" % num_boards_found.value
#     height = c_uint(slm_lib.Get_image_height(board_number));
#     width = c_uint(slm_lib.Get_image_width(board_number));
#     center_x = c_uint(width.value/2);
#     center_y = c_uint(height.value/2);
#
#     # Create two vectors to hold values for two SLM images with opposite ramps.
#     ImageOne = numpy.empty([width.value*height.value], numpy.uint8, 'C');
#     ImageTwo = numpy.empty([width.value*height.value], numpy.uint8, 'C');
# 	# Generate phase gradients
#     VortexCharge = 5;
#     image_lib.Generate_LG(ImageOne.ctypes.data_as(POINTER(c_ubyte)), width.value, height.value, VortexCharge, center_x.value, center_y.value, 0);
#     VortexCharge = 3;
#     image_lib.Generate_LG(ImageTwo.ctypes.data_as(POINTER(c_ubyte)), width.value, height.value, VortexCharge, center_x.value, center_y.value, 0);
#
#     # Set the basic SLM parameters
#     slm_lib.Set_true_frames(5)
#     # A linear LUT must be loaded to the controller for OverDrive Plus
#     slm_lib.Load_linear_LUT(board_number)
#     # Turn the SLM power on
#     slm_lib.SLM_power(c_bool(1))
#
#     # Loop between our ramp images
#     for i in range(0, 10):
#         slm_lib.Write_overdrive_image(board_number, ImageOne.ctypes.data_as(POINTER(c_ubyte)), wait_For_Trigger, OutputPulseImageFlip, timeout_ms)
#         sleep(1.0) # This is in seconds
#         slm_lib.Write_overdrive_image(board_number, ImageTwo.ctypes.data_as(POINTER(c_ubyte)), wait_For_Trigger, OutputPulseImageFlip, timeout_ms)
#         sleep(1.0) # This is in seconds
#
#     # Always call Delete_SDK before exiting
#     slm_lib.Delete_SDK()