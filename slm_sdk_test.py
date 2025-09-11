"""
SLM SDK - 正确的精简版本
加载mlo.bmp到1024x1024 SLM
"""

import numpy as np
from ctypes import *
from PIL import Image as PILImage

# 加载SDK库
cdll.LoadLibrary("C:\\Program Files\\Meadowlark Optics\\Blink OverDrive Plus\\SDK\\Blink_C_wrapper")
slm_lib = CDLL("Blink_C_wrapper")

cdll.LoadLibrary("C:\\Program Files\\Meadowlark Optics\\Blink OverDrive Plus\\SDK\\ImageGen")
image_lib = CDLL("ImageGen")

# 初始化SDK
num_boards_found = c_uint(0)
slm_lib.Create_SDK(c_uint(12), byref(num_boards_found), byref(c_uint(-1)),
                   c_bool(1), c_bool(1), c_bool(1), c_uint(20), 0)

print(f"Found {num_boards_found.value} SLM controller(s)")

# 获取SLM参数
board_number = c_uint(1)
height = slm_lib.Get_image_height(board_number)
width = slm_lib.Get_image_width(board_number)
depth = slm_lib.Get_image_depth(board_number)

print(f"SLM: {width}x{height}, {depth} bits")

# 加载LUT
slm_lib.Load_LUT_file(board_number, b"C:\\Program Files\\Meadowlark Optics\\Blink OverDrive Plus\\LUT Files\\1024x1024_linearVoltage.LUT")

# 加载mlo.bmp图像
img = PILImage.open("C:\\Program Files\\Meadowlark Optics\\Blink OverDrive Plus\\Image Files\\1024\\Central.bmp")
img_array = np.array(img.convert('L'), dtype=np.uint8)

# 准备图像数组
ImgSize = height * width * (depth // 8)
ImageArray = np.zeros([ImgSize], np.uint8, 'C')

# 初始化SLM（写入空白图像）
slm_lib.Write_image(board_number, ImageArray.ctypes.data_as(POINTER(c_ubyte)), ImgSize,
                    c_uint(0), c_uint(0), c_uint(0), c_uint(0), c_uint(5000))

# 填充图像数据
ImageArray[:] = img_array.flatten()

# 写入图像到SLM
slm_lib.Write_image(board_number, ImageArray.ctypes.data_as(POINTER(c_ubyte)), ImgSize,
                    c_uint(0), c_uint(0), c_uint(0), c_uint(0), c_uint(5000))

# 等待写入完成
slm_lib.ImageWriteComplete(board_number, c_uint(5000))

print("Image displayed on SLM. Press Enter to exit...")
input()

# 清理资源
slm_lib.Delete_SDK()