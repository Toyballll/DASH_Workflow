% Example usage of Blink_C_wrapper.dll
% Meadowlark Optics Spatial Light Modulators
% last updated: September 12, 2019

% Load the DLL
% Blink_C_wrapper.dll, Blink_SDK.dll, ImageGen.dll, FreeImage.dll and wdapi1021.dll
% should all be located in the same directory as the program referencing the
% library
if ~libisloaded('Blink_C_wrapper')
    loadlibrary('Blink_C_wrapper.dll', 'Blink_C_wrapper.h');
end

% This loads the image generation functions
if ~libisloaded('ImageGen')
    loadlibrary('ImageGen.dll', 'ImageGen.h');
end

% Basic parameters for calling Create_SDK
bit_depth = 12; %bit depth = 8 for small 512, 12 for 1920
num_boards_found = libpointer('uint32Ptr', 0);
constructed_okay = libpointer('int32Ptr', 0);
is_nematic_type = 1;
RAM_write_enable = 1;
use_GPU = 0;
max_transients = 10;
wait_For_Trigger = 0; % This feature is user-settable; use 1 for 'on' or 0 for 'off'
OutputPulseImageFlip = 0;
timeout_ms = 5000;
board_number = 1;


% Call the constructor
%***SLM_lut.txt is a generic ODP regional calibration. You should replace SLM_lut.txt with your custom calibration***
lut_file = 'C:\\Program Files\\Meadowlark Optics\\Blink OverDrive Plus\\LUT Files\\SLM_lut.txt';
calllib('Blink_C_wrapper', 'Create_SDK', bit_depth, num_boards_found, constructed_okay, is_nematic_type, RAM_write_enable, use_GPU, max_transients, lut_file);

if constructed_okay.value ~= 1  % 1 is success
    disp(calllib('Blink_C_wrapper', 'Get_last_error_message'));
end

if num_boards_found == 1
    disp('Blink SDK was successfully constructed');
	height = calllib('Blink_C_wrapper', 'Get_image_height', board_number);
    width = calllib('Blink_C_wrapper', 'Get_image_width', board_number);
	
    %allocate arrays for our images
	ImageOne = libpointer('uint8Ptr', zeros(width*height,1));
    ImageTwo = libpointer('uint8Ptr', zeros(width*height,1));

    % Generate a fresnel lens
    CenterX = width/2;
    CenterY = height/2;
    Radius = height/2;
    Power = 1;
    cylindrical = true;
    horizontal = false;
    calllib('ImageGen', 'Generate_FresnelLens', ImageOne, width, height, CenterX, CenterY, Radius, Power, cylindrical, horizontal);
    ImageOne = reshape(ImageOne.Value, [width,height]);

    % Generate a blazed grating
    Period = 128;
    Increasing = 1;
    calllib('ImageGen', 'Generate_Grating', ImageTwo, width, height, Period, Increasing, horizontal);
    ImageTwo = reshape(ImageTwo.Value, [width,height]);
	
    fprintf('Found %u SLM controller(s)', num_boards_found.value);
    % Set the basic SLM parameters
    calllib('Blink_C_wrapper', 'Set_true_frames', 5);
    % A linear LUT must be loaded to the controller for OverDrive Plus
    calllib('Blink_C_wrapper', 'Load_linear_LUT', board_number);
    % Turn the SLM power on
    calllib('Blink_C_wrapper', 'SLM_power', board_number);

    % Loop between our ramp images
    for n = 1:5
        calllib('Blink_C_wrapper', 'Write_overdrive_image', board_number, ImageOne, wait_For_Trigger, OutputPulseImageFlip, timeout_ms);
        pause(2.0) % This is in seconds
        calllib('Blink_C_wrapper', 'Write_overdrive_image', board_number, ImageTwo, wait_For_Trigger, OutputPulseImageFlip, timeout_ms);
        pause(2.0) % This is in seconds
    end
    
    % Always call Delete_SDK before exiting
    calllib('Blink_C_wrapper', 'Delete_SDK');
end

unloadlibrary('Blink_C_wrapper');