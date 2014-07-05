/*  Copyright 2014 Aaron Boxer

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>. */

#include "OCLMemoryManager.h"
#include <math.h>
#include "OCLBasic.h"

OCLMemoryManager::OCLMemoryManager(ocl_args_d_t* ocl) :ocl(ocl),      
	                                        rgbaBuffer(NULL),
											width(0),
											height(0),
											preprocessIn(0),
											preprocessOut(0)
{
}


OCLMemoryManager::~OCLMemoryManager(void)
{

}


void OCLMemoryManager::fillHostInputBuffer(std::vector<int*> components,unsigned int w,	unsigned int h){

	int deviceComponents = numDeviceComponents(components);
	if (deviceComponents == 4) {
		//pack channels into RGBA format
		for (int i = 0; i < w*h * deviceComponents; i+=4) {
			 int componentIndex = i >> 2;
			 rgbaBuffer[i] = components[0][componentIndex] & 0xFF;
			 rgbaBuffer[i+1] = components[1][componentIndex] & 0xFF;
			 rgbaBuffer[i+2] = components[2][componentIndex] & 0xFF;
		}
	} else {
		for (int i = 0; i < w*h; i++) {
			 rgbaBuffer[i] = components[0][i] & 0xFF;
		}
	}
}


void OCLMemoryManager::init(std::vector<int*> components,	unsigned int w,	unsigned int h, bool floatingPointOnDevice){
	if (w <=0 || h <= 0 || components.size() == 0)
		return;

	if (w != width || h != height) {

		freeBuffers();
		cl_uint align = requiredOpenCLAlignment(ocl->device);
		rgbaBuffer = (uint16_t*)aligned_malloc(w*h*sizeof(unsigned short) * numDeviceComponents(components), 4*1024);

		fillHostInputBuffer(components,w,h);
		 
		cl_context context  = NULL;
		// Obtain the OpenCL context from the command-queue properties
		cl_int error_code = clGetCommandQueueInfo(ocl->commandQueue, CL_QUEUE_CONTEXT, sizeof(cl_context), &context, NULL);
		if (CL_SUCCESS != error_code)
		{
			LogError("Error: clGetCommandQueueInfo (CL_QUEUE_CONTEXT) returned %s.\n", TranslateOpenCLError(error_code));
			return;
		}

		//allocate input buffer
		cl_image_desc desc;
		desc.image_type = CL_MEM_OBJECT_IMAGE2D;
		desc.image_width = w;
		desc.image_height = h;
		desc.image_depth = 0;
		desc.image_array_size = 0;
		desc.image_row_pitch = 0;
		desc.image_slice_pitch = 0;
		desc.num_mip_levels = 0;
		desc.num_samples = 0;
		desc.buffer = NULL;

		cl_image_format format;
		format.image_channel_order = components.size() == 3 ? CL_RGBA : CL_R;
		format.image_channel_data_type = CL_UNSIGNED_INT16;
		preprocessIn = clCreateImage (context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, &format, &desc,	rgbaBuffer,	&error_code);
		if (CL_SUCCESS != error_code)
		{
			LogError("Error: clCreateImage (CL_QUEUE_CONTEXT) returned %s.\n", TranslateOpenCLError(error_code));
			return;
		}
		format.image_channel_data_type = floatingPointOnDevice ? CL_FLOAT : CL_SIGNED_INT16;
		preprocessOut = clCreateImage (context, CL_MEM_WRITE_ONLY, &format, &desc, NULL,&error_code);
		if (CL_SUCCESS != error_code)
		{
			LogError("Error: clCreateImage (CL_QUEUE_CONTEXT) returned %s.\n", TranslateOpenCLError(error_code));
			return;
		}
		width = w;
	    height = h;
	} else {
		fillHostInputBuffer(components,width,height);

		size_t origin[] = {0,0,0}; // Defines the offset in pixels in the image from where to write.
		size_t region[] = {width, height, 1}; // Size of object to be transferred
		cl_int error_code = clEnqueueWriteImage(ocl->commandQueue, preprocessIn, CL_TRUE, origin, region,0,0, rgbaBuffer, 0, NULL,NULL);
		if (CL_SUCCESS != error_code)
		{
			LogError("Error: clEnqueueWriteImage (CL_QUEUE_CONTEXT) returned %s.\n", TranslateOpenCLError(error_code));
			return;
		}
	}
}

void OCLMemoryManager::freeBuffers(){
	if (rgbaBuffer) {
		aligned_free(rgbaBuffer);
		rgbaBuffer = NULL;
	}

	cl_context context  = NULL;
	// Obtain the OpenCL context from the command-queue properties
	cl_int error_code = clGetCommandQueueInfo(ocl->commandQueue, CL_QUEUE_CONTEXT, sizeof(cl_context), &context, NULL);
	if (CL_SUCCESS != error_code)
	{
		LogError("Error: clGetCommandQueueInfo (CL_QUEUE_CONTEXT) returned %s.\n", TranslateOpenCLError(error_code));
		return;
	}
	// release old buffers
	if (preprocessIn) {
		error_code = clReleaseMemObject(preprocessIn);
		if (CL_SUCCESS != error_code)
		{
			LogError("Error: clReleaseMemObject (CL_QUEUE_CONTEXT) returned %s.\n", TranslateOpenCLError(error_code));
			return;
		}
	}
	if (preprocessOut) {
		error_code = clReleaseMemObject(preprocessOut);
		if (CL_SUCCESS != error_code)
		{
			LogError("Error: clReleaseMemObject (CL_QUEUE_CONTEXT) returned %s.\n", TranslateOpenCLError(error_code));
			return;
		}
	}	

}
