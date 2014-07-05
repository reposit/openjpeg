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

#include "OCLPreprocessorEncode.h"
#include "OCLMemoryManager.h"
#include <stdint.h>

const int32_t factor = (1 << 11);
const int32_t half_factor = (1 << 10);

OCLPreprocessorEncode::OCLPreprocessorEncode(KernelInitInfoBase initInfo, OCLMemoryManager* memMgr) : 
	                OCLPreprocessor(initInfo),
					memoryManager(memMgr),
					ict_with_shift(new OCLKernel( KernelInitInfo(initInfo, "c:\\src\\openjpeg\\src\\lib\\openjp2\\ocl_preprocessor.cl", "ict_with_shift") )),
					ict(new OCLKernel( KernelInitInfo(initInfo, "c:\\src\\openjpeg\\src\\lib\\openjp2\\ocl_preprocessor.cl", "ict") )),
					rct_with_shift(new OCLKernel( KernelInitInfo(initInfo, "c:\\src\\openjpeg\\src\\lib\\openjp2\\ocl_preprocessor.cl", "rct_with_shift") )),
					rct(new OCLKernel( KernelInitInfo(initInfo, "c:\\src\\openjpeg\\src\\lib\\openjp2\\ocl_preprocessor.cl", "rct") )),
					dcShift(new OCLKernel( KernelInitInfo(initInfo, "c:\\src\\openjpeg\\src\\lib\\openjp2\\ocl_preprocessor.cl", "fdc_level_shift") ))
											
{
}


OCLPreprocessorEncode::~OCLPreprocessorEncode(void)
{
	if (ict_with_shift)
		delete ict_with_shift;
	if (ict)
		delete ict;
	if (rct_with_shift)
		delete rct_with_shift;
	if (rct)
		delete rct;
	if (dcShift)
		delete dcShift;
}

void OCLPreprocessorEncode::encode(	bool lossy, std::vector<int*> components,	int w,	int h,	int shift) {

	if (components.size() == 0)
		return;
	OCLKernel* targetKernel = NULL;
	bool floatingPointOnDevice = false;
	if (components.size() == 3) {
		if (lossy) {
			targetKernel = (shift) ? ict_with_shift : ict;
			floatingPointOnDevice = true;
		} else {
			targetKernel = (shift) ? rct_with_shift : rct;
		}
	} else if (components.size() == 1) {
		targetKernel = dcShift;
	}

	memoryManager->init(components,w,h,floatingPointOnDevice);
	setKernelArgs(targetKernel, shift);
	int workGroupDim = 16;
	size_t local_work_size[3] = {workGroupDim,workGroupDim,1};
	int numGroupsX = (size_t)ceil(((float)memoryManager->getWidth())/workGroupDim);
	int numGroupsY = (size_t)ceil(((float)memoryManager->getHeight())/workGroupDim);
	size_t global_work_size[3] = {workGroupDim * numGroupsX, workGroupDim * numGroupsY,1};
	targetKernel->enqueue(2,global_work_size, local_work_size);


}


void OCLPreprocessorEncode::copyFloatsFromDevice(std::vector<int*> components){
	cl_int error_code = CL_SUCCESS;
    float *destination_data = NULL;
	size_t image_dimensions[3] = { memoryManager->getWidth(), memoryManager->getHeight(), 1 };
    size_t image_origin[3] = { 0, 0, 0 };
    size_t image_pitch = 0;

    destination_data = (float*)clEnqueueMapImage(    initInfo.cmd_queue,
                                            *memoryManager->getPreprocessOut(),
                                            CL_FALSE,
                                            CL_MAP_READ,
                                            image_origin,
                                            image_dimensions,
                                            &image_pitch,
                                            NULL,
                                            0,
                                            NULL,
                                            NULL,
                                            &error_code);
    if (CL_SUCCESS != error_code)
    {
        LogError("Error: clEnqueueMapBuffer return %s.\n", TranslateOpenCLError(error_code));
		return;
    }

	error_code  = clFinish(initInfo.cmd_queue);
	if (CL_SUCCESS != error_code)
    {
        LogError("Error: clFinish returned %s.\n", TranslateOpenCLError(error_code));
        return;
    }


	for (int i = 0; i < memoryManager->getWidth() * memoryManager->getHeight()*4; i+=4) {
		float* pixel = destination_data + i;
		unsigned componentIndex = i >> 2;
		components[0][componentIndex] =  (int)((pixel[0] * factor) + half_factor);
		components[1][componentIndex] =  (int)((pixel[1] * factor) + half_factor);
		components[2][componentIndex] =  (int)((pixel[2] * factor) + half_factor);
	}
}

void OCLPreprocessorEncode::copyShortsFromDevice(std::vector<int*> components, bool lossy){
	cl_int error_code = CL_SUCCESS;
    int16_t *destination_data = NULL;
	size_t image_dimensions[3] = { memoryManager->getWidth(), memoryManager->getHeight(), 1 };
    size_t image_origin[3] = { 0, 0, 0 };
    size_t image_pitch = 0;

    destination_data = (int16_t*)clEnqueueMapImage(    initInfo.cmd_queue,
                                            *memoryManager->getPreprocessOut(),
                                            CL_FALSE,
                                            CL_MAP_READ,
                                            image_origin,
                                            image_dimensions,
                                            &image_pitch,
                                            NULL,
                                            0,
                                            NULL,
                                            NULL,
                                            &error_code);
    if (CL_SUCCESS != error_code)
    {
        LogError("Error: clEnqueueMapBuffer return %s.\n", TranslateOpenCLError(error_code));
		return;
    }

	error_code  = clFinish(initInfo.cmd_queue);
	if (CL_SUCCESS != error_code)
    {
        LogError("Error: clFinish returned %s.\n", TranslateOpenCLError(error_code));
        return;
    }

	int numComponents = components.size() == 3 ? 4 : 1;
	if (lossy) {
		for (int i = 0; i < memoryManager->getWidth() * memoryManager->getHeight()*numComponents; i+=numComponents) {
		int16_t* pixel = destination_data + i;
		unsigned componentIndex = i /numComponents;
		if (numComponents == 4) {
			components[0][componentIndex] =  (int)((pixel[0] * factor) + half_factor);
			components[1][componentIndex] =  (int)((pixel[1] * factor) + half_factor);
			components[2][componentIndex] =  (int)((pixel[2] * factor) + half_factor);
		} else {
			components[0][componentIndex] =  (int)((pixel[0] * factor) + half_factor);
		}
	}

	}
	else {
		for (int i = 0; i < memoryManager->getWidth() * memoryManager->getHeight()*numComponents; i+=numComponents) {
		int16_t* pixel = destination_data + i;
		unsigned componentIndex = i /numComponents;
		if (numComponents == 4) {
			components[0][componentIndex] =  pixel[0];
			components[1][componentIndex] =  pixel[1];
			components[2][componentIndex] =  pixel[2];
		} else {
			components[0][componentIndex] =  pixel[0];
		}
	}

	}
	
}


tDeviceRC OCLPreprocessorEncode::setKernelArgs(OCLKernel* myKernel, int shift){

	cl_int error_code =  DeviceSuccess;
	cl_kernel targetKernel = myKernel->getKernel();
	int argNum = 0;
	int width = memoryManager->getWidth();
	int height = memoryManager->getHeight();

	error_code = clSetKernelArg(targetKernel, argNum++, sizeof(cl_mem),  memoryManager->getPreprocessIn());
	if (DeviceSuccess != error_code)
	{
		LogError("Error: setKernelArgs returned %s.\n", TranslateOpenCLError(error_code));
		return error_code;
	}

	error_code = clSetKernelArg(targetKernel, argNum++, sizeof(cl_mem), memoryManager->getPreprocessOut());
	if (DeviceSuccess != error_code)
	{
		LogError("Error: setKernelArgs returned %s.\n", TranslateOpenCLError(error_code));
		return error_code;
	}
	error_code = clSetKernelArg(targetKernel, argNum++, sizeof(width), &width);
	if (DeviceSuccess != error_code)
	{
		LogError("Error: setKernelArgs returned %s.\n", TranslateOpenCLError(error_code));
		return error_code;
	}
	error_code = clSetKernelArg(targetKernel, argNum++, sizeof(height), &height);
	if (DeviceSuccess != error_code)
	{
		LogError("Error: setKernelArgs returned %s.\n", TranslateOpenCLError(error_code));
		return error_code;
	}
	if (shift) {
		error_code = clSetKernelArg(targetKernel,argNum++, sizeof(shift), &shift);
		if (DeviceSuccess != error_code)
		{
			LogError("Error: setKernelArgs returned %s.\n", TranslateOpenCLError(error_code));
			return error_code;
		}
	}
	return DeviceSuccess;
}

