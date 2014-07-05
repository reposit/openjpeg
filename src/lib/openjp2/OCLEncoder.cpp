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

#include "OCLEncoder.h"
#include "OCLPreprocessorEncode.h"
#include "OCLUtil.h"
#include "OCLMemoryManager.h"


OCLEncoder::OCLEncoder(ocl_args_d_t* ocl) : _ocl(ocl), 
	memoryManager(new OCLMemoryManager(ocl)),
	preprocessorEncode(new OCLPreprocessorEncode(KernelInitInfoBase(_ocl->commandQueue, "-I \"c:\\src\\openjpeg\\src\\lib\\openjp2\\\" "), memoryManager))
{

}


void OCLEncoder::preprocess(bool isLossy, std::vector<int*> components,int w,int h,int shift){

	if (preprocessorEncode) {
		preprocessorEncode->encode(isLossy, components,w,h,shift);
		if (components.size() == 3 && isLossy) {
			preprocessorEncode->copyFloatsFromDevice(components);
		} else {
			preprocessorEncode->copyShortsFromDevice(components, isLossy);

		}
	}
}

OCLEncoder::~OCLEncoder(){
	if (preprocessorEncode)
		delete preprocessorEncode;
	if (memoryManager)
		delete memoryManager;

}