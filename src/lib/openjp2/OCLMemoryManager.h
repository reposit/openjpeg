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

#pragma once

#include "ocl_platform.h"
#include "OCLUtil.h"

#include <vector>
#include <stdint.h>

class OCLMemoryManager
{
public:
	OCLMemoryManager(ocl_args_d_t* ocl);
	~OCLMemoryManager(void);
	unsigned short getWidth() {return width;}
	unsigned short getHeight() {return height;}
	cl_mem* getPreprocessIn(){ return &preprocessIn;}
	cl_mem* getPreprocessOut(){ return &preprocessOut;}

	void init(std::vector<int*> components,unsigned int w, unsigned int h, bool floatingPointOnDevice);

private:

	int numDeviceComponents(std::vector<int*> components){ return components.size() ==3 ? 4 : 1;}

	void fillHostInputBuffer(std::vector<int*> components,unsigned int w,	unsigned int h);
	void freeBuffers();
	uint16_t* rgbaBuffer;

	ocl_args_d_t* ocl;
	unsigned int width;
	unsigned int height;
	cl_mem preprocessIn;  // rgba
	cl_mem preprocessOut;  // rgba

};


