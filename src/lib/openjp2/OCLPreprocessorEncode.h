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

#include "OCLKernel.h"
#include "OCLPreprocessor.h"
#include <vector>
class OCLMemoryManager;
#include "OCLInterface.h"


class OCLPreprocessorEncode : public OCLPreprocessor
{
public:
	OCLPreprocessorEncode(KernelInitInfoBase initInfo, OCLMemoryManager* memMgr);
	~OCLPreprocessorEncode(void);

	void encode(bool lossy, std::vector<int*> components,int w,	int h,	int shift);
	void copyShortsFromDevice(std::vector<int*> components, bool lossy);
	void copyFloatsFromDevice(std::vector<int*> components);
private:

	tDeviceRC setKernelArgs(OCLKernel* myKernel, int shift);
	OCLMemoryManager* memoryManager;
	OCLKernel* ict_with_shift;
	OCLKernel* ict;
	OCLKernel* rct_with_shift;
	OCLKernel* rct;
	OCLKernel* dcShift;


};

