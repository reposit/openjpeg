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

#include "OCLInterface.h"

#include "OCLDeviceManager.h"
#include "OCLEncoder.h"

OCLDeviceManager* deviceManager;
OCLEncoder* encoder;

void ocl_destroyDeviceManager() {
	if (deviceManager)
		delete deviceManager;
	deviceManager = NULL;
}

int ocl_createDeviceManager() {
	if (deviceManager)
		return 0;
	deviceManager =  new OCLDeviceManager();
	return 1;

}
int ocl_initializeDeviceManager() {
	if (!deviceManager)
		return 0;
	return deviceManager->init();
}

int ocl_createEncoder() {
	if (encoder || !deviceManager)
		return 0;
	ocl_args_d_t* info = deviceManager->getInfo(GPU);
	if (!info)
		info = deviceManager->getInfo(CPU);
	encoder = new OCLEncoder(info);
	return 1;

}

void ocl_encode_preprocess(int isLossy, int* c0,int* c1,int* c2,int w,int h,int shift){
   if (encoder) {
	   std::vector<int*> comps;
	   if (c0)
		comps.push_back(c0);
	   if (c1)
		comps.push_back(c1);
	   if (c2)
		comps.push_back(c2);
	   encoder->preprocess(isLossy, comps,w,h,shift);
   }
}

void ocl_destroyEncoder() {
	if (!encoder)
		return;
	delete encoder;
}

void ocl_cleanup() {
	ocl_destroyEncoder();
	ocl_destroyDeviceManager();
}
