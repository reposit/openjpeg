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

#include "ocl_platform.cl"

CONSTANT float Wr = 0.299f;
//CONSTANT float Wg = 1.0f - Wr - Wb;
CONSTANT float Wg = 1.0f - 0.299f - 0.114f;
CONSTANT float Wb = 0.114f;

CONSTANT float Umax = 0.436f;
CONSTANT float Vmax = 0.615f;

CONSTANT sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;



void KERNEL fdc_level_shift(__read_only image2d_t idata, __write_only image2d_t odata, 
                                        const unsigned int  width, const unsigned int  height, const int level_shift) {
	const int2 pos = {get_global_id(0), get_global_id(1)};
	if (pos.x >= width || pos.y >= height)
	   return;
     write_imagei(odata, pos, read_imageui(idata, sampler, pos).x - level_shift);
}



void KERNEL idc_level_shift(__read_only image2d_t idata, __write_only image2d_t odata, 
									const unsigned int  width, const unsigned int  height,  const int level_shift) {
	const int2 pos = {get_global_id(0), get_global_id(1)};
	if (pos.x >= width || pos.y >= height)
	   return;
    write_imagei(odata, pos, read_imageui(idata, sampler, pos).x + level_shift  );
}

void KERNEL ict_with_shift(__read_only image2d_t input,	 __write_only image2d_t output,
                      const unsigned int  width, const unsigned int  height, const int level_shift) {

	const int2 pos = {get_global_id(0), get_global_id(1)};
	if (pos.x >= width || pos.y >= height)
	   return;

	uint4 pixel =  read_imageui(input, sampler, pos);
	short r =  pixel.x - level_shift;
	short g =  pixel.y - level_shift;
	short b =  pixel.z - level_shift;

	float y = 0.299*r + 0.587*g + 0.114*b;
	float cb = -0.169*r - 0.331*g + 0.5*b;
	float cr = 0.5*r - 0.419*g - 0.114*b;

	write_imagef(output, pos,(float4)(y,cb,cr,0));
}

void KERNEL ict(__read_only image2d_t input, __write_only image2d_t output,
                      const unsigned int  width, const unsigned int  height) {

	const int2 pos = {get_global_id(0), get_global_id(1)};
	if (pos.x >= width || pos.y >= height)
	   return;

	uint4 pixel =  read_imageui(input, sampler, pos);
	short r =  pixel.x;
	short g =  pixel.y;
	short b =  pixel.z;

	float y = 0.299*r + 0.587*g + 0.114*b;
	float cb = -0.169*r - 0.331*g + 0.5*b;
	float cr = 0.5*r - 0.419*g - 0.114*b;

	write_imagef(output, pos,(float4)(y,cb,cr,0));
}

void KERNEL inverse_ict_with_shift(__read_only image2d_t input,__write_only	 image2d_t output,
                             const unsigned int  width, const unsigned int  height, const int level_shift) {

    const int2 pos = {get_global_id(0), get_global_id(1)};
	if (pos.x >= width || pos.y >= height)
	   return;

	float4 pixel =  read_imagef(input, sampler, pos);
	float y =  pixel.x;
	float cb =  pixel.y ;
	float cr =  pixel.z;

	float r = (cr/0.173f) + y;
	float b = (cb/0.654f) + y;
	float g = (y - Wr*r - Wb*b)/Wg;

	write_imageui(output, pos, (uint4)(r + level_shift,g+level_shift, b+level_shift,0));

}

void KERNEL inverse_ict(__read_only image2d_t input,	__write_only image2d_t output,
                             const unsigned int  width, const unsigned int  height) {

    const int2 pos = {get_global_id(0), get_global_id(1)};
	if (pos.x >= width || pos.y >= height)
	   return;

	float4 pixel =  read_imagef(input, sampler, pos);
	float y =  pixel.x;
	float cb =  pixel.y ;
	float cr =  pixel.z;

	float r = (cr/0.173f) + y;
	float b = (cb/0.654f) + y;
	float g = (y - Wr*r - Wb*b)/Wg;

	write_imageui(output, pos, (uint4)(r,g, b,0));
}

void KERNEL rct_with_shift(__read_only image2d_t input,	__write_only image2d_t output,
								 const unsigned int width, const unsigned int height, const int level_shift) {

    const int2 pos = {get_global_id(0), get_global_id(1)};
	if (pos.x >= width || pos.y >= height)
	   return;

	uint4 pixel =  read_imageui(input, sampler, pos);
	short r =  pixel.x;
	short g =  pixel.y;
	short b =  pixel.z;
	short y =  ((r + (g<<1) + b)>>2) - level_shift;
	write_imagei(output, pos, (int4)( y, b-g, r-g,0)) ;
}

void KERNEL rct(__read_only image2d_t input, __write_only image2d_t output, const unsigned int width, const unsigned int height) {

    const int2 pos = {get_global_id(0), get_global_id(1)};
	if (pos.x >= width || pos.y >= height)
	   return;

	uint4 pixel =  read_imageui(input, sampler, pos);
	short r =  pixel.x;
	short g =  pixel.y;
	short b =  pixel.z;
	short y =  ((r + (g<<1) + b)>>2);
	write_imagei(output, pos, (int4)( y, b-g, r-g,0)) ;
}

void KERNEL inverse_rct(__read_only image2d_t input,	__write_only image2d_t output,
                                         const unsigned int  width, const unsigned int  height, const int level_shift) {

    const int2 pos = {get_global_id(0), get_global_id(1)};
	if (pos.x >= width || pos.y >= height)
	   return;

	int4 pixel =  read_imagei(input, sampler, pos);
	short y =  pixel.x;
	short cb =  pixel.y;
	short cr =  pixel.z;
	int g = y - ((cr + cb)>>2);

	write_imageui(output, pos, (uint4)(cr + g + level_shift,g + level_shift ,cb + g + level_shift ,0));
}