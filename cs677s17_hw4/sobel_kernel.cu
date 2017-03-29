#ifndef _SOBEL_KERNEL_H_
#define _SOBEL_KERNEL_H_


#define TILE_WIDTH 14
#define BLOCK_WIDTH 16
#define MASK_WIDTH 3


__global__ void SobelKernel(int *result,unsigned int *pic, int width, int height, int thresh)
{
	int tx = threadIdx.x, ty = threadIdx.y;

	int row = blockIdx.y * TILE_WIDTH + ty;
	int col = blockIdx.x * TILE_WIDTH + tx;

	int row_fix = row - 1;
	int col_fix = col - 1;

	__shared__ int SD[BLOCK_WIDTH][BLOCK_WIDTH];

	if( row_fix >= 0 && row_fix < height && col_fix >= 0 && col_fix < width)	{
		SD[ty][tx] = pic[row_fix * width + col_fix];
	}
	else{
		SD[ty][tx] = 0;
	}
	__syncthreads();

	int sum1=0, sum2=0, magnitude=0, output=0;

	if(ty < TILE_WIDTH && tx < TILE_WIDTH){
		sum1 =  SD[ty][tx+2] - SD[ty][tx] 
			+ 2 * SD[ty+1][tx+2] - 2 * SD[ty+1][tx]
			+  SD[ty+2][tx+2]-  SD[ty+2][tx];
      
		sum2 =  SD[ty][tx] + 2 * SD[ty][tx+1] + SD[ty][tx+2]
			- SD[ty+2][tx] - 2 * SD[ty+2][tx+1] - SD[ty+2][tx+2];
      
		magnitude =  sum1*sum1 + sum2*sum2;

		if (magnitude > thresh)
			output = 255;
		else 
			output = 0;

		if(row == 0 || row == height-1 || col==0 || col == width-1)
			result[row * width + col] = 0;
		else if(row < height && col < width)
			result[row * width + col] = output;
	}
	__syncthreads();
}

#endif // #ifndef _SOBEL_KERNEL_H_