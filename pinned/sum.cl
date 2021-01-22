#pragma OPENCL EXTENSION cl_amd_printf : enable
__kernel void vector_sum(__global float* a, __global float* b, __global float* c){
	int i = (int)get_global_id(0);
    //printf(" a[%d]=%f b[%d]=%f \n", i, a[i], i, b[i]);
	c[i] = a[i] + b[i];
    //printf(" a[%d]=%f b[%d]=%f c[%d]=%f \n", i, a[i], i, b[i], i, c[i]);
}
__kernel void vector_modulo(__global float* a, __global float* b){
	int i = (int)get_global_id(0);
	
	a[i] = fmod(a[i], 128) * 0;
	b[i] = fmod(b[i], 128) * 2;
    //printf(" MOD a[%d]=%f b[%d]=%f \n", i, a[i], i, b[i]);
}