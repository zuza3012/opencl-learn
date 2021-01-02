// dot product of C = A*B

__kernel void dot_product(__global const int *inputA, __global const int *inputB, __global int *outputC, int widthA, int widthB, int heightA, int heightB){

    int col = get_global_id(0);
    int row = get_global_id(1);

    int sum = 0;
    for(int i = 0; i < widthA; i++){
        sum += inputA[row * widthA + i] * inputB[i * widthB + col];
    }
    outputC[row * widthB + col] = sum;
}
