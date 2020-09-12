__kernel void helloDecode(__global char *inTxt, __global char *outTxt, int N){
    int idx = get_global_id(0);
    if(idx < N){
        outTxt[idx] = inTxt[idx] - 5;
    }
}                        