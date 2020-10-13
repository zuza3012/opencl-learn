__kernel void reduction (__global const int *in_data, __global int *out_data, __local int *shared_data ){
    //uint gid = get_global_id(0);
    uint lid = get_local_id(0);

    uint group_size = get_local_size(0);

    //shared_data[lid] = (gid < N) ? in_data[gid] : 0;
    shared_data[lid] =  in_data[get_global_id(0)];
    // bariera, alby najpierw wszystkie watki w obrebie grupy wypelnily tablice shared_data
    //barrier(CLK_LOCAL_MEM_FENCE);

    for(int stride = group_size / 2; stride > 0; stride /= 2 ){
        barrier(CLK_LOCAL_MEM_FENCE);
        if(lid < stride)
            shared_data[lid] = shared_data[lid] + shared_data[lid + stride];   
    }

    if( 0 == lid)
        out_data[get_group_id(0)] = shared_data[0]; 
}