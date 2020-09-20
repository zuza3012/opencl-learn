__kernel void getFancyArray (__global int *array){
    int gidx = get_global_id(0);
    int gidy = get_global_id(1);

    int lidx = get_local_id(0);
    int lidy = get_local_id(1);

    int value = gidx * 1000 + gidy * 100 + lidx * 10 + lidy;
    int X = gidx - get_global_offset(0);
    int Y = gidy - get_global_offset(1);

    int gsx = get_global_size(0);

    int index = X + gsx * Y; 
    array[index] = value;
}