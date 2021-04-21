__kernel void adjacent_difference(__global float* x, __global float* y)
{
    int gid = get_global_id(0);

    if (gid == 0) {
        y[gid] = x[gid];
    }
    else {
        y[gid] = x[gid]-x[gid-1];
    }
}