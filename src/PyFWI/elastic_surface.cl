
        ////////////////////////////////////////////////////////////////////////
        ////////////////////////////////////////////////////////////////////////
        ////                        source injection                        ////
        ////////////////////////////////////////////////////////////////////////
        ////////////////////////////////////////////////////////////////////////
__kernel void injSrc(__global float *vx,__global float *vz,
                     __global float *taux, __global float *tauz, __global float *tauxz,
                     __global float *seismogram_vxi,__global float *seismogram_vzi,
                     __global float *seismogram_tauxi, __global float *seismogram_tauzi, __global float *seismogram_tauxzi,
                     int dxr,
                     int sourcex, int sourcez,
                     float vsrcx, float vsrcz,
                     float tsrcx, float tsrcz,
                     int nt
                     )


{
  int i = get_global_id(0) ;
  int j = get_global_id(1) ;

  if (i==sourcez && j==sourcex){
    vx[center] += vsrcx;
    vz[center] += vsrcz;
    taux[center] += tsrcx;
    tauz[center] += tsrcz;
  }

  if(j % dxr == 0 && i == rec_surface_const){
      int ir =  j/dxr;
      if (ir < n_main_rec){
        seismogram_vxi[nt * Nr + ir]  =  vx[i*Nx+ (j + rec_surface_var)];
        seismogram_vzi[nt * Nr + ir]  =  vz[i*Nx+ (j + rec_surface_var)];
        seismogram_tauxi[nt * Nr + ir]  =  taux[i*Nx+ (j + rec_surface_var)];
        seismogram_tauzi[nt * Nr + ir]  =  tauz[i*Nx+ (j + rec_surface_var)];
        seismogram_tauxzi[nt * Nr + ir]  =  tauxz[i*Nx+ (j + rec_surface_var)];    
  }
  }

}

        ////////////////////////////////////////////////////////////////////////
        ////////////////////////////////////////////////////////////////////////
        ////                    adjoint    source injection                        ////
        ////////////////////////////////////////////////////////////////////////
        ////////////////////////////////////////////////////////////////////////
__kernel void Adj_injSrc(
                         __global float *Avx, __global float *Avz,
                         __global float *Ataux, __global float *Atauz, __global float *Atauxz,
                         __global float *res_vx, __global float * res_vz,
                         __global float *res_taux, __global float *res_tauz, __global float *res_tauxz,
                         int dxr
                         )


{
  int i = get_global_id(0) ;
  int j = get_global_id(1) ;

  if(j%dxr==0 && i == rec_surface_const){
    int ir =  j/dxr;
    if (ir < n_main_rec){
        Avx[i*Nx + (j+rec_surface_var)] += res_vx[ir];
        Avz[i*Nx + (j+rec_surface_var)] += res_vz[ir];
        Ataux[i*Nx + (j+rec_surface_var)] += res_taux[ir];
        Atauz[i*Nx + (j+rec_surface_var)] += res_tauz[ir];
        Atauxz[i*Nx + (j+rec_surface_var)] += res_tauxz[ir];
  }
  }

}