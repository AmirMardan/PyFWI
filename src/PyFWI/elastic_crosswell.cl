__kernel void injSrc(__global float *vx,__global float *vz,
                     __global float *taux, __global float *tauz, __global float *tauxz,
                     __global float *seismogram_vxi,__global float *seismogram_vzi,
                     __global float *seismogram_tauxi, __global float *seismogram_tauzi, __global float *seismogram_tauxzi,
                     int dxr,
                     int sourcex, int sourcez,
                     float vsrcx, float vsrcz,
                     float tsrcx, float tsrcz,
                      int nt)


{
  int i = get_global_id(0) ;
  int j = get_global_id(1) ;

  if (i==sourcez && j==sourcex){
    vx[center] += vsrcx;
    vz[center] += vsrcz;
    taux[center] += tsrcx;
    tauz[center] += tsrcz;

  }

   if(i%dxr==0 && j == rec_top_right_const){
    int ir =  i/dxr;
      seismogram_vxi[nt * Nr + ir]  =  vx[(i+rec_top_right_var)*Nx + j];
      seismogram_vzi[nt * Nr + ir]  =  vx[(i+rec_top_right_var)*Nx + j];
      seismogram_tauxi[nt * Nr + ir]  =  taux[(i+rec_top_right_var)*Nx + j];
      seismogram_tauzi[nt * Nr + ir]  =  tauz[(i+rec_top_right_var)*Nx + j];
      seismogram_tauxzi[nt * Nr + ir]  =  tauxz[(i+rec_top_right_var)*Nx + j];
  }

}

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

if(i%dxr==0 && j == rec_top_right_const){
    int ir =  i/dxr;
    Avx[(i+rec_top_right_var)*Nx + j] += res_vx[ir];
    Avz[(i+rec_top_right_var)*Nx + j] += res_vz[ir];
    Ataux[(i+rec_top_right_var)*Nx + j] += res_taux[ir];
    Atauz[(i+rec_top_right_var)*Nx + j] += res_tauz[ir];
    Atauxz[(i+rec_top_right_var)*Nx + j] += res_tauxz[ir];
    
  }

}

__kernel void hessian_seismogram(__global float *vx,__global float *vz,
                     __global float *taux, __global float *tauz, __global float *tauxz,
                     __global float *seismogram_vxi,__global float *seismogram_vzi,
                     __global float *seismogram_tauxi, __global float *seismogram_tauzi, __global float *seismogram_tauxzi,
                     int dxr, int recDepth, int first_rec)


{
  int i = get_global_id(0) ;
  int j = get_global_id(1) ;

   if(i%dxr==0 && j == recDepth){
    int ir =  i/dxr;
      seismogram_vxi[ir]  =  vx[(i+first_rec)*Nx + j];
      seismogram_vzi[ir]  =  vx[(i+first_rec)*Nx + j];
      seismogram_tauxi[ir]  =  taux[(i+first_rec)*Nx + j];
      seismogram_tauzi[ir]  =  tauz[(i+first_rec)*Nx + j];
      seismogram_tauxzi[ir]  =  tauxz[(i+first_rec)*Nx + j];
  }

}