__kernel void injSrc(__global float *vx,__global float *vz,
                     __global float *taux, __global float *tauz, __global float *tauxz,
                     __global float *seismogram_vxi,__global float *seismogram_vzi,
                     __global float *seismogram_tauxi, __global float *seismogram_tauzi, __global float *seismogram_tauxzi,
                     int dxr, int recDepth, int first_rec,
                     int sourcex, int sourcez,
                     float srcx, float srcz)


{
  int i = get_global_id(0) ;
  int j = get_global_id(1) ;

  if (i==sourcez && j==sourcex){
    taux[center] += srcx;
    tauz[center] += srcz;
    // printf("%f\n",src );
    // printf("%d, %d,  %d, %d \n",dxr,n_extera_rec, first_rec, n_main_rec );

  }

   if(i%dxr==0 && j == recDepth){
    int ir =  i/dxr;
      seismogram_vxi[ir]  =  vx[(i+first_rec)*Nx + j];
      seismogram_vzi[ir]  =  vx[(i+first_rec)*Nx + j];
      seismogram_tauxi[ir]  =  taux[(i+first_rec)*Nx + j];
      seismogram_tauzi[ir]  =  tauz[(i+first_rec)*Nx + j];
      seismogram_tauxzi[ir]  =  tauxz[(i+first_rec)*Nx + j];
  }

}

__kernel void Adj_injSrc(
                         __global float *Avx, __global float *Avz,
                         __global float *Ataux, __global float *Atauz, __global float *Atauxz,
                         __global float *res_vx, __global float * res_vz,
                         __global float *res_taux, __global float *res_tauz, __global float *res_tauxz,
                         int dxr, int recDepth, int first_rec, int last_rec
                         )


{
  int i = get_global_id(0) ;
  int j = get_global_id(1) ;



if(i%dxr==0 && j == recDepth){
    int ir =  i/dxr;
    // if (ir < n_main_rec){
    Avx[(i+first_rec)*Nx + j] += res_vx[ir];
    Avz[(i+first_rec)*Nx + j] += res_vz[ir];
    Ataux[(i+first_rec)*Nx + j] += res_taux[ir];
    Atauz[(i+first_rec)*Nx + j] += res_tauz[ir];
    Atauxz[(i+first_rec)*Nx + j] += res_tauxz[ir];
    
//      Apx[(i + first_rec)*Nx+ j] += res[ir]; 
//      Apz[(i + first_rec)*Nx+ j] += res[ir];

    // }
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