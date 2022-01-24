
        ////////////////////////////////////////////////////////////////////////
        ////////////////////////////////////////////////////////////////////////
        ////                        source injection                        ////
        ////////////////////////////////////////////////////////////////////////
        ////////////////////////////////////////////////////////////////////////
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
  if(j%dxr==0 && i == recDepth){
    int ir =  j/dxr;
        seismogram_vxi[ir]  =  vx[i*Nx+ (j + first_rec)];
        seismogram_vzi[ir]  =  vz[i*Nx+ (j + first_rec)];
        seismogram_tauxi[ir]  =  taux[i*Nx+ (j + first_rec)];
        seismogram_tauzi[ir]  =  tauz[i*Nx+ (j + first_rec)];
        seismogram_tauxzi[ir]  =  tauxz[i*Nx+ (j + first_rec)];
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
                         int dxr, int recDepth, int first_rec, int last_rec
                         )


{
  int i = get_global_id(0) ;
  int j = get_global_id(1) ;


  if(j%dxr==0 && i == recDepth){
    int ir =  j/dxr;
        Avx[i*Nx + (j+first_rec)] += res_vx[ir];
        Avz[i*Nx + (j+first_rec)] += res_vz[ir];
        Ataux[i*Nx + (j+first_rec)] += res_taux[ir];
        Atauz[i*Nx + (j+first_rec)] += res_tauz[ir];
        Atauxz[i*Nx + (j+first_rec)] += res_tauxz[ir];
  }

  
}