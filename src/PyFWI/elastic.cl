
      ////////////////////////////////////////////////////////////////////////
      ////////////////////////////////////////////////////////////////////////
      ////      Make the matrices zero for each source                    ////
      ////////////////////////////////////////////////////////////////////////
      ////////////////////////////////////////////////////////////////////////

__kernel void MakeAllZero(__global float *vx, __global float *vz,
                          __global float *taux,__global float *tauz, __global float *tauxz
                          )
{

  int i = get_global_id(0);
  int j = get_global_id(1);

  vx[center]=0;
  vz[center]=0;
  taux[center]=0;
  tauz[center]=0;
  tauxz[center]=0;
}

__kernel void MakeGradZero(__global float *Gmu,__global float *Glam, __global float *Grho,
                          __global float *Gmu_precond, __global float *Glam_precond, __global float *Grho_precond)
{

  int i = get_global_id(0);
  int j = get_global_id(1);

  Gmu[center]=0;
  Glam[center]=0;
  Grho[center]=0;
  Gmu_precond[center]=0;
  Glam_precond[center]=0;
  Grho_precond[center]=0;
}

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
  
  if(i%dxr==0 && j == rec_top_left_const && i/dxr<n_extera_rec){
    int ir = i/dxr;    

      seismogram_vxi[nt * Nr + ir]  =  vx[(i+ rec_top_left_var)*Nx + j];
      seismogram_vzi[nt * Nr + ir]  =  vz[(i+ rec_top_left_var)*Nx + j];
      seismogram_tauxi[nt * Nr + ir]  =  taux[(i+ rec_top_left_var)*Nx + j];
      seismogram_tauzi[nt * Nr + ir]  =  tauz[(i+ rec_top_left_var)*Nx + j];
      seismogram_tauxzi[nt * Nr + ir]  =  tauxz[(i+ rec_top_left_var)*Nx + j];
      
  }

  if(j%dxr==0 && i == rec_surface_const){
    int ir =  j/dxr;
    if (ir < n_main_rec){
        seismogram_vxi[nt * Nr + n_extera_rec + ir]  =  vx[i*Nx + (j + rec_surface_var)];
        seismogram_vzi[nt * Nr + n_extera_rec + ir]  =  vz[i*Nx + (j + rec_surface_var)];
        seismogram_tauxi[nt * Nr + n_extera_rec + ir]  =  taux[i*Nx + (j + rec_surface_var)];
        seismogram_tauzi[nt * Nr + n_extera_rec + ir]  =  tauz[i*Nx + (j + rec_surface_var)];
        seismogram_tauxzi[nt * Nr + n_extera_rec + ir]  =  tauxz[i*Nx + (j + rec_surface_var)];
    }
  }

  int startOfThirdRec = n_extera_rec + n_main_rec;

  if(i%dxr==0 && j == rec_top_right_const){
    int ir = i/dxr;
      seismogram_vxi[nt * Nr + startOfThirdRec + ir]  =  vx[(i+ rec_top_right_var)*Nx + j];
      seismogram_vzi[nt * Nr + startOfThirdRec + ir]  =  vz[(i+ rec_top_right_var)*Nx + j];
      seismogram_tauxi[nt * Nr + startOfThirdRec + ir]  =  taux[(i+ rec_top_right_var)*Nx + j];
      seismogram_tauzi[nt * Nr + startOfThirdRec + ir]  =  tauz[(i+ rec_top_right_var)*Nx + j];
      seismogram_tauxzi[nt * Nr + startOfThirdRec + ir]  =  tauxz[(i+ rec_top_right_var)*Nx + j];
  }

}


__kernel void forward_hessian_src_preparing(__global float *w_vx,__global float *w_vz,
                     __global float *w_taux, __global float *w_tauz, __global float *w_tauxz,
                     __global float *vx,__global float *vz,
                     __global float *taux, __global float *tauz, __global float *tauxz,
                     __global float *glam, __global float *gmu, __global float *grho,
                     __global float *rho
                     )


{
  int i = get_global_id(0) ;
  int j = get_global_id(1) ;

  float DxmTx = (c1*(w_taux[center]-w_taux[left]) + c2*(w_taux[right]-w_taux[left2])+
               c3*(w_taux[right2]-w_taux[left3]) + c4*(w_taux[right3]-w_taux[left4]))/dx ;

  float DzmTxz = (c1*(w_tauxz[center]-w_tauxz[above]) + c2*(w_tauxz[below]-w_tauxz[above2])+
               c3*(w_tauxz[below2]-w_tauxz[above3]) + c4*(w_tauxz[below3]-w_tauxz[above4]))/dz;

  float DzpTz = (c1*(w_tauz[below]-w_tauz[center]) + c2*(w_tauz[below2]-w_tauz[above])+
               c3*(w_tauz[below3]-w_tauz[above2]) + c4*(w_tauz[below4]-w_tauz[above3]))/dz;

  float DxpTxz = (c1*(w_tauxz[right]-w_tauxz[center]) + c2*(w_tauxz[right2]-w_tauxz[left])+
               c3*(w_tauxz[right3]-w_tauxz[left2]) + c4*(w_tauxz[right4]-w_tauxz[left3]))/dx;

  float DzmVz= (c1*(w_vz[center]-w_vz[above]) + c2*(w_vz[below]-w_vz[above2])+
               c3*(w_vz[below2]-w_vz[above3]) + c4*(w_vz[below3]-w_vz[above4]))/dz; 

  float DxpVx= (c1*(w_vx[right]-w_vx[center]) + c2*(w_vx[right2]-w_vx[left])+
               c3*(w_vx[right3]-w_vx[left2]) + c4*(w_vx[right4]-w_vx[left3]))/dx;

  float DzpVx= (c1*(w_vz[below]-w_vz[center]) + c2*(w_vz[below2]-w_vz[above])+
               c3*(w_vz[below3]-w_vz[above2]) + c4*(w_vz[below4]-w_vz[above3]))/dz;

  float DxmVz= (c1*(w_vz[center]-w_vz[left]) + c2*(w_vz[right]-w_vz[left2])+
               c3*(w_vz[right2]-w_vz[left3]) + c4*(w_vz[right3]-w_vz[left4]))/dx;


  vx[center] += - dt * (grho[center] * (rho[center] * rho[center] * rho[center])) * (DxmTx + DzmTxz);
  vz[center] += - dt * (grho[center] * (rho[center] * rho[center] * rho[center])) * (DzpTz + DxpTxz);
  taux[center] += dt * glam[center] * (DxpVx + DzmVz) + 2 * dt * gmu[center] * DxpVx;
  tauz[center] += dt * glam[center] * (DxpVx + DzmVz) + 2 * dt * gmu[center] * DzmVz;
  tauxz[center] += dt * gmu[center] * (DzpVx + DxmVz);
  
}

/////////////////////////////// Updating for forward modelling ///////////////////////////////////////

        ////////////////////////////////////////////////////////////////////////
        ////////////////////////////////////////////////////////////////////////
        ////////                 Update velocities  in x-direction      ////////
        ////////////////////////////////////////////////////////////////////////
        ////////////////////////////////////////////////////////////////////////

__kernel void update_velx(int coeff,
             __global float *vx,
            __global float *taux,__global float *tauxz,
            __global float *rho_b,
            __global float *dxpml, __global float *dzpml
            )

{
  int i = get_global_id(0)+sdo ;
  int j = get_global_id(1)+sdo ;

  if(i>Nz-sdo || j>Nx-sdo){return;}

  float DxmTx = (c1*(taux[center]-taux[left]) + c2*(taux[right]-taux[left2])+
               c3*(taux[right2]-taux[left3]) + c4*(taux[right3]-taux[left4]))/dx ;

  float DzmTxz = (c1*(tauxz[center]-tauxz[above]) + c2*(tauxz[below]-tauxz[above2])+
               c3*(tauxz[below2]-tauxz[above3]) + c4*(tauxz[below3]-tauxz[above4]))/dz;


    if (coeff==1){
      vx[center] += coeff * dt * (rho_b[center] * (DxmTx + DzmTxz));
      vx[center] += coeff * dt * (- (dxpml[center] + dzpml[center])*vx[center]);

    }else if(coeff==-1){
      vx[center] += coeff * dt * (+ (dxpml[center] + dzpml[center])*vx[center]);
      vx[center] += coeff * dt * (rho_b[center] * (DxmTx + DzmTxz));
    
    } 

}

        ////////////////////////////////////////////////////////////////////////
        ////////////////////////////////////////////////////////////////////////
        ////////                 Update velocities  in z-direction      ////////
        ////////////////////////////////////////////////////////////////////////
        ////////////////////////////////////////////////////////////////////////

__kernel void update_velz(int coeff,
            __global float *vz,
            __global float *tauz,__global float *tauxz,
            __global float *rho_b,
            __global float *dxpml, __global float *dzpml
            )

{
  int i = get_global_id(0)+sdo ;
  int j = get_global_id(1)+sdo ;

  if(i>Nz-sdo || j>Nx-sdo){return;}



  float DxpTxz= (c1*(tauxz[right]-tauxz[center]) + c2*(tauxz[right2]-tauxz[left])+
               c3*(tauxz[right3]-tauxz[left2]) + c4*(tauxz[right4]-tauxz[left3]))/dx;


  float DzpTz= (c1*(tauz[below]-tauz[center]) + c2*(tauz[below2]-tauz[above])+
               c3*(tauz[below3]-tauz[above2]) + c4*(tauz[below4]-tauz[above3]))/dz;


if (coeff==1){
      vz[center] += coeff * dt * (rho_b[center] * (DzpTz + DxpTxz));
      vz[center] += coeff * dt * (-(dzpml[center] + dxpml[center])*vz[center]);

    }else if(coeff==-1){
      vz[center] += coeff * dt * (+(dzpml[center] + dxpml[center])*vz[center]);
      vz[center] += coeff * dt * (rho_b[center] * (DzpTz + DxpTxz));
    } 


}


        ////////////////////////////////////////////////////////////////////////
        ////////////////////////////////////////////////////////////////////////
        ////////                 Update pressure  in z-direction      ////////
        ////////////////////////////////////////////////////////////////////////
        ////////////////////////////////////////////////////////////////////////

__kernel void update_tauz(int coeff,
            __global float *vx, __global float *vz,
            __global float *taux, __global float *tauz,
            __global float *lambda, __global float *mu,
            __global float *dxpml, __global float *dzpml
            )

{
  int i = get_global_id(0)+sdo ;
  int j = get_global_id(1)+sdo ;

  if(i>Nz-sdo || j>Nx-sdo){return;}

  float DxpVx= (c1*(vx[right]-vx[center]) + c2*(vx[right2]-vx[left])+
               c3*(vx[right3]-vx[left2]) + c4*(vx[right4]-vx[left3]))/dx;

  float DzmVz= (c1*(vz[center]-vz[above]) + c2*(vz[below]-vz[above2])+
               c3*(vz[below2]-vz[above3]) + c4*(vz[below3]-vz[above4]))/dz;


  if (coeff==1){
      taux[center]+= coeff* dt * ((lambda[center] + 2 * mu[center]) * DxpVx + lambda[center] * DzmVz);
      tauz[center]+= coeff* dt * ((lambda[center] + 2 * mu[center]) * DzmVz + lambda[center] * DxpVx);

      taux[center] += coeff * dt * (-(dxpml[center]+dzpml[center])*taux[center]);
      tauz[center] += coeff * dt * (-(dxpml[center]+dzpml[center])*tauz[center]);

    }else if(coeff==-1){
      taux[center] += coeff * dt * (+(dxpml[center]+dzpml[center])*taux[center]);
      tauz[center] += coeff * dt * (+(dxpml[center]+dzpml[center])*tauz[center]);

      taux[center]+= coeff* dt * ((lambda[center] + 2 * mu[center]) * DxpVx + lambda[center] * DzmVz);
      tauz[center]+= coeff* dt * ((lambda[center] + 2 * mu[center]) * DzmVz + lambda[center] * DxpVx);
    } 

}
        ////////////////////////////////////////////////////////////////////////
        ////////////////////////////////////////////////////////////////////////
        ////////                 Update pressure  in xz-direction      ////////
        ////////////////////////////////////////////////////////////////////////
        ////////////////////////////////////////////////////////////////////////

__kernel void update_tauxz(int coeff,
            __global float *vx, __global float *vz,
            __global float *tauxz,
            __global float *mu,
            __global float *dxpml, __global float *dzpml
            )

{
  int i = get_global_id(0)+sdo ;
  int j = get_global_id(1)+sdo ;

  if(i>Nz-sdo || j>Nx-sdo){return;}


  float DxmVz= (c1*(vz[center]-vz[left]) + c2*(vz[right]-vz[left2])+
               c3*(vz[right2]-vz[left3]) + c4*(vz[right3]-vz[left4]))/dx;

  float DzpVx= (c1*(vx[below]-vx[center]) + c2*(vx[below2]-vx[above])+
               c3*(vx[below3]-vx[above2]) + c4*(vx[below4]-vx[above3]))/dz;



  if (coeff==1){
      tauxz[center] += coeff * dt * (mu[center] *(DxmVz + DzpVx));
      tauxz[center] += coeff * dt * (-(dxpml[center] + dzpml[center])*tauxz[center]);

    }else if(coeff==-1){
      tauxz[center] += coeff * dt * (+(dxpml[center] + dzpml[center])*tauxz[center]);
      tauxz[center] += coeff * dt * (mu[center] *(DxmVz + DzpVx));
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
  int j = get_global_id(1);

    if(i%dxr==0 && j == rec_top_left_const && i/dxr<n_extera_rec){
    int ir = i/dxr;
        Avx[(i+ rec_top_left_var)*Nx + j] += res_vx[ir];
        Avz[(i+ rec_top_left_var)*Nx + j] += res_vz[ir];
        Ataux[(i+ rec_top_left_var)*Nx + j] += res_taux[ir];
        Atauz[(i+ rec_top_left_var)*Nx + j] += res_tauz[ir];
        Atauxz[(i+ rec_top_left_var)*Nx + j] += res_tauxz[ir];
}
  if(j%dxr==0 && i == rec_surface_const){
    int ir =  j/dxr;
    if (ir < n_main_rec){

        Avx[i*Nx + (j+ rec_surface_var)] += res_vx[n_extera_rec + ir];
        Avz[i*Nx + (j+ rec_surface_var)] += res_vz[n_extera_rec + ir];
        Ataux[i*Nx + (j+ rec_surface_var)] += res_taux[n_extera_rec + ir];
        Atauz[i*Nx + (j+ rec_surface_var)] += res_tauz[n_extera_rec + ir];
        Atauxz[i*Nx + (j+ rec_surface_var)] += res_tauxz[n_extera_rec + ir];
        }
  }

  int startOfThirdRec = n_extera_rec + n_main_rec;

  if(i%dxr==0 && j == rec_top_right_const){
    int ir = i/dxr;

      Avx[(i+ rec_top_right_var)*Nx + j] += res_vx[startOfThirdRec + ir];
      Avz[(i+ rec_top_right_var)*Nx + j] += res_vz[startOfThirdRec + ir];
      Ataux[(i+ rec_top_right_var)*Nx + j] += res_taux[startOfThirdRec + ir];
      Atauz[(i+ rec_top_right_var)*Nx + j] += res_tauz[startOfThirdRec + ir];
      Atauxz[(i+ rec_top_right_var)*Nx + j] += res_tauxz[startOfThirdRec + ir];
      
}

}


        ////////////////////////////////////////////////////////////////////////
        ////////////////////////////////////////////////////////////////////////
        //////////         Update Ajoint pressure  in x-direction      /////////
        ////////////////////////////////////////////////////////////////////////
        ////////////////////////////////////////////////////////////////////////

__kernel void Adj_update_tau(__global float *Avx, __global float *Avz,
            __global float *Ataux, __global float *Atauz,__global float *Atauxz,
            __global float *rho_b,
            __global float *dxpml, __global float *dzpml
            )

{
  int i = get_global_id(0)+sdo ;
  int j = get_global_id(1)+sdo ;

  if(i>Nz-sdo || j>Nx-sdo){return;}


  float DxpAVx= (c1*(Avx[right]*rho_b[right] -Avx[center]*rho_b[center]) +
                 c2*(Avx[right2]*rho_b[right2] -Avx[left]*rho_b[left])+
                 c3*(Avx[right3]*rho_b[right3] -Avx[left2]*rho_b[left2]) +
                 c4*(Avx[right4]*rho_b[right4] -Avx[left3]*rho_b[left3]))/dx;

  float DzmAVz= (c1*(Avz[center]*rho_b[center] -Avz[above]*rho_b[above]) +
                 c2*(Avz[below]*rho_b[below] -Avz[above2]*rho_b[above2])+
               c3*(Avz[below2]*rho_b[below2] -Avz[above3]*rho_b[above3]) +
               c4*(Avz[below3]*rho_b[below3] -Avz[above4]*rho_b[above4]))/dz;

  Ataux[center]+= - dt * ((dxpml[center])*Ataux[center] + DxpAVx);
  Atauz[center]+= - dt * ((dzpml[center])*Atauz[center] + DzmAVz);
    
    float DzpAvx= (c1*(Avx[below]*rho_b[below] - Avx[center]*rho_b[center]) +
                    c2*(Avx[below2]*rho_b[below2] -Avx[above]*rho_b[above])+
                   c3*(Avx[below3]*rho_b[below3] -Avx[above2]*rho_b[above2]) +
                   c4*(Avx[below4]*rho_b[below4] -Avx[above3]*rho_b[above3]))/dz;

      float DxmAvz= (c1*(Avz[center]*rho_b[center] - Avz[left]*rho_b[left]) +
                    c2*(Avz[right]*rho_b[right] - Avz[left2]*rho_b[left2])+
                    c3*(Avz[right2]*rho_b[right2] - Avz[left3]*rho_b[left3]) +
                    c4*(Avz[right3]*rho_b[right3] - Avz[left4]*rho_b[left4]))/dx;

      Atauxz[center]+= - dt*(((dxpml[center] + dzpml[center]) *Atauxz[center]) + (DzpAvx +  DxmAvz));


}

        ////////////////////////////////////////////////////////////////////////
        ////////////////////////////////////////////////////////////////////////
        ////////          Update Ajoint velocities  in x-direction      ////////
        ////////////////////////////////////////////////////////////////////////
        ////////////////////////////////////////////////////////////////////////

__kernel void Adj_update_v(__global float *Avx, __global float *Avz,
            __global float *Ataux, __global float *Atauz, __global float *Atauxz,
            __global float *lam, __global float *mu, 
            __global float *dxpml,__global float *dzpml
            )

{
  int i = get_global_id(0)+sdo ;
  int j = get_global_id(1)+sdo ;

  if(i>Nz-sdo || j>Nx-sdo){return;}


float DxmAtaux =  (c1*((lam[center]+2*mu[center])*Ataux[center] - (lam[left]+2*mu[left])*Ataux[left]) +
                     c2*((lam[right]+2*mu[right])*Ataux[right] - (lam[left2]+2*mu[left2])*Ataux[left2])+
                    c3*((lam[right2]+2*mu[right2])*Ataux[right2] - (lam[left3]+2*mu[left3])*Ataux[left3]) +
                     c4*((lam[right3]+2*mu[right3])*Ataux[right3] -(lam[left4]+2*mu[left4])*Ataux[left4]))/dx;

  float DxmAtauz = (c1*(lam[center]*Atauz[center] - lam[left]*Atauz[left]) + 
                    c2*(lam[right]*Atauz[right] - lam[left2]*Atauz[left2]) +
                    c3*(lam[right2]*Atauz[right2] - lam[left3]*Atauz[left3]) + 
                    c4*(lam[right3]*Atauz[right3] - lam[left4]*Atauz[left4]))/dx;

  float DzmAtauxz=  (c1*(mu[center]*Atauxz[center] -mu[above]*Atauxz[above]) + 
                    c2*(mu[below]*Atauxz[below] - mu[above2]*Atauxz[above2])+
                    c3*(mu[below2]*Atauxz[below2] -mu[above3]*Atauxz[above3]) + 
                    c4*(mu[below3]*Atauxz[below3] - mu[above4]*Atauxz[above4]))/dz;



  float DzpAtauz =(c1*((lam[below]+2*mu[below])*Atauz[below] -(lam[center]+2*mu[center])*Atauz[center]) +
                   c2*((lam[below2]+2*mu[below2])*Atauz[below2] -(lam[above]+2*mu[above])*Atauz[above])+
                   c3*((lam[below3]+2*mu[below3])*Atauz[below3] -(lam[above2]+2*mu[above2])*Atauz[above2]) +
                   c4*((lam[below4]+2*mu[below4])*Atauz[below4] -(lam[above3]+2*mu[above3])*Atauz[above3]))/dz;

  float DzpAtaux = (c1*(lam[below]*Ataux[below] -lam[center]*Ataux[center]) + 
                    c2*(lam[below2]*Ataux[below2] -lam[above]*Ataux[above])+
                    c3*(lam[below3]*Ataux[below3]-lam[above2]*Ataux[above2]) + 
                    c4*(lam[below4]*Ataux[below4]-lam[above3]*Ataux[above3]))/dz;

  
  float DxpAtauxz= (c1*(mu[right] *Atauxz[right] - mu[center]*Atauxz[center]) +
                    c2*(mu[right2]*Atauxz[right2] - mu[left]*Atauxz[left])+
                    c3*(mu[right3]*Atauxz[right3] - mu[left2]*Atauxz[left2]) + 
                    c4*(mu[right4]*Atauxz[right4] - mu[left3]*Atauxz[left3]))/dx;

  Avx[center]+= -dt* ((2*dxpml[center] + dzpml[center]) * Avx[center] + (DxmAtaux + DxmAtauz + DzmAtauxz));
  Avz[center]+= -dt* ((dxpml[center] + 2*dzpml[center]) * Avz[center] + (DzpAtaux + DzpAtauz + DxpAtauxz));

}

__kernel void Grad(__global float *vx, __global float *vz,
            __global float *taux, __global float *tauz, __global float *tauxz,
            __global float *Avx, __global float *Avz,
            __global float *Ataux, __global float *Atauz,  __global float *Atauxz,
            __global float *lambda, __global float *mu, __global float *rho_b,
            __global float *Gmu, __global float *Gmu_precond,
            __global float *Glam, __global float *Glam_precond,
            __global float *Grho, __global float *Grho_precond
            )
{
  int i = get_global_id(0)+ npml + sdo ;
  int j = get_global_id(1)+ npml + sdo ;

  if(i>Nz-sdo - npml || j>Nx-sdo - npml){return;}

  float DzpVx= (c1*(vz[below]-vz[center]) + c2*(vz[below2]-vz[above])+
               c3*(vz[below3]-vz[above2]) + c4*(vz[below4]-vz[above3]))/dz;

  float DxmVz= (c1*(vz[center]-vz[left]) + c2*(vz[right]-vz[left2])+
               c3*(vz[right2]-vz[left3]) + c4*(vz[right3]-vz[left4]))/dx;

  float DxpVx= (c1*(vx[right]-vx[center]) + c2*(vx[right2]-vx[left])+
               c3*(vx[right3]-vx[left2]) + c4*(vx[right4]-vx[left3]))/dx;

  float DzmVz= (c1*(vz[center]-vz[above]) + c2*(vz[below]-vz[above2])+
               c3*(vz[below2]-vz[above3]) + c4*(vz[below3]-vz[above4]))/dz;    

  float DzpPz= (c1*(tauz[below]-tauz[center]) + c2*(tauz[below2]-tauz[above])+
               c3*(tauz[below3]-tauz[above2]) + c4*(tauz[below4]-tauz[above3]))/dz;

  float DxmPx= (c1*(taux[center]-taux[left]) + c2*(taux[right]-taux[left2])+
               c3*(taux[right2]-taux[left3]) + c4*(taux[right3]-taux[left4]))/dx;

  float DxpPxz= (c1*(tauxz[right]-tauxz[center]) + c2*(tauxz[right2]-tauxz[left])+
               c3*(tauxz[right3]-tauxz[left2]) + c4*(tauxz[right4]-tauxz[left3]))/dx;

  float DzmPxz= (c1*(tauxz[center]-tauxz[above]) + c2*(tauxz[below]-tauxz[above2])+
               c3*(tauxz[below2]-tauxz[above3]) + c4*(tauxz[below3]-tauxz[above4]))/dz;


  Gmu[center]+= 2*dt*(Ataux[center]*DxpVx + Atauz[center]*DzmVz) + dt*Atauxz[center]*(DzpVx+DxmVz);
    
  
  Gmu_precond[center]+= 4*dt*dt*(taux[center]*taux[center]*DxpVx*DxpVx + tauz[center]*tauz[center]*DzmVz*DzmVz)
                          +dt*dt*tauxz[center]*tauxz[center]*(DzpVx*DzpVx+DxmVz*DxmVz);

  Glam[center]+= dt*(Ataux[center] + Atauz[center])*(DxpVx+DzmVz);

  Glam_precond[center] += dt*dt*(taux[center]*taux[center] + tauz[center]*tauz[center])*(DxpVx*DxpVx+DzmVz*DzmVz);

  Grho[center]+= -(dt*(rho_b[center]*rho_b[center]))*(Avx[center]*(DxmPx + DzmPxz)+
                                                  Avz[center]*(DzpPz+DxpPxz));

  Grho_precond[center] += dt*dt*(vx[center]*vx[center]*(DxmPx*DxmPx + DzmPxz*DzmPxz)+
                                            vz[center]*vz[center]*(DzpPz*DzpPz + DxpPxz*DxpPxz));

}