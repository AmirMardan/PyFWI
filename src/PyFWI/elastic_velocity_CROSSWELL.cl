
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
                     __global float *rho_b, 
                     __global float *seismogram_vxi,__global float *seismogram_vzi,
                     __global float *seismogram_tauxi, __global float *seismogram_tauzi, __global float *seismogram_tauxzi,
                     int dxr, int recDepth, int first_rec,
                     int sourcex, int sourcez,
                     float srcx, float srcz)


{
  int i = get_global_id(0) ;
  int j = get_global_id(1) ;

  if (i==sourcez && j==sourcex){
    vx[center] += rho_b[center] * dt * srcx;
    vz[center] += rho_b[center] * dt * srcz;
    taux[center] += dt * srcx;
    tauz[center] += dt * srcz;
    tauxz[center] += dt * srcx;
    // printf("%f\n",src );
  }
  // printf("%d\n",dxr );
  if(i%dxr==0 && j == recDepth){
    int ir = i/dxr;
      seismogram_vxi[ir]  =  vx[(i+first_rec)*Nx + j];
      seismogram_vzi[ir]  =  vz[(i+first_rec)*Nx + j];
      seismogram_tauxi[ir]  =  taux[(i+first_rec)*Nx + j];
      seismogram_tauzi[ir]  =  tauz[(i+first_rec)*Nx + j];
      seismogram_tauxzi[ir]  =  tauxz[(i+first_rec)*Nx + j];
  }
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


//    ====== CPML ===========
//  vx[center]+= coeff * ((dt/rho[center]) * (DxmTx/nux[center] + psi_txxx[center]) +
//             (dt/rho[center]) * (DzmTxz / nuz[center] + psi_txzz[center]));
//
//psi_txxx[center] = bx[center] * psi_txxx[center] + cx[center] * DxmTx;
//psi_txzz[center] = bz[center] * psi_txzz[center] + cz[center] * DzmTxz;


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
    //    ====== CPML ===========

//  vz[center]+= coeff * ((dt/rho[center]) * (DzpTz / nuz[center] + psi_tzzz[center]) +
//                        (dt/rho[center]) * (DxpTxz / nux[center] + psi_txzx[center]));
//
//
//    psi_txzx[center] = bx[center] * psi_txzx[center] + cx[center] * DxpTxz;
//    psi_tzzz[center] = bz[center] * psi_tzzz[center] + cz[center] * DzpTz;


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


    //    ====== CPML ===========
//  taux[center]+= coeff* (dt *(lambda[center] + 2 * mu[center]) * (DxpVx / nux[center]+ psi_vxx[center]) +
//                        dt * lambda[center] * (DzmVz / nuz[center] + psi_vzz[center]));
//
//
//  tauz[center]+= coeff* (dt * (lambda[center] + 2 * mu[center]) * (DzmVz / nuz[center] + psi_vzz[center]) +
//                        dt * lambda[center] * (DxpVx / nux[center] + psi_vxx[center]));
//
//
//  psi_vxx[center] = bx[center] * psi_vxx[center] + cx[center] * DxpVx;
//  psi_vzz[center] = bz[center] * psi_vzz[center] + cz[center] * DzmVz;
    
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
    
    
    //    ====== CPML ===========

//  tauxz[center]+= coeff * (dt * mu[center] *(DxmVz / nux[center] + psi_vzx[center]) +
//                           dt * mu[center] *(DzpVx / nuz[center] + psi_vxz[center]));
//
//
//psi_vxz[center] = bz[center] * psi_vxz[center] + cz[center] * DzpVx;
//psi_vzx[center] = bx[center] * psi_vzx[center] + cx[center] * DxmVz;



}

        ////////////////////////////////////////////////////////////////////////
        ////////////////////////////////////////////////////////////////////////
        ////                        adjoint source injection                ////
        ////////////////////////////////////////////////////////////////////////
        ////////////////////////////////////////////////////////////////////////
__kernel void Adj_injSrc(__global float *Avx, __global float *Avz,
                         __global float *Ataux, __global float *Atauz, __global float *Atauxz,
                         __global float *res_vx, __global float * res_vz,
                         __global float *res_taux, __global float *res_tauz, __global float *res_tauxz,
                         int dxr, int recDepth, int first_rec, int last_rec
                         )


{
  int i = get_global_id(0) ;
  int j = get_global_id(1) ;


  if(i%dxr==0 && j == recDepth){
    int ir = i/dxr;
//    res[ir]=res[ir]/3;
      
      Avx[(i+first_rec)*Nx + j] += res_vx[ir];
      Avz[(i+first_rec)*Nx + j] += res_vz[ir];
      Ataux[(i+first_rec)*Nx + j] += res_taux[ir];
      Atauz[(i+first_rec)*Nx + j] += res_tauz[ir];
      Atauxz[(i+first_rec)*Nx + j] += res_tauxz[ir];
    // printf(" j:%i, i:%i, i2:%i,\n", j, i, i+first_rec);

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



/////////////////////////////// Claculate the dradient ///////////////////////////////////////

        ////////////////////////////////////////////////////////////////////////
        ////////////////////////////////////////////////////////////////////////
        /////////                          GNu                        //////////
        ////////////////////////////////////////////////////////////////////////
        ////////////////////////////////////////////////////////////////////////

__kernel void Grad_mu(__global float *vx, __global float *vz,
            __global float *taux, __global float *tauz, __global float *tauxz,
            __global float *Ataux, __global float *Atauz,  __global float *Atauxz,
            __global float *Gmu, __global float *Gmu_precond
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

  Gmu[center]+= 2*dt*(Ataux[center]*DxpVx + Atauz[center]*DzmVz) + dt*Atauxz[center]*(DzpVx+DxmVz);
    
  Gmu_precond[center]+= 4*dt*dt*(taux[center]*taux[center]*DxpVx*DxpVx + tauz[center]*tauz[center]*DzmVz*DzmVz)
                          +dt*dt*tauxz[center]*tauxz[center]*(DzpVx*DzpVx+DxmVz*DxmVz);

}

                    ////////////////////////////////////////////////////////////////////////
                    ////////////////////////////////////////////////////////////////////////
                    /////////                          G lam                      //////////
                    ////////////////////////////////////////////////////////////////////////
                    ////////////////////////////////////////////////////////////////////////
__kernel void Grad_lam(__global float *vx, __global float *vz,
            __global float *taux, __global float *tauz,
            __global float *Ataux, __global float *Atauz, 
            __global float *Glam, __global float *Glam_precond
            )


{
  int i = get_global_id(0)+ npml + sdo ;
  int j = get_global_id(1)+ npml + sdo ;

  if(i>Nz-sdo - npml || j>Nx-sdo - npml){return;}



  float DxpVx= (c1*(vx[right]-vx[center]) + c2*(vx[right2]-vx[left])+
               c3*(vx[right3]-vx[left2]) + c4*(vx[right4]-vx[left3]))/dx;

  float DzmVz= (c1*(vz[center]-vz[above]) + c2*(vz[below]-vz[above2])+
               c3*(vz[below2]-vz[above3]) + c4*(vz[below3]-vz[above4]))/dz;


  Glam[center]+= dt*(Ataux[center] + Atauz[center])*(DxpVx+DzmVz);

  Glam_precond[center] += dt*dt*(taux[center]*taux[center] + tauz[center]*tauz[center])*(DxpVx*DxpVx+DzmVz*DzmVz);
  // Hlam[center]+= dt*(taux[center] + tauz[center])*(DxpVx+DzmVz);
  // A=glam/(hlam+...)
      

}

                    ////////////////////////////////////////////////////////////////////////
                    ////////////////////////////////////////////////////////////////////////
                    /////////                          G rho                        //////////
                    ////////////////////////////////////////////////////////////////////////
                    ////////////////////////////////////////////////////////////////////////


__kernel void Grad_rho(__global float *vx, __global float *vz,
            __global float *px, __global float *pz, __global float *pxz,
            __global float *Avx, __global float *Avz,
            __global float *rho_b ,
            __global float *Grho, __global float *Grho_precond
            )


{
  int i = get_global_id(0)+ npml + sdo ;
  int j = get_global_id(1)+ npml + sdo ;

  if(i>Nz-sdo - npml || j>Nx-sdo - npml){return;}

  float DzpPz= (c1*(pz[below]-pz[center]) + c2*(pz[below2]-pz[above])+
               c3*(pz[below3]-pz[above2]) + c4*(pz[below4]-pz[above3]))/dz;

  float DxmPx= (c1*(px[center]-px[left]) + c2*(px[right]-px[left2])+
               c3*(px[right2]-px[left3]) + c4*(px[right3]-px[left4]))/dx;

  float DxpPxz= (c1*(pxz[right]-pxz[center]) + c2*(pxz[right2]-pxz[left])+
               c3*(pxz[right3]-pxz[left2]) + c4*(pxz[right4]-pxz[left3]))/dx;

  float DzmPxz= (c1*(pxz[center]-pxz[above]) + c2*(pxz[below]-pxz[above2])+
               c3*(pxz[below2]-pxz[above3]) + c4*(pxz[below3]-pxz[above4]))/dz;


  Grho[center]+= -(dt*(rho_b[center]*rho_b[center]))*(Avx[center]*(DxmPx + DzmPxz)+
                                                  Avz[center]*(DzpPz+DxpPxz));

  Grho_precond[center] += dt*dt*(vx[center]*vx[center]*(DxmPx*DxmPx + DzmPxz*DzmPxz)+
                                            vz[center]*vz[center]*(DzpPz*DzpPz + DxpPxz*DxpPxz));

}
