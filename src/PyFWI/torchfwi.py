import matplotlib.pyplot as plt
import numpy as np

import PyFWI.processing as process
import torch

class Fwi(torch.autograd.Function):
    """
        Fwi is a custom function to integrate PyFWI wo PyTorch.
        The gradient of cost function with respect to model parameters
        are obtained using adjoint state method.
        
        Parameters
        ----------
        W : WavePropagator
            A class of WavePropagator to perform the modeling and backpropagations
        vp : Tensor
            P-wave velocity
        vs : Tensor
            S-wave velocity
        rho : Tensor
            Density

        Returns
        -------
        taux : Tensor
            Normal stress in x-direction 
        tauz : Tensor
            Normal stress in z-direction 
        """
    @staticmethod
    def forward(ctx, W, vp, vs, rho):
        
        model = {'vp': vp.cpu().numpy().astype(np.float32),
                 'vs': vs.cpu().numpy().astype(np.float32),
                 'rho': rho.cpu().numpy().astype(np.float32)
                 }

        # Call the forward modelling
        db_obs = W.forward_modeling(model, show=False)  # show=True can show the propagation of the wave

        # ctx.save_for_backward(vp)        
        ctx.WavePropagator = W
                
        return torch.tensor(db_obs['taux']),\
            torch.tensor(db_obs['tauz'])

    
    @staticmethod
    def backward(ctx, adj_tx, adj_tz):
        """
        backward calculates the gradient of cost function with
        respect to model parameters

        Parameters
        ----------
        adj_tx : Tensor
            Adjoint of recoreded normal stress in x-direction
        adj_tz : Tensor
            Adjoint of recoreded normal stress in z-direction

        Returns
        -------
        Tensor
            Gradient of cost function w.r.t. model parameters
        """
        adj_taux = adj_tx.cpu().detach().numpy()
        adj_tauz = adj_tz.cpu().detach().numpy()
        
        # If shape is (nt, nr, ns) switch it to (nt, nr * ns)
        if adj_tauz.ndim == 3:
            (nt, nr, ns) = adj_taux.shape
            adj_taux = np.reshape(adj_taux, (nt, nr * ns), 'F')
            adj_tauz = np.reshape(adj_tauz, (nt, nr * ns), 'F')
            
        adj_src = process.prepare_residual(
            {
                # 'vx':adj_vx.detach().numpy(),
                # 'vz': adj_vz.detach().numpy(),
                'taux': adj_taux,
                'tauz': adj_tauz,
                # 'tauxz': adj_txz.detach().numpy(),
             }, 1)
                
        result = ctx.WavePropagator.gradient(adj_src, False)
        
        return (None, 
                torch.tensor(result['vp']), 
                torch.tensor(result['vs']), 
                torch.tensor(result['rho'])
                )
    