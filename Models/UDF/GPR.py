def broadcaster(array):
    return np.broadcast_to(array[:, np.newaxis, np.newaxis], (10, 256, 256))

hyp_ell_GREEN = broadcaster(hyp_ell_GREEN)
mx_GREEN = broadcaster(mx_GREEN.ravel())
sx_GREEN = broadcaster(sx_GREEN.ravel())
XDX_pre_calc_GREEN_broadcast = np.broadcast_to(XDX_pre_calc_GREEN.ravel()[:,np.newaxis,np.newaxis],(XDX_pre_calc_GREEN.shape[0],256,256))

init_xr = xr.DataArray()
def apply_datacube(cube: xarray.DataArray, context: dict) -> xarray.DataArray:

    pixel_spectra = (cube.values)
    inspect(data=[pixel_spectra.shape], message="pixel_spectra.shape")

    im_norm_ell2D_hypell  = ((pixel_spectra - mx_GREEN) / sx_GREEN) * hyp_ell_GREEN
    im_norm_ell2D  = ((pixel_spectra - mx_GREEN) / sx_GREEN)

    PtTPt = np.einsum('ijk,ijk->ijk', im_norm_ell2D_hypell, im_norm_ell2D) * (-0.5)
    PtTDX = np.einsum('ij,jkl->ikl',X_train_GREEN,im_norm_ell2D_hypell)
    arg1 = np.exp(PtTPt[0]) * hyp_sig_GREEN

    k_star_im = np.exp(PtTDX - (XDX_pre_calc_GREEN_broadcast * (0.5)))
    mean_pred = (np.einsum('ijk,i->jk',k_star_im,alpha_coefficients_GREEN.ravel()) * arg1) + mean_model_GREEN

    init_xr = mean_pred
    returned = xr.DataArray(init_xr)
    return returned
