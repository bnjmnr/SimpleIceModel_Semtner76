#!/usr/bin/python
# -*- coding: utf-8 -*-
"""Toolbox related to Ice-Ocean Stochastic Modelling study."""
import xarray as xr
import numpy as np
import scipy as scp
import pandas as pd

# Content:
# - Semtner76model_0layiceH: 0-layer model to be used with the scipy.integrate.solve_ivp solver
# - Semtner76model_0layiceH_Euler: Same model implemented as a Euler backward scheme, no need for solver
# - convert2ds: function to convert the dict of numpy array output of Semtner76model_0layiceH into a Xarray.Dataset


def Semtner76model_0layiceH(t, Y_in, param):
    """
    Simple ice-model based on Semtner [1976]. This is the simplest version of the Semtner model, with no snow (0-layer model), described in the appendix.
    We do two  heat balance calculation, for surface and bottom of a single ice slab. 
    At bottom, conductive heat flux inside the ice and sensible ocean-ice heat flux are balanced by latent heat flux growing or melting ice
    At surface, atmosphere-ice heat flux is balanced by conductive flux to calculate the surface ice temperature T_s. 
        If T_s<=0°C, nothing happens (ice cannot grow at surface)
        If T_s=0°C, the result is not physical so the excess temperature is converted into latent heat to melt ice.
        
    Inputs: -t: time of output [s]
            -Y_in: variable state at t-dt, containing old ice thickness H_i, old ice surface temperature T_s and previous time stamp t
            -param: dictionary containing model parameters and forcing. Typical defaul parameters are
                    param['q'] = 300*1e6 # Latent heat capcity, [J/m3]
                    param['k_i'] = 2.3 # Heat conductivity, [J/m/K/s]
                    param['albedo_ice'] = 0.8 # Bare ice albedo, [-]
                    param['albedo_melt'] = 0.5 # Melt ice albedo, [-]
                    # Forcing-related constant parameters
                    param['T_w_in'] = -1.75 # Ice bottom temperature, often assumed constant at the melting point [°C]
                    param['time_in'] = time_int # Time vector of input forcing (like np.arange(0,int(tf/T_day)+1)) [day]
                    # Forcings
                    param['F_wind10_in'] = WindSpd # Wind speed for sensible heat flux (list, same size as time_int) [m/s]
                    param['F_w_in'] = F_w # Bottom ice-ocean heat flux, positive upward. (list, same size as time_int) [W/m2]
                    param['F_lw_in'] = F_lw # Incoming (downward) longwave (thermal) radiation (list, same size as time_int) [W/m2] 
                    param['F_sw_in'] = F_sw # Incoming (downward) shortwave (solar) radiation  (list, same size as time_int) [W/m2]
                    param['T_a_in'] = T2m # Air temperature at 2m (list, same size as time_int) [°C]
                    param['F_lh_in'] = F_lh # Downward latent heat flux due to sublimation or condensation (list, same size as time_int) [W/m2]
    Outputs: -dY_dt = [dHice_dt,dT_dt,1] rate of change of H_i, T_i and time (=1 because multiplied internally by dt).
    
    The model is run using the `scipy` solver `solve_ivp`: https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html
        Ex: sol = scipy.integrate.solve_ivp(Semtner76model_0layiceH,
                                            (t0, tf), # Initial and final time stamps (ex: [0, 86400s*365d*10yrs]) [s]
                                            [H0,T0,0], # Initial conditions (ex: [1.0,-10, 0]) [m, °C, s]
                                            method='Radau', # Solver scheme. Radau works well, RK4(5) as well but slower.
                                            args=(param,), # Model parameters and forcings
                                            dt_eval=np.arange(t0, tf+1, 86400), # Time stamps at which to save the outputs (ex: daily)
                                            max_step=T_day/3) # Maximum time step for the solver scheme (dt is variable in RK)
    
    References: -Semtner [1976] doi:10.1175/1520-0485(1976)006<0379:AMFTTG>2.0.CO;2
                -Maykut & Untersteiner [1971]: doi:10.1029/JC076i006p01550
                -Massonnet et al. [2018]: doi:10.1038/s41558-018-0204-z
    V1: BR, 2023/01/13
    """

    # Assign parameters:
    L_ice = param['q'] # Ice specific latent heat of fusion [J/m^3]
    k_i = param['k_i'] # Ice heat conductivity [J/s/m/K]
    
    # Need to introduce a condition: if H_ice is null or negative, I need to put it back as non-null, else conductivity is infinite
    if Y_in[0]<=1e-3:
        H_i_old = 1e-3
    # Else, I just take the H_ice from previous time stamp
    else:
        H_i_old = Y_in[0]
    # Assign the old surface temperature
    T_s_old = Y_in[1] # [°C]
    # Also get the old time stamp and calculate the time step from there. Very ugly...
    t_ts = Y_in[2]
    dt = t - t_ts  # [s]

    # More parameters
    T_day = 86400 # length of day [s]
    sigma_b = 5.67 * 10 ** (-8) # Stefan-Boltzmann constant [W/m^2/K^4].
    TK2C = 273.15 # To convert from Kelvin to celsius [K]

    # Define some values
    T_w = param['T_w_in'] # Bottom temperature [degC]. #Could be providing as a forcing as well.
    # F_w = param['F_w_in'] # Bottom ice-ocean flux [W/m^2]. #Could be providing as a forcing as well.
    F_w = np.interp(t/T_day, param['time_in'], param['F_w_in']) # Bottom ice-ocean flux [W m^{-2}]
    
    # Surface Heat Fluxes
    # -------------------
    # Calculate net shortwave radiation by changing albedo according to temperature
    F_sw_down = np.interp(t/T_day, param['time_in'], param['F_sw_in']) # [W m^{-2}]
    if T_s_old<0:
        F_sw_net = F_sw_down * (1 - param['albedo_ice']) # [W m^{-2}]
    else:
        F_sw_net = F_sw_down * (1 - param['albedo_melt']) # [W m^{-2}]
    # Get longwave radiation
    F_lw = np.interp(t/T_day, param['time_in'], param['F_lw_in']) # [W m^{-2}]
    # Get latent heat flux
    F_lh = np.interp(t/T_day, param['time_in'], param['F_lh_in'])
    # Prepare sensible heat flux
    # Need to assign the transfer function for sensible heat flux:
    # f_sh = rho_a * c_{p,a} * c_{sh} * Wind speed
    Wd_10m = param['F_wind10_in']
    # Wd_10m = np.interp(t/T_day, param['time_in'], param['F_wind10_in']) # If Wind is variable 
    f_sh = 1.22 * 1005 * 1.75 * 10**(-3) * Wd_10m # [W m^{-2}]
    # Also assign air temperature
    T_a2m =  np.interp(t/T_day, param['time_in'], param['T_a_in']) + TK2C # [K]
    
    # Surface Temperature Calculation
    # -------------------------------
    # I try to balance the surface heat budget Fair-ice + Fconduction = 0.
    # Using that, I can calculate a surface temperature:
    # \sigma_b T^4 +(\frac{k_i}{H_i} + f_{sh}) T - F_sw_net - F_lw - f_sh * T_a - \frac{k_i}{H_i} T_b = 0
    # But first, need a condition if there is no ice!
    # if H_i_old<=1e-3:
    #     T_s_new = 0.
    # else:
    roots = np.roots([sigma_b, 
                      0., 
                      0., 
                      f_sh + k_i/H_i_old, 
                      -(F_sw_net + F_lw + F_lh + f_sh * T_a2m + k_i/H_i_old * (T_w + TK2C))])
    # Looked at the derivative of this polynomial: it's a simple a T^3 +b, which only has a negative root.
    # So I know that the 4th order polynomial only has two real roots, and only one that might be positive.
    roots_real = np.real(roots[np.isreal(roots)])
    T_s_new = np.max(roots_real) - TK2C # Convert back to Celsius

    # Surface Heat Budget
    # -------------------
    # I need to consider two cases: 
    # - If my surface temperature is positive, it is not realistic. 
    #   So I set it to 0. Using that, I can calculate the new heat conductive flux.
    #   Then, I calculate an outgoing longwave radiative flux (\sigma * (0+273.15)^4) to add to the surface heat budget.
    #   I recalculate my surface heat budget with this new temperature of 0°C, and I know it is supposed to be negative,
    #   since I already calculated the polynomial and it gave me a root above this zero surface temperature. 
    #   So calculating the surface budget with T=0 will necessarily give me a negative residual, which is "used" to melt ice.
    if T_s_new>=0.0:
        F_c_new = -k_i * (0.0 - T_w) / H_i_old  # New conductive flux with surface ice temperature Tia=0
        F_lw_out = -sigma_b * (0.0 + TK2C)**4  # New upward longwave radiations with Tia=0
        # Calculate the residual, divide it by latent heat L_ice
        dHice_dt_top = -(F_sw_net + F_lw + F_lh + f_sh * (T_a2m - (0.0 + TK2C)) + F_lw_out + F_c_new)/ L_ice
    # - If my surface temperature is negative, I don't have surface melt and I cannot have surface growth.
    #   So I can just calculate the heat conduction with this new temperature and set the surface ice growth-melt rate to 0.
    else:
        F_c_new = - k_i * (T_s_new - T_w) / H_i_old  # New heat conductivity
        dHice_dt_top = 0.0  # No ice melt at surface
    # Done with the surface budget

    # Bottom Heat Budget
    # -------------------
    # The bottom budget is much simpler, just the balance between the ice-ocean flux and the conductive flux.
    dHice_dt_bot = (F_c_new - F_w) / L_ice
    # Done with the bottom budget

    # Total Heat Budget
    # -------------------
    # Just need to sum both budgets
    dHice_dt = dHice_dt_bot + dHice_dt_top

    # Calculate the ice surface temperature rate
    dT_dt = (T_s_new - T_s_old) / max(dt,1)
    # Need to introduce another condition: if I don't have ice, I cannot have a negative growth rate. So put it back to 0
    if (dHice_dt<0.0) & (Y_in[0]<=0.0):
        dHice_dt=0.0
    # Return the ice change rate, temperature change rate, and time step
    return [dHice_dt,dT_dt,1]


def Semtner76model_0layiceH_Euler(param, begT=0,endT=86400*365,H0=1.1,T_s0=-20,dt=86400/3):
    """
    Exactly the same model as previous, but implemented as a Euler Backward scheme, with fixed time step. 
    This is faster to run than the Radau/RK scheme, but with significant errors when using highly variable stochastic forcing.
    Used by May Wang in our paper about Labrador Shelf ice climatology, trend and variability.
    Inputs: - param: as above
            - begT: initial time stamp (default:0)
            - endT: final time stamp (default: 86400*365 s = 1 year)
            - H0: Initial condition of ice thickness (default: 1.1 m)
            - T_s0: Initial condition of ice surface temperature (default: -20 °C)
            - dt: time step (default: 86400/3 s = 8 hours)
    V1: BR, 2023/06/20
    """
    time_vec = np.arange(begT, endT+1, dt)
    # dHice_dt = np.zeros(len(time_vec))
    H_ice = np.empty(len(time_vec))
    H_ice[:] = np.nan
    # F_airice = np.zeros(t_eval)
    T_s = np.empty(len(time_vec))
    T_s[:] = np.nan
    # LWR_out = np.zeros(time_steps)

    # initial condition
    H_ice[0] = H0
    T_s[0] = T_s0
    # Assign parameters:
    L_ice = param['q'] # Ice specific latent heat of fusion [J/m^3]
    k_i = param['k_i'] # Ice heat conductivity [J/s/m/K]
    # More parameters
    T_day = 86400 # length of day [s]
    sigma_b = 5.67 * 10 ** (-8) # Stefan-Boltzmann constant [W/m^2/K^4].
    TK2C = 273.15 # To convert from Kelvin to celsius [K]
    # One of the strength of the Euler scheme is that I know in advance the time steps so can interpolate the forcing once anf or all.
    # Define some values
    T_w = param['T_w_in'] # Bottom temperature [degC]. #Could be providing as a forcing as well.
    F_w = np.interp(time_vec/T_day, param['time_in'], param['F_w_in']) # Bottom ice-ocean flux [W m^{-2}]
    F_sw_down = np.interp(time_vec/T_day, param['time_in'], param['F_sw_in']) # [W m^{-2}]
    # Get longwave radiation
    F_lw = np.interp(time_vec/T_day, param['time_in'], param['F_lw_in']) # [W m^{-2}]
    # Get latent heat flux
    F_lh = np.interp(time_vec/T_day, param['time_in'], param['F_lh_in'])
    # Prepare sensible heat flux
    # Need to assign the transfer function for sensible heat flux:
    # f_sh = rho_a * c_{p,a} * c_{sh} * Wind speed
    Wd_10m = param['F_wind10_in']
    # Wd_10m = np.interp(t/T_day, param['time_in'], param['F_wind10_in']) # If Wind is variable 
    f_sh = 1.22 * 1005 * 1.75 * 10**(-3) * Wd_10m # [W m^{-2}]
    # Also assign air temperature
    T_a2m =  np.interp(time_vec/T_day, param['time_in'], param['T_a_in']) + TK2C # [K]

    for i in range(1,len(H_ice)):
        # Need to introduce a condition: if H_ice is negative, I need to put it back as non-nul. 
        if H_ice[i-1]<=1e-3:
            H_i_old = 1e-3
        # Else, I just take the H_ice from previous time stamp
        else:
            H_i_old = H_ice[i-1]
        # Assign the old surface temperature
        T_s_old = T_s[i-1] # [°C]

        # Surface Heat Fluxes
        # -------------------
        # Calculate net shortwave radiation by changing albedo according to temperature
        if T_s_old<0:
            F_sw_net = F_sw_down[i] * (1 - param['albedo_ice']) # [W m^{-2}]
        else:
            F_sw_net = F_sw_down[i] * (1 - param['albedo_melt']) # [W m^{-2}]

        # Surface Temperature Calculation
        # -------------------------------
        # I try to balance the surface heat budget Fair-ice + Fconduction = 0.
        # Using that, I can calculate a surface temperature:
        # \sigma_b T^4 +(\frac{k_i}{H_i} + f_{sh}) T - F_sw_net - F_lw - f_sh * T_a - \frac{k_i}{H_i} T_b = 0
        # But first, need a condition if there is no ice!
        # if H_i_old<=1e-3:
        #     T_s_new = 0.
        # else:
        roots = np.roots([sigma_b, 
                          0., 
                          0., 
                          f_sh + k_i/H_i_old, 
                          -(F_sw_net + F_lw[i] + F_lh[i] + f_sh * T_a2m[i] + k_i/H_i_old * (T_w + TK2C))])
        # Looked at the derivative of this polynomial: it's a simple a T^3 +b, which only has a negative root.
        # So I know that the 4th order polynomial only has two real roots, and only one that might be positive.
        roots_real = np.real(roots[np.isreal(roots)])
        T_s_new = np.max(roots_real) - TK2C # Convert back to Celsius

        # Surface Heat Budget
        # -------------------
        # I need to consider two cases: 
        # - If my surface temperature is positive, it is not realistic. 
        #   So I set it to 0. Using that, I can calculate the new heat conductive flux.
        #   Then, I calculate an outgoing longwave radiative flux (\sigma * (0+273.15)^4) to add to the surface heat budget.
        #   I recalculate my surface heat budget with this new temperature of 0°C, and I know it is supposed to be negative,
        #   since I already calculated the polynomial and it gave me a root above this zero surface temperature. 
        #   So calculating the surface budget with T=0 will necessarily give me a negative residual, which is "used" to melt ice.
        if T_s_new>=0.0:
            F_c_new = -k_i * (0.0 - T_w) / H_i_old  # New conductive flux with surface ice temperature Tia=0
            F_lw_out = -sigma_b * (0.0 + TK2C)**4  # New upward longwave radiations with Tia=0
            # Calculate the residual, divide it by latent heat L_ice
            dHice_dt_top = -(F_sw_net + F_lw[i] + F_lh[i] + f_sh * (T_a2m[i] - (0.0 + TK2C)) + F_lw_out + F_c_new)/ L_ice
        # - If my surface temperature is negative, I don't have surface melt and I cannot have surface growth.
        #   So I can just calculate the heat conduction with this new temperature and set the surface ice growth-melt rate to 0.
        else:
            F_c_new = - k_i * (T_s_new - T_w) / H_i_old  # New heat conductivity
            dHice_dt_top = 0.0  # No ice melt at surface
        # Done with the surface budget

        # Bottom Heat Budget
        # -------------------
        # The bottom budget is much simpler, just the balance between the ice-ocean flux and the conductive flux.
        dHice_dt_bot = (F_c_new - F_w[i]) / L_ice
        # Done with the bottom budget

        # Total Heat Budget
        # -------------------
        # Just need to sum both budgets
        dHice_dt = dHice_dt_bot + dHice_dt_top

        # Calculate the ice surface temperature rate
        dT_dt = (T_s_new - T_s_old) / dt
        # Need to introduce another condition: if I don't have ice, I cannot have a negative growth rate. So put it back to 0
        if (dHice_dt<0.0) & (H_ice[i-1]<=0.0):
            dHice_dt=0.0
        # Return the ice change rate, temperature change rate, and time step
        H_ice[i] = H_ice[i-1] + dHice_dt * dt
        T_s[i] = T_s_new
    return {'t': time_vec[::int(T_day/dt)], 'y':[H_ice[::int(T_day/dt)], T_s[::int(T_day/dt)]]}

def convert2ds(sol,params=None,time_origin=0):
    """Create a xarray.Dataset from the Semtner model outputs."""
    # Bulk of the function
    ds_sol = xr.Dataset(coords={'time': pd.to_datetime(sol['t'],origin=time_origin,unit='s')},
                        data_vars={'H_i': ('time',sol['y'][0]),
                                   'T_i': ('time',sol['y'][1]),
                                   'Success':(sol.success)}
                       )
    # Then assign some attributes to describe the solution
    ds_sol['H_i'].attrs['name'] = 'Thickness'
    ds_sol['H_i'].attrs['long_name'] = 'Sea Ice Thickness'
    ds_sol['H_i'].attrs['units'] = 'm'
    ds_sol['T_i'].attrs['name'] = 'Temperature'
    ds_sol['T_i'].attrs['long_name'] = 'Sea Ice Surface Temperature'
    ds_sol['T_i'].attrs['units'] = '°C'
    ds_sol.attrs['description'] = 'Simulation of sea ice thickness and surface temperature, by the Semtner model.'
    # If the user gave some parameters, also stores the forcings
    if params is not None:
        # Longwave Radiation
        ds_sol['F_lw_in'] = ('time', params['F_lw_in'])
        ds_sol['F_lw_in'].attrs['name'] = 'Longwave'
        ds_sol['F_lw_in'].attrs['long_name'] = 'Downward Longwave Radiation (Forcing)'
        ds_sol['F_lw_in'].attrs['units'] = r'W m$^{-2}$'
        # Shortwave Radiation
        ds_sol['F_sw_in'] = ('time', params['F_sw_in'])
        ds_sol['F_sw_in'].attrs['name'] = 'Shortwave'
        ds_sol['F_sw_in'].attrs['long_name'] = 'Downward Shortwave Radiation (Forcing)'
        ds_sol['F_sw_in'].attrs['units'] = r'W m$^{-2}$'
        # Sensible Heat Flux
        ds_sol['F_sh_in'] = ('time', calculate_SensibleHF(sol['t']/86400, sol['y'][1], params))
        ds_sol['F_sh_in'].attrs['name'] = 'Sensible'
        ds_sol['F_sh_in'].attrs['long_name'] = 'Sensible Heat Flux (Forcing)'
        ds_sol['F_sh_in'].attrs['units'] = r'W m$^{-2}$'
        # Latent Heat Flux
        ds_sol['F_lh_in'] = ('time', params['F_lh_in'])
        ds_sol['F_lh_in'].attrs['name'] = 'Latent'
        ds_sol['F_lh_in'].attrs['long_name'] = 'Latent Heat Flux (Forcing)'
        ds_sol['F_lh_in'].attrs['units'] = r'W m$^{-2}$'
        # Air Temperature
        ds_sol['T_a_in'] = ('time', params['T_a_in'])
        ds_sol['T_a_in'].attrs['name'] = 'T2m'
        ds_sol['T_a_in'].attrs['long_name'] = 'Air Temperature (Forcing)'
        ds_sol['T_a_in'].attrs['units'] = '°C'
        # Outgoing Longwave Radiation
        ds_sol['D_lw_out'] = ('time', calculate_OutgoingLW(sol['y'][1]))
        ds_sol['D_lw_out'].attrs['name'] = 'OutLongwave'
        ds_sol['D_lw_out'].attrs['long_name'] = 'Upward Longwave Radiation (Diagnostic)'
        ds_sol['D_lw_out'].attrs['units'] = r'W m$^{-2}$'
    return ds_sol

