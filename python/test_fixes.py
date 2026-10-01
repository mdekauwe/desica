#!/usr/bin/env python

"""
Regression tests for the sub-stepping, soil water balance, root leak rule,
gravity term and water potential bounds. Run as: python test_fixes.py

That's all folks.
"""
__author__ = "Martin De Kauwe"
__version__ = "1.0 (01.10.2026)"
__email__ = "mdekauwe@gmail.com"

import numpy as np
import pandas as pd
from canopy import Canopy
from desica import Desica
from generate_met_data import generate_met_data

def setup(**kwargs):
    params = dict(psi_stem0=0., AL=6., p50=-4., psi_f=-3., gmin=10.,
                  Cl=10000., Cs=120000., g1=4.)
    params.update(kwargs)
    return Desica(F=Canopy(g1=params["g1"]), **params)

def constant_met(N=480, par=1500., vpd=2.0, precip=0.0):
    return pd.DataFrame({'day':np.arange(N) // 96 + 1,
                         'par':np.ones(N) * par, 'tair':np.ones(N) * 25.0,
                         'vpd':np.ones(N) * vpd,
                         'precip':np.ones(N) * precip,
                         'press':np.ones(N) * 101.0,
                         'Ca':np.ones(N) * 400.0})

def soil_water_closure(D, out, met):
    # change in soil water (kg) vs precip in minus root uptake out (kg)
    out = out.dropna(subset=["sw"])
    dt = 60. * D.met_timestep
    d_store = (out.sw.iloc[-1] - D.sw0) * D.soil_volume * 1E03
    p_in = (met.precip[1:len(out)] * D.ground_area * dt).sum()
    q_out = (out.flux_to_stem[1:] * 1E-06 * 18.0 * dt).sum()
    runoff = out.runoff[1:].sum() * D.ground_area
    return d_store, p_in - q_out - runoff

def plant_water_closure(D, out):
    # change in stem + leaf storage (mmol) vs root uptake in minus
    # transpiration out (mmol)
    out = out.dropna(subset=["sw"])
    dt = 60. * D.met_timestep
    d_store = D.Cs * (out.psi_stem.iloc[-1] - D.psi_stem0) + \
              D.Cl * (out.psi_leaf.iloc[-1] - D.psi_leaf0)
    d_flux = ((out.flux_to_stem[1:] - D.AL * out.Eleaf[1:]) * dt).sum()
    return d_store, d_flux

def test_nruns_derived():
    assert setup(met_timestep=15.).nruns == 2
    assert setup(met_timestep=30.).nruns == 3
    assert setup(met_timestep=60.).nruns == 6
    assert setup(met_timestep=30., nruns=1).nruns == 1

def test_soil_water_closure():
    met = generate_met_data(Tmin=10, RH=30, ndays=40, time_step=30)
    D = setup(met_timestep=30., nruns=3)
    out = D.run_simulation(met)
    d_store, d_flux = soil_water_closure(D, out, met)
    assert d_store < -10.0
    assert np.isclose(d_store, d_flux, rtol=1E-06), (d_store, d_flux)

def test_precip_closure():
    # start below saturation so the bucket is not capped; 1 mm per day
    met = constant_met(precip=1.0 / 86400.)
    D = setup(met_timestep=15., sw0=0.3, stop_dead=False)
    out = D.run_simulation(met)
    d_store, d_flux = soil_water_closure(D, out, met)
    p_in = (met.precip[1:] * 900.).sum()
    assert np.isclose(p_in, 479. * 900. / 86400.)
    assert np.isclose(d_store, d_flux, rtol=1E-06), (d_store, d_flux)

def test_runoff():
    # 100 mm in one 15 min step onto a saturated bucket: almost all runs off
    met = constant_met(N=96, par=0.)
    met.loc[10, "precip"] = 100.0 / 900.
    D = setup(met_timestep=15., sw0=0.5, stop_dead=False)
    out = D.run_simulation(met)
    assert out.sw.max() <= D.theta_sat
    assert 99.0 < out.runoff[10] <= 100.0, out.runoff[10]
    assert (out.runoff.drop(10) == 0.0).all()
    d_store, d_flux = soil_water_closure(D, out, met)
    assert np.isclose(d_store, d_flux, rtol=1E-06), (d_store, d_flux)

def test_nruns_independence():
    # the soil should lose the same water whatever the number of sub-steps
    met = generate_met_data(Tmin=10, RH=30, ndays=200, time_step=30)
    sw, death = [], []
    for nruns in (1, 3, 6):
        out = setup(met_timestep=30., nruns=nruns).run_simulation(met)
        sw.append(out.sw[480])
        death.append(out.sw.last_valid_index())
    assert np.allclose(sw, sw[0], rtol=1E-04), sw
    assert max(death) - min(death) <= 48, death

def test_no_root_leak():
    # very dry soil, wet stem: no uptake and no flow back to the soil
    met = constant_met(N=96)
    D = setup(met_timestep=15., sw0=0.1, psi_stem0=-0.5, psi_leaf0=-1.0,
              stop_dead=False)
    out = D.run_simulation(met)
    assert out.psi_soil[0] < out.psi_stem[0]
    assert (out.flux_to_stem[1:] == 0.0).all()
    assert (out.sw == D.sw0).all()
    # psi_stem drains to supply the leaf, by J dt / Cs per sub-step
    assert out.psi_stem[1] < out.psi_stem[0]

def test_gravity_steady_state():
    # at steady state psi_leaf = psi_soil - Eleaf / kplant - psi_h, and the
    # stem store, halfway up, sits half of each drop below the soil
    met = constant_met()
    for height in (0., 20.):
        D = setup(met_timestep=15., keep_wet=True, height=height)
        steady = D.run_simulation(met).iloc[-1]
        x = steady.psi_soil - steady.Eleaf / steady.kplant - D.psi_h
        assert np.isclose(x, steady.psi_leaf, rtol=0.01), (height, x,
                                                           steady.psi_leaf)
        x = steady.psi_soil - steady.Eleaf / steady.ksoil2stem - D.psi_h / 2.
        assert np.isclose(x, steady.psi_stem, rtol=0.01), (height, x,
                                                           steady.psi_stem)
    assert np.isclose(setup(height=20.).psi_h, 0.1962)

def test_gravity_no_uptake():
    # a 20 m plant in wet soil at night with the stem at -0.05 MPa: the soil
    # is wetter than the stem, but not by enough to lift water 10 m to the
    # store, so there is no uptake
    met = constant_met(N=4, par=0., vpd=0.)
    D = setup(met_timestep=15., height=20., psi_stem0=-0.05,
              psi_leaf0=-0.15, gmin=0.)
    out = D.run_simulation(met)
    assert out.psi_soil[0] > out.psi_stem[0]
    assert out.psi_soil[0] - D.psi_h / 2. < out.psi_stem[0]
    assert (out.flux_to_stem[1:] == 0.0).all()

def test_ksoil_units():
    # Ksat in mol m-1 s-1 MPa-1, ksoil in mmol m-2 (leaf) s-1 MPa-1
    D = setup()
    rcyl = 1.0 / np.sqrt(np.pi * D.Lv)
    expected = 1E03 * (D.Lv * D.soil_depth / D.lai) * 2. * np.pi * D.Ksat / \
                np.log(rcyl / D.rroot)
    assert np.isclose(D.calc_ksoil(D.psi_e), expected)

def test_bounds():
    met = generate_met_data(Tmin=10, RH=30, ndays=200, time_step=30)
    D = setup(met_timestep=30., stop_dead=False, gmin=50.)
    out = D.run_simulation(met)
    assert out.psi_stem.min() >= D.psi_stem_min
    assert out.psi_leaf.min() >= D.psi_leaf_min
    assert np.isclose(out.psi_stem.min(), 2.0 * D.p50)
    assert np.isclose(out.psi_leaf.min(), D.psi_leaf_min)
    # the bounds must not create water: plant storage change equals root
    # uptake minus transpiration, and transpiration is cut below the
    # stomatal demand once both pools are at their bounds
    d_store, d_flux = plant_water_closure(D, out)
    assert np.isclose(d_store, d_flux, rtol=1E-06), (d_store, d_flux)
    at_bounds = np.isclose(out.psi_stem, D.psi_stem_min) & \
                np.isclose(out.psi_leaf, D.psi_leaf_min)
    assert at_bounds.any()
    demand = D.gmin * met.vpd / met.press
    assert (out.Eleaf[at_bounds] < demand[at_bounds]).all()

def test_plant_water_closure():
    met = generate_met_data(Tmin=10, RH=30, ndays=200, time_step=30)
    D = setup(met_timestep=30.)
    out = D.run_simulation(met)
    d_store, d_flux = plant_water_closure(D, out)
    assert np.isclose(d_store, d_flux, rtol=1E-06), (d_store, d_flux)

if __name__ == "__main__":

    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for test in tests:
        test()
        print("ok", test.__name__)
