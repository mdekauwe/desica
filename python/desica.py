#!/usr/bin/env python

"""
Desica model: simple plant hydraulics model with mortality.

Key features:
- plant hydraulic conductance depends on xylem water potential, defined by a
  PLC curve
- Stomatal conductance is modelled via the modified Tuzet model. Use of Tuzet
  allows parameterisations of plants with differing stomatal control over
  psi_leaf, i.e. isohydric vs anisohydric.
- During severe drought, root water uptake ceases (due to the decline in
  soil-to-root conductance), but water loss continues due to the minimum
  conductance (gmin). This leads to a gradual decline in the stem water pool.
- The water storage pool is split between the stem and leaf water storage pools
- Xylem water potential is calculated from stem water storage via a simple
  constant capacitance term (and likewise, for the leaf water pool).
- Model solves two fluxes: flux of water from the stem to the leaves and the
  flux of water from the soil to the stem in every timestep. Stomatal
  conductance is set from psi_leaf at the start of the met timestep and held
  fixed; the water potentials are then integrated over sub-steps of <= 10 mins
  (nruns), each pool solved with the other held at its previous value.
- With a plant height set, the gravitational drop rho g h is split equally
  either side of the stem store, as the store is halfway up the plant.
- Root water uptake only occurs when the soil is wetter than the stem (no
  leak back to the soil), and psi_leaf / psi_stem have lower bounds.
- Leaves are assumed to be perfectly coupled, so transpiration = Eleaf * AL,
  where AL is plant leaf area (m2).
- Approach follows Xu to solve the psi_leaf, psi_stem without the need for a
  numerical integrator.  This avoids potential numerical instabilities due to
  various dependancies on water potential. This method works well at short
  timesteps (up to about) 10 to 15 mins. NB we may still get some ocillations.

This is a python implementation of Remko's R code.

References:
----------
* Duursma & Choat (2017). Fitplc - an R package to fit hydraulic vulnerability
  curves. Journal of Plant Hydraulics, 4, e002.
* Tuzet et al. (2003) A coupled model of stomatal conductance,
  photosynthesis and transpiration. Plant, Cell and Environment 26,
  1097–1116.
* Xu X, Medvigy D, Powers JS, Becknell JM, Guan K (2016) Diversity in plant
  hydraulic traits explains seasonal and inter-annual variations of vegetation
  dynamics in seasonally dry tropical forests. New Phytologist, 212, 80–95.

That's all folks.
"""
__author__ = "Martin De Kauwe"
__version__ = "1.0 (19.02.2018)"
__email__ = "mdekauwe@gmail.com"

import pandas as pd
import sys
import numpy as np
import matplotlib.pyplot as plt
import os
from generate_met_data import generate_met_data
from canopy import Canopy, FarquharC3
from math import isclose

class Desica(object):

    def __init__(self, plc_dead=88., soil_depth=1.0, ground_area=1.0,
                 met_timestep=30., sf=8., g1=4., Cs=100000., b=6.,
                 Cl=10000., kp_sat=3., p50=-4., psi_f=-2., s50=30., gmin=10,
                 psi_leaf0=-1., psi_stem0=-0.5, theta_sat=0.5, sw0=0.5, AL=2.5,
                 psi_e=-0.8*1E-03, Ksat=20., Lv=10000., F=None, keep_wet=False,
                 stop_dead=True, nruns=None, rroot=1E-06, height=0.,
                 psi_leaf_min=-20., psi_stem_min=None, dt_max=600.):

        self.keep_wet = keep_wet
        self.stop_dead = stop_dead
        self.plc_dead = plc_dead
        self.dt_max = dt_max # longest sub-step, s (Xu et al. 2016 use 600 s)
        # number of sub-steps per met step, by default derived so that the
        # sub-step is <= dt_max
        if nruns is None:
            nruns = max(1, int(np.ceil(60. * met_timestep / self.dt_max)))
        self.nruns = nruns
        self.soil_depth = soil_depth # depth of soil bucket, m
        self.ground_area = ground_area # m2
        self.soil_volume = self.ground_area * self.soil_depth # m3
        self.sf = sf # sensitivity parameter, MPa-1
        self.g1 = g1 # sensitivity of stomatal conductance to the assimilation
                     # rate, gs = g1 * fw * A / Ca (-)
        self.Cs = Cs # stem capacitance, mmol MPa-1 (total plant)
        self.Cl = Cl # leaf capacitance, mmol MPa-1 (total plant)
        self.kp_sat = kp_sat # plant saturated hydraulic conductance
                             # (mmol m-2 leaf s-1 MPa-1)
        self.p50 = p50 # xylem pressure inducing 50% loss of hydraulic
                       # conductivity due to embolism, MPa
        self.psi_f = psi_f # reference potential for Tuzet model, MPa
        self.s50 = s50 # is slope of the curve at P50 used in weibull model,
                       # % MPa-1
        self.gmin = gmin # minimum stomatal conductance, mmol m-2 leaf s-1
        self.psi_leaf0 = psi_leaf0 # initial leaf water potential, MPa
        self.psi_stem0 = psi_stem0 # initial stem water potential, MPa
        self.theta_sat = theta_sat # soil water content at saturation (m3 m-3)
        self.sw0 = sw0 # initial soil volumetric water content (m3 m-3)
        self.AL = AL # plant leaf area, m2
        self.lai = AL / self.ground_area # leaf area index, m2 m-2
        self.b = b # empirical coefficient related to the clay content of the
                   # soil (Cosby et al. 1984).
        self.psi_e = psi_e # air entry point water potential (MPa)
        self.Ksat = Ksat # saturated conductivity, mol m-1 s-1 MPa-1
        self.Lv = Lv # root length density, m m-3
        self.F = F
        self.rroot = rroot # mean radius of water absorbing roots, m
        self.height = height # plant height, m, for the gravity term
        # gravitational potential drop over the plant height, rho g h, MPa
        # (rho = 1000 kg m-3, g = 9.81 m s-2, Pa -> MPa)
        self.psi_h = 1000. * 9.81 * self.height * 1E-06
        # the stem store is halfway up the plant (see calc_conductances), so
        # half of the drop is between soil and stem, half between stem and leaf
        self.psi_h_half = 0.5 * self.psi_h
        self.psi_leaf_min = psi_leaf_min # lower bound on psi_leaf, MPa
        # lower bound on psi_stem, MPa (2 x P50 as in CABLE-DESICA)
        if psi_stem_min is None:
            psi_stem_min = 2.0 * self.p50
        self.psi_stem_min = psi_stem_min
        self.met_timestep = met_timestep # met timestep, mins
        self.timestep_sec = 60. * self.met_timestep / self.nruns # sub-step, s
        self.mol_2_mmol = 1000.0

    def run_simulation(self, met=None):
        """
        Main wrapper to control everything

        Parameters:
        -----------
        met : object
            met forcing variables: day; Ca; par; precip; press; tair; vpd

        Returns:
        -------
        out : object
            output dataframe containing calculations for each timestep

        """
        (n, out) = self.initialise_model(met)

        for i in range(1, n):

            out = self.run_timestep(i, met, out)

            # Stop the simulation if we've died, i.e. PLC > plc_dead (88%)
            if self.stop_dead:
                plc = self.calc_plc(out.at[i, "kplant"])
                if plc > self.plc_dead:
                    break

        out["plc"] = self.calc_plc(out.kplant)
        # mmol s-1
        out["Eplant"] = self.AL * out.Eleaf
        out["t"] = np.arange(1, n+1)

        return (out)

    def initialise_model(self, met):
        """
        Set everything up: set initial values, build an output dataframe to
        save things

        Parameters:
        -----------
        met : object
            met forcing variables: day; Ca; par; precip; press; tair; vpd

        Returns:
        -------
        n : int
            number of timesteps in the met file
        out : object
            output dataframe to store things as we go along

        """
        n = len(met)

        out = self.setup_out_df(met)
        out.at[0, "psi_leaf"] = self.psi_leaf0
        out.at[0, "psi_stem"] = self.psi_stem0
        out.at[0, "sw"] = self.sw0
        out.at[0, "psi_soil"] = self.calc_swp(self.sw0)
        out.at[0, "Eleaf"] = 0.0
        out.at[0, "runoff"] = 0.0

        # soil-to-root hydraulic conductance (mmol m-2 leaf s-1 MPa-1)
        out.at[0, "ksoil"] = self.calc_ksoil(out.at[0, "psi_soil"])

        return n, out

    def setup_out_df(self, met):
        """
        Create and output dataframe to save things

        Parameters:
        -----------
        met : object
            met forcing variables: day; Ca; par; precip; press; tair; vpd

        Returns:
        -------
        out : object
            output dataframe to store things as we go along.
        """
        dummy = np.ones(len(met)) * np.nan
        out = pd.DataFrame({'Eleaf':dummy,
                            'psi_leaf':dummy,
                            'psi_stem':dummy,
                            'psi_soil':dummy,
                            'sw':dummy,
                            'ksoil':dummy,
                            'kplant':dummy,
                            'flux_to_leaf':dummy,
                            'flux_to_stem':dummy,
                            'ksoil2stem':dummy,
                            'kstem2leaf':dummy,
                            'runoff':dummy})

        return out

    def run_timestep(self, i, met, out):
        """
        Advance the model over one met timestep.

        Stomatal conductance, and so Eleaf, is calculated once per met step
        from psi_leaf at the start of the step and held fixed over the
        sub-steps (as in ED2). psi_leaf, psi_stem and the soil bucket are then
        integrated over nruns sub-steps of timestep_sec, carrying all of the
        states forward between sub-steps. This lets us solve psi_leaf and
        psi_stem without the need for a numerical integrator, which works well
        for short sub-steps (<= 10 mins, Xu et al. 2016).

        The lower bounds on psi_stem and psi_leaf conserve water: when the
        stem reaches psi_stem_min, the flow to the leaf is cut to what the
        stem can supply, and if the leaf is then at psi_leaf_min,
        transpiration is cut to match (so Eleaf can fall below the stomatal
        demand only once both pools are at their bounds).

        States (psi_leaf, psi_stem, sw, psi_soil, ksoil) are stored at the end
        of the met step; fluxes (Eleaf, flux_to_leaf, flux_to_stem) are means
        over the met step and runoff is the total; conductances are from the
        last sub-step.

        Parameters:
        -----------
        i : int
            current index
        met : object
            met forcing variables: day; Ca; par; precip; press; tair; vpd
        out : object
            output dataframe to access previous calculations and update new
            states

        Returns:
        -------
        out : object
            output dataframe with row i filled in
        """
        psi_leaf = out.at[i-1, "psi_leaf"]
        psi_stem = out.at[i-1, "psi_stem"]
        sw = out.at[i-1, "sw"]
        psi_soil = out.at[i-1, "psi_soil"]
        ksoil = out.at[i-1, "ksoil"]

        # modified Tuzet model of stomatal conductance
        mult = (self.g1 / met.Ca[i]) * self.fsig_tuzet(psi_leaf)

        # Calculate photosynthesis and stomatal conductance
        gsw = self.F.canopy(met.Ca[i], met.tair[i], met.par[i],
                            met.vpd[i], mult)

        # Don't add gmin, instead use it as the lower boundary
        gsw = max(self.gmin, self.mol_2_mmol * gsw)

        # Leaf transpiration assuming perfect coupling, mmol m-2 s-1
        Eleaf_demand = gsw * (met.vpd[i] / met.press[i])

        Eleaf_sum = 0.0
        flux_to_leaf_sum = 0.0
        flux_to_stem_sum = 0.0
        runoff_sum = 0.0
        for j in range(self.nruns):

            Eleaf = Eleaf_demand

            (kplant, ksoil2stem,
             kstem2leaf) = self.calc_conductances(psi_stem, ksoil)

            psi_leaf_new = self.calc_lwp(kstem2leaf, psi_stem, psi_leaf,
                                         Eleaf)

            # Flux from stem to leaf (mmol s-1) = change in leaf storage,
            # plus transpiration
            flux_to_leaf = self.calc_flux_to_leaf(psi_leaf_new, psi_leaf,
                                                  Eleaf)

            # Update stem water potential, root uptake only while the soil is
            # wetter than the stem, allowing for the gravitational drop to the
            # store (no leak to the soil, Xu et al. Eqn S1a)
            flux_to_stem = 0.0
            if psi_soil - self.psi_h_half > psi_stem:
                psi_stem_new = self.update_stem_wp(ksoil2stem, psi_soil,
                                                   flux_to_leaf, psi_stem)
                flux_to_stem = self.calc_flux_to_stem(psi_stem_new, psi_stem,
                                                      flux_to_leaf)
            if flux_to_stem <= 0.0:
                flux_to_stem = 0.0
                psi_stem_new = (psi_stem - flux_to_leaf *
                                self.timestep_sec / self.Cs)

            # At the stem bound, the stem can only supply the root uptake plus
            # what is left in the store above the bound. Update the leaf from
            # that supply and, if the leaf is at its bound too, cut
            # transpiration so that no water is created.
            if psi_stem_new < self.psi_stem_min:
                psi_stem_new = self.psi_stem_min
                flux_to_leaf = flux_to_stem + (psi_stem - psi_stem_new) * \
                                self.Cs / self.timestep_sec
                psi_leaf_new = psi_leaf + (flux_to_leaf - self.AL * Eleaf) * \
                                self.timestep_sec / self.Cl
                if psi_leaf_new < self.psi_leaf_min:
                    psi_leaf_new = self.psi_leaf_min
                    Eleaf = (flux_to_leaf - (psi_leaf_new - psi_leaf) * \
                             self.Cl / self.timestep_sec) / self.AL

            # keep_wet holds the soil water fixed, for debugging / comparison
            if not self.keep_wet:
                (sw, runoff) = self.update_sw_bucket(met.precip[i],
                                                     flux_to_stem, sw)
                runoff_sum += runoff

                # Update soil water potential
                psi_soil = self.calc_swp(sw)

                # Update soil-to-root hydraulic conductance
                # (mmol m-2 s-1 MPa-1)
                ksoil = self.calc_ksoil(psi_soil)

            psi_leaf = psi_leaf_new
            psi_stem = psi_stem_new
            Eleaf_sum += Eleaf
            flux_to_leaf_sum += flux_to_leaf
            flux_to_stem_sum += flux_to_stem

        out.at[i, "Eleaf"] = Eleaf_sum / self.nruns
        out.at[i, "psi_leaf"] = psi_leaf
        out.at[i, "psi_stem"] = psi_stem
        out.at[i, "sw"] = sw
        out.at[i, "psi_soil"] = psi_soil
        out.at[i, "ksoil"] = ksoil
        out.at[i, "kplant"] = kplant
        out.at[i, "ksoil2stem"] = ksoil2stem
        out.at[i, "kstem2leaf"] = kstem2leaf
        out.at[i, "flux_to_leaf"] = flux_to_leaf_sum / self.nruns
        out.at[i, "flux_to_stem"] = flux_to_stem_sum / self.nruns
        out.at[i, "runoff"] = runoff_sum

        return out

    def calc_conductances(self, psi_stem_prev, ksoil):
        """
        Calculate the plant hydraulic conductances

        Parameters:
        -----------
        psi_stem_prev : float
            stem water potential from the previous (sub-)step, MPa
        ksoil : float
            soil-to-root hydraulic conductance, mmol m-2 s-1 MPa-1

        All conductances are per unit leaf area.

        Returns:
        -------
        kplant : float
            plant hydraulic conductance, mmol m-2 s-1 MPa-1
        ksoil2stem : float
            conductance from soil to stem, mmol m-2 s-1 MPa-1
        kstem2leaf : float
            conductance from stem to leaf, mmol m-2 s-1 MPa-1
        """

        # Plant hydraulic conductance (mmol m-2 s-1 MPa-1). NB. depends on stem
        # water potential from the previous (sub-)step.
        kplant = self.kp_sat * self.fsig_hydr(psi_stem_prev)

        # Conductance from root surface to the stem water pool (assumed to be
        # halfway to the leaves)
        kroot2stem = 2.0 * kplant

        # Conductance from soil to stem water store (mmol m-2 s-1 MPa-1)
        # (conductances combined in series)
        ksoil2stem = 1.0 / (1.0 / ksoil + 1.0 / kroot2stem)

        # Conductance from stem water store to the leaves (mmol m-2 s-1 MPa-1)
        # assuming the water pool is halfway up the stem
        kstem2leaf = 2.0 * kplant

        return (kplant, ksoil2stem, kstem2leaf)

    def fsig_hydr(self, psi_stem_prev):
        """
        Calculate the relative conductance as a function of xylem pressure
        using the Weibull (sigmoidal) model based on values of P50
        and S50 which are obtained by fitting curves to measured data.

        Higher values for s50 indicate a steeper response to xylem pressure.

        Parameters:
        -----------
        psi_stem_prev : float
            stem water potential from previous timestep, MPa

        Returns:
        --------
        relk : float
            relative conductance (K/Kmax) as a function of xylem pressure (-)

        References:
        -----------
        * Duursma & Choat (2017). Journal of Plant Hydraulics, 4, e002.
        """
        # xylem pressure
        P = np.abs(psi_stem_prev)

        # the xylem pressure (PX) at which X% (here 50%) of the conductivity
        # is lost
        PX = np.abs(self.p50)
        V = (50.0 - 100.) * np.log(1.0 - 50. / 100.)
        p = (P / PX)**((PX * self.s50) / V)

        # relative conductance (K/Kmax) as a function of xylem pressure
        relk = (1. - 50. / 100.)**p

        return (relk)

    def calc_lwp(self, kstem2leaf, psi_stem_prev, psi_leaf_prev, Eleaf):
        """
        Calculate leaf water potential, MPa

        This is a simplified equation based on Xu et al., using the water
        potentials from the previous timestep and the fact that we increase
        the temporal resolution to get around the need to solve the dynamic eqn
        with a numerical approach, i.e., Runge-Kutta.

        NB. kstem2leaf which is the stem conductance in CABLE needs to be a
            function of sapwood, tree height.


        Parameters:
        -----------
        kstem2leaf : float
            conductance from stem to leaf, mmol m-2 s-1 MPa-1
        psi_stem_prev : float
            stem water potential from the previous timestep, MPa
        psi_leaf_prev : float
            leaf water potential from the previous timestep, MPa
        Eleaf : float
            transpiration, mmol m-2 s-1

        Returns:
        --------
        psi_leaf : float
            leaf water potential, MPa

        References:
        -----------
        * Xu et al. (2016) New Phytol, 212: 80–95. doi:10.1111/nph.14009; see
          appendix and code. Can write the dynamic equation as:
          dpsi_leaf_dt = b + a*psi_leaf
        """

        # NB. includes the gravitational drop (Xu et al. Eqn S2a) from the
        # stem store, halfway up the plant, to the leaf: psi_h / 2.
        # Units: ap (s-1) = m2 * mmol m-2 s-1 MPa-1 / mmol MPa-1;
        # bp (MPa s-1)
        ap = -(self.AL * kstem2leaf / self.Cl)
        bp = (self.AL * kstem2leaf * (psi_stem_prev - self.psi_h_half) - \
              self.AL * Eleaf) / self.Cl
        psi_leaf = ((ap * psi_leaf_prev + bp) * \
                    np.exp(ap * self.timestep_sec) - bp) / ap

        return max(psi_leaf, self.psi_leaf_min)

    def calc_swp(self, sw):
        """
        Calculate the soil water potential (MPa). The parameters b and psi_e
        are estimated from a typical soil moisture release function.

        Parameters:
        -----------
        sw : float
            volumetric soil water content (m3 m-3)

        Returns:
        -----------
        psi_soil : float
            soil water potential, MPa

        References:
        -----------
        * Duursma et al. (2008) Tree Physiology 28, 265–276, eqn 10
        """
        return self.psi_e * (sw / self.theta_sat)**-self.b

    def calc_flux_to_stem(self, psi_stem, psi_stem_prev, flux_to_leaf):
        """
        Calculate the flux from the root to the stem, i.e. the root water
        uptake (mmol s-1) = change in stem storage plus flux_to_leaf

        Parameters:
        -----------
        psi_stem : float
            stem water potential, MPa
        psi_stem_prev : float
            stem water potential from the previous timestep, MPa
        flux_to_leaf : float
            flux of water from the stem to leaf, mmol s-1

        Returns:
        -------
        flux_to_stem : float
            flux from soil to the stem, mmol s-1
        """
        return (psi_stem - psi_stem_prev) * \
                self.Cs / self.timestep_sec + flux_to_leaf

    def calc_flux_to_leaf(self, psi_leaf, psi_leaf_prev, Eleaf):
        """
        Calculate the flux from the stem to the leaf = change in leaf storage
        plus transpiration

        Parameters:
        -----------
        psi_leaf : float
            leaf water potential, MPa
        psi_leaf_prev : float
            leaf water potential from the previous timestep, MPa
        Eleaf : float
            transpiration, mmol m-2 s-1

        Returns:
        -------
        flux_to_leaf : float
            flux from stem to the leaf, mmol s-1
        """
        return (psi_leaf - psi_leaf_prev) * \
                self.Cl / self.timestep_sec + self.AL * Eleaf

    def update_stem_wp(self, ksoil2stem, psi_soil_prev, flux_to_leaf,
                       psi_stem_prev):
        """
        Calculate stem water potential, MPa

        This is a simplified equation based on Xu et al., using the water
        potentials from the previous timestep and the fact that we increase
        the temporal resolution to get around the need to solve the dynamic eqn
        with a numerical approach, i.e., Runge-Kutta.

        Parameters:
        -----------
        ksoil2stem : float
            conductance from soil to stem, mmol m-2 s-1 MPa-1
        psi_soil_prev : float
            soil water potential from the previous timestep, MPa
        flux_to_leaf : float
            flux from stem to the leaf, mmol s-1
        psi_stem_prev : float
            stem water potential from the previous timestep, MPa

        Returns:
        -------
        psi_stem : float
            stem water potential at the end of the (sub-)step, MPa

        References:
        -----------
        * Xu et al. (2016) New Phytol, 212: 80–95. doi:10.1111/nph.14009; see
          appendix and code
        """

        # NB. includes the gravitational drop from the soil to the stem store,
        # halfway up the plant: psi_h / 2
        ap = -(self.AL * ksoil2stem / self.Cs)
        bp = (self.AL * ksoil2stem * (psi_soil_prev - self.psi_h_half) - \
              flux_to_leaf) / self.Cs
        psi_stem = ((ap * psi_stem_prev + bp) * \
                    np.exp(ap * self.timestep_sec)-bp) / ap

        return psi_stem

    def update_sw_bucket(self, precip, water_loss, sw_prev):
        """
        Update the simple bucket soil water balance

        Parameters:
        -----------
        precip : float
            precipitation (kg m-2 s-1)
        water_loss : float
            flux of water out of the soil, i.e. root water uptake (mmol s-1)
        sw_prev : float
            volumetric soil water from the previous timestep (m3 m-3)

        Returns:
        -------
        sw : float
            new volumetric soil water (m3 m-3)
        runoff : float
            water that does not fit in the bucket once it is saturated, lost
            as runoff (kg m-2, i.e. mm, per sub-step)
        """
        RHO_W = 1E03 # density of water, kg m-3
        MMOL_2_KG = 18.0 * 1E-06 # mmol H2O -> kg (18 g mol-1)

        # change in soil water over the sub-step (kg)
        delta_sw = (precip * self.ground_area * self.timestep_sec) - \
                    (water_loss * MMOL_2_KG * self.timestep_sec)
        sw = sw_prev + delta_sw / (self.soil_volume * RHO_W)

        runoff = 0.0
        if sw > self.theta_sat:
            runoff = (sw - self.theta_sat) * self.soil_depth * RHO_W
            sw = self.theta_sat

        return (sw, runoff)

    def calc_plc(self, kp):
        """
        Calculates the percent loss of conductivity, PLC (%)

        Parameters:
        -----------
        kp : float
            plant hydraulic conductance (mmol m-2 s-1 MPa-1)

        Returns:
        -------
        plc : float
            percent loss of conductivity (%)

        """
        return 100.0 * (1.0 - kp / self.kp_sat)

    def fsig_tuzet(self, psi_leaf):
        """
        An empirical logistic function to describe the sensitivity of stomata
        to leaf water potential.

        Sigmoid function assumes that stomata are insensitive to psi_leaf at
        values close to zero and that stomata rapidly close with decreasing
        psi_leaf.

        Parameters:
        -----------
        psi_leaf : float
            leaf water potential (MPa)

        Returns:
        -------
        fw : float
            sensitivity of stomata to leaf water potential [0-1]

        Reference:
        ----------
        * Tuzet et al. (2003) A coupled model of stomatal conductance,
          photosynthesis and transpiration. Plant, Cell and Environment 26,
          1097–1116

        """
        num = 1.0 + np.exp(self.sf * self.psi_f)
        den = 1.0 + np.exp(self.sf * (self.psi_f - psi_leaf))
        fw = num / den

        return fw

    def calc_ksoil(self, psi_soil):
        """
        Calculate soil-to-root hydraulic conductance (mmol m-2 s-1 MPa-1),
        per unit leaf area

        Parameters:
        -----------
        psi_soil : float
            soil water potential, MPa

        Returns:
        --------
        Ksoil : float
            soil-to-root hydraulic conductance, mmol m-2 s-1 MPa-1

        References:
        -----------
        * Duursma et al. (2008) Tree Physiology 28, 265–276, eqn 9, 8, 7
        """

        # A simple equation relating Ks to psi_s is given by (Campbell 1974),
        # mol m-1 s-1 MPa-1
        if isclose(psi_soil, 0.0):
            Ks = self.Ksat
        else:
            Ks = self.Ksat * (self.psi_e / psi_soil)**(2.0 + 3.0 / self.b)

        # the radius of a cylinder of soil to which the root has access, m
        rcyl = 1.0 / np.sqrt(np.pi * self.Lv)

        # root length index, m root m-2 ground
        Rl = self.Lv * self.soil_depth

        # conductance per unit root length (Gardner 1960) times root length
        # per unit leaf area (Rl / lai, m root m-2 leaf)
        # mol m-2 s-1 MPa-1 -> mmol m-2 s-1 MPa-1, to be consistent with kplant
        Ksoil = (Rl / self.lai) * 2. * np.pi * Ks / np.log(rcyl / self.rroot)
        Ksoil *= self.mol_2_mmol

        return Ksoil


def make_plot(out, timestep=15):

    ndays = out.t * timestep / (24. * 60.)

    fig = plt.figure(figsize=(9,6))
    fig.subplots_adjust(hspace=0.3)
    fig.subplots_adjust(wspace=0.2)
    plt.rcParams['text.usetex'] = False
    plt.rcParams['font.family'] = "sans-serif"
    plt.rcParams['font.sans-serif'] = "Helvetica"
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['font.size'] = 12
    plt.rcParams['legend.fontsize'] = 10
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12

    ax1 = fig.add_subplot(111)
    ax2 = ax1.twinx()

    ln1 = ax1.plot(ndays, out.psi_leaf, "k-", label="Leaf")
    ln2 = ax1.plot(ndays, out.psi_stem, "r-", label="Stem")
    ln3 = ax1.plot(ndays, out.psi_soil, "b-", label="Soil")
    ln4 = ax2.plot(ndays, out.plc, ls='-', color="darkgrey",
                   label="PLC")

    # added these three lines
    lns = ln1 + ln2 + ln3 + ln4
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc=(0.05,0.08), ncol=2)
    #ax1.legend(numpoints=1, loc="best")
    ax2.set_ylabel(r'PLC (%)')
    ax1.set_xlabel("Time (days)")
    ax1.set_ylabel("Water potential (MPa)")
    fig.savefig("time_to_mortality.pdf", bbox_inches='tight', pad_inches=0.1)

def plot_swp_sw(out):

    fig = plt.figure(figsize=(9,6))
    fig.subplots_adjust(hspace=0.3)
    fig.subplots_adjust(wspace=0.2)
    plt.rcParams['text.usetex'] = False
    plt.rcParams['font.family'] = "sans-serif"
    plt.rcParams['font.sans-serif'] = "Helvetica"
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['font.size'] = 12
    plt.rcParams['legend.fontsize'] = 10
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12

    ax1 = fig.add_subplot(111)

    ax1.plot(out.sw, out.psi_soil, "b.", label="Soil")

    ax1.set_xlabel("Volumetric soil water content (m$^{3}$ m$^{-3}$)")
    ax1.set_ylabel("Soil Water potential (MPa)")
    #ax1.legend(numpoints=1, loc="best")
    fig.savefig("sw_swp.pdf", bbox_inches='tight', pad_inches=0.1)


if __name__ == "__main__":

    time_step = 30

    met = generate_met_data(Tmin=10, RH=30, ndays=200, time_step=time_step)

    psi_stem0 = 0. # initial stem water potential, MPa
    AL = 6.        # plant leaf area, m2
    p50 = -4.      # xylem pressure inducing 50% loss of hydraulic conductivity
                   # due to embolism, MPa
    psi_f = -3.    # reference potential for Tuzet model, MPa
    gmin = 10.     # minimum stomatal conductance, mmol m-2 s-1
    Cl = 10000.    # leaf capacitance, mmol MPa-1 (total plant)
    Cs = 120000.   # stem capacitance, mmol MPa-1
    g1 = 4.0       # sensitivity of stomatal conductance to the assimilation
                   # rate (-)

    F = Canopy(g1=g1)
    D = Desica(psi_stem0=psi_stem0, AL=AL, p50=p50, psi_f=psi_f, gmin=gmin,
               Cl=Cl, Cs=Cs, F=F, g1=g1, met_timestep=time_step,
               stop_dead=True)
    out = D.run_simulation(met)

    make_plot(out, time_step)
    plot_swp_sw(out)
