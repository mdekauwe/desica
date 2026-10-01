
# Units: water potentials MPa; conductances mmol m-2 leaf s-1 MPa-1; capacitances mmol MPa-1
# (total plant); fluxes Jsl and Jrs mmol s-1 (total plant); Eleaf mmol m-2 leaf s-1.
desica <- function(met=NULL,
                   met_timestep = 15, # met timestep (mins)
                   nsub = NULL,     # sub-steps per met timestep (default: so that each is <= dtmax s)
                   runtwice = NULL, # deprecated: TRUE = nsub of 2, FALSE = nsub of 1
                   dtmax = 600,     # longest sub-step (s), Xu et al. (2016)

                   Ca = 400,      # atmospheric CO2 (umol mol-1)
                   sf=8,          # Tuzet sensitivity parameter (MPa-1)
                   g1=10,         # gs = g1 * fw * A / Ca (-)

                   Cs = 100000,   # stem capacitance (mmol MPa-1, total plant)
                   Cl = 10000,    # leaf capacitance (mmol MPa-1, total plant)

                   kpsat=3,       # saturated plant conductance (mmol m-2 leaf s-1 MPa-1)
                   p50 = -4,      # xylem pressure at 50% loss of conductivity (MPa)
                   psiv=-2,       # Tuzet reference potential (MPa)
                   s50 = 30,      # slope of the vulnerability curve at P50 (% MPa-1)
                   gmin = 10,     # minimum stomatal conductance (mmol m-2 leaf s-1)

                   psil0=-1,      # initial leaf water potential (MPa)
                   psist0=-0.5,   # initial stem water potential (MPa)

                   thetasat=0.5,  # soil water content at saturation (m3 m-3)
                   sw0=0.5,       # initial soil water content (m3 m-3)
                   AL=2.5,        # plant leaf area (m2)
                   soildepth=1,   # depth of the soil bucket (m)
                   groundarea=1,  # ground area (m2)
                   b=6,           # soil water retention exponent (-)
                   psie= -0.8*1E-03, # air entry water potential (MPa)
                   Ksat=20,       # saturated soil conductivity (mol m-1 s-1 MPa-1)
                   Lv=10000,      # root length density (m m-3)

                   height=0,         # plant height (m), for the gravity term (split equally
                                     # either side of the stem store, halfway up)
                   psilmin=-20,      # lower bound on psil (MPa)
                   psistmin=2*p50,   # lower bound on psist (MPa), as CABLE-DESICA

                   keepwet=FALSE,
                   stopsimdead=TRUE,
                   plcdead=88,
                   mf=NA,
                   LMA=NA,
                   refill=TRUE){

  if(is.null(met)){
    stop("Must provide met dataframe with VPD, Tair, PPFD, precip (kg m-2 s-1, optional)")
  }
  if(is.null(met$precip))met$precip <- 0
  n <- nrow(met)

  if(is.null(nsub)){
    if(!is.null(runtwice)){
      nsub <- if(runtwice) 2 else 1
    } else {
      nsub <- max(1, ceiling(60*met_timestep / dtmax))
    }
  }
  timestep_sec <- 60*met_timestep / nsub

  LAI <- AL / groundarea
  soilvolume <- groundarea * soildepth

  # Initial conditions
  na <- rep(NA, nrow(met))
  out <- data.frame(Eleaf=na,psil=na,psist=na,psis=na,sw=na,
                    ks=na,kp=na,Jsl=na,Jrs=na,krst=na,kstl=na,runoff=na)

  out$psil[1] <- psil0
  out$psist[1] <- psist0
  out$sw[1] <- sw0
  out$psis[1] <- psie*(sw0/thetasat)^-b
  out$Eleaf[1] <- 0
  out$runoff[1] <- 0

  # soil-to-root conductance
  out$ks[1] <- ksoil_fun(out$psis[1], Ksat, psie, b,
                         LAI, soildepth=soildepth, Lv=Lv)

  pars <- list(timestep_sec=timestep_sec,nsub=nsub,
               psiv=psiv,sf=sf,g1=g1,Ca=Ca,Cs=Cs,
               Cl=Cl,kpsat=kpsat,p50=p50,s50=s50,gmin=gmin,
               thetasat =thetasat,AL=AL,soildepth=soildepth,
               soilvolume=soilvolume,groundarea=groundarea,
               LAI=LAI,b=b,psie=psie,Ksat=Ksat,
               Lv=Lv,keepwet=keepwet,stopsimdead=stopsimdead,
               plcdead=plcdead, refill=refill,
               psih=1000*9.81*height*1E-06, # rho g h (MPa)
               psilmin=psilmin, psistmin=psistmin)

  for(i in 2:n){

    out <- desica_calc_timestep(met, i, out, pars)

    if(stopsimdead){
      plc <- 100*(1 - out$kp[i]/pars$kpsat)
      if(plc > plcdead)break
    }

  }

  d <- cbind(met, out)
  d$plc <- 100 * (1 - d$kp / kpsat)
  d$Eplant <- AL * out$Eleaf
  d$t <- 1:nrow(d)

  # when stopsimdead, last many rows are NA
  d <- d[!is.na(d$psis),]

  attr(d, "met_timestep") <- met_timestep
  attr(d, "soilvolume") <- soilvolume
  attr(d, "groundarea") <- groundarea

  return(d)
}



# Using previous timestep psist, psil, psis, sw and ks, calculate fluxes/pools for next timestep
# can only calculate timestep i when i-1 has been calculated!
#
# Stomatal conductance (and so Eleaf) is calculated once from psil at the start of the
# timestep and held fixed (as in ED2). psil, psist and the soil bucket are then integrated
# over nsub sub-steps, carrying all states forward. Each pool is solved analytically with
# the other held fixed, following Xu et al. (2016), which works for short sub-steps (<= 10 min).
# The lower bounds on psist and psil conserve water: when the stem reaches psistmin, Jsl is
# cut to what the stem can supply, and if the leaf is then at psilmin, transpiration is cut
# to match (so Eleaf can fall below the stomatal demand only once both pools are at their
# bounds).
# States are stored at the end of the timestep, Eleaf, Jsl and Jrs as means over the
# timestep, runoff as the total (kg m-2, i.e. mm), conductances from the last sub-step.
desica_calc_timestep <- function(met, i, out, pars){

  dt <- pars$timestep_sec
  psil <- out$psil[i-1]
  psist <- out$psist[i-1]
  sw <- out$sw[i-1]
  psis <- out$psis[i-1]
  ks <- out$ks[i-1]
  kp_prev <- out$kp[i-1]

  # packageVersion("plantecophys") >= "1.2-6"
  # Estimate stomatal conductance, etc. with known leaf water potential.
  # (Unlike in true Tuzet model, we don't solve for psil but instead use it as input,
  # via the BBmult option in Photosyn)
  p <- Photosyn(VPD=met$VPD[i],
                gsmodel="BBdefine",
                g0=0.0,
                BBmult=(pars$g1/pars$Ca)*fsig_tuzet(psil, pars$psiv, pars$sf),
                Tleaf=met$Tair[i],
                PPFD=met$PPFD[i],
                Ca=pars$Ca)

  # Don't add gmin, instead use it as bottom value.
  gs <- pmax(pars$gmin, 1000 * p$GS)

  # Leaf transpiration (mmol m-2 s-1), assuming perfect coupling and an air pressure of
  # 101 kPa
  Eleaf_demand <- (met$VPD[i]/101)*gs

  Eleaf_sum <- 0
  Jsl_sum <- 0
  Jrs_sum <- 0
  runoff_sum <- 0
  for(j in 1:pars$nsub){

    Eleaf <- Eleaf_demand

    # Plant hydraulic conductance
    # Note how it depends on previous (sub-)step stem water potential.
    kp <- pars$kpsat * fsig_hydr(psist, pars$s50, pars$p50)
    if(!pars$refill && !is.na(kp_prev)){
      kp <- min(kp, kp_prev)
    }
    kp_prev <- kp

    # from soil to stem pool
    krst <- 1 / (1/ks + 1/(2*kp))

    # from stem pool to leaf
    kstl <- 2*kp

    # Xu method.
    # Can write the dynamic equation as: dPsil_dt = b + a*psil
    # Then it follows (Xu et al. 2016, Appendix, and Code). Includes the gravitational
    # drop (Xu et al. Eqn S2a) from the stem store, halfway up the plant, to the leaf: psih/2.
    bp <- (pars$AL * kstl * (psist - pars$psih/2) - pars$AL * Eleaf)/pars$Cl
    ap <- -(pars$AL * kstl / pars$Cl)
    psil_new <- ((ap * psil + bp) * exp(ap * dt) - bp)/ap
    psil_new <- max(psil_new, pars$psilmin)

    # Flux from stem to leaf (mmol s-1) = change in leaf storage, plus transpiration
    Jsl <- (psil_new - psil) * pars$Cl / dt + pars$AL * Eleaf

    # Update stem water potential
    # Also from Xu et al. 2016. Root uptake only while the soil is wetter than the stem,
    # allowing for the gravitational drop to the store halfway up the plant (psih/2), no
    # leak to the soil (Xu et al. Eqn S1a).
    Jrs <- 0
    if(psis - pars$psih/2 > psist){
      bp <- (pars$AL * krst * (psis - pars$psih/2) - Jsl) / pars$Cs
      ap <- -(pars$AL * krst / pars$Cs)
      psist_new <- ((ap*psist + bp) * exp(ap*dt) - bp)/ap

      # flux from soil to stem (mmol s-1) = change in stem storage, plus Jsl
      Jrs <- (psist_new - psist) * pars$Cs / dt + Jsl
    }
    if(Jrs <= 0){
      Jrs <- 0
      psist_new <- psist - Jsl * dt / pars$Cs
    }

    # At the stem bound, the stem can only supply the root uptake plus what is left in the
    # store above the bound. Update the leaf from that supply and, if the leaf is at its
    # bound too, cut transpiration so that no water is created.
    if(psist_new < pars$psistmin){
      psist_new <- pars$psistmin
      Jsl <- Jrs + (psist - psist_new) * pars$Cs / dt
      psil_new <- psil + (Jsl - pars$AL * Eleaf) * dt / pars$Cl
      if(psil_new < pars$psilmin){
        psil_new <- pars$psilmin
        Eleaf <- (Jsl - (psil_new - psil) * pars$Cl / dt) / pars$AL
      }
    }

    # keepwet holds the soil water fixed, for debugging / comparison.
    if(!pars$keepwet){
      # Soil water increase: precip (kg m-2 s-1) - root water uptake (mmol s-1,
      # x 18e-6 kg mmol-1), units kg per sub-step
      water_in <- pars$groundarea * met$precip[i] * dt - dt * 1E-06*18 * Jrs

      # soil water content (sw) in units m3 m-3 (1E03 = density of water, kg m-3); water
      # that does not fit once the bucket is saturated is lost as runoff (kg m-2, i.e. mm)
      sw <- sw + water_in / (pars$soilvolume * 1E03)
      if(sw > pars$thetasat){
        runoff_sum <- runoff_sum + (sw - pars$thetasat) * pars$soildepth * 1E03
        sw <- pars$thetasat
      }

      # Update soil water potential and soil-to-root hydraulic conductance
      psis <- pars$psie * (sw/pars$thetasat)^-pars$b
      ks <- ksoil_fun(psis, Ksat=pars$Ksat, psie=pars$psie,
                      b=pars$b, LAI=pars$LAI, soildepth=pars$soildepth,
                      Lv=pars$Lv)
    }

    psil <- psil_new
    psist <- psist_new
    Eleaf_sum <- Eleaf_sum + Eleaf
    Jsl_sum <- Jsl_sum + Jsl
    Jrs_sum <- Jrs_sum + Jrs
  }

  out$Eleaf[i] <- Eleaf_sum / pars$nsub
  out$psil[i] <- psil
  out$psist[i] <- psist
  out$sw[i] <- sw
  out$psis[i] <- psis
  out$ks[i] <- ks
  out$kp[i] <- kp
  out$krst[i] <- krst
  out$kstl[i] <- kstl
  out$Jsl[i] <- Jsl_sum / pars$nsub
  out$Jrs[i] <- Jrs_sum / pars$nsub
  out$runoff[i] <- runoff_sum

  return(out)
}



diurnal_sim <- function(PPFDmax=2000, RH=30,
                        Tmax=30, Tmin=15,
                        daylength=12,
                        sunrise=8,
                        timestep=15,
                        lag=0.5){

  # PPFD
  crv <- function(relt, ymax)(ymax/2)*(1+sin((2*pi*relt)+1.5*pi))

  r <- seq(0, 24*60 - timestep, by=timestep)
  p <- rep(0, length(r))
  ii <- which(r >= (sunrise*60) & r < (sunrise*60 + daylength*60))
  p[ii] <- crv( (r[ii] - sunrise*60)/(daylength*60), PPFDmax)


  # during daylight
  partondiurnal <- function(timeh, Tmax,Tmin,daylen, lag){
    (Tmax-Tmin)*sin((pi*timeh)/(daylen+2*lag)) + Tmin
  }

  ta <- rep(NA, length(r))
  td <- partondiurnal(seq(0, daylength - timestep/60, by=timestep/60), Tmax, Tmin, daylength,lag)
  ta[ii] <- td

  nnight <- length(r)-length(td)
  tdec <- ta[max(ii)] + (Tmin - ta[max(ii)])* 1:nnight / nnight
  after <- (max(ii)+1):length(r)
  ta[after] <- tdec[1:length(after)]
  before <- 1:(min(ii)-1)
  ta[before] <- tdec[(length(after)+1):length(tdec)]

  vpd <- RHtoVPD(RH, ta)

  return(data.frame(PPFD=p, Tair=ta, VPD=vpd))
}


make_simdfr <- function(..., ndays=40){
  dm <- diurnal_sim(...)
  simdfr <- vector("list", length=ndays)
  for(i in 1:length(simdfr))simdfr[[i]] <- dm
  simdfr <- do.call(rbind, simdfr)
  simdfr$day <- rep(1:ndays, each=nrow(dm))
  simdfr$precip <- 0

return(simdfr)
}




# Soil-to-root hydraulic conductance (mmol m-2 leaf s-1 MPa-1), Ksat in mol m-1 s-1 MPa-1
# (Duursma et al. 2008, Tree Physiology 28, 265-276)
ksoil_fun <- function(psis,
                      Ksat,
                      psie,
                      b,
                      LAI,
                      Lv=50000,
                      rroot=1E-06,
                      soildepth=1
                      ){

  # Campbell (1974), mol m-1 s-1 MPa-1
  Ks <- Ksat*(psie/psis)^(2 + 3/b)
  Ks[psis == 0] <- Ksat

  # radius of the cylinder of soil to which each root has access (m)
  rcyl <- 1/sqrt(pi*Lv)
  # root length index (m root m-2 ground)
  Rl <- Lv * soildepth

  # conductance per unit root length (Gardner 1960) times root length per leaf area
  # (Rl / LAI); mol m-2 s-1 MPa-1 -> mmol m-2 s-1 MPa-1, to be consistent with kp
  1000 * (Rl/LAI)*2*pi*Ks/log(rcyl/rroot)
}




# Tuzet et al. (2003) sensitivity of stomata to leaf water potential (0-1)
fsig_tuzet <- function(psil, psiv, sf){
  (1 + exp(sf*psiv)) / (1 + exp(sf*(psiv - psil)))
}

# Relative plant conductance (K/Kmax) from the Weibull vulnerability curve (Duursma & Choat
# 2017): PX is the xylem pressure (MPa) at which X% of the conductivity is lost and SX the
# slope there (% MPa-1).
fsig_hydr <- function(P, SX, PX, X=50){

  P <- abs(P)
  PX <- abs(PX)
  X <- X[1] # when fitting; vector may be passed but X cannot actually vary.
  V <- (X-100)*log(1-X/100)
  p <- (P/PX)^((PX*SX)/V)
  relk <- (1-X/100)^p

  return(relk)
}



plot_desica <- function(run){

  spd <- steps_per_day(run)

  par(mar=c(4,4,1,4), tcl=0.2, mgp=c(2.2,0.4,0), cex.lab=1.2)
  with(run, plot(t/spd, psil, type='l',
                  xlab="Time (days)",
                  ylab="Water potential (MPa)"))
  with(run, lines(t/spd, psist, col="red"))
  with(run, lines(t/spd, psis, lwd=2, col="cornflowerblue"))
  legend("bottomleft",c("Leaf","Stem","Soil","PLC"), lty=1, lwd=c(1,1,2,1),
         col=c("black","red","cornflowerblue","darkgrey"), inset=0.02)

  par(new=TRUE)
  with(run, plot(t/spd, plc, type='l', col="darkgrey",
                  ann=FALSE,
                  axes=FALSE))
  axis(4)
  mtext(side=4, line=2.2, cex=1.2, text="PLC (%)")


}

plot_desica_2 <- function(d, p50=-3){

  par(mfrow=c(2,2), mar=c(4,4,1,1))

  with(d, {
    plot(t, psis, type='l', ylim=c(-8,0))
    lines(t, psil, col="forestgreen")
    lines(t, psist, col="blue2")
    abline(h=p50)
  })

  plot(d$sw, type='l')

  with(d, {
    plot(t, Jsl, type='l')
    lines(t, Jrs, col="blue2")
  })

  plot(d$plc, type='l')
}



summarize_desica <- function(d){

  if(!is.data.frame(d)){
    res <- lapply(d, summarize_desica)
    return(as.data.frame(do.call(rbind, res)))
  }

  phase2 <- subset(d, ks < 0.05*max(kp, na.rm=TRUE))
  simtot <- nrow(d) / steps_per_day(d)
  sim2 <- nrow(phase2) / steps_per_day(d)
  sim1 <- simtot - sim2

  return(c(phase1=sim1, phase2=sim2, plcfinal=d$plc[nrow(d)]))

}



# Soil water balance check (kg, i.e. litres): the decrease in soil water should equal
# root water uptake plus runoff minus precip.
checkwatbal <- function(run, i=nrow(run)){

  dt <- 60 * met_timestep(run)
  soilvolume <- attr(run, "soilvolume")
  groundarea <- attr(run, "groundarea")
  if(is.null(soilvolume))soilvolume <- 1
  if(is.null(groundarea))groundarea <- 1

  water_lost <- (run$sw[1] - run$sw[i]) * soilvolume * 1000

  # root water uptake (Jrs is the mean over each timestep, mmol s-1; x 1E-03 mol mmol-1
  # x 18 g mol-1 x 1E-03 kg g-1)
  water_used <- sum(run$Jrs[2:i] * dt * 1E-03 * 18 * 1E-03, na.rm=TRUE)
  water_in <- sum(run$precip[2:i] * groundarea * dt, na.rm=TRUE)
  runoff <- sum(run$runoff[2:i] * groundarea, na.rm=TRUE)

  return(c(soilwater_decrease=water_lost, rootwateruptake=water_used, precip=water_in,
           runoff=runoff))

}

met_timestep <- function(run){
  ts <- attr(run, "met_timestep")
  if(is.null(ts)) 15 else ts
}

steps_per_day <- function(run){
  24 * 60 / met_timestep(run)
}
