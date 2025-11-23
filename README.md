# DESI Y3 Quasar Luminosity Function (QLF) — Bayesian Inference with NumPyro

This repository contains the full analysis pipeline used to model the 
DESI Year 3 (Y3) quasar luminosity function in the redshift range 
**2.3 ≤ z ≤ 2.8** using an **inhomogeneous Poisson point-process** 
framework and **Bayesian inference** with NumPyro.

The project includes:
- Construction of DESI synthetic photometry
- K-correction estimation using rest-frame UV continuum (1600–1850 Å)
- Redshift binning and data vector creation
- A normalized double–power–law QLF model
- DESI selection function modeling
- Full posterior inference using NUTS (NumPyro)
- Posterior corner plots and parameter evolution
- Predicted vs. observed magnitude distributions

---
