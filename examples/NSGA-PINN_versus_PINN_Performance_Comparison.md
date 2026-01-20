# NSGA-PINN versus PINN Performance Comparison

Project: NGSA-NODE-PINN ([https://github.com/adills/NGSA-NODE-PINN](https://github.com/adills/NGSA-NODE-PINN)) 

CLI example:  
`python -m examples.verify_hybrid_pinn --device cpu --compare_third_party --plot_comparison --adam_steps 10 --epochs 100 --nsga_gens 100`

The model was trained on the Burger’s PDE.

Traditional PINN approach ([https://github.com/okada39/pinn\_burgers](https://github.com/okada39/pinn_burgers)) 

Key:  
Run 	\= performance test trial number  
E (En)	\= Training Epochs for major loop (n for NSGA)  
Ea	\= NSGA-PINN Adams Epochs for interior loop that feeds NSGA step  
G	\= NSGA generations (iterations)  
ΔD	\= Data loss (y \- y\_pred) (n for NSGA)  
ΔP	\= Physics loss (PDE \- PINN) (n for NSGA)

# PINN Performance Table:

| Run | T | E | ΔD | ΔP | Tn | En | Ea | G | ΔDn | ΔPn |
| ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- |
| 1 | 2 | 100 | 0.096 | 0.0007 | 24 | 20 | 10 | 50 | 0.036 | 0.0082 |
| 2 | 2 | 100 | 0.095 | 0.0008 | 48 | 20 | 10 | 100 | 0.027 | 0.0095 |
| 3 | 2 | 100 | 0.093 | 0.0007 | 95 | 40 | 10 | 100 | 0.021 | 0.0091 |
| 4 | 2 | 100 | 0.108 | 0.0012 | 99 | 40 | 20 | 100 | 0.031 | 0.0115 |
| 5 | 2 | 100 | 0.081 | 0.0004 | 233 | 100 | 10 | 100 | 0.011 | 0.0116 |

Trends in NSGA-PINN show decreasing data loss and increasing PINN loss which suggests one of two possibilities, either the model is learning the noise or the model is learning the physics coefficients better.  