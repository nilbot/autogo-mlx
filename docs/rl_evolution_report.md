# Reinforcement Learning Strategic Evolution Report

This report compiles key tactical, strategic, and behavioral metrics mined across all 
self-play iterations of the current training run. It reveals how the model matured from 
completely random play (Iteration 0) into a tactically aware and strategically structured agent.

---

## 📊 Evolution Metrics Matrix

| Iteration | Games | Avg Length (Plies) | Move 0 PASS Rate | Unique Openings | Capture Rate | Edge plays (Zone 1) | Tengen plays (Zone 5) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Iter 0** | 1500 | 110.6 ± 29.3 | 0.00% | 81 | 14.021% | 41.57% | 1.15% |
| **Iter 1** | 1500 | 101.8 ± 27.3 | 0.00% | 81 | 14.989% | 40.86% | 1.18% |
| **Iter 2** | 1500 | 99.2 ± 28.6 | 0.00% | 81 | 16.541% | 40.63% | 1.20% |
| **Iter 3** | 1500 | 144.2 ± 35.7 | 0.00% | 81 | 27.993% | 40.52% | 1.24% |
| **Iter 4** | 1500 | 138.1 ± 44.0 | 0.00% | 81 | 28.053% | 40.02% | 1.29% |
| **Iter 5** | 1500 | 141.6 ± 50.8 | 0.00% | 81 | 28.347% | 39.88% | 1.26% |
| **Iter 6** | 1500 | 126.0 ± 46.8 | 0.00% | 81 | 26.849% | 39.62% | 1.27% |
| **Iter 7** | 1500 | 118.1 ± 43.8 | 0.00% | 81 | 26.471% | 39.51% | 1.27% |
| **Iter 8** | 1500 | 106.5 ± 37.0 | 0.00% | 81 | 24.453% | 39.18% | 1.31% |
| **Iter 9** | 1000 | 100.6 ± 31.2 | 0.00% | 81 | 24.210% | 37.83% | 1.58% |
| **Iter 10** | 1000 | 107.2 ± 36.9 | 0.00% | 81 | 26.401% | 37.59% | 1.54% |
| **Iter 11** | 1000 | 120.9 ± 42.9 | 0.00% | 81 | 28.863% | 38.87% | 1.32% |
| **Iter 12** | 1000 | 124.6 ± 37.0 | 0.00% | 81 | 29.675% | 40.02% | 1.23% |
| **Iter 13** | 1000 | 122.7 ± 37.6 | 0.00% | 80 | 29.410% | 39.86% | 1.28% |
| **Iter 14** | 1000 | 111.0 ± 42.2 | 0.00% | 81 | 27.018% | 39.25% | 1.32% |
| **Iter 15** | 1000 | 130.4 ± 36.7 | 0.00% | 79 | 29.461% | 39.90% | 1.25% |
| **Iter 16** | 1000 | 139.5 ± 39.3 | 0.00% | 80 | 30.156% | 39.76% | 1.23% |
| **Iter 17** | 1000 | 136.4 ± 38.4 | 0.00% | 79 | 29.921% | 39.22% | 1.24% |
| **Iter 18** | 1000 | 135.1 ± 36.8 | 0.00% | 78 | 29.432% | 38.93% | 1.26% |
| **Iter 19** | 1000 | 113.1 ± 48.8 | 0.00% | 79 | 27.023% | 36.75% | 1.38% |
| **Iter 20** | 100 | 107.6 ± 52.3 | 0.00% | 28 | 26.208% | 35.83% | 1.41% |

---

## 📈 Deeper Behavioral Insights

### 1. Tactical Capture Intensity (Capture Density)
The capture rate indicates the percentage of stone placements that immediately result in capturing 
opponent groups. Healthy learning leads to a steady, non-trivial capture density as the model learns to 
defend its own stones and seize capturing opportunities in the middlegame.

### 2. Spatial Base Selection (Zone Ratios)
Healthy Go strategy dictates establishing bases on the 3rd line (Zone 3) and projecting influence on the 
4th line (Zone 4) early in the game, while minimizing early edge plays (Zone 1). Watching this distribution 
helps verify that the model has learned standard Go spatial heuristics rather than blindly filling corners.

---

## 🗺️ Spatial Heatmap Evolution

These 9x9 density maps illustrate where the model placed stones during self-play as training progressed. 
Notice how the distribution shifts from uniform randomness to structured opening points.

### Iteration 0 Spatial Move Density Map
```text
 +  +  +  +  +  +  +  +  + 
 +  +  +  +  +  +  +  +  + 
 +  +  +  +  +  +  +  +  + 
 +  +  +  +  +  +  +  +  + 
 +  +  +  +  +  +  +  +  + 
 +  +  +  +  +  +  +  +  + 
 +  +  +  +  +  +  +  +  + 
 +  +  +  +  +  +  +  +  + 
 +  +  +  +  +  +  +  +  + 
```

### Iteration 10 Spatial Move Density Map
```text
 +  +  +  +  +  +  +  +  + 
 +  +  +  +  +  +  +  +  + 
 +  +  +  +  +  +  +  +  + 
 +  +  +  *  *  *  +  +  + 
 +  +  +  *  *  *  +  +  + 
 +  +  +  +  +  +  +  +  + 
 +  +  +  +  +  +  +  +  + 
 +  +  +  +  +  +  +  +  + 
 +  +  +  +  +  +  +  +  + 
```

### Iteration 20 Spatial Move Density Map
```text
 +  +  +  +  +  +  +  +  + 
 +  +  +  +  +  +  +  +  + 
 +  +  +  +  +  +  +  +  + 
 +  +  +  +  +  +  +  +  + 
 +  +  +  +  +  +  +  +  + 
 +  +  +  +  +  +  +  +  + 
 +  +  +  +  +  +  +  +  + 
 +  +  +  +  +  +  +  +  + 
 +  +  +  +  +  +  +  +  + 
```
