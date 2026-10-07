# Model evaluation

Selected **extra_trees_w15_capNone** using lowest engine-balanced validation RMSE, before reading NASA test labels.
The fixed split holds out 25 engines. Five deterministic truncation points per held-out engine
(20%, 40%, 60%, 80%, 95% of observed life) cover early and late operation equally by engine.
This proxy endpoint distribution differs from NASA test truncation; scores are not directly comparable.
Full validation trajectories are used only for retrospective warning analysis.

## Validation candidates (raw, uncapped RUL evaluation)

```text
              candidate  window   cap       mae      rmse    nasa_score   n
extra_trees_w15_capNone      15   NaN 21.157631 29.230049   8228.166110 125
    lightgbm_w5_capNone       5   NaN 21.855922 29.901245   5943.304699 125
 extra_trees_w5_capNone       5   NaN 22.039115 29.931077   6051.805180 125
 extra_trees_w15_cap125      15 125.0 20.415075 30.449235   8278.174434 125
    lightgbm_w15_cap125      15 125.0 20.476118 30.496931   7972.178446 125
     lightgbm_w5_cap125       5 125.0 20.516028 30.523472   8650.056144 125
  extra_trees_w5_cap125       5 125.0 20.220311 30.625318   9203.320235 125
      ridge_w15_capNone      15   NaN 25.547732 31.356550   9058.607173 125
       ridge_w5_capNone       5   NaN 26.009858 31.617976   8107.900893 125
   lightgbm_w15_capNone      15   NaN 22.640697 32.004082  40824.737199 125
       ridge_w15_cap125      15 125.0 23.975951 32.749167  17928.096644 125
        ridge_w5_cap125       5 125.0 24.165411 32.856928  20866.066044 125
         mean_w5_cap125       5 125.0 49.725635 58.325941 151507.413046 125
        mean_w15_cap125      15 125.0 49.725635 58.325941 151507.413046 125
        mean_w5_capNone       5   NaN 54.163120 63.569779 517543.361602 125
       mean_w15_capNone      15   NaN 54.163120 63.569779 517543.361602 125
```

## Final test: one endpoint per engine, uncapped ground truth

```json
{
  "mae": 19.50497414021164,
  "rmse": 27.008524920739927,
  "nasa_score": 9986.277321479574,
  "n": 100
}
```

NASA score sums exp(-error/13)-1 for negative errors and exp(error/10)-1 otherwise,
where error = predicted − actual. Lower is better; overprediction costs more. Its scale depends on sample count.
All predictions are clipped at zero. Training caps are compared against uncapped training targets.

## Operational evaluation

Near failure means actual RUL <=20; an actionable warning means predicted RUL <=40.
False alerts mean predicted <=40 and actual >40. These are project assumptions, not industry standards.
```json
{
  "alert_threshold": 40,
  "near_failure_count": 16,
  "near_failure_recall": 0.9375,
  "missed_critical": 1,
  "false_alerts": 0,
  "alerts": 21,
  "false_alert_fraction": 0.0
}
```

## Errors by true RUL band
```text
rul_band       mae      rmse  nasa_score  n
    0-20  6.313046 13.481460  175.703161 16
   21-40 16.580254 23.722739  395.251739 12
   41-80 18.593684 26.845176 1694.470198 17
     >80 24.262418 30.461166 7720.852224 55
```

## Retrospective validation warning analysis
```json
{
  "engines": 25,
  "engines_alerted": 25,
  "mean_first_alert_lead_cycles": 35.28,
  "first_alerts_over_40_cycles": 6
}
```
Lead time is actual cycles remaining at the first <=40 prediction on a held-out complete trajectory.
It is conditional on any alert, can count an early false alarm, and does not measure downtime avoided.
It comes from the selection validation set and is therefore exploratory. NASA test trajectories are
censored and cannot establish actual intervention lead time. Threshold sensitivity is saved separately;
thresholds were not tuned on test results. Feature permutation importance uses validation endpoints,
with correlated features potentially sharing or masking importance; it is not a causal explanation.

## Specific failures and interpretation

Largest absolute endpoint errors:
```text
 engine_id  cycle  actual_rul  predicted_rul     error
         1     31         112     194.041025 82.041025
        15     76          83     160.331485 77.331485
        67     71          77     150.009213 73.009213
        78     72         107     175.406769 68.406769
        37    121          21      76.630417 55.630417
```
Missed near-failure assets under the action rule:
```text
 engine_id  actual_rul  predicted_rul maintenance_status
        41          18      69.372619            Monitor
```
The >80-cycle group has MAE 24.26,
compared with 6.31 at <=20 cycles.
Early-life ambiguity is a material limitation. Large overestimates dominate the exponential NASA
score even when near-failure recall is high. The model was selected on RMSE, not the NASA score;
these objectives need not choose the same candidate. No model choice was revised after this analysis.

Top validation permutation drivers (RMSE increase in cycles):
```text
       feature  rmse_increase      std
         cycle       9.076165 1.529495
sensor_15_mean       0.684142 0.170833
 sensor_8_mean       0.445180 0.262303
setting_2_mean       0.438485 0.194537
 sensor_12_std       0.391956 0.373445
```
A strong age contribution is consistent with useful lifetime-distribution information, but
could be brittle if deployed engine lifetimes shift. Sensor importance is distributed across
correlated raw and smoothed channels. Negative or small values should not be read as precise rankings.
