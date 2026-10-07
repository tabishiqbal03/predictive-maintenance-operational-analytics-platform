# Verified business findings

Training data contains 100 simulated engines and 20631 observations.
Lifetimes range from 128 to 362 cycles
with median 199.0. Age alone cannot capture all lifetime variation.

The largest absolute sensor/RUL Pearson correlations in training are
{'sensor_11': -0.696228101455419, 'sensor_4': -0.6789482333860456, 'sensor_12': 0.6719831036132903, 'sensor_7': 0.6572226620548836}. These pooled associations are not causal and
do not account for repeated observations. See sensor_profile.csv, phase_means.csv,
degradation.png and correlations.png for distributions, phase means and redundancy.

Retained channels: setting_1, setting_2, sensor_2, sensor_3, sensor_4, sensor_7, sensor_8, sensor_9, sensor_11, sensor_12, sensor_13, sensor_14, sensor_15, sensor_17, sensor_20, sensor_21.
The training-only relative-range filter removes channels with <=2 values or relative range <=1e-5;
the complete dropped list is in artifacts/metadata.json. Retained operating settings may still be noise;
their utility should be assessed through a future grouped ablation.

Final test actionable alerts: 21; missed near-failure engines:
1; false alerts: 0.
Operational precision is limited by the assumed 40-cycle planning horizon. These findings support
ranking a review queue, not automatic shutdowns or measured financial savings.

## Observed phase differences and fleet composition

Sensor 11 means: healthy 47.401,
late life 48.024.
Sensor 12 means: healthy 521.790,
late life 520.129.
These pooled means support degradation association, not independence or a causal mechanism.

```text
maintenance_status
Healthy                 57
Monitor                 22
Critical                13
Schedule Maintenance     8
```
Cycle is the leading permutation driver in this run. The large errors listed in model_evaluation.md
show why apparently healthy classifications should not be treated as guarantees. Setting 1/2
vary slightly despite FD001's single operating regime; small importance differences could reflect
sample noise. A future setting-feature ablation belongs on fresh validation splits.
