# AURORA publication-oriented experiment report

## Scope

This is a synthetic, algorithmic evaluation. It does not establish flight readiness, operational performance, or publication acceptance. Claims must be restricted to the scenarios and assumptions recorded in experiment_config.json.

## Pre-registered comparison design

- Monte-Carlo runs per scenario: 24
- Duration per run: 3.0 days
- Decision interval: 600.0 s
- Matched environments: same scenario/run seed for every planner.
- Common random numbers: success/noise stream is indexed by run, time, and target, not policy execution order.
- Scheduler input: imperfect cloud forecast only; latent current cloud remains hidden until execution.
- Primary metric: final field RMSE. Secondary metrics: MAE, information gain, uncertainty calibration, observation efficiency, and resource rejections.

## Scenarios

- **nominal** — Balanced power, storage and cloud regime.
- **battery_stress** — Reduced battery and higher imaging energy; power feasibility must matter.
- **storage_stress** — Small recorder, larger products and restricted downlink throughput.
- **cloud_stress** — Frequent, persistent cloud with imperfect forecast information.

## Per-scenario summary

- battery_stress / dynamic_information_utility: RMSE 0.021445 [0.020452, 0.022476], information gain 16.3622, successful observations 114.75, energy/storage rejections 15232.3/27696.7.
- battery_stress / myopic_information: RMSE 0.025747 [0.024523, 0.026883], information gain 16.5454, successful observations 115.54, energy/storage rejections 15033.9/28095.9.
- battery_stress / resource_aware_myopic: RMSE 0.021796 [0.020670, 0.022964], information gain 16.3429, successful observations 114.46, energy/storage rejections 14713.8/28403.4.
- battery_stress / static_priority: RMSE 0.033506 [0.032190, 0.034869], information gain 0.6974, successful observations 113.38, energy/storage rejections 15382.3/27446.5.
- battery_stress / uniform_random: RMSE 0.024072 [0.022792, 0.025322], information gain 13.1548, successful observations 113.83, energy/storage rejections 15215.8/27526.7.
- cloud_stress / dynamic_information_utility: RMSE 0.020483 [0.019785, 0.021248], information gain 20.0809, successful observations 141.21, energy/storage rejections 7228.9/31396.7.
- cloud_stress / myopic_information: RMSE 0.025937 [0.025108, 0.026740], information gain 20.7484, successful observations 144.54, energy/storage rejections 7306.2/31365.4.
- cloud_stress / resource_aware_myopic: RMSE 0.022154 [0.021229, 0.023171], information gain 20.2284, successful observations 142.17, energy/storage rejections 7211.1/31261.7.
- cloud_stress / static_priority: RMSE 0.035957 [0.034793, 0.037080], information gain 0.6581, successful observations 136.29, energy/storage rejections 6958.4/31513.2.
- cloud_stress / uniform_random: RMSE 0.025132 [0.024324, 0.025943], information gain 15.1176, successful observations 138.00, energy/storage rejections 7043.6/31384.6.
- nominal / dynamic_information_utility: RMSE 0.018723 [0.017692, 0.019787], information gain 20.6637, successful observations 144.54, energy/storage rejections 7318.4/31278.6.
- nominal / myopic_information: RMSE 0.025844 [0.024876, 0.026747], information gain 20.8440, successful observations 144.92, energy/storage rejections 7300.5/31322.6.
- nominal / resource_aware_myopic: RMSE 0.020279 [0.019140, 0.021409], information gain 20.6177, successful observations 143.96, energy/storage rejections 7228.3/31433.4.
- nominal / static_priority: RMSE 0.034993 [0.033646, 0.036318], information gain 0.6558, successful observations 140.96, energy/storage rejections 7163.0/31446.2.
- nominal / uniform_random: RMSE 0.023298 [0.022190, 0.024431], information gain 15.5697, successful observations 141.92, energy/storage rejections 7180.5/31293.9.
- storage_stress / dynamic_information_utility: RMSE 0.024231 [0.023065, 0.025492], information gain 11.0440, successful observations 75.83, energy/storage rejections 9045.4/39307.9.
- storage_stress / myopic_information: RMSE 0.028189 [0.027257, 0.029161], information gain 11.1858, successful observations 76.54, energy/storage rejections 9082.0/39331.0.
- storage_stress / resource_aware_myopic: RMSE 0.023887 [0.022764, 0.025075], information gain 11.1166, successful observations 76.29, energy/storage rejections 9056.9/39234.0.
- storage_stress / static_priority: RMSE 0.034150 [0.032890, 0.035443], information gain 0.4664, successful observations 76.00, energy/storage rejections 9104.5/39058.6.
- storage_stress / uniform_random: RMSE 0.026124 [0.025038, 0.027215], information gain 9.4269, successful observations 75.92, energy/storage rejections 9045.1/39227.8.

## Paired DIU comparisons

Positive differences favor DIU. A CI excluding zero is evidence within this synthetic experiment, not universal superiority.

- battery_stress / rmse vs myopic_information: 0.004302 [0.003346, 0.005175], win rate 95.83%, n=24.
- battery_stress / mae vs myopic_information: 0.004016 [0.003215, 0.004745], win rate 100.00%, n=24.
- battery_stress / information_gain vs myopic_information: -0.183179 [-0.256718, -0.107624], win rate 12.50%, n=24.
- battery_stress / rmse vs resource_aware_myopic: 0.000351 [-0.000530, 0.001214], win rate 50.00%, n=24.
- battery_stress / mae vs resource_aware_myopic: 0.000162 [-0.000458, 0.000778], win rate 50.00%, n=24.
- battery_stress / information_gain vs resource_aware_myopic: 0.019283 [-0.111795, 0.168528], win rate 45.83%, n=24.
- battery_stress / rmse vs static_priority: 0.012061 [0.011216, 0.012878], win rate 100.00%, n=24.
- battery_stress / mae vs static_priority: 0.013098 [0.012155, 0.013961], win rate 100.00%, n=24.
- battery_stress / information_gain vs static_priority: 15.664807 [15.574068, 15.750458], win rate 100.00%, n=24.
- battery_stress / rmse vs uniform_random: 0.002627 [0.001763, 0.003465], win rate 87.50%, n=24.
- battery_stress / mae vs uniform_random: 0.002419 [0.001606, 0.003189], win rate 87.50%, n=24.
- battery_stress / information_gain vs uniform_random: 3.207358 [2.970179, 3.433765], win rate 100.00%, n=24.
- cloud_stress / rmse vs myopic_information: 0.005454 [0.004725, 0.006207], win rate 100.00%, n=24.
- cloud_stress / mae vs myopic_information: 0.005237 [0.004607, 0.005889], win rate 100.00%, n=24.
- cloud_stress / information_gain vs myopic_information: -0.667530 [-0.940923, -0.415298], win rate 8.33%, n=24.
- cloud_stress / rmse vs resource_aware_myopic: 0.001671 [0.000712, 0.002662], win rate 75.00%, n=24.
- cloud_stress / mae vs resource_aware_myopic: 0.001438 [0.000714, 0.002159], win rate 79.17%, n=24.
- cloud_stress / information_gain vs resource_aware_myopic: -0.147511 [-0.480453, 0.163686], win rate 45.83%, n=24.
- cloud_stress / rmse vs static_priority: 0.015473 [0.014606, 0.016351], win rate 100.00%, n=24.
- cloud_stress / mae vs static_priority: 0.016499 [0.015455, 0.017512], win rate 100.00%, n=24.
- cloud_stress / information_gain vs static_priority: 19.422812 [19.158775, 19.674677], win rate 100.00%, n=24.
- cloud_stress / rmse vs uniform_random: 0.004649 [0.003992, 0.005297], win rate 100.00%, n=24.
- cloud_stress / mae vs uniform_random: 0.004026 [0.003468, 0.004629], win rate 100.00%, n=24.
- cloud_stress / information_gain vs uniform_random: 4.963272 [4.577213, 5.341178], win rate 100.00%, n=24.
- nominal / rmse vs myopic_information: 0.007121 [0.006298, 0.007910], win rate 100.00%, n=24.
- nominal / mae vs myopic_information: 0.006467 [0.005701, 0.007162], win rate 100.00%, n=24.
- nominal / information_gain vs myopic_information: -0.180287 [-0.303294, -0.076998], win rate 25.00%, n=24.
- nominal / rmse vs resource_aware_myopic: 0.001555 [0.000496, 0.002564], win rate 75.00%, n=24.
- nominal / mae vs resource_aware_myopic: 0.001183 [0.000387, 0.001967], win rate 70.83%, n=24.
- nominal / information_gain vs resource_aware_myopic: 0.045981 [-0.132951, 0.228552], win rate 50.00%, n=24.
- nominal / rmse vs static_priority: 0.016270 [0.015397, 0.017110], win rate 100.00%, n=24.
- nominal / mae vs static_priority: 0.017295 [0.016228, 0.018295], win rate 100.00%, n=24.
- nominal / information_gain vs static_priority: 20.007913 [19.888757, 20.106578], win rate 100.00%, n=24.
- nominal / rmse vs uniform_random: 0.004575 [0.003771, 0.005402], win rate 100.00%, n=24.
- nominal / mae vs uniform_random: 0.003815 [0.002956, 0.004746], win rate 100.00%, n=24.
- nominal / information_gain vs uniform_random: 5.093958 [4.820704, 5.364985], win rate 100.00%, n=24.
- storage_stress / rmse vs myopic_information: 0.003958 [0.003250, 0.004742], win rate 100.00%, n=24.
- storage_stress / mae vs myopic_information: 0.003676 [0.002943, 0.004443], win rate 100.00%, n=24.
- storage_stress / information_gain vs myopic_information: -0.141757 [-0.254394, -0.042305], win rate 16.67%, n=24.
- storage_stress / rmse vs resource_aware_myopic: -0.000345 [-0.001142, 0.000489], win rate 41.67%, n=24.
- storage_stress / mae vs resource_aware_myopic: -0.000235 [-0.001010, 0.000598], win rate 50.00%, n=24.
- storage_stress / information_gain vs resource_aware_myopic: -0.072576 [-0.150119, 0.001076], win rate 29.17%, n=24.
- storage_stress / rmse vs static_priority: 0.009919 [0.009288, 0.010527], win rate 100.00%, n=24.
- storage_stress / mae vs static_priority: 0.011071 [0.010372, 0.011821], win rate 100.00%, n=24.
- storage_stress / information_gain vs static_priority: 10.577585 [10.486970, 10.653268], win rate 100.00%, n=24.
- storage_stress / rmse vs uniform_random: 0.001893 [0.001002, 0.002798], win rate 79.17%, n=24.
- storage_stress / mae vs uniform_random: 0.001548 [0.000776, 0.002332], win rate 75.00%, n=24.
- storage_stress / information_gain vs uniform_random: 1.617137 [1.463279, 1.756419], win rate 100.00%, n=24.

## Integrity checks

- PASS: All configured integrity checks passed.

## Required external-validation path

1. Replace synthetic truth and cloud forecasts with archived products and a time-respecting train/tune/test split.
2. Validate the geometry, instrument, slew, power, thermal and downlink models against mission-specific sources.
3. Compare against current optimization/MPC/RL baselines and report runtime, complexity, and parameter sensitivity.
4. Perform a literature review before asserting novelty; resource-aware EO scheduling and value-based replanning are established areas.