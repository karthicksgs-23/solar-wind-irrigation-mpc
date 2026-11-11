# nb03 v3 post-hoc analysis

Maize price scenarios (USD/t, World Bank Pink Sheet US maize 2020-2024): {'low': 165.5, 'base': 237.4, 'high': 318.8}. Penalty weights (USD/kWh): [0.0, 5.0, 20.0, 50.0].
Corrected Ymax raised the reference above the 'full' schedule in 18 of 20 site-years; mean raise 0.24 t/ha.

## Paired differences, base price, penalty 0 and 50 USD/kWh

```
     metric        site          contrast     mean     lo95     hi95  pct_of_b   sig
 tc_v0_base    azraq_jo      hybrid - mpc    -26.2    -40.5    -10.5      -0.8  True
 tc_v0_base    azraq_jo         drl - mpc    302.9     99.6    512.4       9.6  True
 tc_v0_base    azraq_jo        mpc - rule  -1173.6  -1321.1   -971.9     -27.1  True
 tc_v0_base    azraq_jo   mpc - two_stage  -1107.4  -1277.8   -933.8     -26.0  True
 tc_v0_base    azraq_jo   mpc - mpc_uncal  -1076.0  -1190.6   -962.7     -25.4  True
 tc_v0_base    azraq_jo mpc - mpc_default     30.4    -30.1     66.5       1.0 False
 tc_v0_base    azraq_jo mpc_perfect - mpc     -1.6    -16.4     12.6      -0.1 False
 tc_v0_base    azraq_jo  two_stage - rule    -66.2   -164.2     27.7      -1.5 False
 tc_v0_base    azraq_jo      hybrid - drl   -329.2   -527.8   -125.9      -9.5  True
 tc_v0_base bushland_tx      hybrid - mpc    -33.0    -58.2     -7.7      -1.4  True
 tc_v0_base bushland_tx         drl - mpc    695.2    366.4    984.0      29.7  True
 tc_v0_base bushland_tx        mpc - rule   -540.9   -920.9   -179.6     -18.8  True
 tc_v0_base bushland_tx   mpc - two_stage   -606.7   -950.6   -272.8     -20.6  True
 tc_v0_base bushland_tx   mpc - mpc_uncal   -967.0  -1144.5   -783.4     -29.2  True
 tc_v0_base bushland_tx mpc - mpc_default    -52.9    -90.4     -8.4      -2.2  True
 tc_v0_base bushland_tx mpc_perfect - mpc     43.4    -33.4    132.7       1.9 False
 tc_v0_base bushland_tx  two_stage - rule     65.9    -12.3    144.6       2.3 False
 tc_v0_base bushland_tx      hybrid - drl   -728.2  -1010.3   -421.4     -24.0  True
 tc_v0_base  jodhpur_in      hybrid - mpc      8.4     -3.4     20.7       0.6 False
 tc_v0_base  jodhpur_in         drl - mpc    924.2   -472.5   2338.3      67.3 False
 tc_v0_base  jodhpur_in        mpc - rule    -87.1   -150.1    -32.2      -6.0  True
 tc_v0_base  jodhpur_in   mpc - two_stage    -30.8    -69.1      7.4      -2.2 False
 tc_v0_base  jodhpur_in   mpc - mpc_uncal   -214.1   -923.7    286.1     -13.5 False
 tc_v0_base  jodhpur_in mpc - mpc_default    -55.5    -87.3    -23.7      -3.9  True
 tc_v0_base  jodhpur_in mpc_perfect - mpc     19.1     -2.6     40.9       1.4 False
 tc_v0_base  jodhpur_in  two_stage - rule    -56.3   -101.4    -12.6      -3.9  True
 tc_v0_base  jodhpur_in      hybrid - drl   -915.8  -2318.1    494.8     -39.9 False
 tc_v0_base    souss_ma      hybrid - mpc     -3.1     -4.6     -1.8      -0.1  True
 tc_v0_base    souss_ma         drl - mpc    254.2    223.8    287.0      10.7  True
 tc_v0_base    souss_ma        mpc - rule   -290.1   -405.5   -192.8     -10.9  True
 tc_v0_base    souss_ma   mpc - two_stage   -252.6   -361.3   -156.0      -9.6  True
 tc_v0_base    souss_ma   mpc - mpc_uncal   -137.9   -229.3    -66.9      -5.5  True
 tc_v0_base    souss_ma mpc - mpc_default    -17.3    -49.1     13.5      -0.7 False
 tc_v0_base    souss_ma mpc_perfect - mpc     34.1      8.3     67.8       1.4  True
 tc_v0_base    souss_ma  two_stage - rule    -37.5    -76.3      2.4      -1.4 False
 tc_v0_base    souss_ma      hybrid - drl   -257.3   -289.3   -228.9      -9.8  True
tc_v50_base    azraq_jo      hybrid - mpc    -25.5    -39.5    -10.3      -0.8  True
tc_v50_base    azraq_jo         drl - mpc    332.5    126.9    529.1      10.1  True
tc_v50_base    azraq_jo        mpc - rule  -9556.3 -14078.1  -6693.6     -74.4  True
tc_v50_base    azraq_jo   mpc - two_stage  -1219.4  -1472.9   -985.1     -27.0  True
tc_v50_base    azraq_jo   mpc - mpc_uncal  -1076.0  -1190.6   -962.7     -24.6  True
tc_v50_base    azraq_jo mpc - mpc_default  -2404.3  -3072.4  -1744.5     -42.2  True
tc_v50_base    azraq_jo mpc_perfect - mpc    -48.5   -130.0      0.0      -1.5 False
tc_v50_base    azraq_jo  two_stage - rule  -8336.9 -12803.3  -5417.8     -64.9  True
tc_v50_base    azraq_jo      hybrid - drl   -358.0   -552.1   -125.8      -9.9  True
tc_v50_base bushland_tx      hybrid - mpc    -22.2    -53.3      8.0      -0.9 False
tc_v50_base bushland_tx         drl - mpc   1915.0   1436.9   2451.3      81.7  True
tc_v50_base bushland_tx        mpc - rule -14047.3 -17401.0 -10149.6     -85.7  True
tc_v50_base bushland_tx   mpc - two_stage   -606.7   -950.6   -287.3     -20.6  True
tc_v50_base bushland_tx   mpc - mpc_uncal   -967.0  -1144.5   -783.4     -29.2  True
tc_v50_base bushland_tx mpc - mpc_default  -6310.5  -7069.2  -5517.7     -72.9  True
tc_v50_base bushland_tx mpc_perfect - mpc     43.4    -33.4    132.7       1.9 False
tc_v50_base bushland_tx  two_stage - rule -13440.6 -17146.5  -9808.5     -82.0  True
tc_v50_base bushland_tx      hybrid - drl  -1937.2  -2491.5  -1476.4     -45.5  True
tc_v50_base  jodhpur_in      hybrid - mpc    194.2     88.1    286.4       8.1  True
tc_v50_base  jodhpur_in         drl - mpc    882.4   -503.4   2336.3      36.8 False
tc_v50_base  jodhpur_in        mpc - rule -28078.4 -34880.5 -21024.2     -92.1  True
tc_v50_base  jodhpur_in   mpc - two_stage    -30.8    -69.0      7.4      -1.3 False
tc_v50_base  jodhpur_in   mpc - mpc_uncal   -214.1   -923.7    286.1      -8.2 False
tc_v50_base  jodhpur_in mpc - mpc_default -11279.1 -14598.3  -7474.4     -82.5  True
tc_v50_base  jodhpur_in mpc_perfect - mpc    -61.6   -170.9     25.3      -2.6 False
tc_v50_base  jodhpur_in  two_stage - rule -28047.6 -35099.7 -20995.4     -92.0  True
tc_v50_base  jodhpur_in      hybrid - drl   -688.2  -2113.7    653.7     -21.0 False
tc_v50_base    souss_ma      hybrid - mpc    -10.5    -18.0     -3.4      -0.1  True
tc_v50_base    souss_ma         drl - mpc   1431.4   1137.5   1764.6      19.1  True
tc_v50_base    souss_ma        mpc - rule -17479.6 -19069.3 -16351.7     -70.0  True
tc_v50_base    souss_ma   mpc - two_stage   -340.8   -414.5   -275.3      -4.4  True
tc_v50_base    souss_ma   mpc - mpc_uncal   -179.8   -283.8    -79.3      -2.3  True
tc_v50_base    souss_ma mpc - mpc_default  -8209.7 -11563.5  -4856.0     -52.3  True
tc_v50_base    souss_ma mpc_perfect - mpc   -240.1   -555.7     -8.1      -3.2  True
tc_v50_base    souss_ma  two_stage - rule -17138.9 -18703.7 -16022.3     -68.6  True
tc_v50_base    souss_ma      hybrid - drl  -1441.9  -1769.3  -1125.6     -16.2  True
```

## Best family per site by scenario

```
                                                                                  best
scenario                                                                              
tc_v0_base   azraq_jo:mpc_default, bushland_tx:hybrid, jodhpur_in:mpc, souss_ma:hybrid
tc_v0_high   azraq_jo:mpc_default, bushland_tx:hybrid, jodhpur_in:mpc, souss_ma:hybrid
tc_v0_low         azraq_jo:hybrid, bushland_tx:hybrid, jodhpur_in:mpc, souss_ma:hybrid
tc_v20_base       azraq_jo:hybrid, bushland_tx:hybrid, jodhpur_in:mpc, souss_ma:hybrid
tc_v20_high       azraq_jo:hybrid, bushland_tx:hybrid, jodhpur_in:mpc, souss_ma:hybrid
tc_v20_low        azraq_jo:hybrid, bushland_tx:hybrid, jodhpur_in:mpc, souss_ma:hybrid
tc_v50_base       azraq_jo:hybrid, bushland_tx:hybrid, jodhpur_in:mpc, souss_ma:hybrid
tc_v50_high       azraq_jo:hybrid, bushland_tx:hybrid, jodhpur_in:mpc, souss_ma:hybrid
tc_v50_low        azraq_jo:hybrid, bushland_tx:hybrid, jodhpur_in:mpc, souss_ma:hybrid
tc_v5_base        azraq_jo:hybrid, bushland_tx:hybrid, jodhpur_in:mpc, souss_ma:hybrid
tc_v5_high        azraq_jo:hybrid, bushland_tx:hybrid, jodhpur_in:mpc, souss_ma:hybrid
tc_v5_low         azraq_jo:hybrid, bushland_tx:hybrid, jodhpur_in:mpc, souss_ma:hybrid
```

## MPC rank / hybrid rank (1 = best, 7 families)

```
site         azraq_jo  bushland_tx  jodhpur_in  souss_ma
scenario                                                
tc_v0_base          3            2           1         2
tc_v0_high          3            2           1         2
tc_v0_low           3            2           1         2
tc_v20_base         2            2           1         2
tc_v20_high         2            2           1         2
tc_v20_low          2            2           1         2
tc_v50_base         2            2           1         2
tc_v50_high         2            2           1         2
tc_v50_low          2            2           1         2
tc_v5_base          2            2           1         2
tc_v5_high          2            2           1         2
tc_v5_low           2            2           1         2
```

```
site         azraq_jo  bushland_tx  jodhpur_in  souss_ma
scenario                                                
tc_v0_base          2            1           2         1
tc_v0_high          2            1           2         1
tc_v0_low           1            1           2         1
tc_v20_base         1            1           3         1
tc_v20_high         1            1           3         1
tc_v20_low          1            1           4         1
tc_v50_base         1            1           3         1
tc_v50_high         1            1           3         1
tc_v50_low          1            1           4         1
tc_v5_base          1            1           2         1
tc_v5_high          1            1           2         1
tc_v5_low           1            1           2         1
```
