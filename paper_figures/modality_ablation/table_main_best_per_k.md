# Main Best-Per-k Table

| endpoint   | model   |    F1_k1 |    F1_k2 |    F1_k3 |    F1_k4 |    F1_k5 |   best_k | best_subset                                     |   best_f1 |
|:-----------|:--------|---------:|---------:|---------:|---------:|---------:|---------:|:------------------------------------------------|----------:|
| rec        | xgb     | 0.983012 | 0.985371 | 0.985371 | 0.985371 | 0.985371 |        5 | clinical+semantic+temporal+pathological+spatial |  0.985371 |
| rec        | lgbm    | 0.982977 | 0.983091 | 0.983126 | 0.983091 | 0.983126 |        5 | clinical+semantic+temporal+pathological+spatial |  0.983126 |
| rec        | rf      | 0.982977 | 0.982978 | 0.982978 | 0.981047 | 0.980615 |        3 | clinical+pathological+spatial                   |  0.982978 |
| rec        | voting  | 0.982977 | 0.982977 | 0.982977 | 0.982977 | 0.982977 |        5 | clinical+semantic+temporal+pathological+spatial |  0.982977 |
| rec        | et      | 0.978684 | 0.981047 | 0.980615 | 0.980615 | 0.981047 |        5 | clinical+semantic+temporal+pathological+spatial |  0.981047 |
| surv       | lgbm    | 0.785912 | 0.785157 | 0.802355 | 0.804845 | 0.784576 |        4 | clinical+temporal+pathological+spatial          |  0.804845 |
| surv       | voting  | 0.785196 | 0.800511 | 0.802013 | 0.798158 | 0.791222 |        3 | clinical+semantic+spatial                       |  0.802013 |
| surv       | et      | 0.788014 | 0.799936 | 0.792143 | 0.787384 | 0.776951 |        2 | clinical+pathological                           |  0.799936 |
| surv       | rf      | 0.777729 | 0.797399 | 0.791104 | 0.798051 | 0.78792  |        4 | clinical+temporal+pathological+spatial          |  0.798051 |
| surv       | xgb     | 0.770063 | 0.797567 | 0.787588 | 0.788734 | 0.779438 |        2 | clinical+spatial                                |  0.797567 |
