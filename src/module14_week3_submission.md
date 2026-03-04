# Module 14 (Week 3) - Query Submission Draft

These queries were generated with an SVM-based surrogate (`SVR` ensemble) from the currently available local dataset.

## Portal-ready query strings

- `function_1`: `0.001680-0.995053`
- `function_2`: `0.999611-0.005957`
- `function_3`: `0.994992-0.007886-0.008697`
- `function_4`: `0.449055-0.459191-0.447611-0.175417`
- `function_5`: `0.422325-0.739748-0.388969-0.993772`
- `function_6`: `0.299818-0.260746-0.770707-0.762085-0.012732`
- `function_7`: `0.008162-0.673426-0.062393-0.029021-0.003312-0.760771`
- `function_8`: `0.028063-0.148217-0.126081-0.239461-0.595120-0.531112-0.149963-0.740981`

## Re-generate command

```bash
python src/generate_week3_queries_svm.py --data-dir initial_data --output src/module14_week3_queries.txt
```

If you receive updated outputs from the portal, update your local data first and rerun the command to produce true Week 3 decisions based on the latest observations.
