# Feature selection benchmark

---

Install requirements:

```bash
pip install -r requirements.txt
```

To run a feature selection method on the synthetic datasets, you simply need to run the following script with the FS method name as argument. For example, use `rf` to use random forests:

```bash
python main-benchmark.py rf
```

Other methods are: `attr`, `attr-t`, `attr-b`, `nn`, `rf`, `relief`, `fsnet`, `mrmr`, `mi`, `lassonet`, `cae`, `treeshap`, `canceloutsigmoid`, `canceloutsoftmax` or `deeppink`.

Results will be stored in `results/`.

To run all methods on the real datasets, just run the following command:
```bash
python real-data-benchmark.py
```
Results will be stored in `results/external-data/`.
