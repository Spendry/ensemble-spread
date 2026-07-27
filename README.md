# Ensemble Spread Is Not Posterior Uncertainty

> Part of my [portfolio](https://github.com/Spendry/portfolio).

A small empirical result that bridges two published papers. One documents that the epistemic uncertainty reported by deep ensembles collapses as networks grow wider, calls it a paradox, and says the cause has yet to be found. The other proves deep ensembles perform empirical Bayes under a prior the architecture learns implicitly. This paper supplies the measurement that joins them: on a regression task with a deliberate hole in the training data, **the correct Bayesian posterior under each width's own implicit prior holds flat (slope +0.004 per doubling) while the ensemble spread falls 7.7x over a 32x range of width.** The ignorance implied by the model does not change with width; the number practitioners read as ignorance does. The collapse belongs to the estimator, not the thing measured.

**Read the paper: [paper.pdf](paper.pdf)** (v1.0, with failure ledger, kill conditions, and full limitations)

## The claim, and the honesty around it

The paper claims no theory. The mechanism belongs to the empirical Bayes paper it cites, the phenomenon belongs to the uncertainty-calibration papers it cites, and the contribution is the measurement connecting them. It carries four recorded failures (each an error made during the work, with its repair), three open tests with pre-registered kill conditions, and a "What Would Change Our Mind" section naming the single result that would kill the central claim outright. Everything is synthetic: one construction, one-dimensional input, one architecture family, widths 8 to 1024. The paper would not defend the result past that scope, and says so.

## Reproducing it

```
# runs on CPU; gp_test.py needs no training and finishes in under a minute
python code/gp_test.py     # implied-posterior width invariance, the result
python code/cbr_c2c3.py    # the support-hole construction
python code/mech.py        # source decomposition + NTK drift
```

- **[code/](code/)**: the scripts. Output paths were changed from the original run's absolute paths to the working directory, so they write their JSON alongside themselves; nothing else was touched.
- **[results/](results/)**: the raw logs and JSON from the run behind the paper's tables, with a [reproduction-order note](results/README.md).

Original run: CPU, single container, PyTorch, July 26 2026. Widths, seed counts, and probe regions are specified in the paper's section A.1.

## Credit and license

Framing and mechanism: Sammuel Pendry. Formalization, code, and the four failure-ledger entries: Claude (Anthropic). The empirical Bayes reading, the collapse phenomenon, and the neural-tangent-kernel results are borrowed and cited in the paper, not claimed. CC BY 4.0.
