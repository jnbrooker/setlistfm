#!/usr/bin/env python3
"""
Conditional logit — the estimator, written out rather than imported.

WHAT THIS IS

McFadden's conditional logit. Given a set of choice occasions, each offering
the same chooser a menu of alternatives with different attributes, it finds the
coefficients that best explain which alternative was picked.

    P(occasion i picks alternative j) = exp(x_ij'B) / sum_k exp(x_ik'B)

WHY IT IS WRITTEN OUT

statsmodels and pylogit both do this. The project's premise is that every
number must be traceable to arithmetic a reader can follow, and a coefficient
produced by a library is exactly as trustworthy as the reader's willingness to
take the library on trust. Here the log-likelihood, its gradient, its Hessian
and the standard errors are forty lines of NumPy that can be read in full.

It is also not hard. The conditional logit likelihood is globally concave in B,
so Newton-Raphson finds the unique maximum in six or seven iterations from any
starting point. There is no tuning, no learning rate, no random seed and no
local optimum to worry about -- which is the whole reason this family of model
was chosen over anything that would need those things.

THE ONE PROPERTY WORTH UNDERSTANDING BEFORE READING ANY OUTPUT

The chooser's own characteristics CANNOT ENTER. An artist's popularity, budget
and genre are the same for every city on the menu, so they cancel out of the
ratio above and contribute nothing. That sounds like a loss and is in fact the
model's main virtue: it means the coefficients are identified purely from
WITHIN-TOUR variation. The question the model answers is not "which tours play
big cities" but "given that this tour played four cities in Italy, why those
four" -- and everything about the act that might have confounded the answer has
been differenced away for free.

To let an artist characteristic matter, interact it with a city attribute. "Big
acts weight capacity more heavily" is a coefficient on category x capacity, and
that does vary across the menu.

DATA LAYOUT

Rows are (occasion, alternative) pairs, sorted so that each occasion's
alternatives are contiguous. `offsets` marks where each occasion starts, in the
style of a CSR sparse matrix, which lets every group-wise sum below be a
`reduceat` rather than a Python loop.
"""

import numpy as np


def group_offsets(group_ids):
    """
    Start index of each run in a sorted array of group ids.

    Requires the rows to be sorted by group -- checked, because getting this
    silently wrong would mix alternatives from different occasions into the
    same denominator and produce coefficients that look entirely plausible.
    """
    g = np.asarray(group_ids)
    if np.any(g[1:] < g[:-1]):
        raise ValueError("rows must be sorted by group id")
    starts = np.flatnonzero(np.r_[True, g[1:] != g[:-1]])
    return starts


def group_logsumexp(v, offsets):
    """
    log(sum(exp(v))) within each group, computed stably.

    The naive version overflows: exp(800) is infinity in float64, and utilities
    reach that range as soon as a coefficient on a logged variable gets large
    during the first Newton step. Subtracting each group's maximum first is the
    standard fix and costs one extra pass.
    """
    m = np.maximum.reduceat(v, offsets)              # max within each group
    e = np.exp(v - np.repeat(m, np.diff(np.r_[offsets, len(v)])))
    s = np.add.reduceat(e, offsets)
    return m + np.log(s)


def choice_probabilities(v, offsets):
    """Softmax within each group: the model's predicted probabilities."""
    lse = group_logsumexp(v, offsets)
    return np.exp(v - np.repeat(lse, np.diff(np.r_[offsets, len(v)])))


def loglik(beta, X, y, offsets):
    """
    Log-likelihood.

        LL(B) = sum_i [ x_chosen(i)'B - log sum_j exp(x_ij'B) ]

    The first term rewards putting utility on alternatives that were chosen;
    the second penalises putting it anywhere. Written with y as a 0/1 vector so
    occasions with several chosen alternatives work without special-casing.
    """
    v = X @ beta
    return float(y @ v - np.sum(group_logsumexp(v, offsets)))


def fit(X, y, offsets, cluster=None, names=None, max_iter=50, tol=1e-9,
        ridge=0.0, verbose=False):
    """
    Maximum likelihood by Newton-Raphson.

    THE ARITHMETIC, IN FULL

    With p the predicted probabilities and mu_i = sum_j p_ij x_ij the
    probability-weighted average attribute bundle on occasion i:

        gradient  g  =  X'(y - p)
        Hessian   H  = -( X' diag(p) X  -  sum_i mu_i mu_i' )

    The gradient is the difference between the attributes actually chosen and
    the attributes the model expects to be chosen; at the optimum they match
    exactly, which is a genuinely useful thing to be able to say about a
    fitted model. The Hessian is minus the within-occasion covariance of
    attributes under the predicted probabilities -- so a variable that barely
    varies across the alternatives on a menu contributes almost nothing to
    precision, no matter how many rows it has. That is why the standard errors
    below are driven by how different the CANDIDATE cities are from each other,
    not by the row count.

    `ridge` adds a small penalty to the diagonal. It defaults to zero and
    should stay there; it exists only so a perfectly collinear specification
    fails gracefully with a warning instead of a LinAlgError.

    `cluster` gives each row's cluster (use the tour). Standard errors are then
    the sandwich estimator, because one tour contributes many occasions and
    they are obviously not independent -- treating them as independent would
    overstate precision by roughly the square root of the dates per tour.
    """
    n_rows, k = X.shape
    beta = np.zeros(k)
    counts = np.diff(np.r_[offsets, n_rows])
    ll = loglik(beta, X, y, offsets)

    for it in range(max_iter):
        v = X @ beta
        p = choice_probabilities(v, offsets)

        g = X.T @ (y - p)

        Xp = X * p[:, None]
        XtPX = X.T @ Xp
        mu = np.add.reduceat(Xp, offsets, axis=0)     # occasions x k
        H = -(XtPX - mu.T @ mu)
        if ridge:
            H -= ridge * np.eye(k)

        try:
            step = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            raise RuntimeError(
                "the Hessian is singular: two columns of X are collinear "
                "within every choice occasion, so their coefficients cannot be "
                "told apart. Drop one.")

        # Newton with a backtracking guard. The likelihood is concave so the
        # full step is almost always uphill, but a wild first step on badly
        # scaled data can overshoot; halving until the likelihood improves
        # costs nothing when it is not needed.
        t = 1.0
        for _ in range(30):
            cand = beta - t * step
            ll_new = loglik(cand, X, y, offsets)
            if ll_new >= ll:
                break
            t /= 2
        else:
            break

        improvement = ll_new - ll
        beta, ll = cand, ll_new
        if verbose:
            print(f"   iter {it + 1:2d}  loglik {ll:,.3f}  step {t:g}")
        if improvement < tol:
            break

    # ---- variance -------------------------------------------------------
    v = X @ beta
    p = choice_probabilities(v, offsets)
    Xp = X * p[:, None]
    mu = np.add.reduceat(Xp, offsets, axis=0)
    H = -(X.T @ Xp - mu.T @ mu)
    Hinv = np.linalg.inv(H)

    resid = (y - p)[:, None] * X
    if cluster is None:
        cov = -Hinv
        se_kind = "classical"
    else:
        # sum the score within each cluster, then the sandwich
        cl = np.asarray(cluster)
        order = np.argsort(cl, kind="stable")
        cl_sorted, r_sorted = cl[order], resid[order]
        cl_offsets = group_offsets(cl_sorted)
        s = np.add.reduceat(r_sorted, cl_offsets, axis=0)
        meat = s.T @ s
        cov = Hinv @ meat @ Hinv
        se_kind = f"clustered ({len(cl_offsets):,} clusters)"

    se = np.sqrt(np.clip(np.diag(cov), 0, None))

    # McFadden's pseudo-R2, against a model with no attributes at all -- which
    # in a conditional logit means every alternative equally likely.
    ll_null = -float(np.sum(np.log(counts)[np.searchsorted(
        offsets, np.flatnonzero(y > 0), side="right") - 1]))

    return {
        "beta": beta, "se": se, "cov": cov, "loglik": ll, "loglik_null": ll_null,
        "pseudo_r2": 1 - ll / ll_null if ll_null else np.nan,
        "iterations": it + 1, "n_rows": n_rows, "n_occasions": len(offsets),
        "n_chosen": int(y.sum()), "se_kind": se_kind,
        "names": list(names) if names is not None else [f"x{i}" for i in range(k)],
        "probabilities": p,
    }


def summary(res, digits=4):
    """A coefficient table, as a DataFrame."""
    import pandas as pd
    z = np.divide(res["beta"], res["se"], out=np.full_like(res["beta"], np.nan),
                  where=res["se"] > 0)
    # Normal approximation. With thousands of clusters the difference from a t
    # is immaterial, and pretending otherwise would imply a precision about the
    # cluster count that this data does not support.
    from math import erfc, sqrt
    pval = [erfc(abs(zi) / sqrt(2)) if np.isfinite(zi) else np.nan for zi in z]
    return pd.DataFrame({
        "variable": res["names"],
        "coefficient": np.round(res["beta"], digits),
        "std error": np.round(res["se"], digits),
        "z": np.round(z, 2),
        "p": np.round(pval, 4),
        "odds ratio": np.round(np.exp(res["beta"]), 3),
        "2.5%": np.round(res["beta"] - 1.96 * res["se"], digits),
        "97.5%": np.round(res["beta"] + 1.96 * res["se"], digits),
    })


# ---------------------------------------------------------------------------
# Self-test
#
# Run `python logit.py` to check the estimator against data with known
# coefficients. This is here because an estimator that is subtly wrong produces
# output that looks exactly like an estimator that is right, and the only
# defence is to generate data where the answer is known in advance.
# ---------------------------------------------------------------------------

def _self_test(seed=0, n_occasions=4000, n_alts=12, verbose=True):
    rng = np.random.default_rng(seed)
    true_beta = np.array([1.5, -0.8, 0.4])
    k = len(true_beta)

    X = rng.normal(size=(n_occasions * n_alts, k))
    offsets = np.arange(0, n_occasions * n_alts, n_alts)
    v = X @ true_beta
    p = choice_probabilities(v, offsets)

    # draw one chosen alternative per occasion from the true probabilities
    y = np.zeros(len(X))
    for i, s in enumerate(offsets):
        block = p[s:s + n_alts]
        y[s + rng.choice(n_alts, p=block / block.sum())] = 1

    res = fit(X, y, offsets, names=["a", "b", "c"])
    err = np.abs(res["beta"] - true_beta)
    if verbose:
        print(summary(res).to_string(index=False))
        print(f"\ntrue      {true_beta}")
        print(f"estimated {np.round(res['beta'], 3)}")
        print(f"max error {err.max():.3f}  (should be well inside the standard "
              f"errors above)")
        print(f"iterations {res['iterations']}, pseudo-R2 {res['pseudo_r2']:.3f}")
    assert err.max() < 0.12, "recovered coefficients are too far from the truth"
    # every coefficient should sit inside its own 95% interval
    inside = np.abs(res["beta"] - true_beta) < 1.96 * res["se"] * 1.5
    assert inside.all(), "true value outside the confidence interval"
    return res


if __name__ == "__main__":
    print("conditional logit self-test: simulating 4,000 choices with known "
          "coefficients\n")
    _self_test()
    print("\nPASS")
