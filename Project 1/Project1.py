import pandas as pd 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from numpy import linalg as la
from tabulate import tabulate
from scipy import stats
import re
from pathlib import Path
import math



def estimate( 
        y: np.ndarray, x: np.ndarray, transform='', T:int=None, robust_se=False
    ) -> list:
    """Uses the provided estimator to perform a regression of y on x, 
    and provides all other necessary statistics such as standard errors, 
    t-values etc.  

    Args:
        >> y (np.ndarray): Dependent variable (Needs to have shape 2D shape)
        >> x (np.ndarray): Independent variable (Needs to have shape 2D shape)
        >> transform (str, optional): Defaults to ''. If the data is 
        transformed in any way, the following transformations are allowed:
            '': No transformations
            'fd': First-difference
            'be': Between transformation
            'fe': Within transformation
            're': Random effects estimation.
        >>T (int, optional): If panel data, T is the number of time periods in
        the panel, and is used for estimating the variance. Defaults to None.
        >> robust_se (bool): Defaults to False. Returns robust standard errors if True.

    Returns:
        list: Returns a dictionary with the following variables:
        'b_hat', 'se', 'sigma2', 't_values', 'R2', 'cov'
    """

    assert y.ndim == 2, 'Input y must be 2-dimensional'
    assert x.ndim == 2, 'Input x must be 2-dimensional'
    assert y.shape[1] == 1, 'y must be a column vector'
    assert y.shape[0] == x.shape[0], 'y and x must have same first dimension'
    
    b_hat = est_ols(y, x)  # Estimated coefficients
    residual = y - x@b_hat  # Calculated residuals
    SSR = residual.T@residual  # Sum of squared residuals
    SST = (y - np.mean(y)).T@(y - np.mean(y))  # Total sum of squares
    R2 = 1 - SSR/SST

    sigma2, cov, se = variance(transform, SSR, x, T)
    # Overwrites cov and se with robust version if specified 'robust_se = True'
    if robust_se:
        cov, se = robust(x, residual, T)
    t_values = b_hat/se
    
    names = ['b_hat', 'se', 'sigma2', 't_values', 'R2', 'cov']
    results = [b_hat, se, sigma2, t_values, R2, cov]
    return dict(zip(names, results))




def est_ols( y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Estimates y on x by ordinary least squares, returns coefficents

    Args:
        >> y (np.ndarray): Dependent variable (Needs to have shape 2D shape)
        >> x (np.ndarray): Independent variable (Needs to have shape 2D shape)

    Returns:
        np.array: Estimated beta coefficients.
    """
    return la.inv(x.T@x)@(x.T@y)

def variance( 
        transform: str, 
        SSR: float, 
        x: np.ndarray, 
        T: int
    ) -> tuple:
    """Calculates the covariance and standard errors from the OLS
    estimation.

    Args:
        >> transform (str): Defaults to ''. If the data is transformed in 
        any way, the following transformations are allowed:
            '': No transformations
            'fd': First-difference
            'be': Between transformation
            'fe': Within transformation
            're': Random effects estimation
        >> SSR (float): Sum of squared residuals
        >> x (np.ndarray): Dependent variables from regression
        >> t (int): The number of time periods in x.

    Raises:
        Exception: If invalid transformation is provided, returns
        an error.

    Returns:
        tuple: Returns the error variance (mean square error), 
        covariance matrix and standard errors.
    """

    # Store n and k, used for DF adjustments.
    K = x.shape[1]
    if transform in ('', 'fd', 'be'):
        N = x.shape[0]
    else:
        N = x.shape[0]/T

    # Calculate sigma2
    if transform in ('', 'fd', 'be'):
        sigma2 = (np.array(SSR/(N - K)))
    elif transform.lower() == 'fe':
        sigma2 = np.array(SSR/(N * (T - 1) - K))
    elif transform.lower() == 're':
        sigma2 = np.array(SSR/(T * N - K))
    else:
        raise Exception('Invalid transform provided.')
    
    cov = sigma2*la.inv(x.T@x)
    se = np.sqrt(cov.diagonal()).reshape(-1, 1)
    return sigma2, cov, se

def robust( x: np.ndarray, residual: np.ndarray, T:int) -> tuple:
    '''Calculates the robust variance estimator 

    Args: 
        x: (NT,K) matrix of regressors. Assumes that rows are sorted 
            so that x[:T, :] is regressors for the first individual, 
            and so forth. 
        residual: (NT,1) vector of residuals 
        T: number of time periods. If T==1 or T==None, assumes cross-sectional 
            heteroscedasticity-robust variance estimator
    
    Returns
        tuple: cov, se 
            cov: (K,K) panel-robust covariance matrix 
            se: (K,1) vector of panel-robust standard errors
    '''

    # If only cross sectional, we can use the diagonal.
    if (not T) or (T == 1):
        Ainv = la.inv(x.T@x) 
        uhat2 = residual ** 2
        uhat2_x = uhat2 * x # elementwise multiplication: avoids forming the diagonal matrix (RAM intensive!)
        cov = Ainv @ (x.T@uhat2_x) @ Ainv
    
    # Else we loop over each individual.
    else:
        nobs,K = x.shape
        N = int(nobs / T)
        B = np.zeros((K, K)) # initialize 

        for i in range(N):
            idx_i = slice(i*T, (i+1)*T) # index values for individual i 
            Omega = residual[idx_i]@residual[idx_i].T # (T,T) matrix of outer product of i's residuals 
            B += x[idx_i].T @ Omega @ x[idx_i] # (K,K) contribution 

        Ainv = la.inv(x.T @ x)
        cov = Ainv @ B @ Ainv
    
    se = np.sqrt(np.diag(cov)).reshape(-1, 1)
    return cov, se


def _escape_latex(text):
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(char, char) for char in str(text))


def _stars(p_value):
    if p_value < 0.01:
        return "***"
    if p_value < 0.05:
        return "**"
    if p_value < 0.1:
        return "*"
    return ""


def _as_list(value, n):
    """Returns value as a list of length n (one entry per model)."""
    if isinstance(value, (list, tuple)):
        if len(value) != n:
            raise ValueError(f"Expected {n} values, got {len(value)}.")
        return list(value)
    return [value] * n


def print_table(
        labels,
        results,
        headers=["", "Beta", "Se", "t-values"],
        title="Results",
        _lambda=None,
        latex_dir="Tables",
        latex_filename=None,
        column_names=None,
        panel_title=None,
        note=None,
        n_obs=None,
        decimals=4,
        **kwargs
    ) -> None:
    """Prints a nice looking table and exports it as a LaTeX table.

    Args:
        >> labels (tuple or list of tuples): (label_y, label_x) for one model,
        or a list with one such tuple per model.
        >> results (dict or list of dicts): Results from one or more
        regressions. Each dict needs at least the keys
            'b_hat', 'se', 't_values', 'R2', 'sigma2'
        and may optionally contain 'p_values' and 'N'.
        >> headers (list, optional): Column headers for the console table.
        >> title (str, optional): Table title, used in the caption.
        >> _lambda (float or list, optional): Only used with Random effects.
        >> column_names (list, optional): LaTeX column headers, one per model.
        Defaults to the dependent variable (one model) or (1), (2), ...
        >> panel_title (str, optional): Bold title above the table,
        e.g. "B: Extensive margin".
        >> note (str, optional): Extra text for the note below the table.
        Written as raw LaTeX, so math like $\\beta$ is allowed.
        >> n_obs (int or list, optional): Number of observations. Falls back
        to results['N'] if not given.
        >> decimals (int, optional): Number of decimals. Defaults to 3.
    """

    # Allow a single model or several models side by side.
    if isinstance(results, dict):
        results = [results]
    n_models = len(results)
    if isinstance(labels, tuple):
        labels = [labels] * n_models
    lambdas = _as_list(_lambda, n_models)
    n_obs = _as_list(n_obs, n_models)

    # ---------------- Console output (as before) ----------------
    for k, (res, (label_y, label_x), lam) in enumerate(
            zip(results, labels, lambdas)):
        b = np.ravel(res.get('b_hat'))
        se = np.ravel(res.get('se'))
        t = np.ravel(res.get('t_values'))
        table = [[name, b[i], se[i], t[i]] for i, name in enumerate(label_x)]

        print(title if n_models == 1 else f"{title} - model {k + 1}")
        print(f"Dependent variable: {label_y}\n")
        print(tabulate(table, headers, **kwargs))
        print(f"R\u00b2 = {np.asarray(res.get('R2')).item():.3f}")
        print(f"\u03C3\u00b2 = {np.asarray(res.get('sigma2')).item():.3f}")
        if lam is not None:
            print(f"\u03bb = {np.asarray(lam).item():.3f}")
        print()

    # ---------------- LaTeX output ----------------
    fmt = f"{{:.{decimals}f}}".format

    if column_names is None:
        if n_models == 1:
            column_names = [_escape_latex(labels[0][0])]
        else:
            column_names = [f"({k + 1})" for k in range(n_models)]

    # All regressors in order of first appearance across models.
    var_order = []
    for _, label_x in labels:
        for name in label_x:
            if name not in var_order:
                var_order.append(name)

    lines = [
        r"\begin{table}[!htbp]",
        r"\centering",
        rf"\begin{{tabular}}{{l{'c' * n_models}}}",
    ]
    if panel_title:
        lines.append(
            rf"\multicolumn{{{n_models + 1}}}{{l}}"
            rf"{{\textbf{{{_escape_latex(panel_title)}}}}} \\"
        )
    lines += [
        r"\toprule",
        " & ".join([""] + list(column_names)) + r" \\",
        r"\midrule",
    ]

    # Coefficients with stars, standard errors in parentheses below.
    for var in var_order:
        coef_cells, se_cells = [], []
        for res, (_, label_x) in zip(results, labels):
            if var not in label_x:
                coef_cells.append("")
                se_cells.append("")
                continue
            i = list(label_x).index(var)
            b = float(np.ravel(res['b_hat'])[i])
            se = float(np.ravel(res['se'])[i])
            if res.get('p_values') is not None:
                p = float(np.ravel(res['p_values'])[i])
            else:
                # Two-sided p-value, normal approximation.
                t = float(np.ravel(res['t_values'])[i])
                p = math.erfc(abs(t) / math.sqrt(2))
            stars = _stars(p)
            coef_cells.append(fmt(b) + (f"$^{{{stars}}}$" if stars else ""))
            se_cells.append(f"({fmt(se)})")
        lines.append(" & ".join([_escape_latex(var)] + coef_cells) + r" \\")
        lines.append(" & ".join([""] + se_cells) + r" \\")
        lines.append(r"\addlinespace")
    lines.pop()  # no extra space before the midrule
    lines.append(r"\midrule")

    # Summary statistics.
    n_values = [n if n is not None else res.get('N')
                for n, res in zip(n_obs, results)]
    if any(n is not None for n in n_values):
        cells = [f"{int(np.asarray(n).item()):,}" if n is not None else ""
                 for n in n_values]
        lines.append(" & ".join(["Observations"] + cells) + r" \\")
    lines.append(" & ".join(
        [r"$R^2$"] + [fmt(np.asarray(r['R2']).item()) for r in results]
    ) + r" \\")
    lines.append(" & ".join(
        [r"$\sigma^2$"] + [fmt(np.asarray(r['sigma2']).item()) for r in results]
    ) + r" \\")
    if any(lam is not None for lam in lambdas):
        cells = [fmt(np.asarray(lam).item()) if lam is not None else ""
                 for lam in lambdas]
        lines.append(" & ".join([r"$\lambda$"] + cells) + r" \\")

    lines += [r"\bottomrule", r"\end{tabular}"]

    # Caption below the table: title, dependent variable, note, stars.
    caption = [_escape_latex(title) + "."]
    dep_vars = {label_y for label_y, _ in labels}
    if len(dep_vars) == 1:
        caption.append(f"Dependent variable: {_escape_latex(dep_vars.pop())}.")
    if note:
        caption.append(note)
    caption.append(
        r"Robust standard errors in parentheses. * $p<0.1$, ** $p<0.05$, "
        r"*** $p<0.01$."
    )

    slug = re.sub(r"[^a-zA-Z0-9]+", "_", title).strip("_").lower()
    lines += [
        rf"\caption{{{' '.join(caption)}}}",
        rf"\label{{tab:{slug}}}",
        r"\end{table}",
    ]

    filename = latex_filename or f"{slug}.tex"
    output_path = Path(latex_dir) / filename
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"LaTeX table saved to: {output_path}")


def perm( Q_T: np.ndarray, A: np.ndarray) -> np.ndarray:
    """Takes a transformation matrix and performs the transformation on 
    the given vector or matrix.

    Args:
        Q_T (np.ndarray): The transformation matrix. Needs to have the same
        dimensions as number of years a person is in the sample.
        
        A (np.ndarray): The vector or matrix that is to be transformed. Has
        to be a 2d array.

    Returns:
        np.array: Returns the transformed vector or matrix.
    """
    # We can infer t from the shape of the transformation matrix.
    M,T = Q_T.shape 
    N = int(A.shape[0]/T)
    K = A.shape[1]

    # initialize output 
    Z = np.empty((M*N, K))
    
    for i in range(N): 
        ii_A = slice(i*T, (i+1)*T)
        ii_Z = slice(i*M, (i+1)*M)
        Z[ii_Z, :] = Q_T @ A[ii_A, :]

    return Z