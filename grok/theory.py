"""The paper's explicit bounds. Equation numbers refer to 2601.19791v4."""
import math


def eq8_t2_bound(n, m, nu2, eps, eta, lam):
    """Eq (8): t2 >= ln((m-n)nu^2 / (8 m eps)) / (2.02 eta lam)."""
    return math.log((m - n) * nu2 / (8 * m * eps)) / (2.02 * eta * lam)


def eq8_t1_bound(n, m, nu2, eps, eta, lam_plus):
    """Eq (8): t1 <= n ln(14 m nu^2 / eps) / (2 eta lam_plus(Phi^T Phi))."""
    return n * math.log(14 * m * nu2 / eps) / (2 * eta * lam_plus)


def eq6_t1_bound(n, b2, theta0_norm2, eps, eta, lam_plus):
    """Eq (6)."""
    return n * math.log(6 * b2 * theta0_norm2 / eps) / (2 * eta * lam_plus)


def eq7_t2_bound(n, m, nu2, c, lam_min_sigma, theta_star_norm, eta, lam):
    """Eq (7)."""
    inner = (math.sqrt(c / lam_min_sigma) + theta_star_norm) ** -2
    return math.log((m - n) * nu2 / 2 * inner) / (4 * eta * lam)


def thm41_train_bound(L, n, eta, lam, lam_plus, theta0_norm2, t):
    """Theorem 4.1(i), on L_n."""
    return L / 2 * (1 - eta * lam_plus / n - eta * lam) ** (2 * t) * theta0_norm2


def thm41_test_bound(lam_min_sigma, eta, lam, m, n, nu2, t):
    """Theorem 4.1(ii), holding with probability at least 1 - 2 exp(-(m-n)/32)."""
    return lam_min_sigma * (1 - eta * lam) ** (2 * t) * (m - n) * nu2 / 2


def thm41_norm_bound(eta, lam, theta0_norm2, t):
    """Theorem 4.1(iii), on the squared norm."""
    return (1 - eta * lam) ** (2 * t) * theta0_norm2
