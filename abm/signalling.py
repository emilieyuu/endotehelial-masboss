
def hill(tau, K, n):
    """
    S: Sensitivity input (sheae nmagnitude / force)
    K: Half activation thershold
    n: Hill coefficiet
    """
    return tau**n / (K**n + tau**n)