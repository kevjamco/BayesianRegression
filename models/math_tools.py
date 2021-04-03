import numpy as np

def normal_pdf(x, mu, sig):
    """
    Gaussian PDF
    """
    pdf = 1/np.sqrt(2*np.pi) * np.exp(-1/2*((x-mu)/sig)**2)
    return pdf

def pdf_dx(x, mu, sig):
    """
    partial of gaussian w.r.t. x
    """
    ddx = -(x - mu)/sig**2 * normal_pdf(x, mu, sig)
    return ddx

def pdf_dmu(x, mu, sig):
    """
    partial of gaussian w.r.t. mu
    """
    ddx = (x - mu) / sig ** 2 * normal_pdf(x, mu, sig)
    return ddx

def pdf_dsig(x, mu, sig):
    """
    partial of gaussian w.r.t. sigma
    """
    ddx = (x - mu) / sig ** 3 * normal_pdf(x, mu, sig)
    return ddx

def ln_pdf_dx(x, mu, sig):
    """
    partial of log P(x), where P(x) is a guassian w.r.t. x
    """
    ddx = -(x - mu)/sig**2
    return ddx

def ln_pdf_dmu(x, mu, sig):
    """
    partial of log P(x), where P(x) is a guassian w.r.t. mu
    """
    ddx = (x - mu) / sig ** 2
    return ddx

def ln_pdf_dsig(x, mu, sig):
    """
    partial of log P(x), where P(x) is a guassian w.r.t. sigma
    """
    ddx = (x - mu) / sig ** 3
    return ddx