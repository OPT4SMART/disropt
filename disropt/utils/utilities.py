import numpy as np


def is_pos_def(P: np.ndarray) -> bool:
    """check if a matrix is positive definite
    
    Args:
        P (numpy.ndarray): matrix
    
    Returns:
        bool: 
    """
    return np.all(np.linalg.eigvals(P) > 0)

def is_semi_pos_def(P: np.ndarray) -> bool:
    """check if a matrix is positive semi-definite
    
    Args:
        P (numpy.ndarray): matrix
    
    Returns:
        bool: 
    """
    return np.all(np.linalg.eigvals(P) >= 0)

def check_symmetric(A: np.ndarray, rtol: float=1e-05, atol: float=1e-08) -> bool:
    """check if a matrix is symetric
    
    Args:
        A (numpy.ndarray): matrix
        rtol (float): Defaults to 1e-05.
        atol (float): Defaults to 1e-08.
    
    Returns:
        bool: 
    """
    return np.allclose(A, A.transpose(), rtol=rtol, atol=atol)