from ..constraints import Constraint, AbstractSet

def check_affine_constraints(constraints: list) -> bool:
    """Check if a list of Constraint objects contains only affine constraints
    
    Args:
        constraints (list): list of Constraint
    
    Returns:
        bool
    """
    affine_constraints = True
    if constraints is not None:
        if not isinstance(constraints, list):
            constraints = [constraints]
        for constraint in constraints:
            if isinstance(constraint, Constraint):
                if not constraint.is_affine:
                    affine_constraints = False
            if isinstance(constraint, AbstractSet):
                constraints_list = constraint.to_constraints()
                for cns in constraints_list:
                    if not cns.is_affine:
                        affine_constraints = False
    return affine_constraints
