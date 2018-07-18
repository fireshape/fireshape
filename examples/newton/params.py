import ROL

def get_params(step_type, max_iter, ksp_type="GMRES"):
    params_dict = {
        'General': {
            'Krylov':
            {
                "Type": ksp_type,
                'Iteration Limit': 200,
                'Use Initial Guess': False
            },
            'Secant': {
                'Type': 'Limited-Memory BFGS',
                'Maximum Storage': 10,
            },
        },
        'Step': {
            'Type': 'Line Search',
            'Line Search': {
                'Descent Method': {
                    "Type": step_type,
                    # 'Type': 'Newton-Krylov'
                    # 'Type': 'Quasi-Newton Method'
                }
            }
        },
        'Status Test': {
            'Gradient Tolerance': 1e-11,
            'Step Tolerance': 1e-10,
            'Iteration Limit': max_iter
        }
    }

    return ROL.ParameterList(params_dict, "Parameters")
