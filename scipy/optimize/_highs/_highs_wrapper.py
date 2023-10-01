from warnings import warn

import numpy as np
from scipy.optimize._highs.highspy import highs as hspy  # type: ignore[attr-defined]
from scipy.optimize._highs.highspy import _highs_options as hopt  # type: ignore[attr-defined]
from scipy.optimize import OptimizeWarning

def _highs_wrapper(c, indptr, indices, data, lhs, rhs, lb, ub, integrality, options):
    numcol = c.size
    numrow = rhs.size
    isMip = integrality is not None and np.sum(integrality) > 0

    # default "null" return values
    res = {
        "x": None,
        "fun": None,
    }

    # Fill up a HighsLp object
    lp = hspy.HighsLp()
    lp.num_col_ = numcol
    lp.num_row_ = numrow
    lp.a_matrix_.num_col_ = numcol
    lp.a_matrix_.num_row_ = numrow
    lp.a_matrix_.format_ = hspy.MatrixFormat.kColwise
    lp.col_cost_ = c
    lp.col_lower_ = lb
    lp.col_upper_ = ub
    lp.row_lower_ = lhs
    lp.row_upper_ = rhs
    lp.a_matrix_.start_ = indptr
    lp.a_matrix_.index_ = indices
    lp.a_matrix_.value_ = data
    if integrality.size > 0:
        lp.integrality_ = [hspy.HighsVarType(i) for i in integrality]

    # Make a Highs object and pass it everything
    highs = hspy.Highs_()
    highs_options = hspy.HighsOptions()
    hoptmanager = hopt.HighsOptionsManager()
    for key, val in options.items():
        # handle filtering of unsupported and default options
        if val is None or key in ("sense",):
            continue

        # ask for the option type
        opt_type = hoptmanager.get_option_type(key)
        if -1 == opt_type:
            warn(f"Unrecognized options detected: {dict({key: val})}", OptimizeWarning)
            continue
        else:
            if key in ("presolve", "parallel"):
                # handle fake bools (require bool -> str conversions)
                if isinstance(val, bool):
                    val = "on" if val else "off"
                else:
                    warn(f'Option f"{key}" is "{val}", but only True or False is '
                         f'allowed. Using default.', OptimizeWarning)
                    continue
            opt_type = hspy.HighsOptionType(opt_type)
            status, msg = check_option(highs, key, val)
            # {
            #     hspy.HighsOptionType.kBool: lambda _x, _y: (0, ""),
            #     hspy.HighsOptionType.kInt: hopt.check_int_option,
            #     hspy.HighsOptionType.kDouble: hopt.check_double_option,
            #     hspy.HighsOptionType.kString: hopt.check_string_option,
            # }[opt_type](key, val)

            # have to do bool checking here because HiGHS doesn't have API
            if opt_type == hspy.HighsOptionType.kBool:
                if not isinstance(val, bool):
                    warn(f'Option f"{key}" is "{val}", but only True or False is '
                         f'allowed. Using default.', OptimizeWarning)
                    continue

            # warn or set option
            if status != 0:
                warn(msg, OptimizeWarning)
            else:
                setattr(highs_options, key, val)

    opt_status = highs.passOptions(highs_options)
    if opt_status == hspy.HighsStatus.kError:
        res.update({
            "status": highs.getModelStatus(),
            "message": highs.modelStatusToString(highs.getModelStatus()),
        })
        return res

    init_status = highs.passModel(lp)
    if init_status == hspy.HighsStatus.kError:
        # if model fails to load, highs.getModelStatus() will be NOT_SET
        err_model_status = hspy.HighsModelStatus.kModelError
        res.update({
            "status": err_model_status,
            "message": highs.modelStatusToString(err_model_status),
        })
        return res

    # Solve the LP
    run_status = highs.run()
    if run_status == hspy.HighsStatus.kError:
        res.update({
            "status": highs.getModelStatus(),
            "message": highs.modelStatusToString(highs.getModelStatus()),
        })
        return res

    # Extract what we need from the solution
    model_status = highs.getModelStatus()

    # it should always be safe to get the info object
    info = highs.getInfo()

    # Failure modes:
    #     LP: if we have anything other than an Optimal status, it
    #         is unsafe (and unhelpful) to read any results
    #    MIP: has a non-Optimal status or has timed out/reached max iterations
    #             1) If not Optimal/TimedOut/MaxIter status, there is no solution
    #             2) If TimedOut/MaxIter status, there may be a feasible solution.
    #                if the objective function value is not Infinity, then the
    #                current solution is feasible and can be returned.  Else, there
    #                is no solution.
    mipFailCondition = model_status not in (
        hspy.HighsModelStatus.kOptimal,
        hspy.HighsModelStatus.kTimeLimit,
        hspy.HighsModelStatus.kIterationLimit,
        hspy.HighsModelStatus.kSolutionLimit,
    ) or (model_status in {
        hspy.HighsModelStatus.kTimeLimit,
        hspy.HighsModelStatus.kIterationLimit,
        hspy.HighsModelStatus.kSolutionLimit,
    } and (info.objective_function_value == hspy.kHighsInf))
    lpFailCondition = model_status != hspy.HighsModelStatus.kOptimal
    if (isMip and mipFailCondition) or (not isMip and lpFailCondition):
        res.update({
            "status": model_status,
            "message": f"model_status is {highs.modelStatusToString(model_status)}; "
                       f"primal_status is "
                       f"{highs.solutionStatusToString(info.primal_solution_status)}",
            "simplex_nit": info.simplex_iteration_count,
            "ipm_nit": info.ipm_iteration_count,
            "crossover_nit": info.crossover_iteration_count,
        })
        return res

    # Should be safe to read the solution:
    solution = highs.getSolution()
    basis = highs.getBasis()

    # Lagrangians for bounds based on column statuses
    marg_bnds = np.zeros((2, numcol))
    for ii in range(numcol):
        if basis.col_status[ii] == hspy.HighsBasisStatus.kLower:
            marg_bnds[0, ii] = solution.col_dual[ii]
        elif basis.col_status[ii] == hspy.HighsBasisStatus.kUpper:
            marg_bnds[1, ii] = solution.col_dual[ii]

    res.update({
        "status": model_status,
        "message": highs.modelStatusToString(model_status),

        # Primal solution
        "x": np.array(solution.col_value),

        # Ax + s = b => Ax = b - s
        # Note: this is for all constraints (A_ub and A_eq)
        "slack": rhs - solution.row_value,

        # lambda are the lagrange multipliers associated with Ax=b
        "lambda": np.array(solution.row_dual),
        "marg_bnds": marg_bnds,

        "fun": info.objective_function_value,
        "simplex_nit": info.simplex_iteration_count,
        "ipm_nit": info.ipm_iteration_count,
        "crossover_nit": info.crossover_iteration_count,
    })

    if isMip:
        res.update({
            "mip_node_count": info.mip_node_count,
            "mip_dual_bound": info.mip_dual_bound,
            "mip_gap": info.mip_gap,
        })

    return res

def check_option(highs_inst, option, value):
    status, option_type = highs_inst.getOptionType(option)
    hoptmanager = hopt.HighsOptionsManager()

    if status != hspy.HighsStatus.kOk:
        return -1, "Invalid option name."

    valid_types = {
        hspy.HighsOptionType.kBool: bool,
        hspy.HighsOptionType.kInt: int,
        hspy.HighsOptionType.kDouble: float,
        hspy.HighsOptionType.kString: str
    }

    expected_type = valid_types.get(option_type, None)

    if expected_type is str:
        if not hoptmanager.check_string_option(option, value):
            return -1, "Invalid option value."
    if expected_type is float:
        if not hoptmanager.check_double_option(option, value):
            return -1, "Invalid option value."
    if expected_type is int:
        if not hoptmanager.check_int_option(option, value):
            return -1, "Invalid option value."

    if expected_type is None:
        return 3, "Unknown option type."

    status, current_value = highs_inst.getOptionValue(option)
    if status != hspy.HighsStatus.kOk:
        return 4, "Failed to validate option value."
    return 0, "Check option succeeded."
