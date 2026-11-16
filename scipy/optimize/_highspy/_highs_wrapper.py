from warnings import warn

import numpy as np
import scipy.optimize._highspy._core as _h
# pyrefly: ignore[missing-module-attribute]
from scipy.optimize._highspy import _highs_options as hopt
from scipy.optimize._optimize import OptimizeResult, OptimizeWarning


def _highs_wrapper(c, indptr, indices, data, lhs, rhs, lb, ub, integrality,
                   options, callback=None, A=None, n_slack=0):
    '''Solve linear programs using HiGHS [1]_.

    Assume problems of the form:

        MIN c.T @ x
        s.t. lhs <= A @ x <= rhs
             lb <= x <= ub

    Parameters
    ----------
    c : 1-D array, (n,)
        Array of objective value coefficients.
    astart : 1-D array
        CSC format index array.
    aindex : 1-D array
        CSC format index array.
    avalue : 1-D array
        Data array of the matrix.
    lhs : 1-D array (or None), (m,)
        Array of left hand side values of the inequality constraints.
        If ``lhs=None``, then an array of ``-inf`` is assumed.
    rhs : 1-D array, (m,)
        Array of right hand side values of the inequality constraints.
    lb : 1-D array (or None), (n,)
        Lower bounds on solution variables x.  If ``lb=None``, then an
        array of all `0` is assumed.
    ub : 1-D array (or None), (n,)
        Upper bounds on solution variables x.  If ``ub=None``, then an
        array of ``inf`` is assumed.
    options : dict
        A dictionary of solver options

    Returns
    -------
    res : dict

        If model_status is one of kOptimal,
        kObjectiveBound, kTimeLimit,
        kIterationLimit:

            - ``status`` : HighsModelStatus
                Model status code.

            - ``message`` : str
                Message corresponding to model status code.

            - ``x`` : list
                Solution variables.

            - ``slack`` : list
                Slack variables.

            - ``lambda`` : list
                Lagrange multipliers associated with the constraints
                Ax = b.

            - ``s`` : list
                Lagrange multipliers associated with the constraints
                x >= 0.

            - ``fun``
                Final objective value.

            - ``simplex_nit`` : int
                Number of iterations accomplished by the simplex
                solver.

            - ``ipm_nit`` : int
                Number of iterations accomplished by the interior-
                point solver.

        If model_status is not one of the above:

            - ``status`` : HighsModelStatus
                Model status code.

            - ``message`` : str
                Message corresponding to model status code.

    Notes
    -----
    If ``options['write_solution_to_file']`` is ``True`` but
    ``options['solution_file']`` is unset or ``''``, then the solution
    will be printed to ``stdout``.

    If any iteration limit is reached, no solution will be
    available.

    ``OptimizeWarning`` will be raised if any option value set by
    the user is found to be incorrect.

    References
    ----------
    .. [1] https://highs.dev/
    .. [2] https://www.maths.ed.ac.uk/hall/HiGHS/HighsOptions.html
    '''
    numcol = c.size
    numrow = rhs.size
    isMip = integrality is not None and np.sum(integrality) > 0

    # default "null" return values
    res = {
        "x": None,
        "fun": None,
    }

    # Fill up a HighsLp object
    lp = _h.HighsLp()
    lp.num_col_ = numcol
    lp.num_row_ = numrow
    lp.a_matrix_.num_col_ = numcol
    lp.a_matrix_.num_row_ = numrow
    lp.a_matrix_.format_ = _h.MatrixFormat.kColwise
    lp.col_cost_ = c
    lp.col_lower_ = lb
    lp.col_upper_ = ub
    lp.row_lower_ = lhs
    lp.row_upper_ = rhs
    lp.a_matrix_.start_ = indptr
    lp.a_matrix_.index_ = indices
    lp.a_matrix_.value_ = data
    if integrality.size > 0:
        lp.integrality_ = [_h.HighsVarType(i) for i in integrality]

    # Make a Highs object and pass it everything
    highs = _h._Highs()
    highs_options = _h.HighsOptions()
    hoptmanager = hopt.HighsOptionsManager()
    for key, val in options.items():
        # handle filtering of unsupported and default options
        if val is None or key in ("sense",):
            continue

        # ask for the option type
        opt_type = hoptmanager.get_option_type(key)
        if -1 == opt_type:
            warn(
                f"Unrecognized options detected: {dict({key: val})}",
                OptimizeWarning,
                stacklevel=2,
            )
            continue
        else:
            if key in ("presolve", "parallel"):
                # handle fake bools (require bool -> str conversions)
                if isinstance(val, bool):
                    val = "on" if val else "off"
                else:
                    warn(
                        f'Option f"{key}" is "{val}", but only True or False is '
                        f"allowed. Using default.",
                        OptimizeWarning,
                        stacklevel=2,
                    )
                    continue
            opt_type = _h.HighsOptionType(opt_type)
            status, msg = check_option(highs, key, val)
            if opt_type == _h.HighsOptionType.kBool:
                if not isinstance(val, bool):
                    warn(
                        f'Option f"{key}" is "{val}", but only True or False is '
                        f"allowed. Using default.",
                        OptimizeWarning,
                        stacklevel=2,
                    )
                    continue

            # warn or set option
            if status != 0:
                warn(msg, OptimizeWarning, stacklevel=2)
            else:
                setattr(highs_options, key, val)

    opt_status = highs.passOptions(highs_options)
    if opt_status == _h.HighsStatus.kError:
        res.update(
            {
                "status": highs.getModelStatus(),
                "message": highs.modelStatusToString(highs.getModelStatus()),
            }
        )
        return res

    init_status = highs.passModel(lp)
    if init_status == _h.HighsStatus.kError:
        # if model fails to load, highs.getModelStatus() will be NOT_SET
        err_model_status = _h.HighsModelStatus.kModelError
        res.update(
            {
                "status": err_model_status,
                "message": highs.modelStatusToString(err_model_status),
            }
        )
        return res

    # No callback is registered unless the caller supplied one.
    state = None
    if callback is not None:
        state = _listen(highs, callback, c, A, rhs, numcol, n_slack,
                        options.get("solver"), isMip)

    run_status = highs.run()
    if state is not None and state["exc"] is not None:
        raise state["exc"]
    if run_status == _h.HighsStatus.kError:
        res.update(
            {
                "status": highs.getModelStatus(),
                "message": highs.modelStatusToString(highs.getModelStatus()),
            }
        )
        return _callback_stop(res, state)

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
    # An interrupted MIP keeps a finite incumbent, like the other limits.
    mipFailCondition = model_status not in (
        _h.HighsModelStatus.kOptimal,
        _h.HighsModelStatus.kTimeLimit,
        _h.HighsModelStatus.kIterationLimit,
        _h.HighsModelStatus.kSolutionLimit,
        _h.HighsModelStatus.kInterrupt,
    ) or (
        model_status
        in {
            _h.HighsModelStatus.kTimeLimit,
            _h.HighsModelStatus.kIterationLimit,
            _h.HighsModelStatus.kSolutionLimit,
            _h.HighsModelStatus.kInterrupt,
        }
        and (info.objective_function_value == _h.kHighsInf)
    )
    lpFailCondition = model_status != _h.HighsModelStatus.kOptimal
    if (isMip and mipFailCondition) or (not isMip and lpFailCondition):
        res.update(
            {
                "status": model_status,
                "message": "model_status is "
                f"{highs.modelStatusToString(model_status)}; "
                "primal_status is "
                f"{highs.solutionStatusToString(info.primal_solution_status)}",
                "simplex_nit": info.simplex_iteration_count,
                "ipm_nit": info.ipm_iteration_count,
                "crossover_nit": info.crossover_iteration_count,
            }
        )
        return _callback_stop(res, state)

    # Should be safe to read the solution:
    solution = highs.getSolution()
    basis = highs.getBasis()

    # Lagrangians for bounds based on column statuses
    marg_bnds = np.zeros((2, numcol))
    basis_col_status = basis.col_status
    solution_col_dual = solution.col_dual
    for ii in range(numcol):
        if basis_col_status[ii] == _h.HighsBasisStatus.kLower:
            marg_bnds[0, ii] = solution_col_dual[ii]
        elif basis_col_status[ii] == _h.HighsBasisStatus.kUpper:
            marg_bnds[1, ii] = solution_col_dual[ii]

    res.update(
        {
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
        }
    )

    if isMip:
        res.update(
            {
                "mip_node_count": info.mip_node_count,
                "mip_dual_bound": info.mip_dual_bound,
                "mip_gap": info.mip_gap,
            }
        )

    return _callback_stop(res, state)


def _known_count(value):
    # HiGHS initializes counts to -1 and leaves them there until it has one.
    value = int(value)
    return None if value < 0 else value


def _callback_stop(res, state):
    # An improving-solution callback cannot set the HiGHS interrupt, so a
    # stop requested there is applied here if the solve returns first.
    if state is not None and state["stop"]:
        res["status"] = _h.HighsModelStatus.kInterrupt
        res["message"] = "Interrupted by user"
    return res


def _listen(highs, callback, c, A, rhs, n_x, n_slack, solver, is_mip):
    """Register ``callback`` and return the mutable solve state.

    Simplex and interior-point callbacks see each iteration count once.
    A count of -1 is unknown (crossover, factorization, or an unfilled
    MIP node count) and is not an event, except on an improving solution,
    which is still delivered with that count reported as 0. An improving
    solution must leave ``user_interrupt`` false. Exceptions are stored:
    IPX would otherwise turn one that escapes into a numerical failure.
    """
    kinds = _h.cb.HighsCallbackType
    simplex = int(kinds.kCallbackSimplexInterrupt)
    ipm = int(kinds.kCallbackIpmInterrupt)
    mip_improv = int(kinds.kCallbackMipImprovingSolution)
    mip_node = int(kinds.kCallbackMipInterrupt)
    if is_mip:
        active = (kinds.kCallbackMipImprovingSolution,
                  kinds.kCallbackMipInterrupt)
    elif solver == "ipm":
        active = (kinds.kCallbackIpmInterrupt,)
    elif solver == "simplex":
        active = (kinds.kCallbackSimplexInterrupt,)
    else:
        active = (kinds.kCallbackSimplexInterrupt, kinds.kCallbackIpmInterrupt)

    seen = {simplex: set(), ipm: set(), mip_node: set()}
    state = {"exc": None, "stop": False}
    n_con = int(np.asarray(rhs).size) - n_slack

    def _highs_callback(callback_type, _message, data_out, data_in, _user):
        report_stop = True
        stop = False
        try:
            kind = int(callback_type)
            # HiGHS does not clear user_interrupt, and asserts that an
            # improving-solution callback leaves it false.
            report_stop = kind != mip_improv
            if state["exc"] is not None:
                return
            improving = False
            if kind == simplex:
                nit = _known_count(data_out.simplex_iteration_count)
                phase = 2
            elif kind == ipm:
                nit = _known_count(data_out.ipm_iteration_count)
                phase = 1
            elif kind == mip_node:
                nit = _known_count(data_out.mip_node_count)
                phase = 2
            elif kind == mip_improv:
                nit = _known_count(data_out.mip_node_count)
                if nit is None:
                    nit = 0
                phase = 2
                improving = True
            else:
                return
            if not improving and (nit is None or nit in seen[kind]):
                return
            if not improving:
                seen[kind].add(nit)

            x = np.full(n_x, np.nan)
            fun = np.nan
            slack = np.full(n_slack, np.nan)
            con = np.full(n_con, np.nan)
            if improving:
                incumbent = np.array(data_out.mip_solution, dtype=float,
                                     copy=True)
                if incumbent.size == n_x:
                    x = incumbent
                    fun = float(np.dot(np.asarray(c, dtype=float), x))
                    residual = np.asarray(rhs, dtype=float) - (A @ x)
                    slack = residual[:n_slack]
                    con = residual[n_slack:]
            result = OptimizeResult({
                "x": x,
                "fun": fun,
                "slack": slack,
                "con": con,
                "success": False,
                "status": 0,
                "nit": int(nit),
                "message": "",
                "phase": phase,
                "complete": False,
            })
            if is_mip:
                result["mip_node_count"] = int(nit)
                result["mip_dual_bound"] = float(data_out.mip_dual_bound)
                result["mip_gap"] = float(data_out.mip_gap)
            stop = callback(result) is True
        except StopIteration:
            stop = True
        except BaseException as exc:
            state["exc"] = exc
            stop = True
        finally:
            if stop:
                state["stop"] = True
            data_in.user_interrupt = bool(report_stop and state["stop"])

    highs.setCallback(_highs_callback, None)
    for kind in active:
        highs.startCallback(kind)
    return state


def check_option(highs_inst, option, value):
    status, option_type = highs_inst.getOptionType(option)
    hoptmanager = hopt.HighsOptionsManager()

    if status != _h.HighsStatus.kOk:
        return -1, "Invalid option name."

    valid_types = {
        _h.HighsOptionType.kBool: bool,
        _h.HighsOptionType.kInt: int,
        _h.HighsOptionType.kDouble: float,
        _h.HighsOptionType.kString: str,
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
    if status != _h.HighsStatus.kOk:
        return 4, "Failed to validate option value."
    return 0, "Check option succeeded."
