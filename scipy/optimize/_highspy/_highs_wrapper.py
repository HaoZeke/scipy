from warnings import warn

import numpy as np
import scipy.optimize._highspy._core as _h
# pyrefly: ignore[missing-module-attribute]
from scipy.optimize._highspy import _highs_options as hopt
from scipy.optimize._optimize import OptimizeResult, OptimizeWarning


def _highs_wrapper(c, indptr, indices, data, lhs, rhs, lb, ub, integrality,
                   options, callback=None, n_slack=0, n_con=0):
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

    # Register a callback only when the caller supplied one. With no
    # callback, HiGHS is not given a callback function at all.
    held_exception = []
    stop_requested = []
    if callback is not None:
        _install_highs_callback(
            highs, callback, numcol, n_slack, n_con, options.get("solver"),
            isMip, c, indptr, indices, data, rhs, held_exception,
            stop_requested)

    run_status = highs.run()
    # Drop the std::function's reference to the Python callback now that
    # the solve has returned. ``highs`` is local, so this also breaks
    # the closure cycle as soon as the wrapper returns.
    if callback is not None:
        try:
            highs.setCallback(None, None)
        except (TypeError, RuntimeError):
            pass
    if held_exception:
        raise held_exception[0]
    if run_status == _h.HighsStatus.kError:
        res.update(
            {
                "status": highs.getModelStatus(),
                "message": highs.modelStatusToString(highs.getModelStatus()),
            }
        )
        return _finish_callback_status(res, stop_requested)

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
    # A MIP interrupt can still carry the postsolved incumbent. Treat it
    # like the other limits: return that incumbent when the objective is
    # finite, and withhold it when HiGHS has not found one.
    mip_with_solution = (
        _h.HighsModelStatus.kOptimal,
        _h.HighsModelStatus.kTimeLimit,
        _h.HighsModelStatus.kIterationLimit,
        _h.HighsModelStatus.kSolutionLimit,
        _h.HighsModelStatus.kInterrupt,
    )
    mip_partial = (
        _h.HighsModelStatus.kTimeLimit,
        _h.HighsModelStatus.kIterationLimit,
        _h.HighsModelStatus.kSolutionLimit,
        _h.HighsModelStatus.kInterrupt,
    )
    mipFailCondition = model_status not in mip_with_solution or (
        model_status in mip_partial
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
        return _finish_callback_status(res, stop_requested)

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

    return _finish_callback_status(res, stop_requested)


def _known_count(value):
    """Return a HiGHS count, or None for the -1 unknown sentinel.

    Iteration counts and ``mip_node_count`` are initialized to -1 and
    stay there until the solver has a real count. That sentinel is not
    an iteration and must not be reported as ``nit``.
    """
    value = int(value)
    if value < 0:
        return None
    return value


def _finish_callback_status(res, stop_requested):
    """Report a callback stop as HiGHS status ``kInterrupt``.

    ``kCallbackMipImprovingSolution`` ignores ``user_interrupt`` (HiGHS
    asserts that it is false). A stop requested there is applied on the
    next interrupt event, and again here if the solve returns before
    that event.
    """
    if stop_requested or res.get("status") == _h.HighsModelStatus.kInterrupt:
        res["status"] = _h.HighsModelStatus.kInterrupt
        res["message"] = "Interrupted by user"
    return res


def _csc_matvec(indptr, indices, values, x, n_row):
    """CSC matrix-vector product. Duplicate column entries accumulate."""
    indptr = np.asarray(indptr)
    indices = np.asarray(indices)
    values = np.asarray(values, dtype=float)
    x = np.asarray(x, dtype=float)
    y = np.zeros(n_row, dtype=float)
    for col in range(x.size):
        start = int(indptr[col])
        end = int(indptr[col + 1])
        if start != end:
            np.add.at(y, indices[start:end], values[start:end] * x[col])
    return y


def _install_highs_callback(highs, callback, n_x, n_slack, n_con, solver,
                            is_mip, c, indptr, indices, data, rhs,
                            held_exception, stop_requested):
    """Register ``callback`` on the simplex and/or IPM interrupt events.

    ``highs-ds`` listens only to ``kCallbackSimplexInterrupt`` and reports
    each distinct iteration count once: ``HEkk::bailout`` runs on every
    pricing pass, so one iteration would otherwise be delivered several
    times. ``highs-ipm`` listens only to ``kCallbackIpmInterrupt``. IPX
    calls ``InterruptCheck()`` with its default iteration count of -1 from
    crossover and factorization; those events are not IPM iterations and
    are dropped. ``method='highs'`` (``solver is None``) registers both,
    and whichever solver HiGHS runs delivers its events.

    IPX catches ``std::exception`` around the solve, which would swallow a
    Python exception raised from this callback. The exception is stored
    and re-raised after ``Highs.run`` returns. pybind11 re-acquires the
    GIL for the callback, so the mutable state below is only touched
    while the GIL is held, including if HiGHS invokes the callback from
    a worker thread.

    A mixed-integer solve listens to ``kCallbackMipImprovingSolution``
    (each new incumbent, in the original variable space) and
    ``kCallbackMipInterrupt`` (``checkLimits``, once per distinct node
    count). ``clearHighsCallbackOutput`` sets the node count to -1, and
    the LP iteration fields on those events are left at that sentinel.
    A node event with an unknown count is not a node and is dropped. An
    improving solution is still delivered; its count is reported as 0.
    HiGHS asserts that an improving-solution callback does not set
    ``user_interrupt``; a stop requested there is applied on the next
    interrupt event. Logging callbacks are never started.
    """
    cb = _h.cb.HighsCallbackType
    simplex_type = int(cb.kCallbackSimplexInterrupt)
    ipm_type = int(cb.kCallbackIpmInterrupt)
    mip_improv_type = int(cb.kCallbackMipImprovingSolution)
    mip_interrupt_type = int(cb.kCallbackMipInterrupt)
    if is_mip:
        active = (cb.kCallbackMipImprovingSolution, cb.kCallbackMipInterrupt)
    elif solver == "ipm":
        active = (cb.kCallbackIpmInterrupt,)
    elif solver == "simplex":
        active = (cb.kCallbackSimplexInterrupt,)
    else:
        active = (cb.kCallbackSimplexInterrupt, cb.kCallbackIpmInterrupt)

    seen_simplex = set()
    seen_ipm = set()
    seen_nodes = set()

    def _set_user_interrupt(data_in, stop):
        # HiGHS does not clear ``user_interrupt`` between callbacks.
        # ``kCallbackMipImprovingSolution`` asserts that it is false, so
        # every return path has to write the flag explicitly.
        if data_in is not None:
            data_in.user_interrupt = bool(stop)

    def _highs_callback(callback_type, _message, data_out, data_in, _user_data):
        kind = int(callback_type)
        # Improving-solution callbacks ignore ``user_interrupt`` and abort
        # the process in a build with assertions if it is left true.
        honors_stop = kind != mip_improv_type

        def _apply_interrupt(stop):
            if stop:
                stop_requested.append(True)
            _set_user_interrupt(
                data_in, honors_stop and bool(stop_requested))

        try:
            if held_exception:
                _set_user_interrupt(data_in, honors_stop)
                return

            improving = False
            if kind == simplex_type:
                nit = _known_count(data_out.simplex_iteration_count)
                # Same unknown sentinel as an IPM crossover event.
                if nit is None or nit in seen_simplex:
                    _apply_interrupt(False)
                    return
                seen_simplex.add(nit)
                phase = 2
            elif kind == ipm_type:
                nit = _known_count(data_out.ipm_iteration_count)
                # Crossover and factorization, not an IPM iteration.
                if nit is None or nit in seen_ipm:
                    _apply_interrupt(False)
                    return
                seen_ipm.add(nit)
                phase = 1
            elif kind == mip_interrupt_type:
                nit = _known_count(data_out.mip_node_count)
                # -1 is "unknown", not a node. Do not record it: a later
                # real count, including 0, still has to be delivered.
                if nit is None or nit in seen_nodes:
                    _apply_interrupt(False)
                    return
                seen_nodes.add(nit)
                phase = 2
            elif kind == mip_improv_type:
                nit = _known_count(data_out.mip_node_count)
                # The incumbent is real even when HiGHS has not filled
                # in a node count. Report that sentinel as 0, matching
                # the LP iteration count on the returned result.
                if nit is None:
                    nit = 0
                phase = 2
                improving = True
            else:
                _apply_interrupt(False)
                return

            x = np.full(n_x, np.nan)
            fun = float("nan")
            slack = np.full(n_slack, np.nan)
            con = np.full(n_con, np.nan)
            if improving:
                incumbent = np.array(
                    data_out.mip_solution, dtype=float, copy=True)
                # The incumbent is stored for the original model. When its
                # length matches, slack and con are A and b of that model.
                if incumbent.size == n_x:
                    x = incumbent
                    fun = float(np.dot(np.asarray(c, dtype=float), x))
                    residual = np.asarray(rhs, dtype=float) - _csc_matvec(
                        indptr, indices, data, x, np.asarray(rhs).size)
                    slack = residual[:n_slack]
                    con = residual[n_slack:n_slack + n_con]

            cb_res = OptimizeResult({
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
            if kind in (mip_improv_type, mip_interrupt_type):
                # ``nit`` is already the sanitized node count.
                cb_res["mip_node_count"] = nit
                cb_res["mip_dual_bound"] = float(data_out.mip_dual_bound)
                cb_res["mip_gap"] = float(data_out.mip_gap)
            try:
                stop = callback(cb_res) is True
            except StopIteration:
                stop = True
            except BaseException as exc:
                held_exception.append(exc)
                stop_requested.append(True)
                _set_user_interrupt(data_in, honors_stop)
                return
            _apply_interrupt(stop)
        except BaseException as exc:
            # IPX catches ``std::exception`` around the solve, which drops
            # a Python exception that escapes this callback. Store it and
            # ask the solver to stop so it can be re-raised after ``run``.
            held_exception.append(exc)
            stop_requested.append(True)
            _set_user_interrupt(data_in, honors_stop)

    # ``None`` is an empty std::function and clears any previous callback.
    highs.setCallback(_highs_callback, None)
    for callback_type in active:
        highs.startCallback(callback_type)


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
