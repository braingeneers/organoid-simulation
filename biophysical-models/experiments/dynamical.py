import numpy as np
import functools
import matplotlib.pyplot as plt
import scipy.integrate as spi


def poincaré_map(F, x0s, P, mus, tmax, progress_bar=lambda x: x):
    """
    Given a vector field F on R^n, an (m,n) array of m initial
    conditions x0s, a function P which changes sign when the
    trajectory crosses the Poincaré section, and a vector mu
    of bifurcation parameter values, computes the Poincaré map
    for each combination of IC and bifurcation parameter.
    Evaluation attempts time out at t = tmax.

    Returns the return time Tret and the Poincaré map Xpcr.
    """
    P.terminal = True
    mus = np.asarray(mus)
    x0s = np.asarray(x0s)

    Tret = np.zeros((mus.shape[0], x0s.shape[0]))
    Xpcr = np.zeros((mus.shape[0],) + x0s.shape)

    for i, mu in progress_bar(enumerate(mus)):
        fun = functools.partial(F, mu)
        for j, x0 in enumerate(x0s):
            out = spi.solve_ivp(fun, (0,tmax), x0, events=P)
            if out.status == 1:
                Tret[i,j] = out.t[-1]
            else:
                Tret[i,j] = np.inf
            Xpcr[i,j,:] = out.y[:,-1]

    return Tret, Xpcr


def bifurcation_diagram(f, fx, x, mu):
    "Plot a bifurcation diagram of a 1D dynamical system."
    xx = x.reshape((-1,1))
    mumu = mu.reshape((1,-1))
    fvals = f(xx, mumu)
    fxvals = fx(xx, mumu)
    fvalsP = np.ma.masked_where(fxvals<0, fvals)
    plt.contour(mu, x, fvals, levels=[0], colors='k')
    plt.contour(mu, x, fvalsP, levels=[0], colors='r')
    plt.xlabel('$\mu^*$')


def hybrid_sim(flow_map, jump_map, jump_event, t_range, x0, other_events=()):
    """
    Given a hybrid system in the form of the following components:
        flow_map(t, x) -> dx/dt
        jump_map(t, x) -> x+
        jump_event(t, x) -> 0 within the jump set

    And an initial condition in the form of:
        evaluation range t_range = (t_start, t_stop)
        initial condition x0

    As well as other_events, an optional list of events to track.
    Note that since these can only occur during flow, their times
    are returned in the continuous (NOT hybrid) time domain.

    Returns
        tout : the evaluation times in a HYBRID time domain
        Xout : the state trajectory corresponding to those times
        tev : the times at which each event occurred (jumps first,
            followed by custom events)
    """
    jump_event.terminal = True
    events = (jump_event,) + other_events

    xouts = []
    touts = []
    tev = [[] for _ in events]

    i = 0
    jumps = 0
    while True:
        # Compute and save the results until an event or the end.
        res = spi.solve_ivp(flow_map, t_range, x0, events=events)

        # Save the latest portion of the state trajectory.
        xouts.append(res.y)

        # Save the evaluation times in a hybrid time domain.
        thist = np.vstack((res.t, np.zeros_like(res.t) + jumps))
        touts.append(thist)

        # Save all the event times.
        for i, arr in enumerate(res.t_events):
            tev[i] += list(arr)

        # Done because of failure or finishing.
        if res.status==0 or res.status==-1:
            break

        # Not done, so we need to jump back into the flow set.
        # x0 = jump_map(res.t[-1], res.y[:,-1])
        # jumps += 1
        # print('Jumped from', res.y[:,-1], 'to', x0)
        x0 = res.y[:,-1]
        while np.isclose(jump_event(res.t[-1], x0), 0):
            x0 = jump_map(res.t[-1], x0)
            jumps += 1
        t_range = res.t[-1], t_range[1]

    touts = np.hstack(touts)
    xouts = np.hstack(xouts)
    return touts, xouts.T, tev
