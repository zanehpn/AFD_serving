"""Observable-only energy prior: replay duration times mean role power.

Local, same-structure log-linear responses. A slope is used only if that
individual coordinate is identifiable from calibration interventions. Neither
TTFT nor NVML is relabelled as attention/dispatch/expert/combine service time.
"""
import copy
import math

import numpy as np

from .space import KNOBS, configuration, digest, structure, positive

MODEL = 'external_power_duration_v1'


def validate_observable(observation):
    if observation.get('selection_split') != 'calibration' or observation.get('status') != 'ok':
        raise ValueError('External model accepts successful calibration measurements only')
    value = observation['external_observables']
    if value.get('provenance') != 'public_replay_and_nvml' or 'four_stage' in observation:
        raise ValueError('External measurements must not contain fabricated stage labels')
    positive(value['duration_s'], 'replay duration')
    for role in ('attention', 'expert'):
        positive(value['role_mean_power_w'][role], 'mean role power')
    energy = value['duration_s'] * sum(value['role_mean_power_w'].values())
    if not math.isclose(energy, observation['metrics']['energy_j'], rel_tol=1e-6):
        raise ValueError('Observable energy differs from the measured power integral')
    return value


def inform(candidates, observations, default_energy_j):
    by_id = {c['id']: c for c in candidates}
    grouped = {}
    for row in observations:
        if row.get('status') != 'ok' or row['candidate_id'] not in by_id:
            continue
        value = validate_observable(row)
        c = by_id[row['candidate_id']]
        grouped.setdefault(digest(structure(c)), []).append((c, value))
    result, reports = copy.deepcopy(candidates), {}
    for candidate in result:
        key = digest(structure(candidate))
        rows = grouped.get(key, [])
        candidate.pop('mechanism', None)
        candidate['prior'] = dict(log_energy=math.log(default_energy_j), energy_sd=1., source='uncovered_neutral')
        if not rows:
            continue
        x = np.array([[math.log(configuration(c)[k]) for k in KNOBS] for c, _ in rows])
        center = x.mean(axis=0)
        dx = x-center
        # Projection onto the measured row space detects individually identifiable
        # controls; correlated f/p interventions cannot identify separate slopes.
        projection = np.linalg.pinv(dx) @ dx
        identified = [i for i in range(4) if np.linalg.norm(projection[:, i]-np.eye(4)[:, i]) < 1e-7]
        targets = np.log([[v['duration_s'], v['role_mean_power_w']['attention'],
                           v['role_mean_power_w']['expert']] for _, v in rows])
        slopes = np.linalg.pinv(dx) @ (targets-targets.mean(axis=0))
        slopes[[i for i in range(4) if i not in identified], :] = 0
        query = np.array([math.log(configuration(candidate)[k]) for k in KNOBS])-center
        prediction = targets.mean(axis=0) + query @ slopes
        log_energy = float(prediction[0] + np.logaddexp(prediction[1], prediction[2]))
        residual = float(np.sqrt(np.mean((targets-targets.mean(axis=0)-dx@slopes)**2)))
        outside = float(np.linalg.norm(np.maximum(x.min(axis=0)-(query+center), 0)
                                       + np.maximum((query+center)-x.max(axis=0), 0)))
        missing = float(np.linalg.norm(query[[i for i in range(4) if i not in identified]]))
        candidate['prior'] = dict(log_energy=log_energy, energy_sd=max(.35, residual)+outside+missing,
                                  source=MODEL)
        reports[key] = dict(observations=len(rows), identified_knobs=[KNOBS[i] for i in identified],
                            slopes=slopes.tolist(), residual_log_rms=residual)
    return result, dict(model=MODEL, selection_split='calibration', structures=reports,
                        stage_times_available=False, actual_microbatch_splits_available=False,
                        extrapolation='wide uncertainty outside same-structure measured controls')
