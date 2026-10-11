"""Experiment-only non-deterministic full decode with normal fixed receive capacity."""
import baseline_graph_experiment as baseline
import fixed_capacity_experiment as fixed


def configure(config,backend,evidence,shadow=False):
    if backend not in ('eager','full'):raise ValueError(backend)
    result=dict(config)
    result.pop('deepep_config',None)
    result['enable_deterministic_inference']=False
    return fixed.configure(result,backend,evidence,shadow)


def environment():
    baseline.environment()


def allow_full_configuration(output):
    fixed.allow_full_configuration(output,deterministic=False)
