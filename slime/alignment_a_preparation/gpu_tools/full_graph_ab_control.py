"""Untimed experiment commands through the normal tokenizer control broadcast."""
KEY='_slime_full_graph_audit'
PHASES={'gate':1,'formal':2}
ACTIONS={'reset':0,'flush':1}


def encode(action,phase):
    if action not in ACTIONS or phase not in PHASES:raise ValueError('Unknown audit command')
    return PHASES[phase]*10+ACTIONS[action]


def decode(value):
    if not isinstance(value,(int,float)) or int(value)!=value:raise ValueError('Integral audit command required')
    phase,action=divmod(int(value),10)
    phases={v:k for k,v in PHASES.items()};actions={v:k for k,v in ACTIONS.items()}
    if phase not in phases or action not in actions:raise ValueError('Unknown audit command')
    return actions[action],phases[phase]


def control(engine,action,phase):
    import asyncio
    from sglang.srt.managers.io_struct import SetInternalStateReq
    req=SetInternalStateReq(server_args={KEY:encode(action,phase)})
    replies=engine.loop.run_until_complete(asyncio.wait_for(
        engine.tokenizer_manager.set_internal_state(req),timeout=60))
    if not replies or not all(replies):raise RuntimeError('Experiment control broadcast was not acknowledged')
