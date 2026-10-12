import torch
import pytest
from pointer_ledger import PointerLedger,parameter_addresses,changed_addresses


def model():return torch.nn.Sequential(torch.nn.Linear(4,3),torch.nn.Linear(3,2))


def test_all_parameters_and_inplace_updates():
    network=model();before=parameter_addresses(network)
    assert len(before)==4
    ledger=PointerLedger();ledger.capture(before)
    with torch.no_grad():
        for p in network.parameters():p.add_(1)
    assert changed_addresses(before,parameter_addresses(network))==[]
    assert ledger.observe_update(parameter_addresses(network))==[]
    ledger.require_replay();assert ledger.value_updates==1


def test_storage_replacement_requires_recapture():
    network=model();ledger=PointerLedger();ledger.capture(parameter_addresses(network))
    network[0].weight.data=network[0].weight.data.clone()
    assert ledger.observe_update(parameter_addresses(network))==['0.weight']
    with pytest.raises(RuntimeError):ledger.require_replay()
    ledger.capture(parameter_addresses(network));ledger.require_replay();assert ledger.capture_generation==2


def test_sleep_wake_and_changed_storage():
    network=model();ledger=PointerLedger();ledger.capture(parameter_addresses(network));ledger.release()
    with pytest.raises(RuntimeError):ledger.require_replay()
    assert ledger.resume(parameter_addresses(network))==[]
    ledger.require_replay();ledger.release()
    network[1].bias.data=network[1].bias.data.clone()
    assert ledger.resume(parameter_addresses(network))==['1.bias']
    with pytest.raises(RuntimeError):ledger.require_replay()


def test_discarded_weights_need_restore_before_same_address_replay():
    network=model();ledger=PointerLedger();ledger.capture(parameter_addresses(network));ledger.release(weights_discarded=True)
    ledger.resume(parameter_addresses(network))
    with pytest.raises(RuntimeError):ledger.require_replay()
    ledger.observe_update(parameter_addresses(network))
    with pytest.raises(RuntimeError):ledger.require_replay()
    ledger.mark_weights_restored();ledger.require_replay()
