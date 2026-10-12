"""Storage-address lifecycle mechanism; CPU checks are not the online RL gate."""
import hashlib
import json


def tensor_address(tensor):
    return dict(data_ptr=tensor.data_ptr(),storage_ptr=tensor.untyped_storage().data_ptr(),storage_offset=tensor.storage_offset(),shape=list(tensor.shape),stride=list(tensor.stride()),dtype=str(tensor.dtype),device=str(tensor.device))


def parameter_addresses(model):
    return {name:tensor_address(value) for name,value in model.named_parameters()}


def address_digest(values):
    return hashlib.sha256(json.dumps(values,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def changed_addresses(before,after):
    return sorted(name for name in set(before)|set(after) if before.get(name)!=after.get(name))


class PointerLedger:
    def __init__(self):
        self.captured=None;self.asleep=False;self.invalidated=True;self.value_updates=0;self.capture_generation=0;self.weight_values_ready=True
    def capture(self,addresses):
        if self.asleep:raise RuntimeError('Cannot capture while memory is offloaded')
        self.captured=addresses;self.invalidated=False;self.capture_generation+=1
    def observe_update(self,addresses):
        self.value_updates+=1
        changes=changed_addresses(self.captured or {},addresses)
        if changes:self.invalidated=True
        return changes
    def release(self,weights_discarded=False):
        self.asleep=True
        if weights_discarded:self.weight_values_ready=False
    def mark_weights_restored(self):self.weight_values_ready=True
    def resume(self,addresses):
        self.asleep=False;changes=changed_addresses(self.captured or {},addresses)
        if changes:self.invalidated=True
        return changes
    def require_replay(self):
        if self.asleep:raise RuntimeError('Graph replay rejected while captured memory is offloaded')
        if not self.weight_values_ready:raise RuntimeError('Graph replay rejected before discarded weights are restored')
        if self.invalidated:raise RuntimeError('Graph replay rejected after storage-address change; recapture required')
