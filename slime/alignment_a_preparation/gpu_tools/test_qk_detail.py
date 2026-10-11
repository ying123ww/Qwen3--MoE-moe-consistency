from summarize_qk_detail import summarize
from summarize_remaining_inference import summarize as basic_summary


def event(name, cat, ts, dur, **args):
    return dict(name=name,cat=cat,ph='X',ts=ts,dur=dur,pid=1,tid=2,args=args)


def test_separates_memory_events_and_nested_launch_api_layers():
    rows = [event('step[DECODE]', 'user_annotation', 0, 20),
            event('P1/q_norm_native/layer=0', 'user_annotation', 1, 10),
            event('aten::pow', 'cpu_op', 2, 4, **{'External id': 5}),
            event('cudaLaunchKernel', 'cuda_runtime', 3, 2, correlation=7),
            event('cuLaunchKernel', 'cuda_driver', 3.5, 1, correlation=7),
            event('square', 'kernel', 25, 2, correlation=7, **{'External id': 5}),
            event('Memcpy DtoD', 'gpu_memcpy', 30, 1),
            event('Memset', 'gpu_memset', 32, 1)]
    d = summarize(rows)
    assert d['gpu_event_categories'] == dict(kernel=1,gpu_memcpy=1,gpu_memset=1)
    assert d['kernel_launch_api_counts'] == {'cuda_runtime:cudaLaunchKernel':1,'cuda_driver:cuLaunchKernel':1}
    assert d['gpu_components']['q_norm_native:decode']['count'] == 1
    assert d['gpu_components']['q_norm_native:decode']['operators'][0]['name'] == 'kernel:aten::pow:square'
    assert d['qk_cpu_ops']['q_norm_native:decode'][0]['name'] == 'aten::pow'


def test_missing_external_ids_require_unique_enclosing_mm_on_launch_thread():
    rows = [event('P1/attention_output/layer=0', 'user_annotation', 0, 20),
            event('aten::mm','cpu_op',1,10,**{'External id':7,'Input Dims':[[4,2048],[2048,2048]]}),
            event('cuLaunchKernelEx','cuda_driver',3,3,correlation=99),
            event('matmul_kernel_persistent','kernel',30,4,correlation=99,device=0)]
    d = basic_summary(rows)
    assert d['unlinked_persistent_matmul'] == 0
    assert d['gpu_families']['attention_output_matmul']['count'] == 1
    assert d['recovered_matrix_links'][0]['cpu_external_id'] == 7
    # Ambiguous nesting must remain explicitly unlinked.
    rows.append(event('aten::mm','cpu_op',0.5,12,**{'External id':8,'Input Dims':[[4,2048],[2048,2048]]}))
    assert basic_summary(rows)['unlinked_persistent_matmul'] == 1
