from summarize_matmul import summarize


def test_gpu_correlation_shapes_and_annotation_categories():
    events = [
        dict(cat='user_annotation',name='step[DECODE bs=4]',ts=0,dur=1000),
        dict(cat='gpu_user_annotation',name='step[EXTEND]',ts=0,dur=1000),
        dict(cat='cpu_op',name='aten::mm',ts=10,dur=50,args={'External id':3,'Input Dims':[[4,2048],[2048,128]],'Input type':['float','float']}),
        dict(cat='cuda_driver',name='cuLaunchKernel',ts=20,dur=5,args={'External id':3,'correlation':8}),
        dict(cat='kernel',name='matmul_kernel_persistent',ts=500,dur=25,args={'External id':3,'correlation':8}),
        dict(cat='kernel',name='matmul_kernel_persistent',ts=700,dur=10,args={'External id':9,'correlation':9}),
    ]
    result=summarize(events)
    assert result['kernel_count']==2 and result['unlinked_kernels']==1
    row=result['rows'][0]
    assert row['family']=='router' and row['phase']=='decode'
    assert row['gpu_ms']==.025 and row['cpu_mm_ms']==.05
    assert row['cpu_launch_ms']==.005
