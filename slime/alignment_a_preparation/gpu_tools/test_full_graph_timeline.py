import pytest
from summarize_full_graph_timeline import associate,resolve_node,union_ns


def test_union_overlapping_streams_and_nested_calls():
    intervals=[(0,10),(3,6),(8,15),(20,25),(25,25)]
    assert union_ns(intervals)==20
    assert sum(b-a for a,b in intervals)==25
    with pytest.raises(ValueError):union_ns([(5,4)])


def test_source_mapping_is_thread_local_and_restores_parent_after_child():
    points=[(4,1,'child'),(8,1,'parent'),(4,2,'other_thread'),(12,1,'outside')]
    ranges=[(0,10,1,'P11/attention'),(3,6,1,'P11/qkv'),(0,10,2,'P11/experts')]
    assert associate(points,ranges)=={'child':'P11/qkv','parent':'P11/attention','other_thread':'P11/experts'}


def test_cloned_nodes_resolve_original_source_and_reject_cycles():
    assert resolve_node(3,{1:'P11/qkv'},{3:2,2:1})=='P11/qkv'
    assert resolve_node(4,{1:'P11/qkv'},{3:2,2:1}) is None
    with pytest.raises(ValueError):resolve_node(3,{}, {3:2,2:3})


def test_process_local_node_ids_do_not_mix_ranks():
    direct={(100,1):'P11/qkv/rank=0',(200,1):'P11/experts/rank=4'}
    parents={(100,2):(100,1),(200,2):(200,1)}
    assert resolve_node((100,2),direct,parents)=='P11/qkv/rank=0'
    assert resolve_node((200,2),direct,parents)=='P11/experts/rank=4'
