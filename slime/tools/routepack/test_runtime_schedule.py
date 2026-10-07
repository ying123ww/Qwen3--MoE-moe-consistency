"""Counterexamples for the runtime schedule acceptance boundary; CPU-only."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from runtime_schedule import make_manifest, apply_schedule


class RuntimeScheduleTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name)/'manifest.json'
        self.lengths = [2]*8
        self.tokens = [[i, i+20] for i in range(8)]
        self.native = ([[0, 1, 4, 5], [2, 3, 6, 7]],
                       [[[0], [1], [2], [3]], [[0], [1], [2], [3]]], [2, 2], [4, 4])
        cells = []
        for step in range(2):
            for rank in range(2):
                for slot in range(2):
                    i = step*4+rank*2+slot
                    cells.append(dict(step=step, group_id=0, source_rank=rank+10, slot=slot, sample_ids=[f'opaque-{i}']))
        self.manifest = make_manifest({'cells':cells}, {'cells':copy.deepcopy(cells)},
                sample_to_global_index={f'opaque-{i}':i for i in range(8)},
                source_rank_to_dp_rank={10:0, 11:1}, total_lengths=self.lengths,
                token_sequences=self.tokens, rollout_indices=list(range(8)),pad_multiplier=4,max_tokens_per_bin=8)

    def run_adapter(self):
        self.path.write_text(json.dumps(self.manifest))
        return apply_schedule(self.native,total_lengths=self.lengths,token_sequences=self.tokens,
                rollout_indices=list(range(8)),dp_size=2,max_tokens_per_bin=8,pad_multiplier=4,path=self.path)

    def test_native_exact_and_no_mutation(self):
        self.manifest['mode']='native'
        original=copy.deepcopy(self.native)
        self.assertEqual(self.run_adapter(),self.native)
        self.assertEqual(self.native,original)

    def test_valid_cross_slot_complete_sample(self):
        c=self.manifest['candidate_cells'];c[0]['sample_ids'],c[3]['sample_ids']=c[3]['sample_ids'],c[0]['sample_ids']
        out=self.run_adapter()
        self.assertEqual(out[0][0],[3,1,4,5]);self.assertEqual(out[0][1],[2,0,6,7])
        self.assertEqual(out[2:],self.native[2:])

    def test_invalid_counterexamples(self):
        mutations = [
            lambda m: m['candidate_cells'][0].update(sample_ids=['opaque-1']),
            lambda m: m['candidate_cells'].pop(),
            lambda m: m['candidate_cells'].append(copy.deepcopy(m['candidate_cells'][0])),
            lambda m: m['candidate_cells'][0].update(group_id=1),
            lambda m: m['candidate_cells'][0].update(sample_ids=['unknown']),
            lambda m: m['sample_to_global_index'].update({'opaque-0':1}),
            lambda m: m['source_rank_to_dp_rank'].update({'10':1}),
            lambda m: m['native_cells'][0].update(sample_ids=['opaque-1']),
            lambda m: m.update(pad_multiplier=8),
            lambda m: m['input_identity']['tokens_sha256'].__setitem__(0,'wrong'),
        ]
        for mutate in mutations:
            with self.subTest(mutation=mutate):
                backup=copy.deepcopy(self.manifest);mutate(self.manifest)
                with self.assertRaises(ValueError):self.run_adapter()
                self.manifest=backup

    def test_cross_step_rejected(self):
        c=self.manifest['candidate_cells'];c[0]['sample_ids'],c[4]['sample_ids']=c[4]['sample_ids'],c[0]['sample_ids']
        with self.assertRaisesRegex(ValueError,'crosses optimizer'):self.run_adapter()

    def test_native_label_cannot_hide_swap(self):
        self.manifest['mode']='native';c=self.manifest['candidate_cells']
        c[0]['sample_ids'],c[3]['sample_ids']=c[3]['sample_ids'],c[0]['sample_ids']
        with self.assertRaisesRegex(ValueError,'Native mode'):self.run_adapter()

    def test_same_lengths_different_tokens_rejected(self):
        self.tokens[0][0]=999
        with self.assertRaisesRegex(ValueError,'Stale input'):self.run_adapter()

    def test_cell_count_rejected(self):
        c=self.manifest['candidate_cells'];c[0]['sample_ids']+=c[1]['sample_ids'];c[1]['sample_ids']=[]
        with self.assertRaises(ValueError):self.run_adapter()

    def test_padding_and_capacity_counterexamples(self):
        # Bins [1,3] and [1,3] have zero padding; swapping 3 and 1 yields 2 and 6, adds4.
        lengths=[1,3,1,3];tokens=[[i]*n for i,n in enumerate(lengths)]
        base={'cells':[dict(step=0,group_id=0,source_rank=0,slot=s,sample_ids=ids)
                       for s,ids in enumerate([['a','b'],['c','d']])]}
        candidate=copy.deepcopy(base);candidate['cells'][0]['sample_ids']=['a','c'];candidate['cells'][1]['sample_ids']=['b','d']
        for capacity,error in [(8,'native cell padded capacity'),(4,'capacity')]:
            m=make_manifest(base,candidate,sample_to_global_index=dict(zip('abcd',range(4))),source_rank_to_dp_rank={0:0},
                total_lengths=lengths,token_sequences=tokens,rollout_indices=list(range(4)),pad_multiplier=4,max_tokens_per_bin=capacity)
            self.path.write_text(json.dumps(m))
            with self.assertRaisesRegex(ValueError,error):
                apply_schedule(([[0,1,2,3]],[[[0,1],[2,3]]],[2],[4]),total_lengths=lengths,token_sequences=tokens,
                    rollout_indices=list(range(4)),dp_size=1,max_tokens_per_bin=capacity,pad_multiplier=4,path=self.path)


    def test_native_cell_cap_even_when_total_padding_unchanged(self):
        lengths=[1,3,3,5]; tokens=[[i]*n for i,n in enumerate(lengths)]
        base={'cells':[dict(step=0,group_id=0,source_rank=0,slot=s,sample_ids=ids)
                       for s,ids in enumerate([['a','b'],['c','d']])]}
        candidate=copy.deepcopy(base)
        candidate['cells'][0]['sample_ids']=['d','b']
        candidate['cells'][1]['sample_ids']=['c','a']
        m=make_manifest(base,candidate,sample_to_global_index=dict(zip('abcd',range(4))),source_rank_to_dp_rank={0:0},
            total_lengths=lengths,token_sequences=tokens,rollout_indices=list(range(4)),pad_multiplier=4,max_tokens_per_bin=8)
        self.path.write_text(json.dumps(m))
        with self.assertRaisesRegex(ValueError,'native cell padded capacity'):
            apply_schedule(([[0,1,2,3]],[[[0,1],[2,3]]],[2],[4]),total_lengths=lengths,token_sequences=tokens,
                rollout_indices=list(range(4)),dp_size=1,max_tokens_per_bin=8,pad_multiplier=4,path=self.path)


if __name__=='__main__':unittest.main()
