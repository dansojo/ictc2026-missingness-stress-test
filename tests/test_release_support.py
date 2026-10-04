import tempfile,unittest,hashlib
from pathlib import Path
import release_support as s
import pandas as pd

class BoundaryTests(unittest.TestCase):
    def test_failure_reports_omit_exception_payload(self):
        report=s.failure_summary(KeyError('synthetic-participant-sensitive-values'))
        self.assertEqual(report,{'status':'failed','error_type':'KeyError'})
    def test_credentials_in_test_files_are_rejected(self):
        import verify_release
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            (root/'test_example.py').write_text("key='"+'ghp_'+'x'*36+"'\n")
            self.assertTrue(any('secret' in x for x in verify_release.scan(root)))
    def test_selected_historical_code_pins_are_checked(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'core.py').write_bytes(b'fixed')
            pins={'core.py':hashlib.sha256(b'fixed').hexdigest(),'unselected-report.md':'0'*64}
            self.assertEqual(s.verify_selected_pins(root,pins),1)
            (root/'core.py').write_bytes(b'changed')
            with self.assertRaises(ValueError):s.verify_selected_pins(root,pins)
    def test_distribution_audit_rejects_data_and_credentials(self):
        import verify_release
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'safe.py').write_text('x=1\n')
            self.assertEqual(verify_release.scan(root),[])
            (root/'records.parquet').write_bytes(b'not public')
            self.assertTrue(any('prohibited' in x for x in verify_release.scan(root)))
            (root/'records.parquet').unlink()
            (root/'secret.txt').write_text('-----BEGIN '+'PRIVATE KEY-----\n')
            self.assertTrue(any('secret' in x for x in verify_release.scan(root)))
    def test_task_names_are_not_api_keys(self):
        import verify_release
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            (root/'source.py').write_text("name='task-12-structured-missingness-preregistered-analysis-plan'\n")
            self.assertEqual(verify_release.scan(root),[])
    def test_window_ids_are_scoped_to_participants(self):
        frame=pd.DataFrame({'dataset':['synthetic','synthetic'],'participant':['person-a','person-b'],'window_id':[0,0],'p_observed':[.2,.8]})
        index=s.prediction_index(frame)
        self.assertEqual(index[('synthetic','person-a',0)],.2)
        self.assertEqual(index[('synthetic','person-b',0)],.8)
        with self.assertRaises(ValueError):s.prediction_index(pd.concat([frame,frame.iloc[:1]]))

    def test_reject_input_and_release_overlap_and_existing_output(self):
        with tempfile.TemporaryDirectory() as d:
            base=Path(d); inputs=base/'inputs';code=base/'code';inputs.mkdir();code.mkdir()
            for p in (inputs/'new',inputs,base,code/'new'):
                with self.assertRaises(ValueError):s.fresh_output(p,[inputs],code)
            out=s.fresh_output(base/'private/run',[inputs],code)
            self.assertTrue(out.is_dir())
            with self.assertRaises(FileExistsError):s.fresh_output(out,[inputs],code)

    def test_hash_tamper_and_manifest_escape_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);file=root/'a';file.write_bytes(b'good')
            expected=hashlib.sha256(b'good').hexdigest()
            self.assertEqual(s.checked_file(root,'a',expected),file)
            file.write_bytes(b'bad')
            with self.assertRaises(ValueError):s.checked_file(root,'a',expected)
            with self.assertRaises(ValueError):s.checked_file(root,'../escape',expected)

    def test_comparison_does_not_silently_accept_numerical_or_status_changes(self):
        self.assertEqual(s.compare_scalar(1.,1.),{'exact':True,'absdiff':0.})
        with self.assertRaises(AssertionError):s.compare_scalar(1.,1.001)
        self.assertTrue(s.compare_scalar(float('nan'),float('nan'))['exact'])
        with self.assertRaises(AssertionError):s.compare_scalar(float('nan'),0.)

if __name__=='__main__':unittest.main()
