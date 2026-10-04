"""Publication and CLI boundaries: unreviewed files and disabled assertions fail."""
from pathlib import Path
import hashlib,json,subprocess,sys,tempfile,unittest
import verify_release as audit

ROOT=Path(__file__).resolve().parents[1]

class PublicationGuards(unittest.TestCase):
    def test_optimized_smoke_and_tests_refuse_execution(self):
        for name in ('smoke.py','run_tests.py'):
            with self.subTest(name=name):
                result=subprocess.run([sys.executable,'-O','-B',str(ROOT/'scripts'/name),'--help'],capture_output=True,text=True)
                self.assertNotEqual(result.returncode,0)
                self.assertIn('Assertions must remain enabled',result.stderr)

    def test_only_exact_reviewed_poster_asset_is_allowed(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);posters=root/'docs/assets';posters.mkdir(parents=True)
            pdf=posters/'a0-en.pdf';pdf.write_bytes(b'%PDF-1.7\nsynthetic approved fixture')
            self.assertTrue(any('prohibited' in issue for issue in audit.scan(root)))
            record={'path':'docs/assets/a0-en.pdf','sha256':hashlib.sha256(pdf.read_bytes()).hexdigest(),
                    'bytes':pdf.stat().st_size,'role':'a0_pdf'}
            manifest={'status':'approved','assets':[record]}
            (posters/'posters.json').write_text(json.dumps(manifest))
            self.assertEqual(audit.scan(root),[])
            pdf.write_bytes(b'%PDF-1.7\nchanged after review')
            self.assertTrue(audit.scan(root))

    def test_poster_manifest_cannot_allow_raw_data_or_escape_its_directory(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);posters=root/'docs/assets';posters.mkdir(parents=True);data=root/'records.parquet';data.write_bytes(b'private')
            for name in ('records.parquet','docs/assets/../../records.parquet','/records.parquet','Z:/records.parquet'):
                record={'path':name,'sha256':hashlib.sha256(data.read_bytes()).hexdigest(),
                        'bytes':data.stat().st_size,'role':'a0_pdf'}
                (posters/'posters.json').write_text(json.dumps({'status':'approved','assets':[record]}))
                self.assertTrue(audit.scan(root))

    def test_pending_poster_manifest_cannot_smuggle_unreviewed_assets(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);posters=root/'docs/assets';posters.mkdir(parents=True)
            (posters/'posters.json').write_text(json.dumps({'status':'pending','assets':[]}))
            self.assertEqual(audit.scan(root),[])
            (posters/'preview.png').write_bytes(b'unreviewed')
            self.assertTrue(any('prohibited' in issue for issue in audit.scan(root)))

    def test_malformed_poster_manifest_reports_an_audit_failure(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);posters=root/'docs/assets';posters.mkdir(parents=True)
            for document in ([],None,{'status':'approved','assets':[None]}):
                (posters/'posters.json').write_text(json.dumps(document))
                self.assertTrue(any('poster manifest' in issue for issue in audit.scan(root)))

if __name__=='__main__':unittest.main()
