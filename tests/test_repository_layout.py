"""Relocation boundaries: public CLI, source pins, CWD and README navigation."""
from pathlib import Path
import json,os,re,subprocess,sys,tempfile,unittest

ROOT=Path(__file__).resolve().parents[1]

class RepositoryLayout(unittest.TestCase):
    def test_root_has_readme_ignore_and_license(self):
        self.assertEqual({p.name for p in ROOT.iterdir() if p.is_file()},{'README.md','.gitignore','LICENSE'})

    def test_selected_research_sources_keep_their_exact_bytes(self):
        import hashlib
        records=json.loads((ROOT/'docs/provenance/SOURCE_MANIFEST.json').read_text(encoding='utf8'))
        self.assertEqual(len(records),136)
        for record in records:
            with self.subTest(path=record['destination']):
                self.assertEqual(hashlib.sha256((ROOT/record['destination']).read_bytes()).hexdigest(),record['sha256'])

    def test_absolute_cli_paths_work_from_unrelated_cwd_without_pythonpath(self):
        env=os.environ.copy();env.pop('PYTHONPATH',None);env.pop('ICTC_RUN_CONFIG',None)
        with tempfile.TemporaryDirectory(prefix='ictc foreign cwd ') as temp:
            for name in ('smoke.py','portable_run.py','fresh_chain.py','verify_release.py'):
                with self.subTest(name=name):
                    result=subprocess.run([sys.executable,'-B',str(ROOT/'scripts'/name),'--help'],cwd=temp,env=env,capture_output=True,text=True)
                    self.assertEqual(result.returncode,0,result.stderr)
                    self.assertIn('usage:',result.stdout)
            self.assertEqual(list(Path(temp).iterdir()),[])

    def test_default_distribution_audit_is_independent_of_cwd(self):
        env=os.environ.copy();env.pop('PYTHONPATH',None)
        with tempfile.TemporaryDirectory(prefix='ictc audit cwd ') as temp:
            # A decoy in CWD must never replace the packaged source ledger.
            decoy=Path(temp)/'SOURCE_MANIFEST.json';decoy.write_text('not valid JSON')
            result=subprocess.run([sys.executable,'-B',str(ROOT/'scripts/verify_release.py')],cwd=temp,env=env,capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr+result.stdout)
            self.assertEqual(json.loads(result.stdout)['source_files_checked'],136)
            self.assertEqual(decoy.read_text(),'not valid JSON')

    def test_readme_navigation_and_local_links_resolve(self):
        md=(ROOT/'README.md').read_text(encoding='utf8')
        headings={re.sub(r'[^a-z0-9 -]','',line.lstrip('# ').lower()).replace(' ','-') for line in md.splitlines() if line.startswith('#')}
        headings.add('top')
        self.assertIn('<a name="top"></a>',md)
        links=re.findall(r'!?\[[^\]]*\]\(([^)]+)\)',md)
        self.assertGreaterEqual(len([x for x in links if x.startswith('#')]),9)
        for link in links:
            if '://' in link:continue
            if link.startswith('#'):self.assertIn(link[1:],headings)
            else:self.assertTrue((ROOT/link.split('#',1)[0]).is_file(),link)
        self.assertIn('python -B scripts/run_tests.py',md)
        self.assertIn('python -B scripts/verify_release.py',md)

if __name__=='__main__':unittest.main()

