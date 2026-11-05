"""Small synthetic regression checks; no original dataset is required."""

import contextlib
import importlib.util
import io
import random
import sys
import tempfile
import unittest
from pathlib import Path

from rdkit import Chem

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from common import (  # noqa: E402
    DEFAULT_TEMPLATE,
    MESO_MAPS,
    build_core_query_from_template,
    make_relaxed_query,
    open_sdf_writer,
    read_mol_file_by_block,
    read_sdf_file_by_block,
    remove_hs_safely,
)


def load_stage(filename):
    spec = importlib.util.spec_from_file_location(filename.removesuffix(".py"), ROOT / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.template = remove_hs_safely(read_mol_file_by_block(DEFAULT_TEMPLATE))

    def test_template_attachment_maps(self):
        core, attachments = build_core_query_from_template(self.template)
        self.assertEqual(core.GetNumAtoms(), 23)
        self.assertEqual(len(attachments), 11)
        self.assertEqual(len(set(attachments.values())), 11)
        self.assertTrue(all(atom.GetAtomicNum() != 0 for atom in core.GetAtoms()))
        self.assertEqual(core.GetNumConformers(), self.template.GetNumConformers())

    def test_missing_or_duplicate_maps_fail_early(self):
        dummies = [a for a in self.template.GetAtoms() if a.GetAtomicNum() == 0]
        first_map = dummies[0].GetAtomMapNum()
        dummies[0].SetAtomMapNum(dummies[1].GetAtomMapNum())
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            build_core_query_from_template(self.template)
        dummies[0].SetAtomMapNum(0)
        with self.assertRaisesRegex(ValueError, str(first_map)):
            build_core_query_from_template(self.template)

    def test_sdf_blank_titles_properties_and_multiple_records(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "空白标题.sdf"
            with path.open("w", encoding="utf-8") as handle:
                with Chem.SDWriter(handle) as writer:
                    for smi in ("CCO", "c1ccccc1"):
                        mol = Chem.AddHs(Chem.MolFromSmiles(smi))
                        mol.SetProp("_Name", "")
                        mol.SetProp("example_property", "preserved")
                        writer.write(mol)
            mols = read_sdf_file_by_block(path)
            self.assertEqual(len(mols), 2)
            self.assertEqual([m.GetProp("_SourceIndex") for m in mols], ["0", "1"])
            for mol in mols:
                self.assertEqual(mol.GetProp("example_property"), "preserved")
                self.assertFalse(any(a.GetAtomicNum() == 1 for a in mol.GetAtoms()))

    def test_extract_generate_and_filter_roundtrip(self):
        generator = load_stage("03_generate_corrole_from_matched.py")
        filtering = load_stage("02_filter_free_base_corrole_by_fixed_template.py")
        core, attachments = build_core_query_from_template(self.template)
        relaxed = make_relaxed_query(core)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "synthetic.sdf"
            seeds = []
            for meso, beta in (("*c1ccccc1", False), ("*c1ccc(F)cc1", True)):
                mol = Chem.Mol(self.template)
                for site in MESO_MAPS:
                    mol = generator.attach_fragment_to_template(mol, site, Chem.MolFromSmiles(meso))
                if beta:
                    mol = generator.attach_fragment_to_template(mol, 102, Chem.MolFromSmiles("*C"))
                mol = generator.remove_all_remaining_dummy_atoms(mol)
                seeds.append(mol)
            with open_sdf_writer(path) as writer:
                for index, mol in enumerate(seeds):
                    mol.SetProp("_Name", f"synthetic_{index}")
                    writer.write(mol)
            with contextlib.redirect_stdout(io.StringIO()):
                libraries = generator.extract_libraries_from_matched_dataset(
                    Path(directory), core, relaxed, attachments
                )
                random.seed(2026)
                generated, _ = generator.generate_molecules(self.template, libraries, 3)
            self.assertEqual(len(libraries["meso_counter"]), 2)
            self.assertEqual(len(libraries["beta_patterns"]), 2)
            self.assertEqual(len(generated), 3)
            self.assertEqual(len({smi for _, smi in generated}), 3)
            for mol, _ in generated:
                ok, reason, _, _ = filtering.validate_molecule(mol, core, relaxed, attachments)
                self.assertTrue(ok, reason)
                self.assertFalse(any(a.GetAtomicNum() == 0 for a in mol.GetAtoms()))


if __name__ == "__main__":
    unittest.main()
