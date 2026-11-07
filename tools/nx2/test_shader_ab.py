#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later

import json
from pathlib import Path
import tempfile
import unittest

import shader_ab


class ExpansionTests(unittest.TestCase):
    def test_variables_expand_and_unset_names_fail(self):
        self.assertEqual(shader_ab.expand("${A}/x/$B", {"A": "/a", "B": "b"}), "/a/x/b")
        with self.assertRaises(shader_ab.CaseError):
            shader_ab.expand("${MISSING}/x", {})

    def test_relative_paths_resolve_against_the_case_file(self):
        base = Path("/cases").resolve()
        self.assertEqual(shader_ab.resolve_path("a.spv", base, {}), base / "a.spv")
        self.assertEqual(shader_ab.resolve_path("/abs/a.spv", base, {}), Path("/abs/a.spv"))


class ModuleSpecTests(unittest.TestCase):
    def test_strings_pick_glsl_by_extension(self):
        self.assertEqual(shader_ab.module_spec("a.comp", "n")["glsl"], "a.comp")
        self.assertEqual(shader_ab.module_spec("a.spv", "n")["spv"], "a.spv")

    def test_names_come_from_defines_or_default(self):
        spec = shader_ab.module_spec({"glsl": "a.comp", "defines": {"V": 1, "K": 8}}, "n")
        self.assertEqual(spec["name"], "K=8,V=1")
        self.assertEqual(shader_ab.module_spec({"spv": "a.spv"}, "n")["name"], "n")
        with self.assertRaises(shader_ab.CaseError):
            shader_ab.module_spec({"spv": "a", "glsl": "b"}, "n")

    def test_glsl_command_carries_defines_and_target(self):
        command = shader_ab.glsl_command(Path("a.comp"), Path("o.spv"), {"V": 1, "F": None},
                                         ["-Os"])
        self.assertEqual(command[1:5], ["-V", "--target-env", "vulkan1.3", "-DF"])
        self.assertEqual(command[5:], ["-DV=1", "-Os", "-o", "o.spv", "a.comp"])


class RunnerArgumentTests(unittest.TestCase):
    def test_case_fields_map_to_runner_flags(self):
        case = {"dispatch": [4, 2, 1], "entry": "main", "cases": 16, "show": 2,
                "buffers": [{"set": 0, "binding": 1, "size": 64, "fill": "random-half",
                             "compare": "f16-ulp", "tolerance": 1, "end": 32}],
                "push": "deadbeef", "timing": {"repeats": 100, "samples": 5}}
        args = shader_ab.runner_arguments(case, "a.spv", "b.spv", 32, Path("/c"), {},
                                          {"validate": True})
        self.assertEqual(args, [
            "--baseline", "a.spv", "--candidate", "b.spv", "--dispatch", "4,2,1",
            "--subgroup", "32", "--entry", "main", "--buffer",
            "set=0,binding=1,size=64,fill=random-half,compare=f16-ulp,tolerance=1,end=32",
            "--push", "hex=deadbeef", "--cases", "16", "--timing", "100", "--samples", "5",
            "--show", "2", "--validate"])

    def test_overrides_replace_case_values(self):
        case = {"cases": 16, "timing": 100}
        args = shader_ab.runner_arguments(case, "a", "b", 0, Path("/c"), {},
                                          {"cases": 2, "timing": 0})
        self.assertNotIn("--subgroup", args)
        self.assertNotIn("--timing", args)
        self.assertEqual(args[args.index("--cases") + 1], "2")

    def test_unknown_buffer_keys_are_rejected(self):
        with self.assertRaises(shader_ab.CaseError):
            shader_ab.buffer_argument({"set": 0, "binding": 0, "sizes": 4}, Path("/c"), {})


class WordsTests(unittest.TestCase):
    def test_lists_and_offset_tables_build_little_endian_images(self):
        self.assertEqual(shader_ab.words_image([1, "0x80000000"]),
                         bytes([1, 0, 0, 0, 0, 0, 0, 0x80]))
        image = shader_ab.words_image({"0x10": [7], 4: ["2"]})
        self.assertEqual(len(image), 20)
        self.assertEqual(image[4], 2)
        self.assertEqual(image[16], 7)
        with self.assertRaises(shader_ab.CaseError):
            shader_ab.words_image({"3": [1]})

    def test_words_become_a_cached_file_argument(self):
        with tempfile.TemporaryDirectory() as directory:
            argument = shader_ab.buffer_argument(
                {"set": 0, "binding": 0, "size": 16, "words": [1, 2]}, Path("/c"), {}, directory)
            self.assertTrue(argument.startswith("set=0,binding=0,size=16,file="))
            path = Path(argument.split("file=")[1])
            self.assertEqual(path.read_bytes(), bytes([1, 0, 0, 0, 2, 0, 0, 0]))
            with self.assertRaises(shader_ab.CaseError):
                shader_ab.buffer_argument({"set": 0, "binding": 0, "words": [1], "fill": "zero"},
                                          Path("/c"), {}, directory)


class BaselineTests(unittest.TestCase):
    def test_default_variables_only_fill_unset_names_with_tagged_directories(self):
        original = (shader_ab.BASELINE_DIR, shader_ab.CANDIDATE_DIR)
        with tempfile.TemporaryDirectory() as directory:
            shader_ab.BASELINE_DIR = Path(directory) / "baseline"
            shader_ab.CANDIDATE_DIR = Path(directory) / "candidate"
            try:
                self.assertEqual(shader_ab.default_variables({}), {})
                shader_ab.BASELINE_DIR.mkdir()
                (shader_ab.BASELINE_DIR / "fixtures.json").write_text(json.dumps(
                    {"label": "baseline", "describe": "abc", "dirty": True, "fixtures": ["a.spv"],
                     "time": "t"}))
                variables = shader_ab.default_variables({"AFTER": "/x"})
                self.assertEqual(variables, {"BEFORE": str(shader_ab.BASELINE_DIR), "AFTER": "/x"})
                self.assertEqual(shader_ab.describe_fixtures(shader_ab.BASELINE_DIR),
                                 "baseline: abc (dirty), 1 fixtures, t")
                self.assertIsNone(shader_ab.describe_fixtures(shader_ab.CANDIDATE_DIR))
            finally:
                shader_ab.BASELINE_DIR, shader_ab.CANDIDATE_DIR = original


class CaseFileTests(unittest.TestCase):
    def test_single_object_or_list_and_required_fields(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "x.json"
            path.write_text(json.dumps({"baseline": "a.spv", "candidate": "b.spv"}))
            cases = shader_ab.load_cases(path)
            self.assertEqual(cases[0]["name"], "x#0")
            self.assertEqual([c["name"] for c in shader_ab.candidates_of(cases[0])],
                             ["candidate"])
            path.write_text(json.dumps([{"baseline": "a.spv"}]))
            with self.assertRaises(shader_ab.CaseError):
                shader_ab.load_cases(path)

    def test_subgroup_lists_and_overrides(self):
        self.assertEqual(shader_ab.subgroups_of({"subgroup": [32, 64]}, None), [32, 64])
        self.assertEqual(shader_ab.subgroups_of({}, None), [0])
        self.assertEqual(shader_ab.subgroups_of({"subgroup": [32, 64]}, 64), [64])


class ReportTests(unittest.TestCase):
    def test_outcome_honours_expectations(self):
        self.assertEqual(shader_ab.outcome({"mismatches": 0}, "pass"), ("PASS", True))
        self.assertEqual(shader_ab.outcome({"mismatches": 3}, "pass"), ("FAIL", False))
        self.assertEqual(shader_ab.outcome({"mismatches": 3}, "fail"), ("DETECTED", True))
        self.assertEqual(shader_ab.outcome({"mismatches": 0}, "fail"), ("UNDETECTED", False))
        self.assertEqual(shader_ab.outcome({"mismatches": 0, "validation_errors": 1}, "pass"),
                         ("FAIL", False))

    def test_table_includes_timing_and_errors(self):
        report = {"subgroup": 32, "compared": 8, "max_distance": 0.0, "pipeline_ms": [1.0, 2.5],
                  "timing": {"baseline": {"median": 10.0}, "candidate": {"median": 5.0},
                             "ratio": 0.5}}
        text = shader_ab.format_rows([
            {"case": "c", "candidate": "k", "subgroup": 32, "result": "PASS", "report": report},
            {"case": "c", "candidate": "e", "subgroup": 0, "report": None}])
        self.assertIn("PASS", text)
        self.assertIn("0.500", text)
        self.assertIn("1.0/2.5", text)
        self.assertIn("ERROR", text)


if __name__ == "__main__":
    unittest.main()
