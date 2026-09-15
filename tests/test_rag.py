"""
Unit tests for Manufacturing Standards RAG Engine and Audit Severity Index Calculator.
"""

import unittest
from core.standards_kb import get_all_standards, get_standard_by_defect_type
from core.rag_engine import StandardsRAGEngine
from core.audit_reporter import AuditReporter


class TestStandardsRAG(unittest.TestCase):
    def setUp(self):
        self.rag = StandardsRAGEngine()
        self.reporter = AuditReporter()

    def test_standards_db_integrity(self):
        standards = get_all_standards()
        self.assertGreaterEqual(len(standards), 6)
        bridge = get_standard_by_defect_type("Solder_Bridge")
        self.assertIsNotNone(bridge)
        self.assertEqual(bridge["id"], "IPC-610-5.2.1")
        self.assertIn("Critical", bridge["severity_level"])

    def test_bm25_search_retrieval(self):
        results = self.rag.search_standards("tombstoning component lifted", top_k=2)
        self.assertGreater(len(results), 0)
        top_match = results[0]
        self.assertIn("Tombstoning", top_match["defect_type"])

    def test_audit_synthesis(self):
        finding = self.rag.synthesize_audit_finding(
            defect_type="Solder_Bridge",
            confidence=0.98,
            operating_class="Class 3"
        )
        self.assertEqual(finding["defect_type"], "Solder_Bridge")
        self.assertEqual(finding["clause_id"], "IPC-610-5.2.1")
        self.assertIn("DEFECT", finding["disposition"])
        self.assertGreaterEqual(finding["severity_score"], 80.0)
        self.assertIn("Procedure 3.1.2", finding["ipc_7721_rework"])

    def test_standards_conversational_qa(self):
        qa_res = self.rag.answer_standards_query(
            user_question="What is the rework procedure for a solder bridge on an IC?",
            operating_class="Class 3"
        )
        self.assertIn("answer", qa_res)
        self.assertIn("IPC-7711/7721", qa_res["answer"])
        self.assertIn("5.2.1", qa_res["primary_clause"]["section"])

    def test_audit_severity_index_calculation(self):
        asi_critical = self.reporter.calculate_audit_severity_index(
            confidence=0.95,
            risk_factor=0.98,
            operating_class="Class 3",
            defect_area_ratio=0.08
        )
        self.assertGreaterEqual(asi_critical["score"], 70.0)
        self.assertIn("CRITICAL", asi_critical["level"])

        asi_pass = self.reporter.calculate_audit_severity_index(
            confidence=0.99,
            risk_factor=0.0,
            operating_class="Class 3",
            defect_area_ratio=0.0
        )
        self.assertLess(asi_pass["score"], 60.0)


if __name__ == "__main__":
    unittest.main()
