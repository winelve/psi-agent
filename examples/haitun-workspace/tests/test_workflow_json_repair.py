from __future__ import annotations

import sys
from importlib import import_module
from pathlib import Path

import pytest

TOOLS_DIR = Path(__file__).parents[1] / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

workflow_tool = import_module("run_flow")


def test_agent_step_result_repairs_small_json_syntax_errors_before_model_retry() -> None:
    malformed = """{
      'candidate_assessments': {
        'resume_summary': ['- 独立完成项目', '- 熟练使用 Python',],
      },
    }"""

    result = workflow_tool._parse_agent_step_result(
        malformed,
        step_id="analyze_resume_step",
        output_ids=("candidate_assessments",),
    )

    assert result == {
        "candidate_assessments": {
            "resume_summary": ["- 独立完成项目", "- 熟练使用 Python"],
        }
    }


@pytest.mark.parametrize(
    "ambiguous",
    [
        '{"candidate_assessments": NaN}',
        '{"candidate_assessments": {}, "candidate_assessments": []}',
    ],
)
def test_agent_step_result_does_not_repair_semantically_ambiguous_json(ambiguous: str) -> None:
    with pytest.raises(workflow_tool._AgentStepResultParseError):
        workflow_tool._parse_agent_step_result(
            ambiguous,
            step_id="analyze_resume_step",
            output_ids=("candidate_assessments",),
        )
