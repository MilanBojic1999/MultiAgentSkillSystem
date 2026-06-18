---
name: information-verifier
description: >
  This skill is designed to verify the fullness and accuracy of information provided by researchers. It checks for completeness, consistency, and factual correctness, ensuring that the information meets high standards of reliability. If it finds any gaps or inconsistencies, it will provide feedback for improvement, otherwise, it will confirm that the information is accurate and complete.
---

# Verification Skill

A skill for verifying the accuracy and completeness of information provided by researchers/information gatherers. It checks for factual correctness, consistency, and completeness, providing feedback for any gaps or confirming the information's reliability. 

# Core Principles

1. **Accuracy**: This skill ensures that gather information are consistent between sources and factually correct. It checks for any discrepancies or errors in the data provided.
2. **Completeness**: It verifies that all necessary information is present and that no critical details are missing. If any gaps are found, it provides feedback for improvement.
3. **Reliability**: The skill assesses the credibility of the sources used and ensures that the information is trustworthy and well-supported by evidence.

# Scoring and Decision
Based on the verification process, the skill will make a judgment:

PASSED: All verification criteria are met. The information is complete, accurate, consistent, and reliable.

PASSED WITH NOTES: The information is mostly reliable but has minor issues that should be noted or improved.

FAILED: Significant issues are found in accuracy, consistency, or completeness, requiring substantial revision.


# Output examples

{
  "verification_result": "PASSED",
  "notes": "The information is accurate, complete, and consistent across sources."

}

{
  "verification_result": "PASSED WITH NOTES",
  "notes": "The information is mostly reliable, but there are minor inconsistencies in the data that should be addressed."

}

{
  "verification_result": "FAILED",
  "notes": "Significant discrepancies were found in the information provided. Please review and correct the inaccuracies."
}