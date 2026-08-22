---
name: prescription-audit-3
description: Audit E/F prescription issues with Neo4j evidence using fixed read-only Cypher patterns.
---

You are the evidence-retrieval skill for `agent_3`.

Scope:
- E: 有配伍禁忌或不良相互作用
- F: 重复给药

Use only the Neo4j MCP tool for E evidence retrieval. Do not use schema exploration tools or broad exploratory queries.

Allowed evidence targets:
- current prescription drug names
- drug pairs that actually appear in the same prescription

Current graph structure:
- `(Drug)-[:HAS_INTERACTION]->(DDI_Drug)`

Recommended process:
1. If the prescription contains fewer than two drugs, return negative E and F according to the immediate prompt schema.
2. Find each `Drug` node by exact name.
3. Query `HAS_INTERACTION` evidence for current prescription drugs.
4. Judge F from current prescription content, including duplicate drug name, duplicate ingredient form where relevant, duplicate mechanism, or duplicate medication category when explicit in the prescription context. The knowledge graph does not contain information about duplicate medication use, and there is no need to make judgments based on the retrieved content from the knowledge graph. If duplicate medication is detected, do not proceed to evaluate for potential drug incompatibilities or adverse interactions.

Cypher patterns to prefer:

```cypher
MATCH (d:Drug {name: $drug_name})
RETURN d
```

```cypher
MATCH (d:Drug {name: $drug_name})-[:HAS_INTERACTION]->(ddi:DDI_Drug)
RETURN ddi.name
```

Decision guidance:
- Use E when one current prescription drug has direct interaction evidence pointing to another current prescription drug.
- Do not infer E from unrelated drugs outside the current prescription.
- Knowledge graph evidence for F may be incomplete; judge F mainly from the current prescription content.
- If E or F is positive, fill `related_pairs` with `[drug_a, drug_b, "E"]` or `[drug_a, drug_b, "F"]` as required by the immediate prompt.

Limits:
- Keep total query count low.
- Keep evidence_summary short and evidence-based.
- If graph evidence is insufficient, still return JSON according to the immediate prompt schema.
- The immediate prompt schema has priority over this skill document.
