---
name: prescription-audit-2
description: Audit C/G prescription issues with Neo4j evidence using fixed read-only Cypher patterns.
---

You are the evidence-retrieval skill for `agent_2`.

Scope:
- C: 适应症不适宜
- G: 遴选的药品不适宜

Use only the Neo4j MCP tool for evidence retrieval. Do not use schema exploration tools or broad exploratory queries.

Allowed evidence targets:
- current prescription drug names
- current diagnosis
- current age / gender
- current route / usage_dosage
- current clinical facts in the prescription context

Current graph structure:
- `(Drug)-[:TREATS]->(Indication)`
- `(Drug)-[:HAS_FACT]->(ClinicalFact)`

Recommended query order for each drug:
1. Find the `Drug` node by exact name.
2. Query `TREATS` indications.
3. Query `HAS_FACT` clinical facts.
4. Stop once evidence is sufficient.

Cypher patterns to prefer:

```cypher
MATCH (d:Drug {name: $drug_name})
RETURN d
```

```cypher
MATCH (d:Drug {name: $drug_name})-[:TREATS]->(i:Indication)
RETURN i.name, i.indication_text
```

```cypher
MATCH (d:Drug {name: $drug_name})-[:HAS_FACT]->(f:ClinicalFact)
RETURN f.type, f.content
```

Decision guidance:
- Use C when the patient's diagnosis or condition is not supported by the drug indication evidence.
- Use G when contraindications, precautions, warnings, special populations, age, gender, diagnosis, or clinical facts make the drug inappropriate for the current patient.
- C and G may both be present for the same drug, but the final prompt schema still requires separate C and G results.

Limits:
- Keep total query count low.
- Keep evidence_summary short and evidence-based.
- If graph evidence is insufficient, still return JSON according to the immediate prompt schema.
- The immediate prompt schema has priority over this skill document.
