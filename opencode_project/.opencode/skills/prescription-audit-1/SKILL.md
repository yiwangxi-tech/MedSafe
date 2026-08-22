---
name: prescription-audit-1
description: Audit B/D prescription issues with Neo4j evidence using fixed read-only Cypher patterns.
---

You are the evidence-retrieval skill for `agent_1`.

Scope:
- B: 用法、用量不适宜
- D: 药品剂型或给药途径不适宜

Use only the Neo4j MCP tool for evidence retrieval. Do not use schema exploration tools or broad exploratory queries.

Allowed evidence targets:
- current prescription drug names
- current diagnosis
- current age / gender
- current route / usage_dosage

Current graph structure:
- `(Drug)-[:HAS_PLAN]->(DosagePlan)`

Recommended query order for each drug:
1. Find the `Drug` node by exact name.
2. Query `HAS_PLAN` dosage plans.
3. Stop once evidence is sufficient.

Cypher patterns to prefer:

```cypher
MATCH (d:Drug {name: $drug_name})
RETURN d
```

```cypher
MATCH (d:Drug {name: $drug_name})-[:HAS_PLAN]->(p:DosagePlan)
RETURN
  p.route,
  p.min_dose,
  p.max_dose,
  p.dose_unit,
  p.min_freq,
  p.max_freq,
  p.freq_unit,
  p.min_age,
  p.max_age,
  p.age_unit,
  p.max_daily_dose,
  p.max_daily_unit,
  p.population,
  p.lab_test,
  p.lab_unit,
  p.lab_min,
  p.lab_max,
  p.indication_text,
  p.remark
```

Decision guidance:
- Use B when dose, frequency, usage timing, usage method, daily dose, special-population limit, or lab-related dose limit is outside supported evidence.
- Use D when administration route or dosage form conflicts with supported evidence or prescription context.
- Treat every `DosagePlan` node as a conditional rule. Prefer the node whose properties best match the patient's explicit diagnosis, age, gender, route, usage_dosage, and available clinical context.
- Do not mark B only by estimated body weight when body weight is absent.
- B and D may share the same evidence, but the final prompt schema still requires separate B and D results.

Limits:
- Keep total query count low.
- Keep evidence_summary short and evidence-based.
- If graph evidence is insufficient, still return JSON according to the immediate prompt schema.
- The immediate prompt schema has priority over this skill document.
