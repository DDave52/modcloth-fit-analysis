# Strategy & Recommendations (Start2Impact Project)
## Fit Mismatch & Sizing Reliability — ModCloth Reviews

This document summarizes the **final, actionable strategy** derived from the analysis in the notebooks (`01_data_understanding` and `02_strategic_eda`).  
It is designed as a stakeholder-friendly deliverable: concise, decision-oriented, and directly connected to evidence.

---

## 1) Executive Summary

- The main issue observed is **fit mismatch** (*small/large* vs *fit*), which varies meaningfully across **size, category, and height segment**.
- The dataset points to immediate opportunities in:
  - **Assortment decisions** (what to promote, limit, or rework)
  - **Sizing UX** (help users pick the correct size)
  - **Return-risk reduction** (intervene where mismatch is systematically high)
  - **Personalization** (different user clusters require different interventions)

---

## 2) Recommended Actions (Prioritized)

### P0 — High impact, low cost (do now)

1. **Sizing risk badge on PDP (product detail page)**
   - Show a *fit reliability* indicator for **category × size × height segment**.
   - If mismatch is high, display a clear message (“runs small/large”) + CTA to sizing guidance.

2. **Default size recommendation with a simple rule**
   - For high-risk segments, recommend *size up/down* when historical distributions support it.
   - Start rule-based; iterate later to a predictive model.

3. **Catalog “hygiene” for problematic segments**
   - Where mismatch is systematically high: improve copy, add real measurements (cm), fabric elasticity, and intended fit guidance.

---

### P1 — High impact, medium cost (1–2 sprints)

1. **Data-driven sizing guide (interactive)**
   - Replace a static chart with guidance computed from real review outcomes.
   - Include “customers like you…” explanations to build trust.

2. **Post-purchase trigger for high-risk profiles**
   - When mismatch risk is high for a user profile: proactive message (e.g., “If you’re between sizes…”).

3. **A/B testing program**
   - KPIs: mismatch rate, return rate, conversion rate, NPS/ratings, customer support contacts.
   - Test: risk badge, size recommendation logic, measurement copy and placement.

---

### P2 — Medium/High impact, medium/high cost (roadmap)

1. **Predictive fit model (classification)**
   - Target: mismatch (0/1) or fit class (fit/small/large).
   - Candidate models: logistic regression, gradient boosting.
   - Features: category, size, height segment, user history, plus (future) brand/material/garment measurements.

2. **Segment-driven merchandising and campaigns**
   - Tailor campaigns, category exposure, and messaging by user cluster.

---

## 3) User Segmentation (KMeans, k = 4)

Below is a compact summary of the clusters (based on `cluster_profile_k4.csv`).

| Cluster | Users | Mismatch rate | Avg. quality | Dominant size | Category diversity | # reviews | Persona |
|---:|---:|---:|---:|---:|---:|---:|---|
| 0 | 2777 | 0.129 | 4.271 | 8.9 | 2.436 | 3.737 | Fit-confident & satisfied |
| 2 | 1057 | 0.308 | 3.939 | 13.4 | 4.106 | 8.837 | Demanding but consistent |
| 1 | 1263 | 0.377 | 3.957 | 26.3 | 2.537 | 3.977 | Mismatch-tolerant |
| 3 | 2005 | 0.544 | 3.418 | 10.2 | 2.446 | 3.698 | High mismatch risk (friction) |

### Cluster-specific strategies

**Cluster 0 — “Fit-confident & satisfied” (low mismatch, high quality)**  
Goal: maximize conversion and retention.  
Actions:
- upsell / cross-sell to adjacent categories
- early access / promos on new drops (high trust)
- reduce checkout friction

**Cluster 2 — “Demanding but consistent” (mid mismatch, many reviews, high category diversity)**  
Goal: guide choice with precision.  
Actions:
- more granular size recommendations (also category-specific)
- highlight materials/fit and real measurements
- incentivize detailed reviews (body measurements + chosen size)

**Cluster 1 — “Mismatch-tolerant” (mid–high mismatch, quality still OK)**  
Goal: reduce mismatch marginally without hurting exploration.  
Actions:
- wider assortment exposure / bundles
- “soft” size suggestions (non-intrusive)

**Cluster 3 — “High mismatch risk (friction)” (high mismatch, lower quality)**  
Goal: reduce mismatch and returns.  
Actions:
- strong warning + sizing guide as primary UI element
- offer alternatives: “similar style with more reliable fit”
- proactive support / easy returns to reduce frustration

---

## 4) Assortment Strategy (What to Promote vs. Review)

- **Promote** category/size segments with:
  - low mismatch
  - high quality ratings
  - sufficient review volume (statistical stability)

- **Review, rework, or limit** segments with systematic high mismatch:
  - improve PDP information (measurements, elasticity, intended fit)
  - update guidance (size up/down rules)
  - consider supplier sizing consistency interventions

- **Pricing and trust**
  - where mismatch risk is high, avoid pushing premium positioning unless you also provide trust signals (clear guidance, guarantees, or frictionless returns).

---

## 5) Notes & Limitations

This project is based on review data, not transactional data. Key missing variables include:
- brand-specific sizing standards
- garment materials and elasticity
- real garment measurements
- body-shape descriptors

Because of this, the strategy focuses on **fit reliability patterns from customer feedback** rather than full causal explanations.

---

## 6) Conclusion

Even without sales data, customer reviews provide a strong signal about **sizing reliability**.  
By targeting the most problematic category–size–profile segments with simple, high-leverage interventions (risk badges, rule-based recommendations, better measurements), an apparel e-commerce platform can plausibly reduce friction and improve customer satisfaction.
