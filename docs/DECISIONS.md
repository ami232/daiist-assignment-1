# Decisions recorded before final evaluation

The student selected these choices in the project discussion on 2026-10-06:

1. Bank marketing and call prioritization.
2. Prioritize customers before a planned call using information available beforehand.
3. Hypothetical costs: EUR 2 for each call and EUR 50 for each missed subscriber.
4. Full date-ordered dataset and chronological evaluation on later campaigns.
5. Exclude duration; engineer prior-contact status and history because returning customers may respond differently.
6. Test a loan combination and count of unknown customer fields, because commitments and incomplete records may distinguish response patterns.
7. English explanatory PDF.

Implementation details: 60/20/20 contiguous row blocks; training-only preprocessing; regularization candidates 0.001, 0.01, 0.1; fixed objective; validation log loss for model configuration and validation cost for the policy threshold. Test metrics must not be used to revise these decisions.

Conservative availability exclusions also remove final-call channel/month/weekday and completed-campaign contact count. Economic indicators are excluded because release-time and revision-vintage information is not provided. Previous-campaign history and customer attributes are assumed available in the CRM before a new call.

AI assists implementation and the separate study guide. The student's own interpretation is required for the analytical and conclusion sections of REPORT.md. No fork push, upstream pull request, Blackboard entry or upload is authorized.

After reviewing validation-only evidence, the student chose the engineered features with correct sentinel semantics over the slightly lower-loss raw control. Select the lowest validation log loss among engineered candidates; retain all raw-control scores for honest comparison. This decision was made without presenting test model metrics.

Data inspection found 4,110 rows with pdays=999 and previous>0. The engineered indicator was renamed to recency_recorded to describe its actual meaning without asserting no historical contact. Its numerical values and all model choices remain unchanged; the conflict is retained and documented.
