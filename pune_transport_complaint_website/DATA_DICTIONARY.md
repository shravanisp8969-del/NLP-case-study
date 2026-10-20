# Dataset data dictionary

| Column | Required | Description | Allowed values |
|---|---|---|---|
| `complaint` | Yes | Anonymized passenger complaint text | Free text |
| `category` | Yes | Manually verified service category | Bus Delay; Overcrowding; Staff Behaviour; Ticketing Issue; Cleanliness; Safety Issue; Service Disruption; Other |
| `sentiment` | Yes | Manually verified sentiment | Positive; Negative; Neutral |
| `priority` | Yes | Manually verified operational priority | High; Medium; Low |

Do not include passenger names, phone numbers, or other unnecessary personal information. Keep an untouched test set for honest evaluation.
