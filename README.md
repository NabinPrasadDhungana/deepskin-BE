# DeepSkin Backend

Django REST Framework backend for the DeepSkin doctor-mediated skin lesion
screening system. Implements the 3-role RBAC workflow (Patient / Doctor /
Admin) described in the system design document, with the EfficientNetB2 +
CBAM model wired in as an isolated `mlservice` inference pipeline.

## Hard design constraint (read this first)

**Patients never see the AI's prediction, confidence score, or attention
map — only a doctor's final verdict.** This is enforced at the serializer
level (`cases/serializers.py` — `PatientCaseListSerializer` /
`PatientCaseDetailSerializer` simply have no fields for `ai_confidence`,
`ai_priority`, or `attention_map_image`), not just hidden in the frontend.
`cases/tests.py` asserts this directly — if you add a new patient-facing
serializer, make sure a similar assertion covers it.

## Project layout

```
accounts/    Custom User model (role field), JWT login, doctor account
             management (admin-only), permission classes for RBAC.
cases/       The core workflow: Case, CaseImage, Verdict, Message models;
             patient upload, doctor queue/pickup/verdict, messaging,
             and admin audit/stats endpoints.
mlservice/   Isolated model inference pipeline. cases/views.py only ever
             calls run_inference_on_case(case) — swap the real trained
             model in here without touching any workflow code.
```

## Setup

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt

python manage.py migrate
python manage.py createsuperuser  # create your first Admin account
python manage.py runserver
```

To point at a real trained model instead of the pipeline stub, set:

```bash
export DEEPSKIN_MODEL_PATH=/path/to/final_model_cbam.keras
```

and install the ML extras noted (commented out) in `requirements.txt`
(`tensorflow`, `opencv-python` — only needed once `mlservice/pipeline.py`'s
`generate_attention_map_overlay()` TODO is replaced with the real CBAM
heatmap extraction from the model development notebook).

### Switching to PostgreSQL (production)

SQLite is used by default for local dev/demo. To switch:

```bash
export DEEPSKIN_DB_ENGINE=postgres
export DEEPSKIN_DB_NAME=deepskin
export DEEPSKIN_DB_USER=deepskin
export DEEPSKIN_DB_PASSWORD=...
export DEEPSKIN_DB_HOST=localhost
```

## Running tests

```bash
python manage.py test
```

The workflow tests (`cases/tests.py`) mock `run_inference_on_case` rather
than loading the real model — see `mlservice/tests.py` for why. This keeps
the test suite fast and GPU-independent while still fully exercising the
RBAC and status-transition logic.

## API surface

All endpoints are under `/api/`. Auth uses JWT (`Authorization: Bearer <token>`).

### Auth (`/api/auth/`)
| Endpoint | Method | Who | Purpose |
|---|---|---|---|
| `register/` | POST | Public | Self-registration — always creates a **Patient** |
| `login/` | POST | Public | Returns access+refresh tokens (role embedded in JWT) |
| `login/refresh/` | POST | Public | Refresh an access token |
| `me/` | GET | Any authenticated user | Own profile |
| `doctors/` | GET | Admin | List doctor accounts |
| `doctors/create/` | POST | Admin | Provision a new doctor account |
| `doctors/<id>/deactivate/` | POST | Admin | Deactivate a doctor account |

### Cases (`/api/cases/`)
| Endpoint | Method | Who | Purpose |
|---|---|---|---|
| `mine/` | GET, POST | Patient | List own cases / upload a new case (multipart: `patient_note`, `images`) |
| `mine/<id>/` | GET | Patient | Own case detail — no AI fields |
| `queue/` | GET | Doctor | Unassigned cases, sorted by AI priority (coarse bucket only) |
| `<id>/pickup/` | POST | Doctor | Self-assign a case |
| `<id>/detail/` | GET | Doctor | Full case detail incl. confidence + attention map (assigned doctor only) |
| `mine-as-doctor/` | GET | Doctor | Cases this doctor has picked up |
| `<id>/verdict/` | POST | Doctor | Record structured decision, marks case Reviewed |
| `patient-history/<patient_id>/` | GET | Doctor | A patient's past reviewed cases |
| `<id>/messages/` | GET, POST | Patient (owner) + Doctor (assigned) | Async message thread |
| `admin/audit/` | GET | Admin | Every case, full detail |
| `admin/stats/` | GET | Admin | Queue length, avg. time-to-review, case volume |

## Known scope gaps (flagged, not silently skipped)

- **Synchronous ML inference.** `PatientCaseListCreateView.create()` calls
  the model inline. For production this belongs on a background task
  queue (Celery) so uploads return instantly. Flagged in code comments.
- **Attention-map generation is a stub.** `generate_attention_map_overlay()`
  currently just resizes the original image. Swap in the real CBAM
  extraction from the model notebook once the trained model is deployed.
- **DullRazor preprocessing is not run inline** at inference time (only
  resize + cast). If this creates a measurable accuracy gap vs. the
  training-time pipeline, port it into `mlservice/pipeline.py`.
