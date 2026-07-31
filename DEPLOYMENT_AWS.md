# Deploying CreditRAG on AWS

This deploys the **cloud tier only** (`cloud/backend/` + `shared/`). The local tier — Streamlit,
privacy pipeline, Phi-3 — is designed to run on the analyst's machine and must **not** be
deployed: moving it to the cloud would defeat the on-device masking guarantee, and the 2.2 GB
GGUF won't fit free-tier compute anyway.

This document describes the deployment that is **actually running**, not a proposal. Every
command here was executed against a real account. Region is `ap-south-1` (Mumbai).

External dependencies (not AWS, both free-tier): **Pinecone** serverless and **Google Gemini**
(AI Studio).

---

## Architecture

```
Analyst machine (Streamlit)
        │  HTTPS + X-API-Key, masked payloads only
        ▼
Lambda Function URL  (AuthType NONE; auth enforced in-app)
        ▼
Lambda container, 2048 MB / 300 s, from ECR
   ├── SSM Parameter Store  GEMINI_API_KEY, PINECONE_API_KEY, CREDITRAG_API_KEY
   ├── Pinecone             regulatory corpus, ns "cbuae-manuals"
   ├── Gemini API           generation
   └── DynamoDB             CreditRAG_Telemetry (intent/path/latency only)

git push to main ──▶ GitHub Actions (OIDC) ──▶ build ──▶ ECR ──▶ update-function-code
```

Building in Actions means the ~2 GB image never crosses a home connection, and no local Docker
daemon is needed.

### Measured characteristics

| | |
|---|---|
| Image size (compressed in ECR) | 767 MB |
| Full CI pipeline (evals + build + push) | ~4.5 min |
| Cold start (image pull + 2 model loads) | ~25 s |
| Warm `/health` | ~0.34 s |
| Warm `/query` end-to-end (BENCHMARK) | 8–15 s, p95 14.6 s |

---

## Four traps, and why they bite

These cost real debugging time. They are the reason this guide exists in this form.

### 1. The base image must be `python:3.12`, not `python:3.11`

Lambda's `python:3.11` base image runs on **Amazon Linux 2 (glibc 2.26)**. AL2023 (glibc 2.34)
arrives only with `python:3.12`.

numpy, faiss-cpu and tiktoken now publish x86_64 Linux wheels exclusively under
`manylinux_2_27` / `manylinux_2_28` tags, which need glibc ≥ 2.27. On the 3.11 image **not one of
them is installable**, so pip silently falls back to source tarballs and the build dies in meson
with `Unknown compiler(s)` — Lambda base images carry no C toolchain. Pinning older package
versions works today and rots quickly; the whole scientific stack has moved to the 2.28 floor.

### 2. pip must be upgraded before anything is installed

The base image ships pip 24.0, which predates `Metadata-Version 2.4` and treats wheels carrying
it as incompatible — the same silent sdist fallback by a different route. `pip install --upgrade
pip` is the first instruction in the Dockerfile for this reason.

### 3. A public Function URL needs **two** policy statements

> Starting October 2025, new function URLs require both `lambda:InvokeFunctionUrl` **and**
> `lambda:InvokeFunction`.

With only the first, every request returns `403 AccessDeniedException` — before reaching your
code — even though `AuthType` is `NONE` and the policy looks textbook-correct. The two statements
use **different CLI flags**; `--function-url-auth-type` is rejected for the `InvokeFunction`
action.

### 4. `.dockerignore` is mandatory here

Both Dockerfiles build from the project root, where the context is **3.8 GB** (2.3 GB Phi-3 GGUF,
1.6 GB `.git`, 17 MB of source PDFs). Without [.dockerignore](.dockerignore), every build ships
all of it to the builder before the first instruction runs.

---

## Files that make this work

| File | Role |
|---|---|
| [.dockerignore](.dockerignore) | Deny-all + allow-list for `shared/` and `cloud/backend/{app,requirements*}`. |
| [Dockerfile.lambda](cloud/backend/Dockerfile.lambda) | AL2023 base, CPU-only torch installed **before** sentence-transformers so pip never resolves the ~2.5 GB CUDA wheel, cross-encoder + MiniLM baked in, `HF_HUB_OFFLINE=1` so cold starts do no network I/O against a read-only filesystem. |
| [requirements-lambda.txt](cloud/backend/requirements-lambda.txt) | Drops `docling` (ingestion-only, pulls torchvision) and `uvicorn` (nothing imports it; Mangum is the adapter). |
| [app/auth.py](cloud/backend/app/auth.py) | `X-API-Key` shared secret from SSM, constant-time compared. Self-disables when unset, so local dev and offline evals are unaffected. |
| [deploy.yml](.github/workflows/deploy.yml) | OIDC → ECR → Lambda, gated on the privacy + adversarial masking evals. |

---

## Step 0 — Prerequisites

- AWS account with an IAM admin user (**not root**), root MFA enabled, AWS CLI v2 configured for
  `ap-south-1`.
- Pinecone index already populated — run `1_extract_and_chunk.py` then `2_embed_and_upload.py`
  once from your machine. Ingestion never runs in AWS. Verify:

  ```bash
  python -c "from pinecone import Pinecone; from shared.env import get_env; \
    print(Pinecone(api_key=get_env('PINECONE_API_KEY')).Index('creditrag').describe_index_stats())"
  ```

  Expect ~1,566 vectors, dimension 1024, namespace `cbuae-manuals`.

## Step 1 — Budget alarm first

Before anything can spend. `$5` cap, alerts at 20 % actual and at forecast-to-exceed:

```bash
aws budgets create-budget --account-id <ACCOUNT_ID> \
  --budget file://budget.json --notifications-with-subscribers file://notifications.json
```

## Step 2 — Secrets in SSM Parameter Store

```bash
for n in GEMINI_API_KEY PINECONE_API_KEY CREDITRAG_API_KEY; do
  aws ssm put-parameter --name $n --type SecureString --value "..." --region ap-south-1
done
```

`CREDITRAG_API_KEY` is a fresh 32-byte random token — the shared secret for the `X-API-Key`
header. Generate it, don't reuse anything. SecureString uses the default AWS-managed KMS key, free.

Both [generation.py](cloud/backend/app/services/generation.py) and
[retrieval.py](cloud/backend/app/services/retrieval.py) resolve their keys through
[`get_secret()`](cloud/backend/app/services/secrets.py), which reads the environment first and
falls back to SSM — so no key ever needs to sit in a plaintext Lambda environment variable, and
local development is unchanged.

## Step 3 — ECR, DynamoDB, IAM

```bash
aws ecr create-repository --repository-name creditrag-backend --region ap-south-1
aws ecr put-lifecycle-policy --repository-name creditrag-backend \
  --lifecycle-policy-text file://ecr-lifecycle.json   # keep last 3 images
```

The lifecycle policy matters: the image is ~767 MB and ECR's free tier is 500 MB, so unbounded
history is the one thing here that would actually cost money.

```bash
aws dynamodb create-table --table-name CreditRAG_Telemetry \
  --attribute-definitions AttributeName=LogId,AttributeType=S \
  --key-schema AttributeName=LogId,KeyType=HASH \
  --billing-mode PAY_PER_REQUEST --region ap-south-1
```

> The partition key **must** be `LogId` — that is what
> [dynamo.py](cloud/backend/app/services/dynamo.py) writes. Any other key name makes every write
> fail with `ValidationException`.

Two roles:

- **`creditrag-lambda-role`** — `AWSLambdaBasicExecutionRole`, plus inline `ssm:GetParameter`
  scoped to the three parameter ARNs, `kms:Decrypt` conditioned on `kms:ViaService = ssm`, and
  `dynamodb:PutItem` on the telemetry table only.
- **`creditrag-github-deploy`** — trusted by the GitHub OIDC provider
  (`token.actions.githubusercontent.com`), scoped to `repo:<owner>/<repo>:*`, permitted only to
  push this ECR repository and update this one function. No long-lived AWS keys in GitHub.

  On the Lambda side it needs `lambda:GetFunction`, **`lambda:GetFunctionConfiguration`**,
  `lambda:UpdateFunctionCode` and `lambda:PublishVersion`. `GetFunctionConfiguration` is easy to
  miss — it is a *separate* IAM action from `GetFunction`, and it is what `aws lambda wait
  function-updated` calls. Without it the deploy step fails **after** the code has already been
  updated, which looks alarming and isn't.

## Step 4 — Build and push (GitHub Actions)

Set repo variables under **Settings → Secrets and variables → Actions → Variables**:
`AWS_ROLE_ARN`, `AWS_REGION`, `ECR_REPOSITORY`, `LAMBDA_FUNCTION`, and `FUNCTION_URL` (optional,
enables the CI smoke test). Then push to `main` or run the workflow manually.

> `workflow_dispatch` only appears once the workflow file exists on the **default branch**.

The workflow sets `provenance: false`. Without it buildx pushes an OCI image index, which Lambda
rejects with *"The image manifest, config or layer media type for the source image is not
supported."*

## Step 5 — Create the function and its URL

```bash
aws lambda create-function --function-name creditrag-backend \
  --package-type Image --code ImageUri=<ACCOUNT>.dkr.ecr.ap-south-1.amazonaws.com/creditrag-backend:<sha> \
  --role arn:aws:iam::<ACCOUNT>:role/creditrag-lambda-role \
  --memory-size 2048 --timeout 300 --architectures x86_64 \
  --environment "Variables={PINECONE_INDEX_NAME=creditrag,GEMINI_MODEL=gemini-2.5-flash,CORS_ALLOW_ORIGINS=*,TELEMETRY_TABLE=CreditRAG_Telemetry}" \
  --region ap-south-1
```

- **2048 MB** — Lambda scales CPU with memory, and cold start here is dominated by loading two
  transformer models.
- **300 s** deliberately matches the client-side `timeout=300` in the compare/EWS handlers.
- **`TELEMETRY_TABLE`** is what switches telemetry on;
  [engines.py](cloud/backend/app/services/engines.py) builds no DynamoDB client when it is unset,
  so developer machines never attempt a doomed write.
- Do **not** set `AWS_REGION` — it is reserved and injected automatically.

Then the URL, **both** statements (see trap 3):

```bash
aws lambda create-function-url-config --function-name creditrag-backend \
  --auth-type NONE --region ap-south-1

aws lambda add-permission --function-name creditrag-backend \
  --statement-id UrlPolicyInvokeURL --action lambda:InvokeFunctionUrl \
  --principal "*" --function-url-auth-type NONE --region ap-south-1

aws lambda add-permission --function-name creditrag-backend \
  --statement-id UrlPolicyInvokeFunction --action lambda:InvokeFunction \
  --principal "*" --invoked-via-function-url --region ap-south-1
```

`AuthType NONE` is safe here because the `X-API-Key` check lives inside the application. IAM auth
would force the analyst machine to carry AWS credentials and sign requests, which it deliberately
does not.

## Step 6 — Point the frontend at it

In the project-root `.env` (gitignored):

```
CLOUD_API_BASE=https://xxxx.lambda-url.ap-south-1.on.aws
CLOUD_API_KEY=<the CREDITRAG_API_KEY value>
```

[config.py](local/app/config.py) derives `/query`, `/compare`, `/ews` and the `X-API-Key` header
from these. Ensure the file ends with a newline — appending to a file without one silently welds
the new variable onto the previous line.

---

## Verification

```bash
curl $URL/health                                   # 200, ~25 s cold / ~0.3 s warm
curl -X POST $URL/query -H 'Content-Type: application/json' \
     --data-binary @query.json                     # 401 — no key
curl -X POST $URL/query -H 'Content-Type: application/json' \
     -H "X-API-Key: $KEY" --data-binary @query.json  # 200 + grounded answer

aws logs tail /aws/lambda/creditrag-backend --follow
aws dynamodb scan --table-name CreditRAG_Telemetry --max-items 5
python eval/ragas_eval.py --api $URL --sleep 13
```

Telemetry rows must contain only `Intent`, `ExecutionPath`, `LatencyMs`, `LogId`, `Timestamp` —
never payload text. `/compare` and `/ews` deliberately log a *fixed* path string rather than the
one they return, because the returned path embeds client-supplied document labels, which
[adversarial_eval.py](eval/adversarial_eval.py) treats as a PII carrier.

### Gemini free tier will fail the eval gate

`gemini-2.5-flash` on AI Studio allows **20 generate-content requests per day**. The 14-question
golden set plus any manual testing exceeds that, surfacing as HTTP 500s and an `error_rate` gate
failure that is **environmental, not a defect**. Confirm by checking for `429 RESOURCE_EXHAUSTED`
in CloudWatch before investigating anything else. Run the full suite on a fresh day, or on a paid
key.

Last recorded run (12 of 14 scored; 2 lost to quota): grounding 0.82, citation coverage 1.00,
expected-term hit rate 1.00, **placeholder leaks 0**, p50 10.0 s / p95 14.6 s.

---

## Cost

| Item | Cost |
|---|---|
| Lambda compute + requests | $0 — always-free tier |
| Function URL, SSM, DynamoDB, CloudWatch (≤5 GB) | $0 |
| ECR storage (767 MB, last-3 lifecycle policy) | ~$0.08–0.30 |
| Pinecone starter, Gemini AI Studio | $0 (rate-limited) |
| **Total** | **≈ $0.30/month** |

## Operational notes

- **Cold starts** are ~25 s after ~15 min idle. If that matters for a demo, add an EventBridge
  rule hitting `/health` every 5 minutes (~9k invocations/month against the 1M free allowance).
- **Function URL payloads cap at 6 MB** each way. `/compare` sends two full masked documents;
  typical markdown memos are 100–300 KB, so there is headroom, but it is not unlimited.
- **Malformed JSON returns 422, not 401.** FastAPI parses the body before solving dependencies,
  so a request with bad JSON is rejected before the API-key check runs. No application logic
  executes either way.
- **Rotating the shared secret**: `aws ssm put-parameter --name CREDITRAG_API_KEY --overwrite`,
  update `CLOUD_API_KEY` in `.env`, then force a new Lambda execution environment (the value is
  cached per process for the life of the container).

## What NOT to deploy

- `local/` — the whole point is on-device masking; deploying it centralises raw PII.
- `base_documents/` and the ingestion scripts — run once from your machine; Pinecone holds the result.
- Any managed vector DB on AWS — none fit the free tier as comfortably as Pinecone's starter plan,
  and the code targets Pinecone.
