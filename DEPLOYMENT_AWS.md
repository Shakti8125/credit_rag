# Deploying CreditRAG on AWS (Free Tier)

This guide deploys the **cloud tier only** (`cloud/backend/` + `shared/`). The local tier (Streamlit + privacy pipeline + Phi-3) is designed to run on the analyst's machine and should **not** be deployed — moving it to the cloud would defeat the on-device masking guarantee, and the 2.2 GB GGUF model won't fit free-tier compute anyway.

The codebase is already AWS-ready:

- [app/main.py](cloud/backend/app/main.py) exposes `handler = Mangum(app)` for Lambda.
- [app/services/secrets.py](cloud/backend/app/services/secrets.py) reads secrets from **SSM Parameter Store** (region `ap-south-1`), falling back to env vars locally.
- [app/services/dynamo.py](cloud/backend/app/services/dynamo.py) logs telemetry to DynamoDB table `CreditRAG_Telemetry` (optional; degrades gracefully).
- [app/routes/health.py](cloud/backend/app/routes/health.py) is a liveness probe usable for warmup pings.
- [cloud/infra/template.yaml](cloud/infra/template.yaml) and [.github/workflows/deploy.yml](.github/workflows/deploy.yml) are currently **empty stubs** — this guide gives you the manual path plus a SAM template to fill the stub.

External dependencies (not AWS, both have free tiers): **Pinecone** (serverless starter) and **Google Gemini** (AI Studio free tier).

---

## Recommended Architecture: Lambda Container + Function URL

```
Analyst machine (Streamlit, CLOUD_API_BASE=<Function URL>)
        │  HTTPS (masked payloads only)
        ▼
Lambda Function URL  ── free, no API Gateway needed
        ▼
Lambda (container image from ECR, 1024–2048 MB)
   ├── SSM Parameter Store  (GEMINI_API_KEY, PINECONE_API_KEY)
   ├── Pinecone  (regulatory corpus, external)
   ├── Gemini API  (generation, external)
   └── DynamoDB CreditRAG_Telemetry  (optional)
```

**Why this shape for free tier:**

| Component | Free-tier coverage |
|---|---|
| Lambda | 1 M requests + 400,000 GB-seconds/month, **always free**. At 1536 MB that's ~74 hours of compute/month — far more than a demo needs. |
| Lambda Function URL | Free (vs. API Gateway, which is only free for 12 months). Gives you a public HTTPS endpoint with zero extra infrastructure. |
| SSM Parameter Store | Standard parameters: free. |
| DynamoDB | 25 GB + 25 RCU/WCU always free. |
| CloudWatch Logs | 5 GB ingestion/month free. |
| ECR | 500 MB private storage free. **This is the one place you'll exceed free tier** — the image (torch + sentence-transformers + baked-in models) is ~2–3 GB, costing roughly **$0.10/GB-month ≈ $0.20–0.30/month**. Effectively free, but not literally zero. |

The honest caveats up front:

1. **Cold starts will be slow** (30–60 s: image pull + cross-encoder/embedder load). Fine for a demo; mitigate with a warmup ping (below). Don't pay for provisioned concurrency on free tier.
2. **`/compare` and `/ews` can run long.** Set Lambda timeout to the max relevant value (up to 900 s; Function URLs cap responses at ~15 min). The frontend already uses generous request timeouts.
3. **Region:** `secrets.py` and `dynamo.py` hardcode `ap-south-1` (Mumbai). Deploy there, or change those two defaults.

---

## Step 0 — Prerequisites

- AWS account (new accounts get the 12-month free tier on top of always-free services), AWS CLI v2 configured (`aws configure`, region `ap-south-1`), Docker running.
- Pinecone index already populated (run `1_extract_and_chunk.py` + `2_embed_and_upload.py` once from your machine — ingestion never needs to run in AWS).

## Step 1 — Store secrets in SSM Parameter Store

```bash
aws ssm put-parameter --name GEMINI_API_KEY   --type SecureString --value "YOUR_GEMINI_KEY"   --region ap-south-1
aws ssm put-parameter --name PINECONE_API_KEY --type SecureString --value "YOUR_PINECONE_KEY" --region ap-south-1
```

`get_secret()` looks up parameters **by these exact names**. SecureString uses the default AWS-managed KMS key — free.

Note: `retrieval.py` reads `PINECONE_API_KEY` via `get_env` (plain env), not via `get_secret`. Easiest fix without code changes: pass it as a Lambda environment variable (Step 4). `GEMINI_API_KEY` goes through SSM properly.

## Step 2 — Build a Lambda-compatible image

The existing [Dockerfile](cloud/backend/Dockerfile) targets uvicorn. Its header comment already describes the Lambda variant; create `cloud/backend/Dockerfile.lambda`:

```dockerfile
FROM public.ecr.aws/lambda/python:3.11

COPY cloud/backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# shared/ next to app/ so `import shared` and `import app` both resolve
COPY shared/   ${LAMBDA_TASK_ROOT}/shared/
COPY cloud/backend/app/ ${LAMBDA_TASK_ROOT}/app/

# Bake models into the image so cold starts don't download them.
# HF cache must be somewhere readable at runtime; /tmp is wiped, so use a
# baked path and point HF_HOME at it.
ENV HF_HOME=/opt/hf-cache
RUN python -c "from sentence_transformers import CrossEncoder, SentenceTransformer; \
    CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2'); \
    SentenceTransformer('all-MiniLM-L6-v2')"

CMD ["app.main.handler"]
```

Build **from the project root** (so `shared/` is in context):

```bash
docker build -f cloud/backend/Dockerfile.lambda -t creditrag-backend:lambda .
```

Tip to shave image size (and ECR cost): install CPU-only torch by adding `--extra-index-url https://download.pytorch.org/whl/cpu` to the pip install, and drop `docling` from the deployed requirements (it's only used by the ingestion scripts, which run on your machine).

## Step 3 — Push to ECR

```bash
aws ecr create-repository --repository-name creditrag-backend --region ap-south-1

ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
aws ecr get-login-password --region ap-south-1 | docker login --username AWS --password-stdin ${ACCOUNT_ID}.dkr.ecr.ap-south-1.amazonaws.com

docker tag creditrag-backend:lambda ${ACCOUNT_ID}.dkr.ecr.ap-south-1.amazonaws.com/creditrag-backend:latest
docker push ${ACCOUNT_ID}.dkr.ecr.ap-south-1.amazonaws.com/creditrag-backend:latest
```

## Step 4 — Create the Lambda function

Execution role (basic logs + SSM read):

```bash
aws iam create-role --role-name creditrag-lambda-role \
  --assume-role-policy-document '{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":{"Service":"lambda.amazonaws.com"},"Action":"sts:AssumeRole"}]}'

aws iam attach-role-policy --role-name creditrag-lambda-role \
  --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole

aws iam put-role-policy --role-name creditrag-lambda-role --policy-name ssm-read \
  --policy-document '{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Action":["ssm:GetParameter"],"Resource":"*"}]}'
```

Function (1536 MB is the sweet spot: enough RAM for the reranker + embedder, and more memory = proportionally more CPU = faster model load):

```bash
aws lambda create-function \
  --function-name creditrag-backend \
  --package-type Image \
  --code ImageUri=${ACCOUNT_ID}.dkr.ecr.ap-south-1.amazonaws.com/creditrag-backend:latest \
  --role arn:aws:iam::${ACCOUNT_ID}:role/creditrag-lambda-role \
  --memory-size 1536 --timeout 300 \
  --environment "Variables={PINECONE_API_KEY=YOUR_PINECONE_KEY,PINECONE_INDEX_NAME=creditrag,GEMINI_MODEL=gemini-2.5-flash,CORS_ALLOW_ORIGINS=*,HF_HOME=/opt/hf-cache}" \
  --region ap-south-1
```

(If you'd rather not put the Pinecone key in env vars, add a `get_secret()` call for it in `retrieval.py` — a two-line change.)

## Step 5 — Public HTTPS endpoint (Function URL)

```bash
aws lambda create-function-url-config \
  --function-name creditrag-backend \
  --auth-type NONE \
  --region ap-south-1

aws lambda add-permission \
  --function-name creditrag-backend \
  --statement-id public-url --action lambda:InvokeFunctionUrl \
  --principal "*" --function-url-auth-type NONE \
  --region ap-south-1
```

This returns a URL like `https://xxxx.lambda-url.ap-south-1.on.aws/`. Test it:

```bash
curl https://xxxx.lambda-url.ap-south-1.on.aws/health
curl -X POST https://xxxx.lambda-url.ap-south-1.on.aws/query \
  -H "Content-Type: application/json" \
  -d '{"query":"What is the minimum CET1 ratio under Basel III?","intent":"BENCHMARK"}'
```

> `--auth-type NONE` makes the endpoint public. For anything beyond a demo, switch to `AWS_IAM` or at minimum set `CORS_ALLOW_ORIGINS` to your real origin. Remember: by design the payloads are already masked, which limits the blast radius — but a public endpoint still spends your Gemini/Pinecone quota.

## Step 6 — Point the frontend at it

On the analyst machine:

```bash
# in .env at project root
CLOUD_API_BASE=https://xxxx.lambda-url.ap-south-1.on.aws
```

[local/app/config.py](local/app/config.py) derives `/query`, `/compare`, `/ews` from this base automatically.

## Step 7 — Optional extras (both within always-free tier)

**Telemetry table** (used by `dynamo.py` if present; logs intent/path/latency only, never text):

```bash
aws dynamodb create-table --table-name CreditRAG_Telemetry \
  --attribute-definitions AttributeName=transaction_id,AttributeType=S \
  --key-schema AttributeName=transaction_id,KeyType=HASH \
  --billing-mode PAY_PER_REQUEST --region ap-south-1
```

Also attach `dynamodb:PutItem` on that table to the Lambda role.

**Warmup ping** to soften cold starts (~9 k invocations/month, negligible against the 1 M free):

```bash
aws events put-rule --name creditrag-warmup --schedule-expression "rate(5 minutes)" --region ap-south-1
aws lambda add-permission --function-name creditrag-backend --statement-id warmup \
  --action lambda:InvokeFunction --principal events.amazonaws.com \
  --source-arn arn:aws:events:ap-south-1:${ACCOUNT_ID}:rule/creditrag-warmup --region ap-south-1
aws events put-targets --rule creditrag-warmup --region ap-south-1 \
  --targets "Id=1,Arn=arn:aws:lambda:ap-south-1:${ACCOUNT_ID}:function:creditrag-backend,Input='{\"warmup\":true}'"
```

(The raw EventBridge payload isn't an HTTP event, so Mangum will log an error and exit — that's fine, the point is keeping the container warm. For a clean warmup, target the `/health` route via a scheduled `curl` from anywhere instead.)

---

## Filling the SAM stub (`cloud/infra/template.yaml`)

Once the manual path works, codify it so redeploys are one command. Drop this into the currently empty [template.yaml](cloud/infra/template.yaml):

```yaml
AWSTemplateFormatVersion: '2010-09-09'
Transform: AWS::Serverless-2016-10-31
Description: CreditRAG cloud tier — Lambda container + Function URL

Globals:
  Function:
    Timeout: 300
    MemorySize: 1536

Resources:
  Backend:
    Type: AWS::Serverless::Function
    Properties:
      PackageType: Image
      ImageUri: !Sub "${AWS::AccountId}.dkr.ecr.${AWS::Region}.amazonaws.com/creditrag-backend:latest"
      FunctionUrlConfig:
        AuthType: NONE
      Environment:
        Variables:
          PINECONE_INDEX_NAME: creditrag
          GEMINI_MODEL: gemini-2.5-flash
          HF_HOME: /opt/hf-cache
      Policies:
        - Statement:
            - Effect: Allow
              Action: [ssm:GetParameter]
              Resource: !Sub "arn:aws:ssm:${AWS::Region}:${AWS::AccountId}:parameter/*"
            - Effect: Allow
              Action: [dynamodb:PutItem]
              Resource: !Sub "arn:aws:dynamodb:${AWS::Region}:${AWS::AccountId}:table/CreditRAG_Telemetry"

Outputs:
  FunctionUrl:
    Value: !GetAtt BackendUrl.FunctionUrl
```

Deploy with `sam deploy --guided` (writes [samconfig.toml](cloud/infra/samconfig.toml)); the GitHub Actions stub ([deploy.yml](.github/workflows/deploy.yml)) can then be `docker build → ecr push → sam deploy` on pushes to `main`, gated on `python eval/privacy_eval.py` passing.

---

## Alternative: single EC2 instance (12-month free tier)

If you'd rather avoid cold starts entirely: one `t3.micro`/`t2.micro` (750 hrs/month free for the first 12 months) running the **existing** [Dockerfile](cloud/backend/Dockerfile) as-is:

```bash
docker build -f cloud/backend/Dockerfile -t creditrag-backend .
docker run -d --restart unless-stopped --env-file .env -p 80:8000 creditrag-backend
```

Trade-offs: 1 GB RAM is tight for torch + sentence-transformers — add a 2 GB swap file (`fallocate -l 2G /swapfile …`) and expect the reranker to be slow; no HTTPS out of the box (put it behind Caddy for a free Let's Encrypt cert, or use Cloudflare); and after 12 months a t3.micro is ~$8/month vs. Lambda staying effectively free. **Recommendation: Lambda.** The workload is bursty and analyst-driven — exactly what Lambda's always-free tier is for.

## What NOT to deploy

- `local/` — the whole point is on-device masking; deploying it centralizes raw PII.
- `base_documents/` + ingestion scripts — run once from your machine; Pinecone holds the result.
- Any managed vector DB on AWS (OpenSearch, pgvector on RDS) — none of them fit the free tier as comfortably as Pinecone's own starter plan, and the code targets Pinecone.

## Monthly cost summary (steady demo usage)

| Item | Cost |
|---|---|
| Lambda compute + requests | $0 (within always-free) |
| Function URL, SSM, DynamoDB, CloudWatch (≤5 GB logs) | $0 |
| ECR image storage (~2–3 GB) | ~$0.20–0.30 |
| Pinecone serverless starter | $0 |
| Gemini AI Studio free tier | $0 (rate-limited) |
| **Total** | **≈ $0.30/month** |
