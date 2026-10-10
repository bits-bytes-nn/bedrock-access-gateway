# Security

This document details the security configuration required for the solution. In particular, it covers:

- **HTTPS on the ALB**
- **Image URL Fetching (SSRF)**

Following these guidelines will help ensure that traffic is encrypted over the public network.

---

## 1. HTTPS on the ALB

The API Gateway + Lambda option is HTTPS only. The ALB + Fargate option serves plain HTTP unless you give it a
certificate, and over HTTP the API key in the `Authorization` header travels unencrypted.

To serve HTTPS:

1. **Request a certificate** in AWS Certificate Manager (ACM), in the Region of the stack, for a domain you
   control (for example `bedrock.example.com`). DNS validation is preferred. Wait until the status is **Issued**.
2. **Set `CertificateArn`** to the certificate ARN when you create or update the Fargate stack. The stack then
   adds an HTTPS listener on port 443 (TLS 1.2 and 1.3 only), opens port 443 in the ALB security group and makes
   the port 80 listener redirect to HTTPS.
3. **Point the domain at the ALB**: create a CNAME (or a Route 53 alias record) from the certificate's domain to
   the `ALBDnsName` stack output.

Then use the base URL:

```
https://<your domain>/api/v1
```

Requests to the ALB's own DNS name over HTTPS fail certificate validation, because the certificate is for your
domain.

---

## 2. Image URL Fetching (SSRF)

### Overview

The multimodal API accepts an `image_url` content part that is either an inline base64 `data:` URL or a
remote URL. When a remote URL is given, the gateway makes the outbound request itself, so a caller who
can reach the API can influence where the gateway connects. Without validation, that is a server-side
request forgery (SSRF) vector: a request could point the gateway at endpoints only reachable from the
task or instance, and the fetched bytes would be sent to the model and returned in the completion.
On ECS/Fargate the notable targets are:

- the container credential endpoint, `169.254.170.2`, which serves the task role's temporary credentials
- the EC2 instance metadata service, `169.254.169.254`
- `localhost` and any VPC-internal service reachable from the task's subnet and security group

On Lambda, credentials are passed as environment variables rather than served over HTTP, but `localhost`
still exposes the Lambda Runtime API (`127.0.0.1:9001`) and the Lambda Web Adapter, and a VPC-attached
function can reach VPC-internal services.

### What the gateway enforces

Remote image URLs are validated in [`src/api/image_url.py`](../src/api/image_url.py) before any request
is made. A URL is only fetched when all of the following hold:

| Check | Behaviour |
|---|---|
| Scheme | Only `http` and `https`. `file:`, `gopher:`, `ftp:`, `dict:` and others are rejected. |
| URL form | URLs containing credentials (`user@host`), backslashes, whitespace or control characters are rejected, because URL parsers disagree on which host such URLs name. |
| Address | Every address the host resolves to must be globally routable. This blocks the link-local range that serves both metadata endpoints, as well as loopback, private, shared address space, reserved and multicast ranges, including their IPv4-mapped, IPv4-compatible, 6to4 and NAT64 IPv6 forms. |
| Connection | Opened to the exact address that passed the check, with the original host name used for the `Host` header and TLS certificate verification. The host is not resolved again, so DNS rebinding cannot swap in an internal address. Proxy environment variables (`HTTP_PROXY`, `HTTPS_PROXY`) are not used. |
| Redirects | Followed manually, up to 3 hops, and re-validated at every hop so a public URL cannot redirect to an internal one. |
| Response size | Capped, so a large or endless response cannot exhaust task memory. |
| Time | Each fetch, redirects included, must finish within 60 seconds, and a request may reference at most 20 remote image URLs, so a slow server cannot hold a worker. |
| Content type | Normalised to a format Bedrock accepts (`image/jpeg`, `image/png`, `image/gif`, `image/webp`). |

Rejected requests return `400` with a generic message, so error responses cannot be used to probe which
internal hosts exist. An image over the size cap returns `413`.

### Recommended configuration

Any authenticated caller can still make the gateway fetch an arbitrary public URL. To reduce that, apply
the controls that fit your deployment:

```bash
# Only allow images from hosts you trust. Strongest option; recommended if you know the hosts.
IMAGE_URL_ALLOWED_HOSTS=images.example.com,cdn.example.com

# Or refuse remote URLs entirely and require callers to send base64 data URLs.
ENABLE_IMAGE_URL_FETCH=false

# Lower the response size cap from the 10 MB default if your images are smaller.
IMAGE_URL_MAX_SIZE_MB=5
```

For the Lambda stack, set them with the `EnableImageUrlFetch` and `ImageUrlAllowedHosts` stack parameters, so a
later stack update does not undo them. For the Fargate stack, set them as environment variables on the task
definition.

In addition:

- **Scope the task role.** This is the control that limits the damage if the credential endpoint is ever
  reached. The Lambda template grants the role only `bedrock:InvokeModel`,
  `bedrock:InvokeModelWithResponseStream`, `bedrock:ListFoundationModels`,
  `bedrock:ListInferenceProfiles` and `secretsmanager:GetSecretValue` on the API key secret (plus
  CloudWatch Logs). Keep it that narrow; credentials obtained through any future SSRF-style issue then grant
  nothing beyond model invocation.
- **Restrict egress** to limit which VPC-internal services and internet hosts the task can reach. Note
  the limit of this control: security groups do **not** filter link-local traffic, so an egress rule
  cannot protect the container credential endpoint at `169.254.170.2` or the EC2 instance metadata
  service at `169.254.169.254`. Those are served locally and never traverse the ENI where the security
  group applies.
- **Require IMDSv2** if you run the gateway on EC2 or ECS-on-EC2, so instance metadata cannot be read
  with a plain `GET`. On ECS-on-EC2 you can also block container access to instance metadata entirely
  with `ECS_AWSVPC_BLOCK_IMDS=true` in `/etc/ecs/ecs.config`. Neither setting applies to Fargate.

### If you deployed an earlier version

Images built before the image URL checks pass the caller-supplied URL straight to the HTTP client with no
validation. Images built from the first version of the checks (before 2026-10-10 in this fork) can be bypassed
with URLs such as `http://169.254.169.254\@example.com/`, and are open to DNS rebinding. If you run either,
treat it as exposed to the SSRF vector described above.

**Redeploy** from the current version, or rebuild your container image from the current source. See
[How to upgrade?](../README.md#how-to-upgrade) in the README.

Redeploying is the mitigation, not merely the preferred one. Be aware that on Fargate there is no
network-level workaround: as noted above, security groups cannot filter the link-local credential
endpoint, and `ECS_AWSVPC_BLOCK_IMDS` is not available on Fargate. The environment variables in this
section only exist in the hardened code, so they cannot mitigate an older image either.

Until you can redeploy, the controls that do still apply are:

- Confirm the task role carries no permissions beyond Bedrock invocation, so any credentials reachable
  through the vector are of limited value.
- Treat the API key as the only thing standing between a caller and this vector: rotate it if it may have
  been shared beyond its intended users, and restrict who can reach the ALB or API Gateway endpoint.

Note that you cannot determine retrospectively whether the vector was probed. The gateway does not log
request bodies, so an older image leaves no record of the `image_url` values it fetched; absence of log
evidence is not evidence that nothing was attempted. After upgrading, rejected URLs are recorded as
warnings in CloudWatch, so probing becomes visible from that point on.

---

By following the steps outlined in this guide, you can configure a secure environment that uses HTTPS via ALB for encrypted traffic.